import pytest
from eth_account.messages import encode_defunct
from tests.test_account_import_music import published
from tests.test_account_import_execution import ready
from tests.test_account_import import inventory, platform, keys, token, PREPARE, challenge_request
import app
import account_import_preferences as preferences


@pytest.fixture
def saved(published):
    _, _, account, signer = published[:4]
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("INSERT INTO music_publication_likes VALUES ('legacy-song',?,10)", (signer.address.lower(),))
            conn.execute("INSERT INTO music_library_saves VALUES (?,'legacy-song',20)", (signer.address.lower(),))
            conn.execute("UPDATE music_publications SET like_count=1 WHERE id='legacy-song'")
    return account, signer.address.lower()


@pytest.mark.parametrize("kind", ["likes", "saves"])
def test_selected_preference_transfer_preserves_date_and_requires_outer_transaction(saved, kind):
    account, wallet = saved
    with app.app.app_context():
        conn = app.get_db()
        reviewed = preferences.review(conn, account, wallet, kind, ["legacy-song"])
        assert reviewed[0]["already_in_account"] is False
        assert preferences.inventory(conn, account, wallet, kind, 50, 0)[1] == 1
        with pytest.raises(RuntimeError):
            preferences.transfer(conn, account, wallet, kind, reviewed)
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            preferences.transfer(conn, account, wallet, kind, reviewed)
        source, destination, _ = preferences.TABLES[kind]
        assert conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0] == 0
        assert conn.execute(f"SELECT created_at FROM {destination} WHERE account_id=?", (account,)).fetchone()[0] == (10 if kind == "likes" else 20)
        assert conn.execute("SELECT like_count FROM music_publications WHERE id='legacy-song'").fetchone()[0] == 1


def test_deduplicates_existing_account_like_without_inflating_count(saved):
    account, wallet = saved
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("INSERT INTO account_music_likes VALUES (?,'legacy-song',5)", (account,))
            conn.execute("UPDATE music_publications SET like_count=2 WHERE id='legacy-song'")
        reviewed = preferences.review(conn, account, wallet, "likes", ["legacy-song"])
        assert reviewed[0]["already_in_account"]
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            preferences.transfer(conn, account, wallet, "likes", reviewed)
        assert conn.execute("SELECT like_count FROM music_publications WHERE id='legacy-song'").fetchone()[0] == 1
        assert conn.execute("SELECT created_at FROM account_music_likes").fetchone()[0] == 5


def test_outer_failure_rolls_back_both_preferences_and_public_count(saved):
    account, wallet = saved
    with app.app.app_context():
        conn = app.get_db()
        reviewed = preferences.review(conn, account, wallet, "likes", ["legacy-song"])
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError), conn:
            preferences.transfer(conn, account, wallet, "likes", reviewed)
            raise RuntimeError("receipt failed")
        assert conn.execute("SELECT COUNT(*) FROM music_publication_likes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_music_likes").fetchone()[0] == 0
        assert conn.execute("SELECT like_count FROM music_publications WHERE id='legacy-song'").fetchone()[0] == 1


@pytest.mark.parametrize("mutation", ["DELETE FROM music_library_saves", "UPDATE music_library_saves SET saved_at=21",
    "UPDATE music_publications SET state='unpublished'", "INSERT INTO account_music_saves SELECT id,'legacy-song',5 FROM accounts"])
def test_changed_source_or_target_cannot_use_previous_review(saved, mutation):
    account, wallet = saved
    with app.app.app_context():
        conn = app.get_db()
        reviewed = preferences.review(conn, account, wallet, "saves", ["legacy-song"])
        with conn:
            conn.execute(mutation)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(preferences.PreferenceImportError), conn:
            preferences.transfer(conn, account, wallet, "saves", reviewed)


def test_preferences_only_signed_http_import_is_atomic_and_retryable(saved, published):
    harness, headers, account, signer = published[:4]
    preview = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers).json
    assert preview["like_total"] == preview["save_total"] == 1
    result = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "preferences-only"},
        json={"job_ids": [], "include_credits": False, "like_ids": ["legacy-song"], "save_ids": ["legacy-song"]})
    assert result.status_code == 201, result.json
    snapshot = result.json
    assert snapshot["scope"] == ["music_likes", "music_saves"]
    challenge = challenge_request(harness, headers, snapshot).json
    assert 'like_ids: ["legacy-song"]' in challenge["message"]
    assert 'save_ids: ["legacy-song"]' in challenge["message"]
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    path = f'/v2/account/import-snapshots/{snapshot["id"]}/execute'
    body = {"challenge_id": challenge["challenge_id"], "signature": signature, "chain_id": 11155111}
    auth = {**headers, "Origin": "https://joinhavn.io"}
    with app.app.app_context():
        app.get_db().execute("CREATE TRIGGER fail_preference_receipt BEFORE INSERT ON account_import_receipts BEGIN SELECT RAISE(ABORT,'receipt failed'); END")
    failed = harness.client.post(path, headers=auth, json=body)
    assert failed.status_code == 500 and failed.is_json
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM music_publication_likes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM music_library_saves").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_music_likes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM account_music_saves").fetchone()[0] == 0
        assert conn.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (challenge["challenge_id"],)).fetchone()[0] is None
        conn.execute("DROP TRIGGER fail_preference_receipt")
    first = harness.client.post(path, headers=auth, json=body)
    assert first.status_code == 200, first.json
    assert first.json["receipt"]["like_ids"] == first.json["receipt"]["save_ids"] == ["legacy-song"]
    assert harness.client.post(path, headers=auth, json=body).json == first.json
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM music_publication_likes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM music_library_saves").fetchone()[0] == 0
        assert conn.execute("SELECT owner_account_id FROM music_publications WHERE id='legacy-song'").fetchone()[0] is None
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert conn.execute("SELECT like_count FROM music_publications WHERE id='legacy-song'").fetchone()[0] == 1
        assert conn.execute("SELECT created_at FROM account_music_saves WHERE account_id=?", (account,)).fetchone()[0] == 20

import pytest
from eth_account.messages import encode_defunct

from tests.test_account_import_execution import ready, execute
from tests.test_account_import import inventory, platform, keys, token, PREPARE, SELECTION, challenge_request
import app
import account_import


@pytest.fixture
def published(ready):
    with app.app.app_context():
        conn = app.get_db()
        path = app.OUTPUTS_DIR / "legacy-song.mp3"
        path.write_bytes(b"legacy song audio")
        with conn:
            conn.execute("UPDATE jobs SET task_type='TEXT_TO_MUSIC' WHERE id='01-ready'")
            conn.execute("""INSERT INTO artifacts
                (id,job_id,kind,filename,content_type,path,size_bytes,sha256,created_at)
                VALUES ('song-audio','01-ready','audio','legacy-song.mp3','audio/mpeg',?,17,'fixture',1)""", (str(path),))
            conn.execute("""INSERT INTO music_publications
                (id,job_id,audio_artifact_id,creator_wallet,title,cover_art_seed,play_count,like_count,published_at,updated_at)
                VALUES ('legacy-song','01-ready','song-audio',?,'Keep this song','seed',12,3,1,1)""", (ready[3].address.lower(),))
    return ready


def select_publication(published):
    harness, headers, account, signer, _, _, _, principal = published
    selection = {**SELECTION, "publication_ids": ["legacy-song"]}
    response = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "with-song"}, json=selection)
    assert response.status_code == 201, response.json
    snapshot = response.json
    response = challenge_request(harness, headers, snapshot)
    assert response.status_code == 201, response.json
    challenge = response.json
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    return harness, headers, account, signer, snapshot, challenge, signature, principal


def test_publication_is_explicit_and_keeps_its_link_counts_and_account_controls(published, keys):
    harness, headers, account, signer, _, _, _, _ = published
    preview = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers).json
    assert preview["publications"] == [{"id": "legacy-song", "job_id": "01-ready", "title": "Keep this song", "state": "published", "eligible": True}]
    missing = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "missing-song"}, json=SELECTION)
    assert missing.status_code == 409 and missing.json["error"]["code"] == "import_publication_selection_required"
    proof = select_publication(published)
    assert proof[4]["scope"][-1] == "music_publications"
    assert proof[4]["publications"][0]["title"] == "Keep this song"
    assert 'publication_ids: ["legacy-song"]' in proof[5]["message"]
    # Public engagement during review is preserved, not interpreted as an edit
    # to the user's ownership/content selection.
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE music_publications SET play_count=15,like_count=4,updated_at=2")
    receipt = execute(proof)
    assert receipt["publication_ids"] == ["legacy-song"]
    assert execute(proof) == receipt
    public_url = "/music/publications/legacy-song/audio"
    assert harness.client.get(public_url).data == b"legacy song audio"
    publications = harness.client.get("/v2/music/publications", headers=headers).json["publications"]
    assert len(publications) == 1 and publications[0]["id"] == "legacy-song"
    with app.app.app_context():
        conn = app.get_db()
        row = conn.execute("SELECT creator_wallet,owner_account_id,creator_account_id,play_count,like_count FROM music_publications").fetchone()
        assert tuple(row) == (signer.address.lower(), account, account, 15, 4)
        profile = conn.execute("SELECT id FROM account_public_profiles WHERE account_id=?", (account,)).fetchone()[0]
        assert not app.music_discover.unpublish_song("legacy-song", signer.address.lower())["ok"]
    assert harness.client.get(f"/music/creators/{profile}").json["track_count"] == 1
    other = {"Authorization": token(keys, sub="other", sid="other-session")}
    assert harness.client.delete("/v2/music/publications/legacy-song", headers=other).status_code == 404
    assert harness.client.delete("/v2/music/publications/legacy-song", headers=headers).status_code == 200
    assert harness.client.get(public_url).status_code == 404


@pytest.mark.parametrize("mutation", [
    "UPDATE music_publications SET title='Changed title'",
    "UPDATE music_publications SET state='unpublished'",
    "UPDATE music_publications SET creator_wallet='0x2222222222222222222222222222222222222222'",
    "UPDATE music_publications SET audio_artifact_id='different-audio'",
    "DELETE FROM music_publications",
])
def test_changed_publication_cannot_be_imported_under_old_signature(published, mutation):
    proof = select_publication(published)
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute(mutation)
    with pytest.raises(account_import.MigrationError):
        execute(proof)
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert conn.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (proof[5]["challenge_id"],)).fetchone()[0] is None


def test_cannot_select_publication_without_its_job_or_from_another_wallet(published):
    harness, headers, _, _, _, _, _, _ = published
    response = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "unselected-job"},
        json={"job_ids": ["05-purchased"], "include_credits": False, "publication_ids": ["legacy-song"]})
    assert response.status_code == 409
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE music_publications SET creator_wallet='0x2222222222222222222222222222222222222222'")
    response = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "wrong-owner"},
        json={**SELECTION, "publication_ids": ["legacy-song"]})
    assert response.status_code == 409 and response.json["error"]["code"] == "import_publication_unavailable"

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from tests.test_account_import import inventory, platform, keys, token, WALLET, PREPARE, SELECTION, challenge_request
import app
import account_identity
import account_import
import account_ledger


@pytest.fixture
def ready(inventory):
    harness, headers, account, other_headers = inventory
    signer = Account.create()
    wallet = signer.address.lower()
    with app.app.app_context():
        conn = app.get_db()
        app.music_discover.init_music_discover_tables(conn)
        app.stripe_payments.init_stripe_tables(conn)
        with conn:
            for table, column in [("jobs", "wallet"), ("credits", "wallet"), ("wallet_links", "wallet"),
                                  ("gallery_listings", "owner_wallet"), ("gallery_listings", "seller_wallet")]:
                conn.execute(f"UPDATE {table} SET {column}=? WHERE LOWER({column})=?", (wallet, WALLET))
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    challenge = challenge_request(harness, headers, snapshot).json
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    principal = account_identity.VerifiedPrincipal("https://auth.example", "user_alice", "sess_alice")
    return harness, headers, account, signer, snapshot, challenge, signature, principal


def execute(ready, **overrides):
    _, _, _, _, snapshot, challenge, signature, principal = ready
    args = {"challenge_id": challenge["challenge_id"], "signature": signature,
            "origin": "https://joinhavn.io", "chain_id": 11155111, **overrides}
    with app.app.app_context():
        return account_import.execute(app.get_db(), principal, snapshot["id"], **args)


def test_real_signed_import_moves_ownership_and_credit_once(ready):
    harness, headers, account, signer, snapshot, challenge, _, _ = ready
    receipt = execute(ready)
    assert receipt["credit_units"] == 2125
    assert {job["id"] for job in receipt["jobs"]} == {"01-ready", "05-purchased"}
    assert execute(ready) == receipt
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT owner_account_id,creator_account_id FROM jobs WHERE id='01-ready'").fetchone()[:] == (account, account)
        assert conn.execute("SELECT owner_account_id,creator_account_id FROM jobs WHERE id='05-purchased'").fetchone()[:] == (account, None)
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='08-failed'").fetchone()[0] is None
        assert conn.execute("SELECT balance FROM credits WHERE wallet=?", (signer.address.lower(),)).fetchone()[0] == 0
        assert account_ledger.balance(conn, account)["settled_units"] == 12125
        assert conn.execute("SELECT COUNT(*) FROM account_import_receipts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_audit_events WHERE operation='legacy_import'").fetchone()[0] == 1
        assert conn.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (challenge["challenge_id"],)).fetchone()[0] is not None
        # Legacy gallery reads can no longer reveal/manage the acquired job.
        listing = conn.execute("SELECT id FROM gallery_listings WHERE job_id='05-purchased'").fetchone()[0]
        assert app.gallery.get_listing(listing) is None
    assert harness.client.get("/v2/jobs/01-ready", headers=headers).status_code == 200
    assert harness.client.get("/v1/jobs/01-ready", headers=harness.owner_headers).status_code == 404


@pytest.mark.parametrize("overrides", [{"signature": "invalid"}, {"chain_id": 1}, {"origin": "https://evil.example"}, {"challenge_id": "missing"}])
def test_invalid_proof_never_consumes_or_transfers(ready, overrides):
    with pytest.raises(account_import.MigrationError):
        execute(ready, **overrides)
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None
        assert app.get_db().execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None


def test_signature_from_another_wallet_is_not_authorization(ready):
    signature = Account.create().sign_message(encode_defunct(text=ready[5]["message"])).signature.hex()
    with pytest.raises(account_import.MigrationError, match="invalid_import_signature"):
        execute(ready, signature=signature)


def test_mutation_between_signature_recovery_and_write_lock_is_rejected(ready, monkeypatch):
    recover = Account.recover_message

    def recover_then_change(*args, **kwargs):
        result = recover(*args, **kwargs)
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET data='{}' WHERE id='01-ready'")
        return result

    monkeypatch.setattr(Account, "recover_message", recover_then_change)
    with pytest.raises(account_import.MigrationError, match="import_snapshot_changed"):
        execute(ready)
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None


def test_late_receipt_failure_rolls_back_signature_jobs_and_both_balances(ready):
    with app.app.app_context():
        app.get_db().execute("""CREATE TRIGGER fail_import_receipt BEFORE INSERT ON account_import_receipts
            BEGIN SELECT RAISE(ABORT, 'receipt unavailable'); END""")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError, match="receipt unavailable"):
        execute(ready)
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 2.125
        assert account_ledger.balance(conn, ready[2])["settled_units"] == 10000
        assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 0
        conn.execute("DROP TRIGGER fail_import_receipt")
    assert execute(ready)["credit_units"] == 2125


def test_unlink_and_session_changes_cannot_execute(ready):
    changed = (*ready[:-1], account_identity.VerifiedPrincipal("https://auth.example", "user_alice", "other-session"))
    with pytest.raises(account_import.MigrationError, match="invalid_import_proof"):
        execute(changed)
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=2")
    with pytest.raises(account_import.MigrationError, match="import_snapshot_not_found"):
        execute(ready)


def test_legacy_payment_provenance_requires_its_own_migration(ready):
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("""INSERT INTO stripe_payments(stripe_session_id,wallet,package_id,credits_amount,price_cents,status,created_at)
                VALUES ('historical-checkout',?,'starter',50,500,'completed',1)""", (ready[3].address.lower(),))
    with pytest.raises(account_import.MigrationError, match="import_payment_provenance_required"):
        execute(ready)
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None


def test_publication_dependency_is_not_silently_left_behind(ready):
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("""INSERT INTO music_publications
                (id,job_id,audio_artifact_id,creator_wallet,title,cover_art_seed,published_at,updated_at)
                VALUES ('pub','01-ready','artifact',?,'A song','seed',1,1)""", (ready[3].address.lower(),))
    with pytest.raises(account_import.MigrationError, match="import_publication_migration_required"):
        execute(ready)
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None


def test_real_concurrent_signature_retries_share_one_receipt(ready):
    import concurrent.futures
    import sqlite3
    import threading
    with app.app.app_context():
        path = app.get_db().execute("PRAGMA database_list").fetchone()[2]
    barrier = threading.Barrier(4)

    def retry(_):
        conn = sqlite3.connect(path, timeout=10)
        try:
            barrier.wait(timeout=10)
            return account_import.execute(conn, ready[7], ready[4]["id"], challenge_id=ready[5]["challenge_id"],
                signature=ready[6], origin="https://joinhavn.io", chain_id=11155111)
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(retry, range(4)))
    assert all(receipt == receipts[0] for receipt in receipts)
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM account_import_receipts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 1
        assert account_ledger.balance(conn, ready[2])["settled_units"] == 12125


@pytest.fixture
def imported_artwork(ready):
    from PIL import Image
    harness, headers, account, signer, _, _, _, principal = ready
    with app.app.app_context():
        path = app.OUTPUTS_DIR / "legacy-import.png"
        Image.new("RGB", (24, 24), "blue").save(path)
        conn = app.get_db()
        with conn:
            conn.execute("""INSERT INTO artifacts
                (id,job_id,kind,filename,content_type,path,size_bytes,sha256,created_at)
                VALUES ('legacy-artifact','01-ready','image','legacy-import.png','image/png',?,?,'fixture',1)""",
                (str(path), path.stat().st_size))
    snapshot = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "with-artifact"}, json=SELECTION).json
    challenge = challenge_request(harness, headers, snapshot).json
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    proof = harness, headers, account, signer, snapshot, challenge, signature, principal
    execute(proof)
    return proof


def test_imported_artwork_can_publish_sell_and_relist_without_fictitious_generation_charge(imported_artwork, keys):
    harness, headers, seller, _, snapshot, _, _, _ = imported_artwork
    body = {"job_id": "01-ready", "artifact_id": "legacy-artifact", "title": "Imported blue", "price_units": 500}
    listed = harness.client.post("/v2/marketplace/listings", headers=headers, json=body)
    assert listed.status_code == 201, listed.json
    listing_id = listed.json["listing_id"]
    assert harness.client.get(f"/v2/marketplace/listings/{listing_id}/preview").status_code == 200
    original = "/v2/artifacts/legacy-artifact/content"
    assert harness.client.get(original, headers=headers).status_code == 200
    assert harness.client.get("/static/outputs/legacy-import.png").status_code == 404
    buyer_headers = {"Authorization": token(keys, sub="buyer", sid="buyer-session"), "Idempotency-Key": "buy-import"}
    buyer = harness.client.get("/v2/account", headers=buyer_headers).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, buyer, 1000, payment_id="buyer-checkout")
    sold = harness.client.post(f"/v2/marketplace/listings/{listing_id}/purchase", headers=buyer_headers,
                              json={"expected_price_units": 500})
    assert sold.status_code == 200, sold.json
    assert harness.client.get(original, headers=buyer_headers).status_code == 200
    assert harness.client.get(original, headers=headers).status_code == 404
    relisted = harness.client.post("/v2/marketplace/listings", headers={**buyer_headers, "Idempotency-Key": "relist"}, json=body)
    assert relisted.status_code == 201, relisted.json
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM account_credit_reservations WHERE job_id='01-ready'").fetchone()[0] == 0
        assert tuple(conn.execute("SELECT creator_account_id,owner_account_id FROM jobs WHERE id='01-ready'").fetchone()) == (seller, buyer)
        assert tuple(conn.execute("SELECT snapshot_id,account_id FROM account_import_job_transfers WHERE job_id='01-ready'").fetchone()) == (snapshot["id"], seller)


def test_import_provenance_cannot_override_uncaptured_account_charge(imported_artwork):
    harness, headers, account, _, _, _, _, _ = imported_artwork
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.reserve_in_transaction(conn, account, 1000, job_id="01-ready")
    body = {"job_id": "01-ready", "artifact_id": "legacy-artifact", "title": "Imported blue", "price_units": 500}
    response = harness.client.post("/v2/marketplace/listings", headers=headers, json=body)
    assert response.status_code == 409 and response.json["error"]["code"] == "marketplace_unsettled"


def test_import_job_provenance_is_immutable_and_initialization_replays_safely(imported_artwork):
    import sqlite3
    with app.app.app_context():
        conn = app.get_db()
        account_import.initialize(conn)
        assert conn.execute("SELECT COUNT(*) FROM account_import_job_transfers").fetchone()[0] == 2
        for query in ("DELETE FROM account_import_job_transfers", "UPDATE account_import_job_transfers SET creator_wallet='forged'"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"), conn:
                conn.execute(query)


def test_older_signed_receipts_backfill_provenance_without_wallet_inference(imported_artwork):
    with app.app.app_context():
        conn = app.get_db()
        before = [tuple(row) for row in conn.execute("SELECT * FROM account_import_job_transfers ORDER BY job_id")]
        # Simulate upgrading the prior schema, which has signed receipts but no
        # indexed transfer table. This is confined to the temporary test DB.
        conn.execute("DROP TABLE account_import_job_transfers")
        account_import.initialize(conn)
        assert [tuple(row) for row in conn.execute("SELECT * FROM account_import_job_transfers ORDER BY job_id")] == before
        assert conn.execute("SELECT COUNT(*) FROM account_import_job_transfers WHERE job_id='08-failed'").fetchone()[0] == 0

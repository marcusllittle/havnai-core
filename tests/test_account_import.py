import pytest
import json
import sqlite3
from tests.test_account_jobs import platform, keys, token
import app
import account_import

WALLET = "0x" + "1" * 40
OTHER = "0x" + "2" * 40


@pytest.fixture
def inventory(platform, keys):
    harness, headers, account = platform
    other_headers = {"Authorization": token(keys, sub="other", sid="other-session")}
    other = harness.client.get("/v2/account", headers=other_headers).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        app.gallery.init_gallery_tables(conn)
        with conn:
            conn.execute("INSERT INTO wallet_links(id,account_id,wallet,verified_at,linked_at) VALUES ('link-import',?,?,1,1)", (account, WALLET))
            for job_id, wallet, status, owner in [
                ("01-ready", WALLET, "completed", None), ("02-running", WALLET, "running", None),
                ("03-account", WALLET, "completed", other), ("04-sold-away", WALLET, "completed", None),
                ("05-purchased", OTHER, "completed", None), ("06-listed", WALLET, "completed", None),
                ("07-unrelated", OTHER, "completed", None), ("08-failed", WALLET, "failed", None),
                ("09-unknown", WALLET, "custom_state", None),
            ]:
                conn.execute("""INSERT INTO jobs(id,wallet,model,task_type,status,timestamp,weight,owner_account_id,data)
                    VALUES (?,?,'model','IMAGE_GEN',?,1,1,?,'{"prompt":"private"}')""", (job_id, wallet, status, owner))
        first = app.gallery.create_listing("04-sold-away", WALLET, "Sold", 1)
        app.gallery.purchase_listing(first["id"], OTHER)
        bought = app.gallery.create_listing("05-purchased", OTHER, "Bought", 1)
        app.gallery.purchase_listing(bought["id"], WALLET)
        app.gallery.create_listing("06-listed", WALLET, "Active", 1)
        app.credits.deposit_credits(WALLET, 2.125)
    return harness, headers, account, other_headers


def test_preview_is_read_only_and_resolves_current_gallery_owner(inventory):
    harness, headers, account, _ = inventory
    with app.app.app_context():
        conn = app.get_db()
        before = conn.total_changes
        result = account_import.preview(conn, account, "link-import")
        assert conn.total_changes == before
        assert not conn.in_transaction
        assert conn.execute("SELECT balance FROM credits WHERE wallet=?", (WALLET,)).fetchone()[0] == 2.125
    rows = {row["id"]: row for row in result["jobs"]}
    assert result["read_only"] and result["confirmation_required"]
    assert result["total"] == 8 and result["eligible_count"] == 3
    assert rows["01-ready"]["eligible"] and rows["05-purchased"]["eligible"] and rows["08-failed"]["eligible"]
    assert rows["02-running"]["exclusion"] == "job_not_final"
    assert rows["03-account"]["exclusion"] == "already_account_owned"
    assert rows["04-sold-away"]["exclusion"] == "not_current_owner"
    assert rows["06-listed"]["exclusion"] == "active_listing"
    assert rows["09-unknown"]["exclusion"] == "job_not_final"
    assert "07-unrelated" not in rows
    assert result["credits"] == {"available_units": 2125, "scale": 1000, "exclusion": None}
    assert all(set(row) == {"id", "status", "type", "created_at", "eligible", "exclusion"} for row in rows.values())
    response = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.json == result


def test_preview_is_scoped_to_active_link_and_session(inventory):
    harness, headers, _, other_headers = inventory
    path = "/v2/account/wallet-links/link-import/import-preview"
    assert harness.client.get(path).status_code == 401
    assert harness.client.get(path, headers=harness.owner_headers).status_code == 401
    assert harness.client.get(path + "?wallet=" + WALLET, headers=other_headers).status_code == 404
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=2 WHERE id='link-import'")
    assert harness.client.get(path, headers=headers).status_code == 404


def test_preview_paginates_without_changing_totals(inventory):
    harness, headers, _, _ = inventory
    path = "/v2/account/wallet-links/link-import/import-preview"
    pages = [harness.client.get(path + f"?limit=3&offset={offset}", headers=headers).json for offset in (0, 3, 6, 9)]
    assert [len(page["jobs"]) for page in pages] == [3, 3, 2, 0]
    assert all(page["total"] == 8 and page["eligible_count"] == 3 for page in pages)
    assert len({job["id"] for page in pages for job in page["jobs"]}) == 8
    for query in ("limit=0", "limit=101", "offset=-1", "offset=no"):
        assert harness.client.get(path + "?" + query, headers=headers).status_code == 422


@pytest.mark.parametrize("balance,reason", [(1.00001, "balance_precision_review"), (1e14, "invalid_wallet_balance"), (float("inf"), "invalid_wallet_balance")])
def test_preview_never_rounds_or_overflows_legacy_credits(inventory, balance, reason):
    harness, headers, _, _ = inventory
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE credits SET balance=? WHERE wallet=?", (balance, WALLET))
    result = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers).json
    assert result["credits"]["available_units"] is None
    assert result["credits"]["exclusion"] == reason


def test_ambiguous_case_variant_balances_require_review(inventory):
    harness, headers, _, _ = inventory
    # A checksum/case variant is the same EVM identity, but must not silently
    # duplicate or add two independently stored historical credit records.
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("INSERT INTO credits(wallet,balance,updated_at) VALUES (?,3,1)", ("0X" + WALLET[2:],))
    response = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers)
    assert response.json["credits"] == {"available_units": None, "scale": 1000, "exclusion": "ambiguous_wallet_balance"}


def test_new_sale_invalidates_preview_eligibility_without_reassigning_job(inventory):
    harness, headers, _, _ = inventory
    with app.app.app_context():
        created = app.gallery.create_listing("01-ready", WALLET, "New sale", 1)
        app.gallery.purchase_listing(created["id"], OTHER)
    response = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers).json
    assert next(job for job in response["jobs"] if job["id"] == "01-ready")["exclusion"] == "not_current_owner"
    assert response["eligible_count"] == 2
    with app.app.app_context():
        assert app.get_db().execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None


PREPARE = "/v2/account/wallet-links/link-import/import-snapshots"
SELECTION = {"job_ids": ["05-purchased", "01-ready"], "include_credits": True}


def test_snapshot_is_exact_immutable_and_does_not_move_resources(inventory):
    harness, headers, account, _ = inventory
    response = harness.client.post(PREPARE, headers=headers, json=SELECTION)
    assert response.status_code == 201, response.json
    snapshot = response.json
    assert not snapshot["transfer_authorized"]
    assert snapshot["account_id"] == account
    assert snapshot["expires_at"] - snapshot["created_at"] == 300
    assert [job["id"] for job in snapshot["jobs"]] == ["01-ready", "05-purchased"]
    assert snapshot["credits"]["available_units"] == 2125
    assert snapshot["scope"] == ["generation_history", "available_credits"]
    assert snapshot["digest"] == account_import._digest({key: value for key, value in snapshot.items()
        if key not in {"digest", "transfer_authorized"}})
    assert "prompt" not in json.dumps(snapshot)
    path = "/v2/account/import-snapshots/" + snapshot["id"]
    assert harness.client.get(path, headers=headers).json == snapshot
    assert response.headers["Cache-Control"] == "private, no-store"
    # Exact retries recover the original selection; later inventory never leaks
    # into the user's already prepared snapshot.
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET data='{}' WHERE id='01-ready'")
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert conn.execute("SELECT balance FROM credits WHERE wallet=?", (WALLET,)).fetchone()[0] == 2.125
        for query in ("UPDATE account_import_snapshots SET expires_at=0", "DELETE FROM account_import_snapshots"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"), conn:
                conn.execute(query)
    retry = harness.client.post(PREPARE, headers=headers,
        json={**SELECTION, "job_ids": list(reversed(SELECTION["job_ids"]))})
    assert retry.json == snapshot
    fresh = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "fresh"}, json=SELECTION).json
    assert fresh["jobs"][0]["state_digest"] != snapshot["jobs"][0]["state_digest"]
    assert harness.client.get(path, headers=headers).json == snapshot
    conflict = harness.client.post(PREPARE, headers=headers, json={**SELECTION, "include_credits": False})
    assert conflict.status_code == 409 and conflict.json["error"]["code"] == "idempotency_conflict"


def test_snapshot_access_requires_original_session_and_active_link(inventory, keys):
    harness, headers, _, other_headers = inventory
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    path = "/v2/account/import-snapshots/" + snapshot["id"]
    changed_session = {**headers, "Authorization": token(keys, sid="new-session")}
    assert harness.client.get(path).status_code == 401
    assert harness.client.get(path, headers=harness.owner_headers).status_code == 401
    assert harness.client.get(path, headers=other_headers).status_code == 404
    assert harness.client.get(path, headers=changed_session).status_code == 404
    assert harness.client.post(PREPARE, headers=changed_session, json=SELECTION).status_code == 404
    assert harness.client.post(PREPARE, headers={**other_headers, "Idempotency-Key": "other"}, json=SELECTION).status_code == 404
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=2 WHERE id='link-import'")
    assert harness.client.get(path, headers=headers).status_code == 404
    assert harness.client.post(PREPARE, headers=headers, json=SELECTION).status_code == 404


def test_snapshot_requires_recent_auth_and_expires_without_mutation(inventory, keys, monkeypatch):
    harness, headers, _, _ = inventory
    stale_headers = {**headers, "Authorization": token(keys, fva=[6, -1])}
    assert harness.client.post(PREPARE, headers=stale_headers, json=SELECTION).status_code == 403
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    # Patch this module's clock only; token verification still uses real time.
    from types import SimpleNamespace
    monkeypatch.setattr(account_import, "time", SimpleNamespace(time=lambda: snapshot["expires_at"]))
    path = "/v2/account/import-snapshots/" + snapshot["id"]
    expired = harness.client.get(path, headers=headers)
    assert expired.status_code == 409 and expired.json["error"]["code"] == "import_snapshot_expired"
    assert harness.client.post(PREPARE, headers=headers, json=SELECTION).status_code == 409


@pytest.mark.parametrize("ids", [["missing"], ["04-sold-away"], ["02-running"], ["03-account"], ["06-listed"], ["07-unrelated"], ["01-ready", "missing"]])
def test_snapshot_rejects_entire_selection_if_any_resource_is_ineligible(inventory, ids):
    harness, headers, _, _ = inventory
    response = harness.client.post(PREPARE, headers=headers, json={**SELECTION, "job_ids": ids})
    assert response.status_code == 409
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_import_snapshots").fetchone()[0] == 0


@pytest.mark.parametrize("selection", [None, [], {}, {"job_ids": [], "include_credits": False},
    {"job_ids": ["01-ready", "01-ready"], "include_credits": True},
    {"job_ids": [{}], "include_credits": True}, {"job_ids": ["01-ready"], "include_credits": 1},
    {**SELECTION, "wallet": OTHER}, {"job_ids": [str(i) for i in range(101)], "include_credits": False}])
def test_snapshot_rejects_ambiguous_or_unbounded_selection(inventory, selection):
    harness, headers, _, _ = inventory
    assert harness.client.post(PREPARE, headers=headers, json=selection).status_code == 422


def test_snapshot_allows_independent_jobs_and_credit_scope(inventory):
    harness, headers, _, _ = inventory
    response = harness.client.post(PREPARE, headers=headers, json={"job_ids": [], "include_credits": True})
    assert response.status_code == 201, response.json
    assert response.json["jobs"] == [] and response.json["scope"] == ["available_credits"]
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE credits SET balance=1.00001 WHERE wallet=?", (WALLET,))
    headers = {**headers, "Idempotency-Key": "jobs-only"}
    jobs = harness.client.post(PREPARE, headers=headers, json={**SELECTION, "include_credits": False})
    assert jobs.status_code == 201
    assert jobs.json["credits"] is None and jobs.json["scope"] == ["generation_history"]
    rejected = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "bad-credits"}, json=SELECTION)
    assert rejected.status_code == 409 and rejected.json["error"]["code"] == "import_credits_require_review"


def challenge_request(harness, headers, snapshot):
    return harness.client.post("/v2/account/import-snapshots/" + snapshot["id"] + "/challenge",
        headers={**headers, "Origin": "https://joinhavn.io"}, json={"chain_id": 11155111})


def test_import_challenge_binds_selection_and_reuses_nonce_without_transfer(inventory):
    harness, headers, account, _ = inventory
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    response = challenge_request(harness, headers, snapshot)
    assert response.status_code == 201, response.json
    challenge = response.json
    for field in ("purpose: legacy_import", "origin: https://joinhavn.io", "chain_id: 11155111",
                  "account_id: " + account, "wallet: " + WALLET, "snapshot_digest: " + snapshot["digest"],
                  "session_binding: " + snapshot["session_binding"], 'job_ids: ["01-ready","05-purchased"]',
                  "credit_units: 2125", "credit_scale: 1000", "nonce: " + challenge["challenge_id"]):
        assert field in challenge["message"]
    assert challenge["expires_at"] == snapshot["expires_at"]
    assert challenge_request(harness, headers, snapshot).json == challenge
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM account_import_challenges").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_wallet_challenges").fetchone()[0] == 0
        assert conn.execute("SELECT balance FROM credits WHERE wallet=?", (WALLET,)).fetchone()[0] == 2.125
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None


@pytest.mark.parametrize("mutation", [
    "UPDATE jobs SET data='{}' WHERE id='01-ready'",
    "UPDATE jobs SET status='running' WHERE id='01-ready'",
    "UPDATE gallery_listings SET title='Changed' WHERE job_id='05-purchased'",
    "UPDATE gallery_listings SET owner_wallet='0x2222222222222222222222222222222222222222' WHERE job_id='05-purchased'",
    "UPDATE credits SET balance=balance+1",
    "UPDATE credits SET updated_at=updated_at+1",
    "UPDATE credits SET balance=1.00001",
    "INSERT INTO artifacts(id,job_id,kind,filename,content_type,path,size_bytes,sha256,created_at) VALUES ('new-artifact','01-ready','image','new.png','image/png','/private/new.png',1,'abc',1)",
])
def test_changed_snapshot_cannot_issue_or_recover_confirmation(inventory, mutation):
    harness, headers, _, _ = inventory
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    assert challenge_request(harness, headers, snapshot).status_code == 201
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute(mutation)
    response = challenge_request(harness, headers, snapshot)
    assert response.status_code == 409
    assert response.json["error"]["code"] == "import_snapshot_changed"


def test_unselected_changes_do_not_expand_or_invalidate_confirmation(inventory):
    harness, headers, _, _ = inventory
    snapshot = harness.client.post(PREPARE, headers=headers, json={**SELECTION, "include_credits": False}).json
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET status='completed',data='{}' WHERE id='02-running'")
            conn.execute("UPDATE credits SET balance=balance+1")
    response = challenge_request(harness, headers, snapshot)
    assert response.status_code == 201
    assert "02-running" not in response.json["message"]
    assert "credit_units: 0" in response.json["message"]


def test_challenge_enforces_session_origin_recent_auth_and_chain(inventory, keys):
    harness, headers, _, other_headers = inventory
    snapshot = harness.client.post(PREPARE, headers=headers, json=SELECTION).json
    path = "/v2/account/import-snapshots/" + snapshot["id"] + "/challenge"
    assert challenge_request(harness, other_headers, snapshot).status_code == 404
    for claims, status in [({"sid": "another-session"}, 404), ({"fva": [6, -1]}, 403)]:
        assert challenge_request(harness, {**headers, "Authorization": token(keys, **claims)}, snapshot).status_code == status
    assert harness.client.post(path, headers=headers, json={"chain_id": 11155111}).status_code == 403
    assert harness.client.post(path, headers={**headers, "Origin": "https://evil.example"}, json={"chain_id": 11155111}).status_code == 403
    for body in ({"chain_id": True}, {"chain_id": "11155111"}, {"chain_id": 137}, {"chain_id": 1, "digest": "spoof"}):
        assert harness.client.post(path, headers={**headers, "Origin": "https://joinhavn.io"}, json=body).status_code == 422
    assert challenge_request(harness, headers, snapshot).status_code == 201
    assert harness.client.post(path, headers={**headers, "Origin": "https://joinhavn.io"}, json={"chain_id": 1}).status_code == 409
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=2 WHERE id='link-import'")
    assert challenge_request(harness, headers, snapshot).status_code == 404

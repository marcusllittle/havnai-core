from types import SimpleNamespace

import pytest

from tests.test_account_import_execution import ready, execute
from tests.test_account_import import inventory, platform, keys, token
import app
import account_import

PATH = "/v2/account/import-receipts"


def test_receipt_recovers_after_lost_response_expiry_unlink_and_new_session(ready, keys, monkeypatch):
    harness, headers, _, _, snapshot, _, _, _ = ready
    result = execute(ready)
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=5")
    monkeypatch.setattr(account_import, "time", SimpleNamespace(time=lambda: snapshot["expires_at"] + 1000))
    auth = {"Authorization": token(keys, sid="new-session", fva=[60, -1])}
    recovered = harness.client.get(PATH + "/" + snapshot["id"], headers=auth)
    assert recovered.status_code == 200, recovered.json
    assert recovered.json == {"receipt": result, "scale": 1000}
    assert recovered.headers["Cache-Control"] == "private, no-store"
    summary = harness.client.get(PATH, headers=auth).json
    assert summary["total"] == 1 and summary["scale"] == 1000
    assert summary["receipts"] == [{"id": result["id"], "created_at": result["created_at"],
        "job_count": 2, "publication_count": 0, "playlist_count": 0, "credit_units": 2125}]
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 1


def test_receipts_are_account_private_and_pending_snapshot_is_not_a_receipt(ready, keys):
    harness, headers, account, _, snapshot, _, _, _ = ready
    path = PATH + "/" + snapshot["id"]
    assert harness.client.get(path, headers=headers).status_code == 404
    execute(ready)
    for auth in ({}, harness.owner_headers):
        assert harness.client.get(path, headers=auth).status_code == 401
        assert harness.client.get(PATH, headers=auth).status_code == 401
    other = {"Authorization": token(keys, sub="other", sid="other-session")}
    assert harness.client.get(path, headers=other).status_code == 404
    assert harness.client.get(PATH, headers=other).json["receipts"] == []
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (account,))
    assert harness.client.get(path, headers=headers).status_code == 403
    assert harness.client.get(PATH, headers=headers).status_code == 403


def test_receipt_pagination_and_input_bounds(ready):
    harness, headers, _, _, _, _, _, _ = ready
    execute(ready)
    page = harness.client.get(PATH + "?limit=1&offset=0", headers=headers).json
    assert len(page["receipts"]) == 1 and page["total"] == 1
    end = harness.client.get(PATH + "?limit=1&offset=1", headers=headers).json
    assert end["receipts"] == [] and end["total"] == 1
    for query in ("limit=0", "limit=101", "limit=no", "offset=-1", "offset=1000001"):
        assert harness.client.get(PATH + "?" + query, headers=headers).status_code == 422

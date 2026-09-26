import pytest

from tests.test_account_import_execution import ready
from tests.test_account_import import inventory, platform, keys, token, challenge_request
import app


def submit(ready, *, headers=None, body=None):
    harness, auth, _, _, snapshot, challenge, signature, _ = ready
    return harness.client.post("/v2/account/import-snapshots/" + snapshot["id"] + "/execute",
        headers=headers if headers is not None else {**auth, "Origin": "https://joinhavn.io"},
        json=body if body is not None else {"challenge_id": challenge["challenge_id"], "signature": signature, "chain_id": 11155111})


def test_signed_http_execution_replays_and_recovers_receipt(ready):
    harness, headers, _, _, snapshot, _, _, _ = ready
    response = submit(ready)
    assert response.status_code == 200, response.json
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.json["receipt"]["credit_units"] == 2125
    assert submit(ready).json == response.json
    assert harness.client.get("/v2/account/import-receipts/" + snapshot["id"], headers=headers).json == response.json


def test_rollout_switch_blocks_proof_and_execution_but_not_receipt_recovery(ready, monkeypatch):
    harness, headers, _, _, snapshot, _, _, _ = ready
    monkeypatch.delenv("HAVNAI_ACCOUNT_IMPORT_ENABLED")
    caps = harness.client.get("/v2/account/import-capabilities", headers=headers)
    assert caps.status_code == 200 and caps.json["execution_enabled"] is False
    assert submit(ready).status_code == 503
    assert challenge_request(harness, headers, snapshot).status_code == 503
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None
    monkeypatch.setenv("HAVNAI_ACCOUNT_IMPORT_ENABLED", "1")
    assert submit(ready).status_code == 200
    monkeypatch.delenv("HAVNAI_ACCOUNT_IMPORT_ENABLED")
    assert harness.client.get("/v2/account/import-receipts/" + snapshot["id"], headers=headers).status_code == 200


def test_http_execution_requires_recent_account_auth_and_allowlisted_origin(ready, keys):
    for auth, status in [({}, 401), (ready[0].owner_headers, 401),
                         ({"Authorization": token(keys, fva=[6, -1]), "Origin": "https://joinhavn.io"}, 403),
                         ({**ready[1], "Origin": "https://evil.example"}, 403), (ready[1], 403)]:
        assert submit(ready, headers=auth).status_code == status
    with app.app.app_context():
        assert app.get_db().execute("SELECT used_at FROM account_import_challenges").fetchone()[0] is None


@pytest.mark.parametrize("body", [{}, {"challenge_id": "x", "signature": "x", "chain_id": True},
    {"challenge_id": "x", "signature": "x", "chain_id": 137},
    {"challenge_id": "x", "signature": "x", "chain_id": 1, "account_id": "spoof"}])
def test_http_execution_rejects_ambiguous_client_identity_or_network(ready, body):
    assert submit(ready, body=body).status_code == 422

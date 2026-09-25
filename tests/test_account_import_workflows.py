import sqlite3
import pytest
from eth_account.messages import encode_defunct
from tests.test_account_import_execution import ready, execute
from tests.test_account_import import inventory, platform, keys, token, PREPARE, challenge_request
import app
import account_import


@pytest.fixture
def workflow_proof(ready):
    harness, headers, account, signer, _, _, _, principal = ready
    with app.app.app_context():
        chosen = app.workflows.create_workflow(signer.address.lower(), "My workflow", config={"prompt_template": "private prompt"})
        app.workflows.create_workflow(signer.address.lower(), "Unselected")
        app.workflows.publish_workflow(chosen["id"], signer.address.lower())
    preview = harness.client.get("/v2/account/wallet-links/link-import/import-preview", headers=headers).json
    assert preview["workflow_total"] == 2 and len(preview["workflows"]) == 2
    selection = {"job_ids": [], "include_credits": False, "workflow_ids": [str(chosen["id"])]}
    result = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "workflow-only"}, json=selection)
    assert result.status_code == 201, result.json
    snapshot = result.json
    assert snapshot["scope"] == ["workflows"]
    assert "private prompt" not in str(snapshot)
    challenge = challenge_request(harness, headers, snapshot).json
    assert f'workflow_ids: ["{chosen["id"]}"]' in challenge["message"]
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    return harness, headers, account, signer, snapshot, challenge, signature, principal


def test_signed_workflow_import_preserves_config_sharing_and_retries(workflow_proof):
    receipt = execute(workflow_proof)
    assert execute(workflow_proof) == receipt
    identifier = workflow_proof[4]["workflows"][0]["id"]
    assert receipt["workflow_ids"] == [identifier] and receipt["credit_units"] == 0
    harness, headers, account, signer = workflow_proof[:4]
    with app.app.app_context():
        conn = app.get_db()
        row = conn.execute("SELECT creator_wallet,owner_account_id,creator_account_id,published,config FROM workflow_registry WHERE id=?", (identifier,)).fetchone()
        assert row[:4] == (signer.address.lower(), account, account, 1)
        assert "private prompt" in row[4]
        assert app.workflows.get_workflow(int(identifier)) is None
        assert app.workflows.update_workflow(int(identifier), signer.address.lower(), name="Hijack") is None
        assert conn.execute("SELECT owner_account_id FROM workflow_registry WHERE name='Unselected'").fetchone()[0] is None
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=3")
    assert harness.client.get(f"/v2/account/workflows/{identifier}", headers=headers).status_code == 200
    assert harness.client.get(f"/v2/workflows/{identifier}").status_code == 200
    assert harness.client.get("/v2/account/import-receipts", headers=headers).json["receipts"][0]["workflow_count"] == 1


@pytest.mark.parametrize("assignment", ["config='{}'", "published=0", "name='Changed'", "creator_wallet='other'"])
def test_changed_workflow_rejects_old_signature(workflow_proof, assignment):
    identifier = workflow_proof[4]["workflows"][0]["id"]
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute(f"UPDATE workflow_registry SET {assignment} WHERE id=?", (identifier,))
    with pytest.raises(account_import.MigrationError):
        execute(workflow_proof)
    with app.app.app_context():
        assert app.get_db().execute("SELECT owner_account_id FROM workflow_registry WHERE id=?", (identifier,)).fetchone()[0] is None


def test_receipt_failure_rolls_back_workflow_and_signature(workflow_proof):
    with app.app.app_context():
        app.get_db().execute("CREATE TRIGGER fail_workflow_import BEFORE INSERT ON account_import_receipts BEGIN SELECT RAISE(ABORT,'fail receipt'); END")
    with pytest.raises(sqlite3.IntegrityError):
        execute(workflow_proof)
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT COUNT(*) FROM workflow_registry WHERE owner_account_id IS NOT NULL").fetchone()[0] == 0
        assert conn.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (workflow_proof[5]["challenge_id"],)).fetchone()[0] is None


@pytest.mark.parametrize("ids", [[1], ["01"], ["1", "1"], ["-1"], ["1 OR 1=1"]])
def test_workflow_ids_must_be_unique_canonical_strings(ready, ids):
    result = ready[0].client.post(PREPARE, headers=ready[1], json={"job_ids": [], "include_credits": False, "workflow_ids": ids})
    assert result.status_code == 422


def test_another_wallets_workflow_cannot_be_selected(ready):
    with app.app.app_context():
        other = app.workflows.create_workflow("0x" + "2" * 40, "Other wallet")
    result = ready[0].client.post(PREPARE, headers={**ready[1], "Idempotency-Key": "other-workflow"},
        json={"job_ids": [], "include_credits": False, "workflow_ids": [str(other["id"])]})
    assert result.status_code == 409 and result.json["error"]["code"] == "import_workflow_unavailable"

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from tests import test_platform_v1 as platform_fixture
from tests.test_account_auth import keys, token, config
import app
import account_auth
import account_ledger
import account_lifecycle
from tests.test_account_lifecycle import signed, CONFIG as lifecycle_config
import job_helpers


@pytest.fixture
def platform(keys, monkeypatch):
    harness = platform_fixture.PlatformApiContractTests(methodName="runTest")
    harness.setUp()
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: config(keys))
    headers = {"Authorization": token(keys), "Idempotency-Key": "create-one"}
    alice = harness.client.get("/v2/account", headers=headers).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, alice, 10000, payment_id="test-paid-checkout")
    yield harness, headers, alice
    harness.tearDown()


def create(harness, headers, **extra):
    return harness.client.post("/v2/jobs", headers=headers,
        json={"type": "image", "model": platform_fixture.IMAGE_MODEL, "prompt": "A blue sky", **extra})


def test_revoked_session_cannot_recover_or_submit_studio_jobs(platform):
    harness, headers, _ = platform
    created = create(harness, headers)
    assert created.status_code == 202
    payload, proof = signed("session.revoked", {"id": "sess_alice"})
    with app.app.app_context():
        account_lifecycle.webhook(app.get_db(), payload, proof, config=lifecycle_config)
    assert harness.client.get("/v2/account/jobs", headers=headers).status_code == 403
    assert harness.client.get(f"/v2/jobs/{created.json['id']}", headers=headers).status_code == 403
    assert create(harness, headers).status_code == 403


def test_account_generation_retry_and_recovery_without_wallet(platform):
    harness, headers, alice = platform
    first = create(harness, headers)
    assert first.status_code == 202, first.json
    second = create(harness, headers)
    assert first.json["id"] == second.json["id"]
    assert first.json["wallet"] == ""
    assert first.json["owner_account_id"] == alice
    recovery = harness.client.get("/v2/account/jobs", headers=headers)
    assert recovery.json["count"] == 1
    assert recovery.json["jobs"][0]["id"] == first.json["id"]
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert balance["reserved_units"] == 1000
    assert balance["available_units"] == 9000
    conflict = create(harness, headers, prompt="Different request")
    assert conflict.status_code == 409
    assert conflict.json["error"]["code"] == "idempotency_conflict"


def test_guest_different_account_and_legacy_routes_cannot_access(platform, keys):
    harness, headers, _ = platform
    job = create(harness, headers).json
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.get(f"/v2/jobs/{job['id']}").status_code == 401
    assert harness.client.get(f"/v2/jobs/{job['id']}", headers=bob).status_code == 404
    assert harness.client.post(f"/v2/jobs/{job['id']}/cancel", headers=bob).status_code == 404
    assert harness.client.get("/v2/jobs", headers=bob).json["jobs"] == []
    for path in (f"/jobs/{job['id']}", f"/result/{job['id']}", f"/v1/jobs/{job['id']}"):
        assert harness.client.get(path, headers=harness.owner_headers).status_code == 404
    assert harness.client.post(f"/jobs/{job['id']}/cancel", json={}).status_code == 404
    assert harness.client.get("/v1/jobs", headers=harness.owner_headers).json["jobs"] == []


def test_cannot_forge_identity_or_spend_without_funding(platform, keys):
    harness, headers, _ = platform
    assert create(harness, headers, wallet=platform_fixture.WALLET).status_code == 422
    assert create(harness, headers, owner_account_id="forged").status_code == 422
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob"), "Idempotency-Key": "bob-job"}
    response = create(harness, bob)
    assert response.status_code == 409
    assert response.json["error"]["code"] == "insufficient_credits"
    assert harness.client.get("/v2/jobs", headers=bob).json["jobs"] == []


def test_cancellation_releases_reservation_once(platform):
    harness, headers, _ = platform
    job = create(harness, headers).json
    assert harness.client.post(f"/v2/jobs/{job['id']}/cancel", headers=headers).status_code == 202
    assert harness.client.post(f"/v2/jobs/{job['id']}/cancel", headers=headers).status_code == 409
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert balance["available_units"] == 10000
    assert balance["reserved_units"] == 0


def test_actual_worker_completion_captures_account_charge_once(platform):
    harness, headers, _ = platform
    job = create(harness, headers).json
    with app.app.app_context():
        attempt = job_helpers.assign_job_to_node(job["id"], "node-test")
        assert job_helpers.complete_job(job["id"], "node-test", "succeeded", attempt)
        assert not job_helpers.complete_job(job["id"], "node-test", "succeeded", attempt)
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert balance["available_units"] == 9000
    assert balance["reserved_units"] == 0


def test_source_assets_and_worker_artifacts_are_private(platform, keys):
    harness, headers, _ = platform
    upload = harness.client.post("/v2/assets", headers=headers,
        data={"kind": "image", "file": (io.BytesIO(b"image-fixture"), "source.png")})
    assert upload.status_code == 201
    asset = upload.json
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.get(asset["content_url"], headers=bob).status_code == 404
    assert harness.client.get(asset["content_url"], headers=headers).data == b"image-fixture"
    assert harness.client.get(f"/v1/assets/{asset['id']}/content", headers=harness.owner_headers).status_code == 404
    assert harness.client.get(f"/v1/assets/{asset['id']}/content", headers=harness.node_headers).status_code == 200
    reuse = harness.client.post("/v2/jobs", headers={**bob, "Idempotency-Key": "steal"}, json={
        "type": "image_to_video", "model": platform_fixture.VIDEO_MODEL,
        "prompt": "stolen image", "source_asset_id": asset["id"]})
    assert reuse.status_code == 400
    job = create(harness, headers).json
    with app.app.app_context():
        attempt = job_helpers.assign_job_to_node(job["id"], "node-test")
    artifact = harness.client.post(f"/v1/node/jobs/{job['id']}/artifacts", headers=harness.node_headers,
        data={"node_id": "node-test", "attempt_id": attempt, "kind": "image", "file": (io.BytesIO(b"private-result"), "out.png")})
    assert artifact.status_code == 201, artifact.json
    assert harness.client.get(artifact.json["url"]).status_code == 404
    detail = harness.client.get(f"/v2/jobs/{job['id']}", headers=headers).json
    url = detail["artifacts"][0]["url"]
    assert harness.client.get(url).status_code == 401
    assert harness.client.get(url, headers=bob).status_code == 404
    assert harness.client.get(url, headers=headers).data == b"private-result"

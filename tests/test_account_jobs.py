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
import account_jobs
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


def test_lost_response_retry_recovers_before_changed_model_configuration(platform, monkeypatch):
    harness, headers, _ = platform
    first = create(harness, headers)
    assert first.status_code == 202
    monkeypatch.setattr(app, "get_model_config", lambda name: None)
    retry = create(harness, headers)
    assert retry.status_code == 202
    assert retry.json["id"] == first.json["id"]
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 1000


@pytest.mark.parametrize("job_type,model", [
    ("image", platform_fixture.MUSIC_MODEL),
    ("image", platform_fixture.VIDEO_MODEL),
    ("text_to_music", platform_fixture.IMAGE_MODEL),
    ("text_to_music", platform_fixture.VIDEO_MODEL),
    ("image_to_video", platform_fixture.IMAGE_MODEL),
    ("image_to_video", platform_fixture.MUSIC_MODEL),
])
def test_model_task_mismatch_does_not_enqueue_or_reserve_credits(platform, job_type, model):
    harness, headers, _ = platform
    for route, auth in (("/v2/jobs", headers), ("/v1/jobs", harness.owner_headers)):
        response = harness.client.post(route, headers=auth,
            json={"type": job_type, "model": model, "prompt": "A blue sky"})
        assert response.status_code == 400
        error = response.json["error"]
        assert (error["code"] if isinstance(error, dict) else error) == "model_task_mismatch"
    assert harness.client.get("/v2/jobs", headers=headers).json["jobs"] == []
    assert harness.client.get("/v1/jobs", headers=harness.owner_headers).json["jobs"] == []
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert balance["reserved_units"] == 0
    assert balance["available_units"] == 10000
    # A rejected request has no durable submission: correcting its model is safe.
    assert create(harness, headers).status_code == 202


def test_type_and_status_history_filters_run_before_pagination(platform, keys):
    harness, headers, account = platform
    with app.app.app_context():
        conn = app.get_db()
        for index in range(42):
            video = index < 2
            conn.execute("""INSERT INTO jobs(id,wallet,model,data,task_type,status,timestamp,updated_at,creator_account_id,owner_account_id,weight)
                VALUES (?,'','fixture',?,?,?,?,?,?,?,1)""", (f"history-{index}",
                '{"v1_type":"image_to_video"}' if video else '{"v1_type":"image"}',
                "LTX_VIDEO_GEN" if video else "IMAGE_GEN", "completed" if video else "queued", index, index, account, account))
        conn.commit()
    # More than the old limit*3 scan window of newer images must not hide old video renders.
    response = harness.client.get("/v2/jobs?type=image_to_video&status=succeeded&limit=1", headers=headers)
    assert response.status_code == 200
    assert [job["id"] for job in response.json["jobs"]] == ["history-1"]
    response = harness.client.get("/v2/jobs?type=image_to_video&status=succeeded&limit=1&offset=1", headers=headers)
    assert [job["id"] for job in response.json["jobs"]] == ["history-0"]
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.get("/v2/jobs?type=image_to_video", headers=bob).json["jobs"] == []


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


@pytest.mark.parametrize("field", ["audio_asset_id", "source_audio_asset_id", "reference_asset_id"])
@pytest.mark.parametrize("source", ["other_account", "missing", "wrong_kind"])
def test_music_source_spellings_enforce_account_ownership_and_kind(platform, keys, field, source):
    harness, headers, _ = platform
    owner_headers = ({"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
                     if source == "other_account" else headers)
    kind = "image" if source == "wrong_kind" else "audio"
    upload = harness.client.post("/v2/assets", headers=owner_headers,
        data={"kind": kind, "file": (io.BytesIO(b"private-source"), "source.png" if kind == "image" else "source.wav")})
    assert upload.status_code == 201
    response = harness.client.post("/v2/jobs", headers=headers, json={
        "type": "text_to_music", "model": platform_fixture.MUSIC_MODEL,
        "prompt": "A song", field: "missing-asset" if source == "missing" else upload.json["id"],
    })
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_asset"
    assert harness.client.get("/v2/jobs", headers=headers).json["jobs"] == []
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


def test_owned_music_source_alias_resolves_and_replays(platform):
    harness, headers, _ = platform
    asset = harness.client.post("/v2/assets", headers=headers,
        data={"kind": "audio", "file": (io.BytesIO(b"own-audio"), "source.wav")}).json
    body = {"type": "text_to_music", "model": platform_fixture.MUSIC_MODEL,
            "prompt": "A song", "source_audio_asset_id": asset["id"]}
    first = harness.client.post("/v2/jobs", headers=headers, json=body)
    assert first.status_code == 202, first.json
    assert first.json["resolved_spec"]["parameters"]["audio_asset_id"] == asset["id"]
    assert harness.client.post("/v2/jobs", headers=headers, json=body).json["id"] == first.json["id"]


@pytest.mark.parametrize("source", ["other_account", "wrong_kind", "missing"])
def test_enqueue_checks_resolved_assets_under_transaction(platform, keys, source):
    harness, headers, account = platform
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    kind = "image" if source == "wrong_kind" else "audio"
    asset = harness.client.post("/v2/assets", headers=bob if source == "other_account" else headers,
        data={"kind": kind, "file": (io.BytesIO(b"private-source"), "source.png" if kind == "image" else "source.wav")}).json
    asset_id = "missing-asset" if source == "missing" else asset["id"]
    with app.app.app_context():
        with pytest.raises(ValueError, match="asset_not_found"):
            account_jobs.enqueue(app.get_db(), account, request_key="resolved-source-test",
                request_payload={"source_audio_asset_id": asset_id},
                model=platform_fixture.MUSIC_MODEL, task_type="MUSIC_GEN",
                settings={"audio_asset_id": asset_id}, resolved_spec={}, weight=1, units=1000)
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


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


def test_owned_image_refinement_survives_enqueue_and_worker_claim(platform):
    harness, headers, _ = platform
    assets = [harness.client.post("/v2/assets", headers=headers,
        data={"kind": "image", "file": (io.BytesIO(b"private-image"), name)}).json["id"]
        for name in ("source.png", "mask.png")]
    response = create(harness, headers, source_asset_id=assets[0], mask_asset_id=assets[1],
                      img2img_strength=0.4, preserve_reference_aspect=True, seed=42)
    assert response.status_code == 202, response.json
    task = harness._claim(response.json["id"])
    for field, expected in {"source_asset_id": assets[0], "mask_asset_id": assets[1],
                            "img2img_strength": 0.4, "preserve_reference_aspect": True, "seed": 42}.items():
        assert task[field] == expected
        assert task["resolved_spec"]["parameters"][field] == expected
    assert "init_image" not in task


def test_image_history_preserves_prompt_loras_and_sfw_controls(platform, monkeypatch):
    harness, headers, _ = platform
    monkeypatch.setattr(app, "SFW_NEGATIVE_PROMPT", "sfw-negative-fixture")
    response = create(harness, headers, negative_prompt="blur", sfw_mode=True,
                      loras=[{"name": "style", "weight": 0.5}])
    assert response.status_code == 202
    parameters = response.json["resolved_spec"]["parameters"]
    assert parameters["prompt"] == "A blue sky"
    assert parameters["sfw_mode"] is True
    assert "blur" in parameters["negative_prompt"]
    assert "sfw-negative-fixture" in parameters["negative_prompt"]
    assert parameters["loras"] == [{"name": "style", "weight": 0.5}]
    task = harness._claim(response.json["id"])
    assert task["prompt"] == parameters["prompt"]
    assert task["negative_prompt"] == parameters["negative_prompt"]
    assert task["loras"] == parameters["loras"]


@pytest.mark.parametrize("field", ["source_asset_id", "mask_asset_id", "face_asset_id"])
def test_foreign_image_conditioning_is_rejected(platform, keys, field):
    harness, headers, _ = platform
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    asset = harness.client.post("/v2/assets", headers=bob,
        data={"kind": "image", "file": (io.BytesIO(b"private-image"), "source.png")}).json
    response = create(harness, headers, **{field: asset["id"]})
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_asset"
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


@pytest.mark.parametrize("field", ["init_image", "init_image_url", "inpaint_mask", "reference_face_url"])
def test_account_images_require_owned_assets_instead_of_raw_worker_sources(platform, field):
    harness, headers, _ = platform
    response = create(harness, headers, **{field: "http://127.0.0.1/private-source"})
    assert response.status_code == 400
    assert response.json["error"]["code"] == "owned_image_asset_required"
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


@pytest.mark.parametrize("seed", ["invalid", True, 2.5, {}, -2, 2**32])
def test_invalid_image_seed_is_rejected_before_charging(platform, seed):
    harness, headers, _ = platform
    response = create(harness, headers, seed=seed)
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_seed"
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


@pytest.mark.parametrize("strength", [0, -1, 1.1, "invalid", None, 0.001])
def test_invalid_refinement_strength_is_rejected_before_charging(platform, strength):
    harness, headers, _ = platform
    asset = harness.client.post("/v2/assets", headers=headers,
        data={"kind": "image", "file": (io.BytesIO(b"source"), "source.png")}).json
    response = create(harness, headers, source_asset_id=asset["id"], img2img_strength=strength)
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_image_strength"
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0


def test_owned_face_reference_requires_a_face_capable_worker(platform):
    harness, headers, _ = platform
    asset = harness.client.post("/v2/assets", headers=headers,
        data={"kind": "image", "file": (io.BytesIO(b"face"), "face.png")}).json
    response = create(harness, headers, face_asset_id=asset["id"])
    assert response.status_code == 202
    harness._register_node()
    with app.app.app_context():
        assert job_helpers.fetch_next_job_for_node("node-test") is None
    app.NODES["node-test"]["supports"].append("face_swap")
    task = harness.client.get("/tasks/creator?node_id=node-test", headers=harness.node_headers).json["tasks"][0]
    assert task["task_id"] == response.json["id"]
    assert task["face_asset_id"] == asset["id"]

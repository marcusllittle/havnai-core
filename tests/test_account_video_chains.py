import io

import pytest

from tests.test_account_jobs import platform, keys
from tests.test_account_auth import token
from tests import test_platform_v1 as fixtures
import app
import account_video
import account_video_chains


def plan(platform):
    harness, headers, _ = platform
    app.MANIFEST_MODELS[fixtures.VIDEO_MODEL]["capabilities"].append("text_to_video")
    body = {"template": {"type": "text_to_video", "model": fixtures.VIDEO_MODEL, "prompt": "A coast", "seed": 42},
            "total": 2, "auto_stitch": True}
    response = harness.client.post("/v2/video-chains", headers=headers, json=body)
    assert response.status_code == 200, response.json
    return response.json, body


def complete(harness, job):
    task = harness._claim(job["id"])
    response = harness.client.post(f"/v1/node/jobs/{job['id']}/artifacts", headers=harness.node_headers,
        data={"node_id": "node-test", "attempt_id": task["attempt_id"], "kind": "video",
              "file": (io.BytesIO(b"clip"), "clip.mp4")})
    assert response.status_code == 201
    completed = harness.client.post("/results", headers=harness.node_headers, json={
        "node_id": "node-test", "task_id": job["id"], "attempt_id": task["attempt_id"],
        "status": "success", "metrics": {"inference_time_ms": 12}})
    assert completed.status_code == 200, completed.json


def test_chain_recovers_across_sessions_and_next_is_idempotent(platform, keys, monkeypatch):
    harness, headers, _ = platform
    chain, body = plan(platform)
    url = f"/v2/video-chains/{chain['id']}"
    other_session = {"Authorization": token(keys, sid="sess_second_device")}
    assert harness.client.get(url, headers=other_session).json == chain
    assert harness.client.post("/v2/video-chains", headers=headers, json=body).json["id"] == chain["id"]
    assert harness.client.post("/v2/video-chains", headers=headers, json={**body, "total": 3}).status_code == 409
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0
    first = harness.client.post(url + "/next", headers=other_session)
    assert first.status_code == 202, first.json
    job = first.json["job"]
    first_balance = harness.client.get("/v2/account/credits", headers=headers).json
    cost = first_balance["reserved_units"]
    assert cost > 0
    for _ in range(2):
        retry = harness.client.post(url + "/next", headers=other_session)
        assert retry.status_code == 200
        assert retry.json["job"]["id"] == job["id"]
    assert harness.client.get("/v2/account/credits", headers=headers).json == first_balance
    complete(harness, job)
    monkeypatch.setattr(account_video, "extract_frame", lambda video, output: output.write_bytes(b"frame"))
    second = harness.client.post(url + "/next", headers=headers)
    assert second.status_code == 202, second.json
    second_job = second.json["job"]
    assert second_job["id"] != job["id"]
    assert second_job["type"] == "image_to_video"
    params = second_job["resolved_spec"]["parameters"]
    assert params["seed"] == 43
    assert params["source_asset_id"].startswith("asset-")
    assert harness.client.post(url + "/next", headers=headers).json["job"]["id"] == second_job["id"]
    complete(harness, second_job)
    finished = harness.client.post(url + "/next", headers=headers)
    assert finished.status_code == 200
    assert finished.json["chain"]["state"] == "rendered"
    assert len(finished.json["chain"]["jobs"]) == 2
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert balance["available_units"] == 10000 - 2 * cost and balance["reserved_units"] == 0


def test_chain_is_private_and_stopping_never_cancels_an_accepted_clip(platform, keys):
    harness, headers, _ = platform
    chain, _ = plan(platform)
    url = f"/v2/video-chains/{chain['id']}"
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.get("/v2/video-chains", headers=bob).json == {"chains": []}
    assert harness.client.get(url, headers=bob).status_code == 404
    assert harness.client.post(url + "/next", headers=bob).status_code == 404
    assert harness.client.delete(url, headers=bob).status_code == 404
    job = harness.client.post(url + "/next", headers=headers).json["job"]
    assert harness.client.delete(url, headers=headers).json["state"] == "stopped"
    assert harness.client.post(url + "/next", headers=headers).status_code == 409
    assert harness.client.get(f"/v2/jobs/{job['id']}", headers=headers).json["status"] == "queued"
    assert harness.client.get("/v2/jobs", headers=headers).json["count"] == 1


def test_stop_during_frame_preparation_prevents_new_reservation(platform, monkeypatch):
    harness, headers, account = platform
    chain, _ = plan(platform)
    url = f"/v2/video-chains/{chain['id']}"
    first = harness.client.post(url + "/next", headers=headers).json["job"]
    complete(harness, first)
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    def extract(video, output):
        account_video_chains.stop(app.get_db(), account, chain["id"])
        output.write_bytes(b"frame")
    monkeypatch.setattr(account_video, "extract_frame", extract)
    assert harness.client.post(url + "/next", headers=headers).status_code == 409
    assert harness.client.get("/v2/account/credits", headers=headers).json == balance
    assert len(harness.client.get(url, headers=headers).json["jobs"]) == 1


@pytest.mark.parametrize("total", [1, 8, True, 2.5, "2"])
def test_invalid_chain_sizes_have_no_effect(platform, total):
    harness, headers, _ = platform
    response = harness.client.post("/v2/video-chains", headers=headers, json={
        "template": {"type": "text_to_video", "model": fixtures.VIDEO_MODEL, "prompt": "Clouds"},
        "total": total, "auto_stitch": True})
    assert response.status_code == 422
    assert harness.client.get("/v2/video-chains", headers=headers).json == {"chains": []}
    assert harness.client.get("/v2/account/credits", headers=headers).json["reserved_units"] == 0

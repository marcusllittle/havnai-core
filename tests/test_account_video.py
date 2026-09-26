import io
import shutil
import subprocess

import pytest

from tests.test_account_jobs import platform, keys
from tests.test_account_auth import token
from tests import test_platform_v1 as fixtures
import app
import account_video
import job_helpers


def completed_video(platform):
    harness, headers, account = platform
    app.MANIFEST_MODELS[fixtures.VIDEO_MODEL]["capabilities"].append("text_to_video")
    job = harness.client.post("/v2/jobs", headers=headers, json={"type": "text_to_video",
        "model": fixtures.VIDEO_MODEL, "prompt": "Clouds"}).json
    task = harness._claim(job["id"])
    artifact = harness.client.post(f"/v1/node/jobs/{job['id']}/artifacts", headers=harness.node_headers,
        data={"node_id": "node-test", "attempt_id": task["attempt_id"], "kind": "video",
              "file": (io.BytesIO(b"video-fixture"), "clip.mp4")})
    assert artifact.status_code == 201
    with app.app.app_context():
        assert job_helpers.complete_job(job["id"], "node-test", "succeeded", task["attempt_id"])
    return job, artifact.json


def test_last_frame_is_private_idempotent_and_reusable_without_a_charge(platform, keys, monkeypatch):
    harness, headers, account = platform
    job, artifact = completed_video(platform)
    calls = []
    def extract(video, output):
        calls.append(video)
        output.write_bytes(b"private-frame")
    monkeypatch.setattr(account_video, "extract_frame", extract)
    route = f"/v2/jobs/{job['id']}/last-frame"
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    assert harness.client.post(route, headers=bob).status_code == 404
    assert harness.client.post(route, headers=harness.owner_headers).status_code == 401
    assert not calls
    first = harness.client.post(route, headers=headers)
    assert first.status_code == 200, first.json
    assert first.json["kind"] == "image"
    assert harness.client.post(route, headers=headers).json == first.json
    assert len(calls) == 1
    assert harness.client.get("/v2/account/credits", headers=headers).json == balance
    asset_id = first.json["id"]
    assert harness.client.get(f"/v2/assets/{asset_id}/content", headers=headers).data == b"private-frame"
    assert harness.client.get(f"/v2/assets/{asset_id}/content", headers=bob).status_code == 404
    assert harness.client.get(f"/v1/assets/{asset_id}/content", headers=harness.owner_headers).status_code == 404
    continued = harness.client.post("/v2/jobs", headers={**headers, "Idempotency-Key": "next-clip"}, json={
        "type": "image_to_video", "model": fixtures.VIDEO_MODEL, "prompt": "Continue", "source_asset_id": asset_id})
    assert continued.status_code == 202, continued.json
    assert continued.json["resolved_spec"]["parameters"]["source_asset_id"] == asset_id


def test_ownership_change_during_extraction_discards_the_private_frame(platform, keys, monkeypatch):
    harness, headers, _ = platform
    job, _ = completed_video(platform)
    bob = harness.client.get("/v2/account", headers={"Authorization": token(keys, sub="user_bob", sid="sess_bob")}).json["id"]
    def extract(video, output):
        output.write_bytes(b"private-frame")
        conn = app.get_db()
        conn.execute("UPDATE jobs SET owner_account_id=? WHERE id=?", (bob, job["id"]))
        conn.commit()
    monkeypatch.setattr(account_video, "extract_frame", extract)
    result = harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers)
    assert result.status_code == 404
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_video_frames").fetchone()[0] == 0
        assert app.get_db().execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0
    assert list(app.ASSETS_DIR.iterdir()) == []


def test_extraction_failure_does_not_cache_or_charge(platform, monkeypatch):
    harness, headers, _ = platform
    job, _ = completed_video(platform)
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    def fail(video, output):
        output.write_bytes(b"partial")
        raise account_video.VideoInputError("video_processing_unavailable", 503)
    monkeypatch.setattr(account_video, "extract_frame", fail)
    assert harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers).status_code == 503
    assert harness.client.get("/v2/account/credits", headers=headers).json == balance
    assert list(app.ASSETS_DIR.iterdir()) == []


def test_an_interleaved_extraction_reuses_the_winner_and_cleans_its_duplicate(platform, monkeypatch):
    harness, headers, account = platform
    job, _ = completed_video(platform)
    interleaved = False
    winner = None
    def extract(video, output):
        nonlocal interleaved, winner
        if not interleaved:
            interleaved = True
            winner = account_video.last_frame(app.get_db(), account, job["id"],
                outputs=app.OUTPUTS_DIR, assets=app.ASSETS_DIR, max_bytes=app.ASSET_MAX_BYTES)
        output.write_bytes(b"private-frame")
    monkeypatch.setattr(account_video, "extract_frame", extract)
    response = harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers)
    assert response.status_code == 200
    assert response.json == winner
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 1
        assert app.get_db().execute("SELECT COUNT(*) FROM account_video_frames").fetchone()[0] == 1
    assert len(list(app.ASSETS_DIR.iterdir())) == 1


def test_unfinished_and_outside_root_videos_are_not_decoded(platform, monkeypatch, tmp_path):
    harness, headers, _ = platform
    job, artifact = completed_video(platform)
    monkeypatch.setattr(account_video, "extract_frame", lambda *_: pytest.fail("Should not decode"))
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],)); conn.commit()
    assert harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers).status_code == 409
    outside = tmp_path / "outside.mp4"; outside.write_bytes(b"private")
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("UPDATE jobs SET status='succeeded' WHERE id=?", (job["id"],))
        conn.execute("UPDATE artifacts SET path=? WHERE id=?", (str(outside), artifact["id"])); conn.commit()
    assert harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers).status_code == 410


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_real_ffmpeg_extracts_last_color_from_short_video(tmp_path):
    from PIL import Image
    video = tmp_path / "clip.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=red:s=32x32:r=10:d=0.3",
        "-f", "lavfi", "-i", "color=blue:s=32x32:r=10:d=0.3", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v", "mpeg4", "-threads", "1", str(video)], check=True, timeout=30)
    output = tmp_path / "frame.png"
    account_video.extract_frame(video, output)
    with Image.open(output) as frame:
        r, g, b = frame.convert("RGB").getpixel((16, 16))
        assert b > 200 and r < 30 and g < 30

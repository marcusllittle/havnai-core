import json
import shutil
import subprocess

import pytest

from tests.test_account_jobs import platform, keys
from tests.test_account_auth import token
from tests.test_account_video_chains import plan, complete
import app
import account_video
import account_video_stitch


def rendered(platform, monkeypatch):
    harness, headers, _ = platform
    chain, _ = plan(platform)
    url = f"/v2/video-chains/{chain['id']}"
    monkeypatch.setattr(account_video, "extract_frame", lambda video, output: output.write_bytes(b"frame"))
    for _ in range(2):
        complete(harness, harness.client.post(url + "/next", headers=headers).json["job"])
    return chain, url


def test_stitched_result_is_private_idempotent_and_keeps_source_provenance(platform, keys, monkeypatch):
    harness, headers, _ = platform
    chain, url = rendered(platform, monkeypatch)
    calls = []
    def merge(paths, output):
        calls.append(paths)
        output.write_bytes(b"private-merged-video")
    monkeypatch.setattr(account_video_stitch, "concatenate", merge)
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.post(url + "/stitch", headers=bob).status_code == 404
    assert not calls
    balance = harness.client.get("/v2/account/credits", headers=headers).json
    response = harness.client.post(url + "/stitch", headers=headers)
    assert response.status_code == 200, response.json
    job = response.json["job"]
    assert job["type"] == "video_stitch" and job["status"] == "succeeded"
    assert response.json["chain"]["state"] == "complete"
    params = job["resolved_spec"]["parameters"]
    assert params["chain_id"] == chain["id"]
    assert len(params["source_job_ids"]) == len(params["source_artifact_ids"]) == 2
    assert harness.client.post(url + "/stitch", headers=headers).json["job"]["id"] == job["id"]
    assert len(calls) == 1
    assert harness.client.get("/v2/account/credits", headers=headers).json == balance
    media = job["artifacts"][0]["url"]
    assert media.startswith("/v2/artifacts/")
    assert harness.client.get(media, headers=headers).data == b"private-merged-video"
    assert harness.client.get(media, headers=bob).status_code == 404
    assert harness.client.get(f"/static/outputs/private-chains/{job['id']}/result.mp4").status_code == 404
    visual = harness.client.get("/v2/jobs?collection=1&type=image_to_video", headers=headers).json
    assert job["id"] in {entry["id"] for entry in visual["jobs"]}


def test_stitch_before_completion_does_not_create_a_result(platform, monkeypatch):
    harness, headers, _ = platform
    chain, _ = plan(platform)
    monkeypatch.setattr(account_video_stitch, "concatenate", lambda *_: pytest.fail("Should not merge"))
    response = harness.client.post(f"/v2/video-chains/{chain['id']}/stitch", headers=headers)
    assert response.status_code == 409
    assert harness.client.get("/v2/jobs", headers=headers).json["jobs"] == []


def test_interleaved_stitch_requests_publish_only_one_result(platform, monkeypatch):
    harness, headers, account = platform
    chain, url = rendered(platform, monkeypatch)
    entered = False
    winner = None
    def merge(paths, output):
        nonlocal entered, winner
        if not entered:
            entered = True
            winner = account_video_stitch.stitch(app.get_db(), account, chain["id"], outputs=app.OUTPUTS_DIR, max_bytes=app.ARTIFACT_MAX_BYTES)
        output.write_bytes(b"merged")
    monkeypatch.setattr(account_video_stitch, "concatenate", merge)
    response = harness.client.post(url + "/stitch", headers=headers)
    assert response.status_code == 200
    assert response.json["job"]["id"] == winner
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_video_chain_outputs").fetchone()[0] == 1
        assert app.get_db().execute("SELECT COUNT(*) FROM jobs WHERE task_type='VIDEO_STITCH'").fetchone()[0] == 1
    assert len(list((app.OUTPUTS_DIR / "private-chains").iterdir())) == 1


def test_ownership_change_while_merging_prevents_result_publication(platform, keys, monkeypatch):
    harness, headers, _ = platform
    chain, url = rendered(platform, monkeypatch)
    bob = harness.client.get("/v2/account", headers={"Authorization": token(keys, sub="user_bob", sid="sess_bob")}).json["id"]
    def merge(paths, output):
        output.write_bytes(b"merged")
        conn = app.get_db()
        conn.execute("UPDATE jobs SET owner_account_id=? WHERE id=(SELECT job_id FROM account_video_chain_clips WHERE chain_id=? AND clip_index=0)", (bob, chain["id"]))
        conn.commit()
    monkeypatch.setattr(account_video_stitch, "concatenate", merge)
    assert harness.client.post(url + "/stitch", headers=headers).status_code == 404
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_video_chain_outputs").fetchone()[0] == 0
    assert list((app.OUTPUTS_DIR / "private-chains").iterdir()) == []


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable")
@pytest.mark.parametrize("audio", [False, True])
def test_real_stitch_preserves_clip_order_and_audio(tmp_path, audio):
    from PIL import Image
    paths = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"clip-{index}.mp4"
        cmd = [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=32x32:r=10:d=0.3"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.3", "-c:a", "aac"]
        subprocess.run([*cmd, "-c:v", "mpeg4", "-threads", "1", str(path)], check=True, timeout=30)
        paths.append(path)
    output = tmp_path / "merged.mp4"
    account_video_stitch.concatenate(paths, output)
    metadata = json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(output)], timeout=15))
    streams = metadata["streams"]
    assert any(stream["codec_type"] == "audio" for stream in streams) is audio
    assert int(next(stream for stream in streams if stream["codec_type"] == "video")["nb_frames"]) == 6
    last = tmp_path / "last.png"
    account_video.extract_frame(output, last)
    with Image.open(last) as image:
        r, g, b = image.convert("RGB").getpixel((16, 16))
        assert b > 200 and r < 30 and g < 30

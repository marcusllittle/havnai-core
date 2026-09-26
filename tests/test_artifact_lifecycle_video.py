from tests.test_account_jobs import platform, keys
from tests.test_account_video import completed_video
from tests.test_account_video_chains import plan, complete
import app
import account_video


def test_deleted_video_cannot_extract_or_read_cached_frame(platform, monkeypatch):
    harness, headers, account = platform
    job, _ = completed_video(platform)
    def extract(video, output):
        output.write_bytes(b"frame")
    monkeypatch.setattr(account_video, "extract_frame", extract)
    route = f"/v2/jobs/{job['id']}"
    asset = harness.client.post(route + "/last-frame", headers=headers).json
    with app.app.app_context():
        app.gallery.init_gallery_tables(app.get_db())
        app.music_discover.init_music_discover_tables(app.get_db())
        app.get_db().commit()
    assert harness.client.delete(route, headers=headers).status_code == 200
    assert harness.client.post(route + "/last-frame", headers=headers).status_code == 404
    assert harness.client.get(f"/v2/assets/{asset['id']}/content", headers=headers).status_code == 404
    assert harness.client.post(route + "/restore", headers=headers).status_code == 200
    assert harness.client.post(route + "/last-frame", headers=headers).json["id"] == asset["id"]
    assert harness.client.get(f"/v2/assets/{asset['id']}/content", headers=headers).status_code == 200


def test_deletion_during_frame_extraction_does_not_publish_a_derived_asset(platform, monkeypatch):
    harness, headers, account = platform
    job, _ = completed_video(platform)
    with app.app.app_context():
        app.gallery.init_gallery_tables(app.get_db())
        app.music_discover.init_music_discover_tables(app.get_db())
        app.get_db().commit()
    def extract(video, output):
        output.write_bytes(b"frame")
        app.artifact_lifecycle.delete(app.get_db(), account, job["id"])
    monkeypatch.setattr(account_video, "extract_frame", extract)
    response = harness.client.post(f"/v2/jobs/{job['id']}/last-frame", headers=headers)
    assert response.status_code == 404
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_video_frames").fetchone()[0] == 0


def test_deleted_clip_hides_sequence_and_blocks_continuation(platform):
    harness, headers, account = platform
    chain, body = plan(platform)
    url = f"/v2/video-chains/{chain['id']}"
    job = harness.client.post(url + "/next", headers=headers).json["job"]
    complete(harness, job)
    with app.app.app_context():
        app.gallery.init_gallery_tables(app.get_db())
        app.music_discover.init_music_discover_tables(app.get_db())
        app.get_db().commit()
    assert harness.client.delete(f"/v2/jobs/{job['id']}", headers=headers).status_code == 200
    assert harness.client.get(url, headers=headers).status_code == 404
    assert harness.client.post(url + "/next", headers=headers).status_code == 404
    assert harness.client.post(url + "/stitch", headers=headers).status_code == 404
    assert harness.client.get("/v2/video-chains", headers=headers).json["chains"] == []
    assert harness.client.post("/v2/video-chains", headers=headers, json=body).status_code == 404

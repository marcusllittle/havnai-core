import pytest
import uuid
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.test_account_auth import keys, token
from tests.test_account_music import music, publish
import app
import artifact_lifecycle as lifecycle
import music_discover


def test_publish_and_delete_race_leaves_no_public_song(music, monkeypatch):
    harness, headers, account = music
    database = app.DB_PATH
    local = threading.local()
    barrier = threading.Barrier(2)
    original_get_db = music_discover.get_db
    monkeypatch.setattr(music_discover, "get_db", lambda: getattr(local, "conn", None) or original_get_db())
    def run(operation):
        local.conn = sqlite3.connect(database, timeout=10)
        local.conn.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=10)
            if operation == "publish":
                return music_discover.publish_song(job_id="job-1", creator_account_id=account, title="Race song")
            lifecycle.delete(local.conn, account, "job-1")
            return {"deleted": True}
        finally:
            local.conn.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        published, removed = pool.map(run, ["publish", "delete"])
    assert removed == {"deleted": True}
    assert published.get("ok") is True or published.get("error") == "job_not_found"
    conn = app.get_db()
    assert lifecycle.deleted(conn, "job-1")
    assert conn.execute("SELECT count(*) FROM music_publications WHERE state='published' AND job_id='job-1'").fetchone()[0] == 0
    assert harness.client.get("/v2/artifacts/artifact-1/content", headers=headers).status_code == 404
    if published.get("ok"):
        publication = published["publication"]["id"]
        assert harness.client.get(f"/music/publications/{publication}/audio").status_code == 404


def test_concurrent_delete_retries_keep_one_recovery_window(music):
    _, _, account = music
    database = app.DB_PATH
    barrier = threading.Barrier(4)
    def run(index):
        conn = sqlite3.connect(database, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=10)
            return lifecycle.delete(conn, account, "job-1", now=100 + index)
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert all(result == results[0] for result in results)
    assert results[0]["recover_until"] - results[0]["deleted_at"] == lifecycle.RECOVERY_SECONDS
    assert app.get_db().execute("SELECT count(*) FROM artifact_lifecycle_events WHERE action='delete'").fetchone()[0] == 1


def test_delete_unpublishes_blocks_reads_and_restores_only_private(music):
    harness, headers, account = music
    publication = publish(harness, headers).json
    path = "/v2/jobs/job-1"
    response = harness.client.delete(path, headers=headers)
    assert response.status_code == 200, response.json
    assert response.json["recover_until"] - response.json["deleted_at"] == 30 * 86400
    assert harness.client.delete(path, headers=headers).json == response.json
    assert harness.client.get(path, headers=headers).status_code == 404
    assert harness.client.get("/v2/artifacts/artifact-1/content", headers=headers).status_code == 404
    assert harness.client.get(f"/music/publications/{publication['id']}/audio").status_code == 404
    assert publish(harness, headers).status_code == 404
    history = harness.client.get("/v2/jobs?type=text_to_music", headers=headers).json
    assert history["jobs"] == []
    assert harness.client.post(path + "/restore", headers=headers).status_code == 200
    assert harness.client.post(path + "/restore", headers=headers).status_code == 200
    assert harness.client.get("/v2/artifacts/artifact-1/content", headers=headers).status_code == 200
    assert harness.client.get(f"/music/publications/{publication['id']}/audio").status_code == 404
    events = app.get_db().execute("SELECT action FROM artifact_lifecycle_events ORDER BY created_at").fetchall()
    assert [row[0] for row in events] == ["delete", "restore"]


def test_owner_required_and_active_jobs_cannot_be_deleted(music, keys):
    harness, headers, account = music
    path = "/v2/jobs/job-1"
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert harness.client.delete(path).status_code == 401
    assert harness.client.delete(path, headers=bob).status_code == 404
    assert harness.client.post(path + "/restore", headers=bob).status_code == 404
    assert harness.client.delete("/v2/jobs/missing", headers=headers).status_code == 404
    conn = app.get_db()
    conn.execute("UPDATE jobs SET status='running' WHERE id='job-1'")
    conn.commit()
    assert harness.client.delete(path, headers=headers).status_code == 409
    assert conn.execute("SELECT COUNT(*) FROM artifact_lifecycle_events").fetchone()[0] == 0


def test_recovery_deadline_and_new_delete_window(music):
    _, _, account = music
    conn = app.get_db()
    lifecycle.delete(conn, account, "job-1", now=100)
    with pytest.raises(lifecycle.LifecycleError, match="recovery_window_expired"):
        lifecycle.restore(conn, account, "job-1", now=100 + lifecycle.RECOVERY_SECONDS)
    lifecycle.restore(conn, account, "job-1", now=101)
    row = lifecycle.delete(conn, account, "job-1", now=200)
    assert row["recover_until"] == 200 + lifecycle.RECOVERY_SECONDS


def test_purge_respects_deadline_holds_and_keeps_audit(music):
    _, _, account = music
    conn = app.get_db()
    path = Path(conn.execute("SELECT path FROM artifacts WHERE id='artifact-1'").fetchone()[0])
    lifecycle.delete(conn, account, "job-1", now=100)
    with pytest.raises(lifecycle.LifecycleError, match="purge_not_eligible"):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=101)
    conn.execute("INSERT INTO artifact_lifecycle_holds VALUES ('job-1','support-1','support',100,NULL)")
    conn.commit()
    with pytest.raises(lifecycle.LifecycleError, match="artifact_on_hold"):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=100 + lifecycle.RECOVERY_SECONDS)
    assert path.exists()
    conn.execute("UPDATE artifact_lifecycle_holds SET released_at=200")
    conn.commit()
    lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=100 + lifecycle.RECOVERY_SECONDS)
    lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=100 + lifecycle.RECOVERY_SECONDS)
    assert not path.exists()
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE id='job-1'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM artifact_lifecycle_events WHERE action='purge'").fetchone()[0] == 1


def test_purge_rejects_storage_outside_output_root(music, tmp_path):
    _, _, account = music
    conn = app.get_db()
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"keep")
    conn.execute("UPDATE artifacts SET path=? WHERE id='artifact-1'", (str(outside),))
    conn.commit()
    lifecycle.delete(conn, account, "job-1", now=100)
    with pytest.raises(lifecycle.LifecycleError, match="unsafe_artifact_path"):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=100 + lifecycle.RECOVERY_SECONDS)
    assert outside.read_bytes() == b"keep"


def test_delete_detaches_playlists_but_retains_likes_and_plays(music):
    harness, headers, account = music
    publication = publish(harness, headers).json["id"]
    playlist = "playlist-" + str(uuid.uuid4())
    assert harness.client.post("/v2/music/playlists", headers=headers,
        json={"id": playlist, "title": "Test", "is_public": True}).status_code == 201
    assert harness.client.put(f"/v2/music/playlists/{playlist}/items/{publication}", headers=headers).status_code == 200
    assert harness.client.put(f"/v2/music/publications/{publication}/like", headers=headers, json={"liked": True}).status_code == 200
    conn = app.get_db()
    before = [tuple(row) for row in conn.execute("SELECT * FROM account_music_likes")]
    lifecycle.delete(conn, account, "job-1")
    assert conn.execute("SELECT COUNT(*) FROM music_playlist_items").fetchone()[0] == 0
    assert [tuple(row) for row in conn.execute("SELECT * FROM account_music_likes")] == before
    lifecycle.restore(conn, account, "job-1")
    assert conn.execute("SELECT COUNT(*) FROM music_playlist_items").fetchone()[0] == 0

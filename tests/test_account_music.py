"""Account music publication must not require or trust wallet ownership."""
import concurrent.futures
import json
import sqlite3
import threading

import pytest

from tests.test_account_auth import keys, token, config
from tests import test_music_discover_api as music_fixture
import app
import account_auth
import music_discover


@pytest.fixture
def music(keys, monkeypatch):
    harness = music_fixture.MusicDiscoverApiTests(methodName="runTest")
    harness.setUp()
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: config(keys))
    monkeypatch.setattr(app, "OUTPUTS_DIR", app.STATIC_DIR / "outputs")
    headers = {"Authorization": token(keys)}
    account = harness.client.get("/v2/account", headers=headers).json["id"]
    conn = app.get_db()
    # Keep the old wallet as provenance to prove it cannot override account ownership.
    conn.execute("UPDATE jobs SET owner_account_id=?,creator_account_id=? WHERE id='job-1'", (account, account))
    conn.commit()
    yield harness, headers, account
    harness.tearDown()


def publish(harness, headers, **extra):
    return harness.client.post("/v2/music/publications", headers=headers,
                               json={"job_id": "job-1", "title": "My account song", **extra})


def test_account_publish_play_unpublish_and_private_artifact(music):
    harness, headers, account = music
    assert harness.client.get("/v2/artifacts/artifact-1/content").status_code == 401
    assert harness.client.get("/static/outputs/artifacts/job-1/song.mp3").status_code == 404
    audio = harness.client.get("/v2/artifacts/artifact-1/content", headers={**headers, "Range": "bytes=0-2"})
    assert audio.status_code == 206
    assert audio.data == b"aud"
    response = publish(harness, headers)
    assert response.status_code == 201, response.json
    publication = response.json
    assert publication["creator_wallet"] == ""
    assert publication["creator_profile_id"].startswith("creator_")
    assert publication["job_id"] == "job-1"
    retry = publish(harness, headers)
    assert retry.status_code == 200
    assert retry.json["id"] == publication["id"]
    public = harness.client.get(f"/music/discover/{publication['id']}").json
    for value in (account, "job-1", "artifact-1", "private prompt", "user_alice", "sess_alice"):
        assert value not in json.dumps(public)
    assert harness.client.get(f"/music/publications/{publication['id']}/audio").data == b"audio"
    assert harness.client.get("/static/outputs/artifacts/job-1/song.mp3").status_code == 404
    profile = harness.client.get(f"/music/creators/{publication['creator_profile_id']}")
    assert profile.status_code == 200
    assert profile.json["track_count"] == 1
    assert account not in json.dumps(profile.json)
    own = harness.client.get("/v2/music/publications", headers=headers)
    assert own.json["publications"][0]["job_id"] == "job-1"
    path = f"/v2/music/publications/{publication['id']}"
    assert harness.client.delete(path, headers=headers).status_code == 200
    assert harness.client.delete(path, headers=headers).status_code == 200
    assert harness.client.get(f"/music/publications/{publication['id']}/audio").status_code == 404
    assert harness.client.get("/v2/artifacts/artifact-1/content", headers=headers).status_code == 200


def test_other_account_and_provenance_wallet_cannot_publish_or_unpublish(music, keys):
    harness, headers, _ = music
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    assert publish(harness, {}).status_code == 401
    assert publish(harness, bob).status_code == 404
    assert publish(harness, headers, wallet=harness.wallet).status_code == 422
    assert publish(harness, headers, owner_account_id="forged").status_code == 422
    assert harness.client.get("/v2/artifacts/artifact-1/content", headers=bob).status_code == 404
    assert not music_discover.publish_song(job_id="job-1", creator_wallet=harness.wallet, title="Wallet bypass")["ok"]
    publication = publish(harness, headers).json
    assert harness.client.get("/v2/music/publications", headers=bob).json["publications"] == []
    assert harness.client.delete(f"/v2/music/publications/{publication['id']}", headers=bob).status_code == 404
    assert not music_discover.unpublish_song(publication["id"], "")["ok"]
    assert harness.client.get(f"/music/publications/{publication['id']}/audio").status_code == 200


def test_rejects_unfinished_or_wrong_artifact_without_publication(music):
    harness, headers, _ = music
    assert publish(harness, headers, artifact_id="other-artifact").status_code == 422
    conn = app.get_db()
    conn.execute("UPDATE jobs SET status='running' WHERE id='job-1'")
    conn.commit()
    assert publish(harness, headers).status_code == 422
    assert conn.execute("SELECT count(*) FROM music_publications").fetchone()[0] == 0


def test_concurrent_publish_returns_same_publication(music, monkeypatch):
    harness, _, account = music
    path = app.DB_PATH
    local = threading.local()
    monkeypatch.setattr(music_discover, "get_db", lambda: local.conn)
    barrier = threading.Barrier(2)
    def worker():
        local.conn = sqlite3.connect(path, timeout=10)
        local.conn.row_factory = sqlite3.Row
        try:
            barrier.wait()
            return music_discover.publish_song(job_id="job-1", creator_account_id=account, title="One song")["publication"]["id"]
        finally:
            local.conn.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: worker(), range(2)))
    assert results[0] == results[1]
    assert app.get_db().execute("SELECT count(*) FROM music_publications").fetchone()[0] == 1

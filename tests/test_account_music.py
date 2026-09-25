"""Account music publication must not require or trust wallet ownership."""
import concurrent.futures
import json
import sqlite3
import threading
import uuid

import pytest

from tests.test_account_auth import keys, token, config
from tests import test_music_discover_api as music_fixture
import app
import account_auth
import music_discover
import account_playlists


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


def test_account_library_is_private_idempotent_and_separate_from_wallet(music, keys):
    harness, alice, account = music
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    publication = publish(harness, alice).json["id"]
    path = f"/v2/music/publications/{publication}"
    for action, key in (("save", "saved"), ("like", "liked")):
        assert harness.client.put(f"{path}/{action}", json={key: True}).status_code == 401
        assert harness.client.put(f"{path}/{action}", headers=alice, json={key: 1}).status_code == 422
        assert harness.client.put(f"{path}/{action}", headers=alice, json={key: True, "account_id": "other"}).status_code == 422
        for _ in range(2):
            assert harness.client.put(f"{path}/{action}", headers=alice, json={key: True}).status_code == 200
    assert harness.client.get("/v2/music/library").status_code == 401
    assert harness.client.get("/v2/music/library?wallet=forged", headers=alice).status_code == 422
    assert harness.client.get("/v2/music/library?limit=bad", headers=alice).status_code == 422
    library = harness.client.get("/v2/music/library", headers=alice).json
    assert library["total"] == 1
    assert library["publications"][0]["saved_by_me"]
    assert library["recent_liked"][0]["liked_by_me"]
    assert library["publications"][0]["like_count"] == 1
    preference_path = "/v2/music/preferences"
    assert harness.client.post(preference_path, json={"publication_ids": [publication]}).status_code == 401
    assert harness.client.post(preference_path, headers=alice, json={"publication_ids": [publication], "wallet": harness.wallet}).status_code == 422
    assert harness.client.post(preference_path, headers=alice, json={"publication_ids": [publication] * 101}).status_code == 422
    assert harness.client.post(preference_path, headers=alice, json={"publication_ids": [None]}).status_code == 422
    assert harness.client.post(preference_path, headers=alice, json={"publication_ids": [publication]}).json["preferences"][publication] == {
        "liked_by_me": True, "saved_by_me": True, "like_count": 1}
    assert harness.client.post(preference_path, headers=bob, json={"publication_ids": [publication]}).json["preferences"][publication] == {
        "liked_by_me": False, "saved_by_me": False, "like_count": 1}
    assert "job-1" not in json.dumps(library)
    assert account not in json.dumps(library)
    assert harness.client.get("/v2/music/library", headers=bob).json["total"] == 0
    assert harness.client.get("/v2/music/library", headers=bob).json["recent_liked"] == []
    assert harness.client.get("/v2/music/library?search=missing", headers=alice).json["total"] == 0
    assert harness.client.get("/v2/music/library?offset=1", headers=alice).json["publications"] == []
    # A legacy wallet's like must neither replace nor remove an account's like.
    assert music_discover.set_like(publication, harness.wallet)["like_count"] == 2
    assert music_discover.set_like(publication, harness.wallet, liked=False)["like_count"] == 1
    assert music_discover.list_saved(wallet=harness.wallet)["total"] == 0
    assert harness.client.delete(path, headers=alice).status_code == 200
    assert harness.client.get("/v2/music/library", headers=alice).json["total"] == 0
    assert harness.client.post(preference_path, headers=alice, json={"publication_ids": [publication]}).json["preferences"] == {}
    assert harness.client.put(f"{path}/save", headers=bob, json={"saved": True}).status_code == 404
    # Users may remove their own stale preference after a creator unpublishes.
    assert harness.client.put(f"{path}/save", headers=alice, json={"saved": False}).status_code == 200
    assert harness.client.put(f"{path}/like", headers=alice, json={"liked": False}).json["like_count"] == 0


def test_concurrent_account_like_retry_counts_once(music, monkeypatch):
    harness, headers, account = music
    publication = publish(harness, headers).json["id"]
    path = app.DB_PATH
    local = threading.local()
    monkeypatch.setattr(music_discover, "get_db", lambda: local.conn)
    barrier = threading.Barrier(2)
    def worker():
        local.conn = sqlite3.connect(path, timeout=10)
        local.conn.row_factory = sqlite3.Row
        try:
            barrier.wait()
            return music_discover.set_account_music_preference(account, publication, kind="like", enabled=True)
        finally:
            local.conn.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: worker(), range(2)))
    assert [result["like_count"] for result in results] == [1, 1]
    assert app.get_db().execute("SELECT like_count FROM music_publications WHERE id=?", (publication,)).fetchone()[0] == 1


def test_account_playlist_private_share_edit_reorder_and_delete(music, keys):
    harness, alice, account = music
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    publication = publish(harness, alice).json["id"]
    playlist_id = f"playlist-{uuid.uuid4()}"
    body = {"id": playlist_id, "title": "My private collection"}
    path = f"/v2/music/playlists/{playlist_id}"
    public_path = f"/music/playlists/{playlist_id}"
    assert harness.client.post("/v2/music/playlists", json=body).status_code == 401
    assert harness.client.post("/v2/music/playlists", headers=alice, json={**body, "wallet": harness.wallet}).status_code == 422
    response = harness.client.post("/v2/music/playlists", headers=alice, json=body)
    assert response.status_code == 201, response.json
    assert response.json["is_owner"] is True
    assert response.json["is_public"] is False
    assert harness.client.post("/v2/music/playlists", headers=alice, json=body).status_code == 200
    assert harness.client.post("/v2/music/playlists", headers=alice, json={**body, "title": "changed"}).status_code == 409
    assert harness.client.post("/v2/music/playlists", headers=bob, json=body).status_code == 404
    assert harness.client.get(public_path).status_code == 404
    assert harness.client.get(path, headers=bob).status_code == 404
    item = f"{path}/items/{publication}"
    for _ in range(2):
        response = harness.client.put(item, headers=alice)
        assert response.status_code == 200, response.json
        assert response.json["track_count"] == 1
    assert harness.client.put(item, headers=bob).status_code == 404
    assert harness.client.put(item, headers=alice, json={"account_id": account}).status_code == 422
    assert harness.client.put(f"{path}/reorder", headers=alice, json={"publication_ids": []}).status_code == 409
    assert harness.client.put(f"{path}/reorder", headers=alice, json={"publication_ids": [publication]}).status_code == 200
    assert harness.client.patch(path, headers=alice, json={"is_public": True}).status_code == 200
    public = harness.client.get(public_path)
    assert public.status_code == 200
    assert public.json["is_owner"] is False
    assert public.json["track_count"] == 1
    assert account not in json.dumps(public.json)
    assert "job-1" not in json.dumps(public.json)
    profile = app.get_db().execute("SELECT id FROM account_public_profiles WHERE account_id=?", (account,)).fetchone()[0]
    assert harness.client.get(f"/music/creators/{profile}").json["playlists"][0]["id"] == playlist_id
    # Even preserved wallet provenance after an explicit import must not grant access.
    app.get_db().execute("UPDATE music_playlists SET owner_wallet=? WHERE id=?", (harness.wallet, playlist_id))
    app.get_db().commit()
    assert harness.client.get(path, headers=bob).json["is_owner"] is False
    assert harness.client.patch(path, headers=bob, json={"title": "take over"}).status_code == 404
    assert harness.client.delete(path, headers=bob).status_code == 404
    assert not music_discover.update_playlist(playlist_id, owner_wallet=harness.wallet, title="wallet bypass")["ok"]
    assert len(harness.client.get("/v2/music/library", headers=alice).json["playlists"]) == 1
    assert harness.client.get("/v2/music/playlists", headers=bob).json["playlists"] == []
    assert harness.client.patch(path, headers=alice, json={"is_public": False}).status_code == 200
    assert harness.client.get(public_path).status_code == 404
    assert music_discover.get_playlist(playlist_id, requester_wallet=harness.wallet) is None
    assert music_discover.list_playlists(harness.wallet, requester_wallet=harness.wallet)["playlists"] == []
    assert harness.client.get(f"/music/creators/{profile}").json["playlists"] == []
    # An unpublished track disappears from public/private playback, but can be removed.
    assert harness.client.delete(f"/v2/music/publications/{publication}", headers=alice).status_code == 200
    assert harness.client.get(path, headers=alice).json["track_count"] == 0
    assert harness.client.put(f"{path}/reorder", headers=alice, json={"publication_ids": []}).status_code == 200
    assert harness.client.delete(item, headers=alice).status_code == 200
    assert harness.client.delete(path, headers=alice).status_code == 200
    assert harness.client.get(path, headers=alice).status_code == 404
    assert app.get_db().execute("SELECT COUNT(*) FROM music_playlist_items WHERE playlist_id=?", (playlist_id,)).fetchone()[0] == 0


def test_concurrent_account_playlist_creation_is_one_record(music, monkeypatch):
    _, _, account = music
    path = app.DB_PATH
    local = threading.local()
    monkeypatch.setattr(music_discover, "get_db", lambda: local.conn)
    barrier = threading.Barrier(2)
    data = {"id": f"playlist-{uuid.uuid4()}", "title": "One playlist"}
    def worker():
        local.conn = sqlite3.connect(path, timeout=10)
        local.conn.row_factory = sqlite3.Row
        try:
            barrier.wait()
            return account_playlists.create(account, data)
        finally:
            local.conn.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: worker(), range(2)))
    assert {result[0]["id"] for result in results} == {data["id"]}
    assert sorted(result[1] for result in results) == [False, True]
    assert app.get_db().execute("SELECT COUNT(*) FROM music_playlists WHERE owner_account_id=?", (account,)).fetchone()[0] == 1

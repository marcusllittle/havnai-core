"""Tests for the public HavnAI music Discover publication layer."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "server" / "music_discover.py"
SPEC = importlib.util.spec_from_file_location("havnai_server_music_discover", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
music_discover = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = music_discover
SPEC.loader.exec_module(music_discover)


WALLET = "0x" + "a" * 40
OTHER_WALLET = "0x" + "b" * 40


class MusicDiscoverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                wallet TEXT NOT NULL,
                model TEXT NOT NULL,
                data TEXT,
                task_type TEXT NOT NULL,
                weight REAL NOT NULL,
                status TEXT NOT NULL,
                node_id TEXT,
                timestamp REAL NOT NULL,
                assigned_at REAL,
                completed_at REAL,
                invite_code TEXT,
                resolved_spec TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                attempt_id TEXT,
                kind TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        music_discover.get_db = lambda: self.conn
        music_discover.log_event = lambda *args, **kwargs: None
        music_discover.WALLET_REGEX = type("Regex", (), {"match": staticmethod(lambda value: value.startswith("0x") and len(value) == 42)})()
        music_discover.artifact_url = lambda path: "/static/outputs/music.mp3" if path.startswith("/safe/") else None
        music_discover.init_music_discover_tables(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def _insert_music_job(
        self,
        *,
        job_id: str = "job-1",
        artifact_id: str = "artifact-1",
        wallet: str = WALLET,
        status: str = "succeeded",
        task_type: str = "MUSIC_GEN",
    ) -> None:
        resolved = {
            "parameters": {
                "prompt": "private prompt should not leak",
                "style": "Dream pop, cinematic",
                "duration": 90,
                "bpm": 118,
                "key": "A Minor",
                "instrumental": True,
            }
        }
        self.conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, task_type, weight, status, timestamp, completed_at, resolved_spec)
            VALUES (?, ?, 'ace_step_1_5_turbo', ?, 1, ?, ?, ?, ?)
            """,
            (job_id, wallet, task_type, status, time.time(), time.time(), json.dumps(resolved)),
        )
        self.conn.execute(
            """
            INSERT INTO artifacts (id, job_id, kind, filename, content_type, path, size_bytes, sha256, metadata, created_at)
            VALUES (?, ?, 'audio', 'song.mp3', 'audio/mpeg', '/safe/song.mp3', 20, 'hash', ?, ?)
            """,
            (artifact_id, job_id, json.dumps({"duration": 91, "bpm": 120, "keyscale": "B Minor"}), time.time()),
        )
        self.conn.commit()

    def test_publish_completed_music_job_without_exposing_prompt_or_paths(self) -> None:
        self._insert_music_job()
        result = music_discover.publish_song(
            job_id="job-1",
            creator_wallet=WALLET,
            title="Neon Coast",
            style="Dream pop, night drive",
        )

        self.assertTrue(result["ok"])
        publication = result["publication"]
        self.assertEqual(publication["title"], "Neon Coast")
        self.assertEqual(publication["audio_url"], f"/api/music/publications/{publication['id']}/audio")
        self.assertEqual(publication["bpm"], 120)
        self.assertEqual(publication["key"], "B Minor")
        self.assertEqual(publication["job_id"], "job-1")
        self.assertNotIn("prompt", publication)
        self.assertNotIn("/safe/song.mp3", json.dumps(publication))

        public_publication = music_discover.browse_publications()["publications"][0]
        self.assertEqual(public_publication["audio_url"], f"/api/music/publications/{publication['id']}/audio")
        self.assertNotIn("job_id", public_publication)
        self.assertNotIn("audio_artifact_id", public_publication)
        self.assertNotIn("model", public_publication)
        self.assertNotIn("cover_art_seed", public_publication)
        self.assertNotIn("prompt", public_publication)

    def test_rejects_wrong_wallet_incomplete_or_non_music_jobs(self) -> None:
        self._insert_music_job()
        wrong_wallet = music_discover.publish_song(job_id="job-1", creator_wallet=OTHER_WALLET, title="Nope")
        self.assertEqual(wrong_wallet["error"], "not_your_job")

        self.conn.execute("UPDATE jobs SET status='running' WHERE id='job-1'")
        incomplete = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Nope")
        self.assertEqual(incomplete["error"], "job_not_completed")

        self.conn.execute("UPDATE jobs SET status='succeeded', task_type='IMAGE_GEN' WHERE id='job-1'")
        non_music = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Nope")
        self.assertEqual(non_music["error"], "not_music_job")

    def test_prevents_duplicate_active_publication_for_job(self) -> None:
        self._insert_music_job()
        first = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="First")
        second = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Second")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(second["publication"]["already_published"])
        self.assertEqual(first["publication"]["id"], second["publication"]["id"])

    def test_likes_are_idempotent_and_wallet_scoped(self) -> None:
        self._insert_music_job()
        publication_id = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Song")["publication"]["id"]

        first = music_discover.set_like(publication_id, OTHER_WALLET, liked=True)
        second = music_discover.set_like(publication_id, OTHER_WALLET, liked=True)
        unlike = music_discover.set_like(publication_id, OTHER_WALLET, liked=False)

        self.assertEqual(first["like_count"], 1)
        self.assertEqual(second["like_count"], 1)
        self.assertEqual(unlike["like_count"], 0)

    def test_play_count_requires_listening_and_dedupes_recent_listener(self) -> None:
        self._insert_music_job()
        publication_id = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Song")["publication"]["id"]

        short = music_discover.record_play(publication_id, listener_key="listener", seconds_listened=2)
        counted = music_discover.record_play(publication_id, listener_key="listener", seconds_listened=8)
        duplicate = music_discover.record_play(publication_id, listener_key="listener", seconds_listened=12)

        self.assertFalse(short["counted"])
        self.assertTrue(counted["counted"])
        self.assertFalse(duplicate["counted"])
        self.assertEqual(music_discover.get_publication(publication_id)["play_count"], 1)

    def test_library_saves_are_idempotent_and_do_not_change_ownership(self) -> None:
        self._insert_music_job()
        publication_id = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Song")["publication"]["id"]

        first = music_discover.set_saved(publication_id, OTHER_WALLET, saved=True)
        second = music_discover.set_saved(publication_id, OTHER_WALLET, saved=True)
        saved = music_discover.list_saved(wallet=OTHER_WALLET)
        removed = music_discover.set_saved(publication_id, OTHER_WALLET, saved=False)

        self.assertTrue(first["saved"])
        self.assertTrue(second["saved"])
        self.assertEqual(saved["total"], 1)
        self.assertEqual(saved["publications"][0]["creator_wallet"], WALLET)
        self.assertTrue(saved["publications"][0]["saved_by_me"])
        self.assertFalse(removed["saved"])
        self.assertEqual(music_discover.list_saved(wallet=OTHER_WALLET)["total"], 0)

    def test_library_hides_saved_songs_after_unpublish(self) -> None:
        self._insert_music_job()
        publication_id = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Song")["publication"]["id"]
        music_discover.set_saved(publication_id, OTHER_WALLET, saved=True)

        before = music_discover.list_saved(wallet=OTHER_WALLET)
        music_discover.unpublish_song(publication_id, WALLET)
        after = music_discover.list_saved(wallet=OTHER_WALLET)

        self.assertEqual([item["id"] for item in before["publications"]], [publication_id])
        self.assertEqual(after["total"], 0)
        self.assertEqual(after["publications"], [])

    def test_playlist_owner_access_ordering_and_public_visibility(self) -> None:
        self._insert_music_job(job_id="job-1", artifact_id="artifact-1")
        self._insert_music_job(job_id="job-2", artifact_id="artifact-2")
        pub_one = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="One")["publication"]["id"]
        pub_two = music_discover.publish_song(job_id="job-2", creator_wallet=WALLET, title="Two")["publication"]["id"]

        created = music_discover.create_playlist(owner_wallet=WALLET, title="Night Set")
        playlist_id = created["playlist"]["id"]
        self.assertIsNone(music_discover.get_playlist(playlist_id, requester_wallet=None))
        self.assertEqual(created["playlist"]["artwork_url"], "")
        self.assertEqual(
            music_discover.add_playlist_item(playlist_id, pub_one, owner_wallet=OTHER_WALLET)["error"],
            "playlist_not_found",
        )

        music_discover.add_playlist_item(playlist_id, pub_one, owner_wallet=WALLET)
        music_discover.add_playlist_item(playlist_id, pub_one, owner_wallet=WALLET)
        music_discover.add_playlist_item(playlist_id, pub_two, owner_wallet=WALLET)
        private = music_discover.get_playlist(playlist_id, requester_wallet=WALLET)
        self.assertEqual(private["artwork_url"], "")
        self.assertEqual([item["id"] for item in private["publications"]], [pub_one, pub_two])
        self.assertEqual(private["artwork_tiles"], [
            f"/api/music/publications/{pub_one}/cover.svg",
            f"/api/music/publications/{pub_two}/cover.svg",
        ])
        reordered = music_discover.reorder_playlist_items(playlist_id, [pub_two, pub_one], owner_wallet=WALLET)
        self.assertEqual([item["id"] for item in reordered["playlist"]["publications"]], [pub_two, pub_one])
        self.assertEqual(reordered["playlist"]["artwork_tiles"], [
            f"/api/music/publications/{pub_two}/cover.svg",
            f"/api/music/publications/{pub_one}/cover.svg",
        ])
        deduped_reorder = music_discover.reorder_playlist_items(
            playlist_id,
            [pub_one, pub_one, "missing-publication"],
            owner_wallet=WALLET,
        )
        self.assertEqual([item["id"] for item in deduped_reorder["playlist"]["publications"]], [pub_one, pub_two])

        music_discover.update_playlist(playlist_id, owner_wallet=WALLET, is_public=True)
        public = music_discover.get_playlist(playlist_id, requester_wallet=None)
        self.assertIsNotNone(public)
        self.assertTrue(public["artwork_url"].endswith(f"/music/playlists/{playlist_id}/cover.svg"))
        self.assertEqual(public["artwork_tiles"], [
            f"/api/music/publications/{pub_one}/cover.svg",
            f"/api/music/publications/{pub_two}/cover.svg",
        ])
        self.assertEqual(public["track_count"], 2)

        music_discover.unpublish_song(pub_two, WALLET)
        after_unpublish = music_discover.get_playlist(playlist_id, requester_wallet=None)
        self.assertEqual([item["id"] for item in after_unpublish["publications"]], [pub_one])

    def test_creator_summary_uses_public_tracks_and_public_playlists(self) -> None:
        self._insert_music_job()
        publication_id = music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Creator Song")["publication"]["id"]
        music_discover.set_like(publication_id, OTHER_WALLET, liked=True)
        playlist = music_discover.create_playlist(owner_wallet=WALLET, title="Public Mix", is_public=True)["playlist"]
        music_discover.add_playlist_item(playlist["id"], publication_id, owner_wallet=WALLET)

        creator = music_discover.get_creator(WALLET, requester_wallet=OTHER_WALLET)

        self.assertEqual(creator["wallet"], WALLET)
        self.assertEqual(creator["track_count"], 1)
        self.assertEqual(creator["like_count"], 1)
        self.assertEqual(creator["publications"][0]["id"], publication_id)
        self.assertEqual(creator["playlists"][0]["id"], playlist["id"])

    def test_public_creator_summary_never_includes_private_playlists(self) -> None:
        self._insert_music_job()
        music_discover.publish_song(job_id="job-1", creator_wallet=WALLET, title="Creator Song")
        private_playlist = music_discover.create_playlist(owner_wallet=WALLET, title="Private Mix")["playlist"]

        creator = music_discover.get_creator(WALLET, requester_wallet=WALLET)

        self.assertEqual(private_playlist["is_public"], False)
        self.assertEqual(creator["playlists"], [])


if __name__ == "__main__":
    unittest.main()

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

    def _insert_music_job(self, *, wallet: str = WALLET, status: str = "succeeded", task_type: str = "MUSIC_GEN") -> None:
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
            VALUES ('job-1', ?, 'ace_step_1_5_turbo', ?, 1, ?, ?, ?, ?)
            """,
            (wallet, task_type, status, time.time(), time.time(), json.dumps(resolved)),
        )
        self.conn.execute(
            """
            INSERT INTO artifacts (id, job_id, kind, filename, content_type, path, size_bytes, sha256, metadata, created_at)
            VALUES ('artifact-1', 'job-1', 'audio', 'song.mp3', 'audio/mpeg', '/safe/song.mp3', 20, 'hash', ?, ?)
            """,
            (json.dumps({"duration": 91, "bpm": 120, "keyscale": "B Minor"}), time.time()),
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
        self.assertEqual(publication["audio_url"], "/static/outputs/music.mp3")
        self.assertEqual(publication["bpm"], 120)
        self.assertEqual(publication["key"], "B Minor")
        self.assertNotIn("prompt", publication)
        self.assertNotIn("/safe/song.mp3", json.dumps(publication))

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


if __name__ == "__main__":
    unittest.main()

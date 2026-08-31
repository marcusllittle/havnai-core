"""Route-level tests for Music Discover publication APIs."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module


class MusicDiscoverApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self.account = app_module.Account.create()
        self.wallet = self.account.address.lower()
        self._orig_db_path = app_module.DB_PATH
        self._orig_db_conn = app_module.DB_CONN
        self._orig_nodes = copy.deepcopy(app_module.NODES)
        self._orig_tasks = copy.deepcopy(app_module.TASKS)

        self._tmpdir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = Path(self._tmpdir.name) / "ledger.db"
        app_module.DB_CONN = None
        app_module.NODES.clear()
        app_module.TASKS.clear()
        app_module.init_db()
        app_module.stripe_payments.init_stripe_tables(app_module.get_db())
        app_module.settlement.init_settlement_tables(app_module.get_db())
        app_module.hai_funding.init_hai_funding_tables(app_module.get_db())
        app_module.blockchain.init_blockchain_tables(app_module.get_db())
        app_module.validators.init_validator_tables(app_module.get_db())
        app_module.workflows.init_workflow_tables(app_module.get_db())
        app_module.gallery.init_gallery_tables(app_module.get_db())
        app_module.music_discover.init_music_discover_tables(app_module.get_db())
        self._insert_completed_music_job()

    def tearDown(self) -> None:
        if app_module.DB_CONN is not None:
            app_module.DB_CONN.close()
        app_module.DB_PATH = self._orig_db_path
        app_module.DB_CONN = self._orig_db_conn
        app_module.NODES.clear()
        app_module.NODES.update(self._orig_nodes)
        app_module.TASKS.clear()
        app_module.TASKS.update(self._orig_tasks)
        self._tmpdir.cleanup()

    def _insert_completed_music_job(self) -> None:
        now = time.time()
        resolved = {
            "parameters": {
                "prompt": "private prompt",
                "style": "Synthwave, cinematic",
                "duration": 60,
                "instrumental": True,
            }
        }
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (
                id, wallet, model, data, task_type, weight, status, timestamp,
                completed_at, stage, progress, updated_at, resolved_spec
            ) VALUES (?, ?, ?, '{}', 'MUSIC_GEN', 1, 'succeeded', ?, ?, 'succeeded', 100, ?, ?)
            """,
            ("job-1", self.wallet, "ace_step_1_5_turbo", now, now, now, json.dumps(resolved)),
        )
        artifact_path = app_module.STATIC_DIR / "outputs" / "artifacts" / "job-1" / "song.mp3"
        conn.execute(
            """
            INSERT INTO artifacts (id, job_id, kind, filename, content_type, path, size_bytes, sha256, metadata, created_at)
            VALUES ('artifact-1', 'job-1', 'audio', 'song.mp3', 'audio/mpeg', ?, 10, 'hash', ?, ?)
            """,
            (str(artifact_path), json.dumps({"duration": 61, "bpm": 122, "keyscale": "C Minor"}), now),
        )
        conn.commit()

    def _signed_payload(self, purpose: str, **extra: object) -> dict[str, object]:
        nonce = self.client.post(
            "/wallet/nonce",
            json={"wallet": self.wallet, "purpose": purpose, **extra},
        )
        self.assertEqual(nonce.status_code, 200, nonce.get_data(as_text=True))
        challenge = nonce.get_json()
        signature = self.account.sign_message(app_module.encode_defunct(text=challenge["message"])).signature.hex()
        return {
            "wallet": self.wallet,
            "nonce": challenge["nonce"],
            "signature": signature,
        }

    def test_publish_discover_like_and_count_genuine_play(self) -> None:
        publish_payload = {
            **self._signed_payload("music_publish", job_id="job-1"),
            "job_id": "job-1",
            "title": "Midnight Relay",
            "style": "Synthwave, cinematic",
            "tags": ["Synthwave", "Cinematic"],
        }
        publish = self.client.post("/music/publications", json=publish_payload)
        self.assertEqual(publish.status_code, 201, publish.get_data(as_text=True))
        publication = publish.get_json()
        publication_id = publication["id"]
        self.assertEqual(publication["audio_url"], "/static/outputs/artifacts/job-1/song.mp3")
        self.assertNotIn("prompt", publication)

        discover = self.client.get(f"/music/discover?wallet={self.wallet}")
        self.assertEqual(discover.status_code, 200)
        self.assertEqual(discover.get_json()["publications"][0]["title"], "Midnight Relay")

        like_payload = {
            **self._signed_payload("music_like", publication_id=publication_id),
            "liked": True,
        }
        like = self.client.post(f"/music/publications/{publication_id}/like", json=like_payload)
        self.assertEqual(like.status_code, 200, like.get_data(as_text=True))
        self.assertEqual(like.get_json()["like_count"], 1)

        short_play = self.client.post(
            f"/music/publications/{publication_id}/play",
            json={"seconds_listened": 2, "session_id": "listener"},
        )
        counted_play = self.client.post(
            f"/music/publications/{publication_id}/play",
            json={"seconds_listened": 7, "session_id": "listener"},
        )
        self.assertFalse(short_play.get_json()["counted"])
        self.assertTrue(counted_play.get_json()["counted"])


if __name__ == "__main__":
    unittest.main()

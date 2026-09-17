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
        self.other_account = app_module.Account.create()
        self.other_wallet = self.other_account.address.lower()
        self._orig_db_path = app_module.DB_PATH
        self._orig_db_conn = app_module.DB_CONN
        self._orig_static_dir = app_module.STATIC_DIR
        self._orig_nodes = copy.deepcopy(app_module.NODES)
        self._orig_tasks = copy.deepcopy(app_module.TASKS)

        self._tmpdir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = Path(self._tmpdir.name) / "ledger.db"
        app_module.DB_CONN = None
        app_module.STATIC_DIR = Path(self._tmpdir.name) / "static"
        app_module.STATIC_DIR.mkdir(parents=True, exist_ok=True)
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
        app_module.STATIC_DIR = self._orig_static_dir
        app_module.NODES.clear()
        app_module.NODES.update(self._orig_nodes)
        app_module.TASKS.clear()
        app_module.TASKS.update(self._orig_tasks)
        self._tmpdir.cleanup()

    def _insert_completed_music_job(self, job_id: str = "job-1", artifact_id: str = "artifact-1") -> None:
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
            (job_id, self.wallet, "ace_step_1_5_turbo", now, now, now, json.dumps(resolved)),
        )
        artifact_path = app_module.STATIC_DIR / "outputs" / "artifacts" / job_id / "song.mp3"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"audio")
        conn.execute(
            """
            INSERT INTO artifacts (id, job_id, kind, filename, content_type, path, size_bytes, sha256, metadata, created_at)
            VALUES (?, ?, 'audio', 'song.mp3', 'audio/mpeg', ?, 10, 'hash', ?, ?)
            """,
            (artifact_id, job_id, str(artifact_path), json.dumps({"duration": 61, "bpm": 122, "keyscale": "C Minor"}), now),
        )
        conn.commit()

    def _signed_payload(
        self,
        purpose: str,
        *,
        wallet: str | None = None,
        account: object | None = None,
        **extra: object,
    ) -> dict[str, object]:
        signer_wallet = wallet or self.wallet
        signer_account = account or self.account
        nonce = self.client.post(
            "/wallet/nonce",
            json={"wallet": signer_wallet, "purpose": purpose, **extra},
        )
        self.assertEqual(nonce.status_code, 200, nonce.get_data(as_text=True))
        challenge = nonce.get_json()
        signature = signer_account.sign_message(app_module.encode_defunct(text=challenge["message"])).signature.hex()
        return {
            "wallet": signer_wallet,
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
        self.assertEqual(publication["audio_url"], f"/api/music/publications/{publication_id}/audio")
        self.assertNotIn("prompt", publication)

        discover = self.client.get(f"/music/discover?wallet={self.wallet}")
        self.assertEqual(discover.status_code, 200)
        public_publication = discover.get_json()["publications"][0]
        self.assertEqual(public_publication["title"], "Midnight Relay")
        self.assertEqual(public_publication["audio_url"], f"/api/music/publications/{publication_id}/audio")
        self.assertNotIn("job_id", public_publication)
        self.assertNotIn("audio_artifact_id", public_publication)
        self.assertNotIn("model", public_publication)
        self.assertNotIn("cover_art_seed", public_publication)
        self.assertNotIn("/artifacts/job-1/", json.dumps(public_publication))

        audio = self.client.get(f"/music/publications/{publication_id}/audio")
        self.assertEqual(audio.status_code, 200)
        audio_data = audio.get_data()
        audio.close()
        self.assertEqual(audio_data, b"audio")

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

    def test_save_playlist_private_access_and_creator_routes(self) -> None:
        publish_payload = {
            **self._signed_payload("music_publish", job_id="job-1"),
            "job_id": "job-1",
            "title": "Library Track",
            "style": "Synthwave",
        }
        publish = self.client.post("/music/publications", json=publish_payload)
        self.assertEqual(publish.status_code, 201, publish.get_data(as_text=True))
        publication_id = publish.get_json()["id"]
        self._insert_completed_music_job(job_id="job-2", artifact_id="artifact-2")
        second_publish_payload = {
            **self._signed_payload("music_publish", job_id="job-2"),
            "job_id": "job-2",
            "title": "Library Track Two",
            "style": "Synthwave",
        }
        second_publish = self.client.post("/music/publications", json=second_publish_payload)
        self.assertEqual(second_publish.status_code, 201, second_publish.get_data(as_text=True))
        second_publication_id = second_publish.get_json()["id"]

        save_payload = self._signed_payload("music_save", publication_id=publication_id)
        saved = self.client.post(f"/music/publications/{publication_id}/save", json=save_payload)
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        self.assertTrue(saved.get_json()["saved"])
        spoofed_discover = self.client.get(f"/music/discover?wallet={self.wallet}")
        self.assertEqual(spoofed_discover.status_code, 200, spoofed_discover.get_data(as_text=True))
        self.assertFalse(spoofed_discover.get_json()["publications"][0]["saved_by_me"])
        signed_discover_payload = self._signed_payload("music_library_read")
        signed_discover = self.client.post(
            "/music/discover",
            json={**signed_discover_payload, "creator_wallet": self.wallet, "limit": 3},
        )
        self.assertEqual(signed_discover.status_code, 200, signed_discover.get_data(as_text=True))
        signed_owner_publication = next(
            item for item in signed_discover.get_json()["publications"] if item["id"] == publication_id
        )
        self.assertTrue(signed_owner_publication["saved_by_me"])
        self.assertEqual(signed_owner_publication["job_id"], "job-1")
        self.assertEqual(signed_owner_publication["audio_artifact_id"], "artifact-1")
        spoofed_detail = self.client.get(f"/music/discover/{publication_id}?wallet={self.wallet}")
        self.assertEqual(spoofed_detail.status_code, 200, spoofed_detail.get_data(as_text=True))
        self.assertFalse(spoofed_detail.get_json()["saved_by_me"])
        signed_detail_payload = self._signed_payload("music_library_read", publication_id=publication_id)
        signed_detail = self.client.post(f"/music/discover/{publication_id}", json=signed_detail_payload)
        self.assertEqual(signed_detail.status_code, 200, signed_detail.get_data(as_text=True))
        self.assertTrue(signed_detail.get_json()["saved_by_me"])
        self.assertEqual(self.client.get(f"/music/library?wallet={self.wallet}").status_code, 401)
        saved_state_payload = self._signed_payload("music_library_read", publication_id=publication_id)
        saved_state = self.client.post(f"/music/publications/{publication_id}/saved", json=saved_state_payload)
        self.assertEqual(saved_state.status_code, 200, saved_state.get_data(as_text=True))
        self.assertTrue(saved_state.get_json()["saved"])
        library_payload = self._signed_payload("music_library_read")
        library = self.client.post("/music/library", json=library_payload)
        self.assertEqual(library.status_code, 200)
        self.assertEqual(library.get_json()["publications"][0]["id"], publication_id)
        self.assertNotIn("job_id", library.get_json()["publications"][0])

        create_payload = {
            **self._signed_payload("playlist_create", playlist_id="new"),
            "title": "Private Set",
            "description": "Owner only",
        }
        created = self.client.post("/music/playlists", json=create_payload)
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        playlist_id = created.get_json()["id"]
        private_library_payload = self._signed_payload("music_library_read")
        private_library = self.client.post("/music/library", json=private_library_payload)
        self.assertEqual(private_library.status_code, 200, private_library.get_data(as_text=True))
        self.assertEqual(private_library.get_json()["playlists"][0]["id"], playlist_id)
        self.assertTrue(private_library.get_json()["playlists"][0]["is_owner"])
        self.assertFalse(private_library.get_json()["playlists"][0]["is_public"])

        add_payload = {
            **self._signed_payload("playlist_add", playlist_id=playlist_id, publication_id=publication_id),
            "publication_id": publication_id,
        }
        added = self.client.post(f"/music/playlists/{playlist_id}/items", json=add_payload)
        self.assertEqual(added.status_code, 200, added.get_data(as_text=True))
        self.assertEqual(added.get_json()["publications"][0]["id"], publication_id)
        duplicate_add_payload = {
            **self._signed_payload("playlist_add", playlist_id=playlist_id, publication_id=publication_id),
            "publication_id": publication_id,
        }
        duplicate_added = self.client.post(f"/music/playlists/{playlist_id}/items", json=duplicate_add_payload)
        self.assertEqual(duplicate_added.status_code, 200, duplicate_added.get_data(as_text=True))
        self.assertEqual([item["id"] for item in duplicate_added.get_json()["publications"]], [publication_id])
        second_add_payload = {
            **self._signed_payload("playlist_add", playlist_id=playlist_id, publication_id=second_publication_id),
            "publication_id": second_publication_id,
        }
        second_added = self.client.post(f"/music/playlists/{playlist_id}/items", json=second_add_payload)
        self.assertEqual(second_added.status_code, 200, second_added.get_data(as_text=True))
        self.assertEqual(
            [item["id"] for item in second_added.get_json()["publications"]],
            [publication_id, second_publication_id],
        )
        reorder_payload = {
            **self._signed_payload("playlist_reorder", playlist_id=playlist_id),
            "publication_ids": [second_publication_id, publication_id],
        }
        reordered = self.client.post(f"/music/playlists/{playlist_id}/reorder", json=reorder_payload)
        self.assertEqual(reordered.status_code, 200, reordered.get_data(as_text=True))
        self.assertEqual(
            [item["id"] for item in reordered.get_json()["publications"]],
            [second_publication_id, publication_id],
        )
        self.assertEqual(self.client.get(f"/music/playlists/{playlist_id}").status_code, 404)

        access_payload = self._signed_payload("playlist_read", playlist_id=playlist_id)
        private_detail = self.client.post(f"/music/playlists/{playlist_id}/access", json=access_payload)
        self.assertEqual(private_detail.status_code, 200, private_detail.get_data(as_text=True))
        self.assertTrue(private_detail.get_json()["is_owner"])
        self.assertEqual(private_detail.get_json()["artwork_url"], "")
        self.assertEqual(self.client.get(f"/music/playlists/{playlist_id}/cover.svg").status_code, 404)
        other_add_payload = {
            **self._signed_payload(
                "playlist_add",
                wallet=self.other_wallet,
                account=self.other_account,
                playlist_id=playlist_id,
                publication_id=second_publication_id,
            ),
            "publication_id": second_publication_id,
        }
        other_add = self.client.post(f"/music/playlists/{playlist_id}/items", json=other_add_payload)
        self.assertEqual(other_add.status_code, 404, other_add.get_data(as_text=True))
        other_update_payload = {
            **self._signed_payload(
                "playlist_update",
                wallet=self.other_wallet,
                account=self.other_account,
                playlist_id=playlist_id,
            ),
            "title": "Takeover",
        }
        other_update = self.client.patch(f"/music/playlists/{playlist_id}", json=other_update_payload)
        self.assertEqual(other_update.status_code, 404, other_update.get_data(as_text=True))
        other_reorder_payload = {
            **self._signed_payload(
                "playlist_reorder",
                wallet=self.other_wallet,
                account=self.other_account,
                playlist_id=playlist_id,
            ),
            "publication_ids": [publication_id, second_publication_id],
        }
        other_reorder = self.client.post(f"/music/playlists/{playlist_id}/reorder", json=other_reorder_payload)
        self.assertEqual(other_reorder.status_code, 404, other_reorder.get_data(as_text=True))
        other_remove_payload = self._signed_payload(
            "playlist_remove",
            wallet=self.other_wallet,
            account=self.other_account,
            playlist_id=playlist_id,
            publication_id=publication_id,
        )
        other_remove = self.client.delete(f"/music/playlists/{playlist_id}/items/{publication_id}", json=other_remove_payload)
        self.assertEqual(other_remove.status_code, 404, other_remove.get_data(as_text=True))
        other_delete_payload = self._signed_payload(
            "playlist_delete",
            wallet=self.other_wallet,
            account=self.other_account,
            playlist_id=playlist_id,
        )
        other_delete = self.client.delete(f"/music/playlists/{playlist_id}", json=other_delete_payload)
        self.assertEqual(other_delete.status_code, 404, other_delete.get_data(as_text=True))

        spoofed_creator = self.client.get(f"/music/creator/{self.wallet}?wallet={self.wallet}")
        self.assertEqual(spoofed_creator.status_code, 200, spoofed_creator.get_data(as_text=True))
        self.assertEqual(spoofed_creator.get_json()["playlists"], [])
        self.assertFalse(spoofed_creator.get_json()["publications"][0]["saved_by_me"])

        update_payload = {
            **self._signed_payload("playlist_update", playlist_id=playlist_id),
            "is_public": True,
        }
        updated = self.client.patch(f"/music/playlists/{playlist_id}", json=update_payload)
        self.assertEqual(updated.status_code, 200, updated.get_data(as_text=True))
        public_detail = self.client.get(f"/music/playlists/{playlist_id}")
        self.assertEqual(public_detail.status_code, 200)
        self.assertFalse(public_detail.get_json()["is_owner"])
        self.assertTrue(public_detail.get_json()["artwork_url"].endswith(f"/music/playlists/{playlist_id}/cover.svg"))
        playlist_cover = self.client.get(f"/music/playlists/{playlist_id}/cover.svg")
        self.assertEqual(playlist_cover.status_code, 200)
        playlist_cover_svg = playlist_cover.get_data(as_text=True)
        self.assertIn(f'/api/music/publications/{second_publication_id}/cover.svg', playlist_cover_svg)
        self.assertIn(f'/api/music/publications/{publication_id}/cover.svg', playlist_cover_svg)
        self.assertEqual(playlist_cover_svg.count("<image href="), 2)

        creator = self.client.get(f"/music/creator/{self.wallet}")
        self.assertEqual(creator.status_code, 200, creator.get_data(as_text=True))
        self.assertEqual(creator.get_json()["track_count"], 2)
        self.assertEqual(creator.get_json()["playlists"][0]["id"], playlist_id)
        signed_creator_payload = self._signed_payload("music_library_read")
        signed_creator = self.client.post(f"/music/creator/{self.wallet}", json={**signed_creator_payload, "sort": "newest"})
        self.assertEqual(signed_creator.status_code, 200, signed_creator.get_data(as_text=True))
        self.assertTrue(
            next(item for item in signed_creator.get_json()["publications"] if item["id"] == publication_id)["saved_by_me"]
        )
        play_all_seed = self.client.post(
            f"/music/publications/{second_publication_id}/play",
            json={"seconds_listened": 8, "session_id": "playlist-listener"},
        )
        self.assertEqual(play_all_seed.status_code, 200, play_all_seed.get_data(as_text=True))
        self.assertTrue(play_all_seed.get_json()["counted"])

        remove_payload = self._signed_payload("playlist_remove", playlist_id=playlist_id, publication_id=publication_id)
        removed = self.client.delete(f"/music/playlists/{playlist_id}/items/{publication_id}", json=remove_payload)
        self.assertEqual(removed.status_code, 200, removed.get_data(as_text=True))
        self.assertEqual([item["id"] for item in removed.get_json()["publications"]], [second_publication_id])

        unsave_payload = self._signed_payload("music_unsave", publication_id=publication_id)
        unsaved = self.client.delete(f"/music/publications/{publication_id}/save", json=unsave_payload)
        self.assertEqual(unsaved.status_code, 200, unsaved.get_data(as_text=True))
        self.assertFalse(unsaved.get_json()["saved"])

        delete_payload = self._signed_payload("playlist_delete", playlist_id=playlist_id)
        deleted = self.client.delete(f"/music/playlists/{playlist_id}", json=delete_payload)
        self.assertEqual(deleted.status_code, 200, deleted.get_data(as_text=True))
        self.assertEqual(self.client.get(f"/music/discover/{publication_id}").status_code, 200)


if __name__ == "__main__":
    unittest.main()

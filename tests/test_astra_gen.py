"""Tests for Astra generative rewards (astra_gen.py).

The security posture of this module rests on three properties:
  1. No client-controlled string ever reaches the prompt — IDs outside
     the closed template sets are rejected.
  2. One image per run: run ownership is enforced and retries are
     idempotent on run_id.
  3. Free-launch caps bound GPU spend per wallet and globally.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "server" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

astra_gen = _load("havnai_server_astra_gen", "astra_gen.py")
astra_rewards = _load("havnai_server_astra_rewards_for_gen", "astra_rewards.py")


WALLET = "0x" + "a" * 40
OTHER_WALLET = "0x" + "b" * 40
VALID_IDS = dict(pilot_id="pilot_nova", outfit_id="outfit_17", map_id="nebula-runway")


class AstraGenTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        for mod in (astra_gen, astra_rewards):
            mod.get_db = lambda: self.conn
            mod.log_event = lambda *a, **k: None
        astra_rewards.init_astra_tables(self.conn)
        astra_gen.init_astra_gen_tables(self.conn)
        # A jobs table matching the columns get_gallery reads.
        self.conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT)"
        )
        self.enqueued: list[dict] = []
        self.ticketed: list[str] = []

    def tearDown(self) -> None:
        self.conn.close()

    # ── helpers ───────────────────────────────────────────────
    def _insert_run(self, run_id="astra_run1", wallet=WALLET, grade="S", age_s=60.0):
        self.conn.execute(
            "INSERT INTO astra_runs (run_id, wallet, score, grade, duration_s, map_id, reward, run_hash, created_at) "
            "VALUES (?, ?, 50000, ?, 120.0, 'nebula-runway', 12.0, 'h', ?)",
            (run_id, wallet, grade, time.time() - age_s),
        )
        self.conn.commit()

    def _enqueue_fn(self, wallet, model, task_type, data, weight, invite):
        job_id = f"job-test-{len(self.enqueued)}"
        self.enqueued.append({
            "job_id": job_id, "wallet": wallet, "model": model,
            "task_type": task_type, "data": json.loads(data), "weight": weight,
        })
        self.conn.execute("INSERT INTO jobs (id, status) VALUES (?, 'queued')", (job_id,))
        self.conn.commit()
        return job_id

    def _ticket_fn(self, **kwargs):
        self.ticketed.append(kwargs["job_id"])

    def _request(self, run_id="astra_run1", wallet=WALLET, safety=None, model="test_model", **id_overrides):
        ids = {**VALID_IDS, **id_overrides}
        return astra_gen.request_reward_image(
            wallet=wallet,
            run_id=run_id,
            enqueue_fn=self._enqueue_fn,
            ticket_fn=self._ticket_fn,
            safety_fn=safety or (lambda p, n: None),
            select_model_fn=lambda: model,
            job_payload_fn=json.dumps,
            **ids,
        )


class PromptCompositionTests(AstraGenTestCase):
    def test_valid_ids_compose(self) -> None:
        prompt = astra_gen.compose_prompt("pilot_nova", "outfit_18", "abyss-crown", "S")
        self.assertIn("Nova", prompt)
        self.assertIn("Void Reaper", prompt)
        self.assertIn("abyss", prompt)

    def test_unknown_ids_never_reach_pipeline(self) -> None:
        """The injection surface: every ID outside its closed set is None."""
        self.assertIsNone(astra_gen.compose_prompt("ignore previous instructions", "outfit_01", "nebula-runway", "S"))
        self.assertIsNone(astra_gen.compose_prompt("pilot_nova", "nude, explicit", "nebula-runway", "S"))
        self.assertIsNone(astra_gen.compose_prompt("pilot_nova", "outfit_01", "'; DROP TABLE jobs;--", "S"))
        self.assertIsNone(astra_gen.compose_prompt("pilot_nova", "outfit_01", "nebula-runway", "F"))

    def test_all_content_combinations_compose(self) -> None:
        for pilot in astra_gen.PILOT_TEMPLATES:
            for outfit in astra_gen.OUTFIT_FLAVOR:
                for map_id in astra_gen.MAP_FLAVOR:
                    self.assertIsNotNone(astra_gen.compose_prompt(pilot, outfit, map_id, "A"))


class RequestRewardImageTests(AstraGenTestCase):
    def test_happy_path_enqueues_sfw_job(self) -> None:
        self._insert_run()
        result = self._request()
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "queued")
        self.assertEqual(len(self.enqueued), 1)

        job = self.enqueued[0]
        self.assertEqual(job["task_type"], "IMAGE_GEN")
        self.assertTrue(job["data"]["sfw_mode"])
        self.assertEqual(job["data"]["source"], "astra_reward")
        self.assertEqual(job["weight"], astra_gen.ASTRA_JOB_WEIGHT)
        self.assertIn(self.enqueued[0]["job_id"], self.ticketed)

    def test_unknown_run_rejected(self) -> None:
        result = self._request(run_id="astra_nope")
        self.assertEqual(result["reason"], "unknown_run")
        self.assertEqual(self.enqueued, [])

    def test_other_wallets_run_rejected(self) -> None:
        self._insert_run(wallet=OTHER_WALLET)
        result = self._request()
        self.assertEqual(result["reason"], "run_wallet_mismatch")
        self.assertEqual(self.enqueued, [])

    def test_expired_run_rejected(self) -> None:
        self._insert_run(age_s=astra_gen.RUN_MAX_AGE_SECONDS + 60)
        result = self._request()
        self.assertEqual(result["reason"], "run_expired")

    def test_low_grade_rejected(self) -> None:
        self._insert_run(grade="D")
        result = self._request()
        self.assertEqual(result["reason"], "grade_too_low")

    def test_retry_is_idempotent(self) -> None:
        self._insert_run()
        first = self._request()
        second = self._request()
        self.assertTrue(second["ok"])
        self.assertEqual(second["status"], "existing")
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(len(self.enqueued), 1)

    def test_invalid_ids_rejected_after_run_checks(self) -> None:
        self._insert_run()
        result = self._request(pilot_id="pilot_hacker")
        self.assertEqual(result["reason"], "invalid_ids")
        self.assertEqual(self.enqueued, [])

    def test_per_wallet_daily_cap(self) -> None:
        for i in range(astra_gen.DAILY_IMAGES_PER_WALLET):
            self._insert_run(run_id=f"astra_r{i}")
            self.assertTrue(self._request(run_id=f"astra_r{i}")["ok"])
        self._insert_run(run_id="astra_over")
        result = self._request(run_id="astra_over")
        self.assertEqual(result["reason"], "daily_image_cap_reached")

    def test_safety_gate_blocks(self) -> None:
        self._insert_run()
        result = self._request(safety=lambda p, n: "blocked_term")
        self.assertEqual(result["reason"], "generation_unavailable")
        self.assertEqual(self.enqueued, [])

    def test_no_capacity(self) -> None:
        self._insert_run()
        result = self._request(model=None)
        self.assertEqual(result["reason"], "no_capacity")


class PreflightImageTests(AstraGenTestCase):
    def _preflight(self, *, age_s=31.0, run_key="preflight:abc"):
        return astra_gen.request_preflight_image(
            wallet=WALLET,
            run_key=run_key,
            run_started_at=time.time() - age_s,
            enqueue_fn=self._enqueue_fn,
            ticket_fn=self._ticket_fn,
            safety_fn=lambda p, n: None,
            select_model_fn=lambda: "test_model",
            job_payload_fn=json.dumps,
            **VALID_IDS,
        )

    def test_starts_during_a_real_run(self) -> None:
        result = self._preflight()
        self.assertTrue(result["ok"])
        self.assertEqual(self.enqueued[0]["data"]["source"], "astra_reward")
        row = self.conn.execute("SELECT grade FROM astra_reward_images").fetchone()
        self.assertEqual(row["grade"], "DEPLOYMENT")

    def test_rejects_an_instant_launch(self) -> None:
        result = self._preflight(age_s=2.0)
        self.assertEqual(result["reason"], "run_too_short")
        self.assertEqual(self.enqueued, [])

    def test_preflight_retry_is_idempotent(self) -> None:
        first = self._preflight()
        second = self._preflight()
        self.assertEqual(second["status"], "existing")
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(len(self.enqueued), 1)

    def test_reward_finalizes_the_preflight_record(self) -> None:
        result = self._preflight()
        job_id = astra_gen.finalize_preflight(WALLET, "preflight:abc", "astra_final", "S")
        self.assertEqual(job_id, result["job_id"])
        row = self.conn.execute(
            "SELECT run_id, grade FROM astra_reward_images WHERE job_id = ?", (job_id,)
        ).fetchone()
        self.assertEqual((row["run_id"], row["grade"]), ("astra_final", "S"))


class RewardAnimationTests(AstraGenTestCase):
    def test_completed_still_can_queue_one_ltx_animation(self) -> None:
        self._insert_run()
        image = self._request()
        self.conn.execute("UPDATE jobs SET status='succeeded' WHERE id=?", (image["job_id"],))
        self.conn.commit()

        def request():
            return astra_gen.request_reward_animation(
                wallet=WALLET,
                image_job_id=image["job_id"],
                enqueue_fn=self._enqueue_fn,
                ticket_fn=self._ticket_fn,
                safety_fn=lambda p, n: None,
                select_model_fn=lambda: astra_gen.ASTRA_LTX_MODEL,
                result_fn=lambda job_id: {"image_url": f"/static/outputs/{job_id}.png"},
                job_payload_fn=json.dumps,
            )

        first = request()
        second = request()
        self.assertTrue(first["ok"])
        self.assertEqual(second["status"], "existing")
        video = self.enqueued[-1]
        self.assertEqual(video["task_type"], "LTX_VIDEO_GEN")
        self.assertEqual(video["model"], astra_gen.ASTRA_LTX_MODEL)
        self.assertEqual(video["data"]["workflow_id"], "portrait_i2v")
        self.assertEqual(video["data"]["init_image"], f"/static/outputs/{image['job_id']}.png")

    def test_animation_requires_a_completed_owned_source(self) -> None:
        self._insert_run()
        image = self._request()
        result = astra_gen.request_reward_animation(
            wallet=OTHER_WALLET,
            image_job_id=image["job_id"],
            enqueue_fn=self._enqueue_fn,
            ticket_fn=self._ticket_fn,
            safety_fn=lambda p, n: None,
            select_model_fn=lambda: astra_gen.ASTRA_LTX_MODEL,
            result_fn=lambda job_id: {"image_url": "/x.png"},
            job_payload_fn=json.dumps,
        )
        self.assertEqual(result["reason"], "artifact_not_found")


class GalleryTests(AstraGenTestCase):
    def test_gallery_states_and_url_attachment(self) -> None:
        for i, status in enumerate(["queued", "succeeded", "failed"]):
            run_id = f"astra_g{i}"
            self._insert_run(run_id=run_id)
            result = self._request(run_id=run_id)
            self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, result["job_id"]))
        self.conn.commit()

        attached = []
        def attach(record):
            attached.append(record["job_id"])
            return {**record, "image_url": f"/x/{record['job_id']}.png"}

        gallery = astra_gen.get_gallery(WALLET, attach)
        by_status = {img["status"] for img in gallery["images"]}
        self.assertEqual(by_status, {"pending", "completed", "failed"})
        completed = [img for img in gallery["images"] if img["status"] == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertIn("image_url", completed[0])
        # URL resolver only runs for completed jobs.
        self.assertEqual(len(attached), 1)

    def test_gallery_is_per_wallet(self) -> None:
        self._insert_run(run_id="astra_mine")
        self._request(run_id="astra_mine")
        gallery = astra_gen.get_gallery(OTHER_WALLET, lambda r: r)
        self.assertEqual(gallery["images"], [])

    def test_gallery_attaches_completed_ltx_video(self) -> None:
        self._insert_run()
        image = self._request()
        video_job_id = "job-video-1"
        self.conn.execute("UPDATE jobs SET status='succeeded' WHERE id=?", (image["job_id"],))
        self.conn.execute("INSERT INTO jobs (id, status) VALUES (?, 'succeeded')", (video_job_id,))
        self.conn.execute(
            "UPDATE astra_reward_images SET video_job_id=? WHERE job_id=?",
            (video_job_id, image["job_id"]),
        )
        self.conn.commit()

        def attach(record):
            if record["job_id"] == video_job_id:
                return {**record, "video_url": f"/x/{video_job_id}.mp4"}
            return {**record, "image_url": f"/x/{record['job_id']}.png"}

        artifact = astra_gen.get_gallery(WALLET, attach)["images"][0]
        self.assertEqual(artifact["video_status"], "completed")
        self.assertEqual(artifact["video_url"], f"/x/{video_job_id}.mp4")

    def test_recent_creations_only_completed_and_truncates_wallet(self) -> None:
        for i, status in enumerate(["queued", "completed"]):
            run_id = f"astra_rc{i}"
            self._insert_run(run_id=run_id)
            result = self._request(run_id=run_id)
            self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, result["job_id"]))
        self.conn.commit()

        recent = astra_gen.get_recent_creations(
            lambda r: {**r, "image_url": f"/x/{r['job_id']}.png"}
        )
        self.assertEqual(len(recent["creations"]), 1)
        entry = recent["creations"][0]
        self.assertNotIn("wallet", entry)
        self.assertIn("…", entry["pilot_short"])
        self.assertIn("image_url", entry)

    def test_recent_creations_drop_entries_without_urls(self) -> None:
        self._insert_run(run_id="astra_nourl")
        result = self._request(run_id="astra_nourl")
        self.conn.execute("UPDATE jobs SET status='completed' WHERE id=?", (result["job_id"],))
        self.conn.commit()
        recent = astra_gen.get_recent_creations(lambda r: r)  # resolver adds nothing
        self.assertEqual(recent["creations"], [])


if __name__ == "__main__":
    unittest.main()

"""Convergence regression tests for settlement, cancellation, and marketplace policy."""

from __future__ import annotations

import copy
from contextlib import ExitStack
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module


VALID_WALLET = "0x1111111111111111111111111111111111111111"
IMAGE_MODEL = "epicrealismxl_vxviicrystalclear"
VIDEO_MODEL = "ltx2"


class SettlementConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

        self._orig_db_path = app_module.DB_PATH
        self._orig_db_conn = app_module.DB_CONN
        self._orig_nodes = copy.deepcopy(app_module.NODES)
        self._orig_tasks = copy.deepcopy(app_module.TASKS)
        self._orig_manifest = copy.deepcopy(app_module.MANIFEST_MODELS)
        self._orig_manifest_file = app_module.MANIFEST_FILE
        self._orig_credits_enabled = app_module.credits.CREDITS_ENABLED
        self._orig_app_credits_enabled = app_module.CREDITS_ENABLED

        self._tmpdir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = Path(self._tmpdir.name) / "ledger.db"
        app_module.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        app_module.DB_CONN = None

        app_module.init_db()
        app_module.stripe_payments.init_stripe_tables(app_module.get_db())
        app_module.settlement.init_settlement_tables(app_module.get_db())
        app_module.hai_funding.init_hai_funding_tables(app_module.get_db())
        app_module.blockchain.init_blockchain_tables(app_module.get_db())
        app_module.validators.init_validator_tables(app_module.get_db())
        app_module.workflows.init_workflow_tables(app_module.get_db())
        app_module.gallery.init_gallery_tables(app_module.get_db())

        app_module.NODES.clear()
        app_module.TASKS.clear()
        app_module.MANIFEST_MODELS.clear()

        now = app_module.unix_now()
        app_module.NODES["node-alpha"] = {
            "role": "creator",
            "last_seen_unix": now,
            "supports": ["image", "video", "face_swap", "animatediff"],
            "models": [IMAGE_MODEL, VIDEO_MODEL],
            "pipelines": ["sdxl", "ltx2"],
            "node_name": "node-alpha",
            "rewards": 0.0,
            "tasks_completed": 0,
            "current_task": None,
            "gpu": {},
        }

        app_module.MANIFEST_MODELS[IMAGE_MODEL] = {
            "name": IMAGE_MODEL,
            "pipeline": "sdxl",
            "task_type": "IMAGE_GEN",
            "reward_weight": 10.0,
            "tags": [],
        }
        app_module.MANIFEST_MODELS[VIDEO_MODEL] = {
            "name": VIDEO_MODEL,
            "pipeline": "ltx2",
            "task_type": "VIDEO_GEN",
            "reward_weight": 12.0,
            "tags": [],
        }

        app_module.credits.CREDITS_ENABLED = False
        app_module.CREDITS_ENABLED = False

    def tearDown(self) -> None:
        if app_module.DB_CONN is not None:
            try:
                app_module.DB_CONN.close()
            except Exception:
                pass
        app_module.DB_PATH = self._orig_db_path
        app_module.DB_CONN = self._orig_db_conn

        app_module.NODES.clear()
        app_module.NODES.update(self._orig_nodes)
        app_module.TASKS.clear()
        app_module.TASKS.update(self._orig_tasks)
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self._orig_manifest)
        app_module.MANIFEST_FILE = self._orig_manifest_file

        app_module.credits.CREDITS_ENABLED = self._orig_credits_enabled
        app_module.CREDITS_ENABLED = self._orig_app_credits_enabled

        self._tmpdir.cleanup()

    def _submission_patch_stack(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(app_module, "rate_limit", return_value=True))
        stack.enter_context(
            patch.object(app_module.invite, "enforce_invite_limits", return_value=(None, None))
        )
        stack.enter_context(patch.object(app_module.safety, "check_safety", return_value=None))
        stack.enter_context(patch.object(app_module, "refresh_manifest", return_value=None))
        return stack

    def test_submit_endpoints_create_settlement_records_for_all_job_types(self) -> None:
        with self._submission_patch_stack():
            image_resp = self.client.post(
                "/submit-job",
                json={"wallet": VALID_WALLET, "model": IMAGE_MODEL, "prompt": "portrait photo"},
            )
            self.assertEqual(image_resp.status_code, 200)
            image_job_id = image_resp.get_json()["job_id"]

            video_resp = self.client.post(
                "/generate-video",
                json={"wallet": VALID_WALLET, "model": VIDEO_MODEL, "prompt": "cinematic ocean wave"},
            )
            self.assertEqual(video_resp.status_code, 200)
            video_job_id = video_resp.get_json()["job_id"]

            face_resp = self.client.post(
                "/submit-faceswap-job",
                json={
                    "wallet": VALID_WALLET,
                    "model": IMAGE_MODEL,
                    "prompt": "face swap",
                    "base_image_url": "https://example.com/base.png",
                    "face_source_url": "https://example.com/face.png",
                },
            )
            self.assertEqual(face_resp.status_code, 200)
            face_job_id = face_resp.get_json()["job_id"]

        image_settlement = app_module.settlement.get_job_settlement(image_job_id)
        video_settlement = app_module.settlement.get_job_settlement(video_job_id)
        face_settlement = app_module.settlement.get_job_settlement(face_job_id)

        self.assertIsNotNone(image_settlement)
        self.assertIsNotNone(video_settlement)
        self.assertIsNotNone(face_settlement)

        self.assertEqual(image_settlement["job_type"], "IMAGE_GEN")
        self.assertEqual(video_settlement["job_type"], "VIDEO_GEN")
        self.assertEqual(face_settlement["job_type"], "FACE_SWAP")

        self.assertEqual(image_settlement["execution_status"], "queued")
        self.assertEqual(video_settlement["execution_status"], "queued")
        self.assertEqual(face_settlement["execution_status"], "queued")

    def test_job_detail_exposes_canonical_model_metadata(self) -> None:
        with self._submission_patch_stack():
            submit_resp = self.client.post(
                "/submit-job",
                json={"wallet": VALID_WALLET, "model": IMAGE_MODEL, "prompt": "metadata check"},
            )
        self.assertEqual(submit_resp.status_code, 200)
        job_id = submit_resp.get_json()["job_id"]

        detail_resp = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(detail_resp.status_code, 200)
        body = detail_resp.get_json()
        model_metadata = body.get("model_metadata")
        self.assertIsInstance(model_metadata, dict)
        self.assertEqual(model_metadata.get("model_name"), IMAGE_MODEL)
        self.assertEqual(model_metadata.get("model_key"), IMAGE_MODEL.lower())
        self.assertEqual(model_metadata.get("task_type"), "IMAGE_GEN")
        self.assertEqual(model_metadata.get("tier"), "A")
        self.assertEqual(float(model_metadata.get("reward_weight")), 10.0)

    def test_models_list_prefers_reward_weight_for_tier(self) -> None:
        manifest_path = Path(self._tmpdir.name) / "manifest_test.json"
        manifest_payload = {
            "models": [
                {
                    "name": "perfectdeliberate_v60",
                    "pipeline": "sdxl",
                    "path": "/tmp/perfectdeliberate_v60.safetensors",
                    "task_type": "IMAGE_GEN",
                    "weight": 3,
                    "reward_weight": 10,
                }
            ]
        }
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        app_module.MANIFEST_FILE = manifest_path
        app_module.load_manifest()

        with patch.object(app_module, "refresh_manifest", return_value=None), patch.object(
            app_module, "_eligible_online_node_count", return_value=1
        ):
            resp = self.client.get("/models/list")
        self.assertEqual(resp.status_code, 200)
        models = resp.get_json().get("models", [])
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["name"], "perfectdeliberate_v60")
        self.assertEqual(model["reward_weight"], 10.0)
        self.assertEqual(model["tier"], "A")

    def test_cancel_endpoint_releases_reserved_credits_and_is_idempotent(self) -> None:
        app_module.credits.CREDITS_ENABLED = True
        app_module.CREDITS_ENABLED = True

        app_module.credits.deposit_credits(VALID_WALLET, 10.0, reason="test-seed")
        starting_balance = app_module.credits.get_credit_balance(VALID_WALLET)
        self.assertEqual(starting_balance, 10.0)

        with self._submission_patch_stack():
            submit_resp = self.client.post(
                "/submit-job",
                json={"wallet": VALID_WALLET, "model": IMAGE_MODEL, "prompt": "cancel me"},
            )
        self.assertEqual(submit_resp.status_code, 200)
        job_id = submit_resp.get_json()["job_id"]

        after_submit_balance = app_module.credits.get_credit_balance(VALID_WALLET)
        self.assertLess(after_submit_balance, starting_balance)

        with patch.object(app_module, "rate_limit", return_value=True):
            cancel_resp = self.client.post(f"/jobs/{job_id}/cancel", json={"wallet": VALID_WALLET})
        self.assertEqual(cancel_resp.status_code, 200)
        cancel_body = cancel_resp.get_json()
        self.assertEqual(cancel_body["status"], "cancelled")

        record = app_module.settlement.get_job_settlement(job_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["settlement_outcome"], "released")

        restored_balance = app_module.credits.get_credit_balance(VALID_WALLET)
        self.assertAlmostEqual(restored_balance, starting_balance)

        with patch.object(app_module, "rate_limit", return_value=True):
            second_cancel = self.client.post(f"/jobs/{job_id}/cancel", json={"wallet": VALID_WALLET})
        self.assertEqual(second_cancel.status_code, 200)
        self.assertTrue(bool(second_cancel.get_json().get("already_cancelled")))

        conn = app_module.get_db()
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        self.assertEqual(str(row["status"]).lower(), "cancelled")

    def test_cancel_endpoint_rejects_finalized_jobs(self) -> None:
        conn = app_module.get_db()
        now = time.time()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                "job-finalized",
                VALID_WALLET,
                IMAGE_MODEL,
                json.dumps({"prompt": "done"}),
                "IMAGE_GEN",
                10.0,
                "success",
                now,
            ),
        )
        conn.commit()

        with patch.object(app_module, "rate_limit", return_value=True):
            resp = self.client.post("/jobs/job-finalized/cancel", json={"wallet": VALID_WALLET})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("error"), "job_finalized")

    def test_gallery_listing_enforces_settlement_marketplace_eligibility(self) -> None:
        conn = app_module.get_db()
        now = time.time()

        image_job_id = "job-image-eligible"
        face_job_id = "job-face-ineligible"

        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                image_job_id,
                VALID_WALLET,
                IMAGE_MODEL,
                json.dumps({"prompt": "eligible image"}),
                "IMAGE_GEN",
                10.0,
                "success",
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                face_job_id,
                VALID_WALLET,
                IMAGE_MODEL,
                json.dumps({"prompt": "ineligible face"}),
                "FACE_SWAP",
                10.0,
                "success",
                now,
                now,
            ),
        )
        conn.commit()

        app_module.settlement.create_job_ticket(
            job_id=image_job_id,
            wallet=VALID_WALLET,
            job_type="IMAGE_GEN",
            model=IMAGE_MODEL,
            estimated_cost=1.0,
            prompt="eligible image",
        )
        app_module.settlement.create_job_ticket(
            job_id=face_job_id,
            wallet=VALID_WALLET,
            job_type="FACE_SWAP",
            model=IMAGE_MODEL,
            estimated_cost=1.0,
            prompt="ineligible face",
        )

        conn.execute(
            """
            UPDATE job_settlement
            SET execution_status = 'settled',
                quality_status = 'valid',
                settlement_outcome = 'spent',
                reserved_amount = 1.0,
                spent_amount = 1.0,
                updated_at = ?
            WHERE job_id IN (?, ?)
            """,
            (now, image_job_id, face_job_id),
        )
        conn.commit()

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module,
            "_verify_wallet_signature",
            return_value=(True, None),
        ):
            eligible_resp = self.client.post(
                "/gallery/listings",
                json={
                    "wallet": VALID_WALLET,
                    "job_id": image_job_id,
                    "title": "Eligible",
                    "price_credits": 1.0,
                    "nonce": "nonce",
                    "signature": "sig",
                },
            )
            self.assertEqual(eligible_resp.status_code, 201)

            ineligible_resp = self.client.post(
                "/gallery/listings",
                json={
                    "wallet": VALID_WALLET,
                    "job_id": face_job_id,
                    "title": "Ineligible",
                    "price_credits": 1.0,
                    "nonce": "nonce",
                    "signature": "sig",
                },
            )
            self.assertEqual(ineligible_resp.status_code, 400)
            self.assertEqual(ineligible_resp.get_json().get("error"), "marketplace_ineligible")

    def test_gallery_listing_uses_canonical_settlement_model_metadata(self) -> None:
        conn = app_module.get_db()
        now = time.time()
        job_id = "job-canonical-listing"

        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                job_id,
                VALID_WALLET,
                "legacy_model_name",
                json.dumps({"prompt": "legacy prompt"}),
                "IMAGE_GEN",
                10.0,
                "success",
                now,
                now,
            ),
        )
        conn.commit()

        app_module.settlement.create_job_ticket(
            job_id=job_id,
            wallet=VALID_WALLET,
            job_type="IMAGE_GEN",
            model=IMAGE_MODEL,
            estimated_cost=1.0,
            prompt="canonical prompt",
            input_metadata={
                "model_key": IMAGE_MODEL.lower(),
                "model_name": IMAGE_MODEL,
                "pipeline": "sdxl",
                "task_type": "IMAGE_GEN",
                "reward_weight": 10.0,
                "tier": "A",
                "credit_cost": 1.0,
                "prompt": "canonical prompt",
                "source": "manifest",
            },
        )
        conn.execute(
            """
            UPDATE job_settlement
            SET execution_status = 'settled',
                quality_status = 'valid',
                settlement_outcome = 'spent',
                reserved_amount = 1.0,
                spent_amount = 1.0,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, job_id),
        )
        conn.commit()

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module,
            "_verify_wallet_signature",
            return_value=(True, None),
        ):
            create_resp = self.client.post(
                "/gallery/listings",
                json={
                    "wallet": VALID_WALLET,
                    "job_id": job_id,
                    "title": "Canonical listing",
                    "price_credits": 1.0,
                    "nonce": "nonce",
                    "signature": "sig",
                },
            )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.get_json()
        self.assertEqual(created.get("model"), IMAGE_MODEL)
        self.assertEqual(created.get("model_tier"), "A")
        self.assertEqual(created.get("prompt"), "canonical prompt")

        listing_id = int(created["id"])
        detail_resp = self.client.get(f"/gallery/listings/{listing_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.get_json()
        self.assertEqual(detail.get("model"), IMAGE_MODEL)
        self.assertEqual(detail.get("model_tier"), "A")

    def test_gallery_listing_duplicate_active_returns_existing_listing(self) -> None:
        conn = app_module.get_db()
        now = time.time()
        job_id = "job-duplicate-active"

        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                job_id,
                VALID_WALLET,
                IMAGE_MODEL,
                json.dumps({"prompt": "duplicate active"}),
                "IMAGE_GEN",
                10.0,
                "success",
                now,
                now,
            ),
        )
        conn.commit()

        app_module.settlement.create_job_ticket(
            job_id=job_id,
            wallet=VALID_WALLET,
            job_type="IMAGE_GEN",
            model=IMAGE_MODEL,
            estimated_cost=1.0,
            prompt="duplicate active",
        )
        conn.execute(
            """
            UPDATE job_settlement
            SET execution_status = 'settled',
                quality_status = 'valid',
                settlement_outcome = 'spent',
                reserved_amount = 1.0,
                spent_amount = 1.0,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, job_id),
        )
        conn.commit()

        payload = {
            "wallet": VALID_WALLET,
            "job_id": job_id,
            "title": "Duplicate active",
            "price_credits": 2.0,
            "nonce": "nonce",
            "signature": "sig",
        }

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module,
            "_verify_wallet_signature",
            return_value=(True, None),
        ):
            first_resp = self.client.post("/gallery/listings", json=payload)
            second_resp = self.client.post("/gallery/listings", json=payload)

        self.assertEqual(first_resp.status_code, 201)
        self.assertEqual(second_resp.status_code, 201)
        first = first_resp.get_json()
        second = second_resp.get_json()
        self.assertEqual(first.get("id"), second.get("id"))
        self.assertTrue(bool(second.get("already_listed")))

    def test_gallery_listing_can_relist_after_previous_sale(self) -> None:
        conn = app_module.get_db()
        now = time.time()
        job_id = "job-relist-after-sale"

        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                job_id,
                VALID_WALLET,
                IMAGE_MODEL,
                json.dumps({"prompt": "relist after sale"}),
                "IMAGE_GEN",
                10.0,
                "success",
                now,
                now,
            ),
        )
        conn.commit()

        app_module.settlement.create_job_ticket(
            job_id=job_id,
            wallet=VALID_WALLET,
            job_type="IMAGE_GEN",
            model=IMAGE_MODEL,
            estimated_cost=1.0,
            prompt="relist after sale",
        )
        conn.execute(
            """
            UPDATE job_settlement
            SET execution_status = 'settled',
                quality_status = 'valid',
                settlement_outcome = 'spent',
                reserved_amount = 1.0,
                spent_amount = 1.0,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, job_id),
        )
        conn.commit()

        payload = {
            "wallet": VALID_WALLET,
            "job_id": job_id,
            "title": "Relist after sale",
            "price_credits": 3.0,
            "nonce": "nonce",
            "signature": "sig",
        }

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module,
            "_verify_wallet_signature",
            return_value=(True, None),
        ):
            first_resp = self.client.post("/gallery/listings", json=payload)

        self.assertEqual(first_resp.status_code, 201)
        first_listing = first_resp.get_json()
        first_id = int(first_listing["id"])

        conn.execute(
            "UPDATE gallery_listings SET sold = 1, updated_at = ? WHERE id = ?",
            (time.time(), first_id),
        )
        conn.commit()

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module,
            "_verify_wallet_signature",
            return_value=(True, None),
        ):
            second_resp = self.client.post("/gallery/listings", json=payload)

        self.assertEqual(second_resp.status_code, 201)
        second_listing = second_resp.get_json()
        second_id = int(second_listing["id"])

        self.assertNotEqual(first_id, second_id)
        self.assertFalse(bool(second_listing.get("already_listed")))

        sold_row = conn.execute(
            "SELECT sold FROM gallery_listings WHERE id = ?",
            (first_id,),
        ).fetchone()
        relisted_row = conn.execute(
            "SELECT sold, listed FROM gallery_listings WHERE id = ?",
            (second_id,),
        ).fetchone()

        self.assertEqual(int(sold_row["sold"]), 1)
        self.assertEqual(int(relisted_row["sold"]), 0)
        self.assertEqual(int(relisted_row["listed"]), 1)

        browse_resp = self.client.get("/gallery/browse")
        self.assertEqual(browse_resp.status_code, 200)
        browse_listings = browse_resp.get_json().get("listings", [])
        active_ids = [int(item["id"]) for item in browse_listings]
        self.assertIn(second_id, active_ids)


if __name__ == "__main__":
    unittest.main()

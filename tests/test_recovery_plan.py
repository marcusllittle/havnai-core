"""Regression tests for dashboard telemetry, capacity checks, and scheduling."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

try:
    from eth_account import Account  # type: ignore
    from eth_account.messages import encode_defunct  # type: ignore
except Exception:  # pragma: no cover
    Account = None  # type: ignore[assignment]
    encode_defunct = None  # type: ignore[assignment]

import app as app_module
import job_helpers


VALID_WALLET = "0x1111111111111111111111111111111111111111"
SDXL_MODEL = "epicrealismxl_vxviicrystalclear"
LTX2_MODEL = "ltx2"


class JobHelperSupportMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_get_db = job_helpers.get_db
        self._orig_get_model_config = job_helpers.get_model_config
        self._orig_nodes = job_helpers.NODES
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
                invite_code TEXT
            )
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        job_helpers.get_db = self._orig_get_db
        job_helpers.get_model_config = self._orig_get_model_config
        job_helpers.NODES = self._orig_nodes
        self.conn.close()

    def test_faceswap_not_assignable_without_face_swap_support(self) -> None:
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
            """,
            ("job-face-swap", VALID_WALLET, SDXL_MODEL, "{}", "FACE_SWAP", 10.0, now),
        )
        self.conn.commit()

        job_helpers.get_db = lambda: self.conn
        job_helpers.get_model_config = lambda model_name: {"pipeline": "sdxl"} if model_name == SDXL_MODEL else None
        job_helpers.NODES = {
            "node-a": {
                "role": "creator",
                "supports": ["image"],  # no face_swap support advertised
                "models": [SDXL_MODEL],
                "pipelines": ["sdxl"],
            }
        }

        job = job_helpers.fetch_next_job_for_node("node-a")
        self.assertIsNone(job)


class CoordinatorCapacityEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._manifest_backup = copy.deepcopy(app_module.MANIFEST_MODELS)
        self._nodes_backup = copy.deepcopy(app_module.NODES)
        app_module.MANIFEST_MODELS.clear()
        app_module.NODES.clear()

    def tearDown(self) -> None:
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self._manifest_backup)
        app_module.NODES.clear()
        app_module.NODES.update(self._nodes_backup)

    def _common_submission_patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(app_module, "rate_limit", return_value=True))
        stack.enter_context(
            patch.object(app_module.invite, "enforce_invite_limits", return_value=(None, None))
        )
        stack.enter_context(patch.object(app_module.safety, "check_safety", return_value=None))
        stack.enter_context(patch.object(app_module.credits, "check_and_deduct_credits", return_value=None))
        stack.enter_context(patch.object(app_module, "refresh_manifest", return_value=None))
        return stack

    def test_submit_job_returns_no_capacity(self) -> None:
        app_module.MANIFEST_MODELS[SDXL_MODEL] = {
            "name": SDXL_MODEL,
            "pipeline": "sdxl",
            "task_type": "IMAGE_GEN",
            "reward_weight": 10.0,
            "tags": [],
        }
        with self._common_submission_patches():
            resp = self.client.post(
                "/submit-job",
                json={
                    "wallet": VALID_WALLET,
                    "model": SDXL_MODEL,
                    "prompt": "portrait photo",
                },
            )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "no_capacity")
        self.assertEqual(body.get("task_type"), "IMAGE_GEN")
        self.assertEqual(body.get("model"), SDXL_MODEL)

    def test_generate_video_returns_no_capacity(self) -> None:
        app_module.MANIFEST_MODELS[LTX2_MODEL] = {
            "name": LTX2_MODEL,
            "pipeline": "ltx2",
            "task_type": "VIDEO_GEN",
            "reward_weight": 12.0,
            "tags": [],
        }
        with self._common_submission_patches():
            resp = self.client.post(
                "/generate-video",
                json={
                    "wallet": VALID_WALLET,
                    "model": LTX2_MODEL,
                    "prompt": "cinematic ocean wave",
                },
            )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "no_capacity")
        self.assertEqual(body.get("task_type"), "VIDEO_GEN")
        self.assertEqual(body.get("model"), LTX2_MODEL)

    def test_submit_faceswap_returns_no_capacity(self) -> None:
        app_module.MANIFEST_MODELS[SDXL_MODEL] = {
            "name": SDXL_MODEL,
            "pipeline": "sdxl",
            "task_type": "IMAGE_GEN",
            "reward_weight": 10.0,
            "tags": [],
        }
        with self._common_submission_patches():
            resp = self.client.post(
                "/submit-faceswap-job",
                json={
                    "wallet": VALID_WALLET,
                    "model": SDXL_MODEL,
                    "prompt": "face swap",
                    "base_image_url": "https://example.com/base.png",
                    "face_source_url": "https://example.com/source.png",
                },
            )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "no_capacity")
        self.assertEqual(body.get("task_type"), "FACE_SWAP")
        self.assertEqual(body.get("model"), SDXL_MODEL)


class ExplicitLoraPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._manifest_backup = copy.deepcopy(app_module.MANIFEST_MODELS)
        self._nodes_backup = copy.deepcopy(app_module.NODES)
        app_module.MANIFEST_MODELS.clear()
        app_module.NODES.clear()
        now = app_module.unix_now()
        app_module.MANIFEST_MODELS[SDXL_MODEL] = {
            "name": SDXL_MODEL,
            "pipeline": "sdxl",
            "task_type": "IMAGE_GEN",
            "reward_weight": 10.0,
            "tags": [],
        }
        app_module.NODES["node-image"] = {
            "role": "creator",
            "last_seen_unix": now,
            "supports": ["image"],
            "models": [SDXL_MODEL],
            "pipelines": ["sdxl"],
        }

    def tearDown(self) -> None:
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self._manifest_backup)
        app_module.NODES.clear()
        app_module.NODES.update(self._nodes_backup)

    def test_submit_job_does_not_inject_default_loras_or_auto_anatomy(self) -> None:
        captured: dict = {}

        def _capture_enqueue(wallet, model, task_type, data, weight, invite_code=None):
            captured["wallet"] = wallet
            captured["model"] = model
            captured["task_type"] = task_type
            captured["data"] = data
            return "job-captured"

        with patch.object(app_module, "rate_limit", return_value=True), patch.object(
            app_module.invite, "enforce_invite_limits", return_value=(None, None)
        ), patch.object(app_module.safety, "check_safety", return_value=None), patch.object(
            app_module.credits, "check_and_deduct_credits", return_value=None
        ), patch.object(
            app_module, "refresh_manifest", return_value=None
        ), patch.object(
            app_module.job_helpers, "enqueue_job", side_effect=_capture_enqueue
        ):
            resp = self.client.post(
                "/submit-job",
                json={
                    "wallet": VALID_WALLET,
                    "model": SDXL_MODEL,
                    "prompt": "portrait",
                    "auto_anatomy": True,
                },
            )
        self.assertEqual(resp.status_code, 200)
        payload = json.loads(captured["data"])
        self.assertNotIn("auto_anatomy", payload)
        self.assertNotIn("loras", payload)

    def _submit_and_capture_job_data(self, payload: dict, *, capture_log_events: bool = False):
        captured: dict = {}

        def _capture_enqueue(wallet, model, task_type, data, weight, invite_code=None):
            captured["wallet"] = wallet
            captured["model"] = model
            captured["task_type"] = task_type
            captured["data"] = data
            return "job-captured"

        with ExitStack() as stack:
            stack.enter_context(patch.object(app_module, "rate_limit", return_value=True))
            stack.enter_context(
                patch.object(app_module.invite, "enforce_invite_limits", return_value=(None, None))
            )
            stack.enter_context(patch.object(app_module.safety, "check_safety", return_value=None))
            stack.enter_context(patch.object(app_module.credits, "check_and_deduct_credits", return_value=None))
            stack.enter_context(patch.object(app_module, "refresh_manifest", return_value=None))
            stack.enter_context(
                patch.object(app_module.job_helpers, "enqueue_job", side_effect=_capture_enqueue)
            )
            log_event_mock = None
            if capture_log_events:
                log_event_mock = stack.enter_context(patch.object(app_module, "log_event"))
            resp = self.client.post("/submit-job", json=payload)
        self.assertEqual(resp.status_code, 200)
        parsed = json.loads(captured["data"])
        return parsed, log_event_mock

    def test_hardcore_lora_submission_caps_auto_steps_to_30(self) -> None:
        payload, log_event_mock = self._submit_and_capture_job_data(
            {
                "wallet": VALID_WALLET,
                "model": SDXL_MODEL,
                "prompt": "doggy style sex portrait",
                "loras": [{"name": "perfectionstyle", "weight": 0.6}],
            },
            capture_log_events=True,
        )
        self.assertEqual(int(payload.get("steps", 0)), 30)
        self.assertEqual(float(payload.get("guidance", 0)), 7.5)
        self.assertIsNotNone(log_event_mock)
        event_names = [str(call.args[0]) for call in log_event_mock.call_args_list if call.args]
        self.assertIn("hardcore_lora_step_cap_applied", event_names)

    def test_hardcore_without_loras_keeps_steps_40(self) -> None:
        payload, _ = self._submit_and_capture_job_data(
            {
                "wallet": VALID_WALLET,
                "model": SDXL_MODEL,
                "prompt": "doggy style sex portrait",
            }
        )
        self.assertEqual(int(payload.get("steps", 0)), 40)

    def test_hardcore_with_lora_preserves_explicit_steps_override(self) -> None:
        payload, _ = self._submit_and_capture_job_data(
            {
                "wallet": VALID_WALLET,
                "model": SDXL_MODEL,
                "prompt": "doggy style sex portrait",
                "loras": [{"name": "perfectionstyle", "weight": 0.6}],
                "steps": 26,
            }
        )
        self.assertEqual(int(payload.get("steps", 0)), 26)


class CreditsFallbackCostTests(unittest.TestCase):
    def test_animatediff_task_fallback_uses_animatediff_default(self) -> None:
        with patch.object(app_module.credits, "get_model_config", return_value=None):
            cost = app_module.credits.resolve_credit_cost("missing-model", "ANIMATEDIFF")
        self.assertEqual(cost, app_module.credits.DEFAULT_CREDIT_COSTS["animatediff"])


class CreditConversionNonceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        app_module.RATE_LIMIT_BUCKETS.clear()
        if Account is None or encode_defunct is None:
            self.skipTest("eth-account not installed")
        self.account = Account.create()
        self.wallet = self.account.address
        app_module.credits.deposit_credits(self.wallet, 100.0, reason="test-seed")

    def _issue_nonce(self, amount: float) -> dict:
        resp = self.client.post(
            "/wallet/nonce",
            json={
                "wallet": self.wallet,
                "purpose": "convert_credits_to_hai",
                "amount": amount,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("nonce", body)
        self.assertIn("message", body)
        return body

    def _sign_message(self, message: str, private_key: bytes) -> str:
        signed = Account.sign_message(encode_defunct(text=message), private_key=private_key)
        return signed.signature.hex()

    def test_convert_requires_nonce_and_signature(self) -> None:
        resp = self.client.post(
            "/credits/convert",
            json={"wallet": self.wallet, "amount": 1.0},
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "malformed_payload")

    def test_nonce_signature_success_and_replay_rejected(self) -> None:
        amount = 2.5
        challenge = self._issue_nonce(amount)
        signature = self._sign_message(challenge["message"], self.account.key)
        payload = {
            "wallet": self.wallet,
            "amount": amount,
            "nonce": challenge["nonce"],
            "signature": signature,
        }

        resp = self.client.post("/credits/convert", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("balance", body)
        self.assertIn("remaining", body)
        self.assertEqual(body.get("remaining"), body.get("balance"))

        replay = self.client.post("/credits/convert", json=payload)
        self.assertEqual(replay.status_code, 409)
        replay_body = replay.get_json()
        self.assertEqual(replay_body.get("error"), "nonce_used")

    def test_nonce_expired_rejected(self) -> None:
        amount = 1.0
        challenge = self._issue_nonce(amount)
        conn = app_module.get_db()
        conn.execute(
            "UPDATE wallet_nonces SET expires_at=? WHERE wallet=? AND nonce=?",
            (app_module.unix_now() - 1.0, self.wallet, challenge["nonce"]),
        )
        conn.commit()
        signature = self._sign_message(challenge["message"], self.account.key)
        resp = self.client.post(
            "/credits/convert",
            json={
                "wallet": self.wallet,
                "amount": amount,
                "nonce": challenge["nonce"],
                "signature": signature,
            },
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "nonce_expired")

    def test_signature_wallet_mismatch_rejected(self) -> None:
        amount = 1.25
        challenge = self._issue_nonce(amount)
        other = Account.create()
        wrong_signature = self._sign_message(challenge["message"], other.key)
        resp = self.client.post(
            "/credits/convert",
            json={
                "wallet": self.wallet,
                "amount": amount,
                "nonce": challenge["nonce"],
                "signature": wrong_signature,
            },
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "invalid_signature")


class ModelsListAvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._manifest_backup = copy.deepcopy(app_module.MANIFEST_MODELS)
        self._nodes_backup = copy.deepcopy(app_module.NODES)
        app_module.MANIFEST_MODELS.clear()
        app_module.NODES.clear()

    def tearDown(self) -> None:
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self._manifest_backup)
        app_module.NODES.clear()
        app_module.NODES.update(self._nodes_backup)

    def test_models_list_includes_capacity_fields(self) -> None:
        now = app_module.unix_now()
        app_module.MANIFEST_MODELS.update(
            {
                SDXL_MODEL: {
                    "name": SDXL_MODEL,
                    "pipeline": "sdxl",
                    "task_type": "IMAGE_GEN",
                    "reward_weight": 10.0,
                    "tags": [],
                },
                LTX2_MODEL: {
                    "name": LTX2_MODEL,
                    "pipeline": "ltx2",
                    "task_type": "VIDEO_GEN",
                    "reward_weight": 12.0,
                    "tags": [],
                },
            }
        )
        app_module.NODES.update(
            {
                "node-image": {
                    "role": "creator",
                    "last_seen_unix": now,
                    "supports": ["image", "face_swap"],
                    "models": [SDXL_MODEL],
                    "pipelines": ["sdxl"],
                },
                "node-video": {
                    "role": "creator",
                    "last_seen_unix": now,
                    "supports": ["video"],
                    "models": [LTX2_MODEL],
                    "pipelines": ["ltx2"],
                },
                "node-offline-video": {
                    "role": "creator",
                    "last_seen_unix": now - (app_module.ONLINE_THRESHOLD + 10),
                    "supports": ["video"],
                    "models": [LTX2_MODEL],
                    "pipelines": ["ltx2"],
                },
            }
        )

        with patch.object(app_module, "refresh_manifest", return_value=None):
            resp = self.client.get("/models/list")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        models = {str(item["name"]).lower(): item for item in body.get("models", [])}

        sdxl = models[SDXL_MODEL]
        self.assertTrue(sdxl["available"])
        self.assertEqual(sdxl["online_nodes"], 1)
        self.assertTrue(sdxl["face_swap_available"])
        self.assertEqual(sdxl["face_swap_online_nodes"], 1)

        ltx2 = models[LTX2_MODEL]
        self.assertTrue(ltx2["available"])
        self.assertEqual(ltx2["online_nodes"], 1)
        self.assertFalse(ltx2["face_swap_available"])
        self.assertEqual(ltx2["face_swap_online_nodes"], 0)


class DashboardProxyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_dashboard_uses_relative_static_script(self) -> None:
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn('window.NEXT_PUBLIC_API_BASE_URL = window.location.pathname.startsWith("/api/") ? "/api" : "";', html)
        self.assertIn('<script src="static/dashboard.js" type="module"></script>', html)


class JobCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._token_backup = app_module.SERVER_JOIN_TOKEN
        app_module.SERVER_JOIN_TOKEN = "test-join-token"
        self.job_id = f"job-cancel-{int(time.time() * 1000)}"
        payload = json.dumps(
            {
                "prompt": "portrait",
                "loras": [{"name": "incase_style", "weight": 0.74}],
                "requested_loras": [{"name": "incase_style", "weight": 0.74}],
            }
        )
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
            """,
            (self.job_id, VALID_WALLET, SDXL_MODEL, payload, "IMAGE_GEN", 10.0, time.time()),
        )
        conn.commit()

    def tearDown(self) -> None:
        app_module.SERVER_JOIN_TOKEN = self._token_backup
        conn = app_module.get_db()
        conn.execute("DELETE FROM rewards WHERE task_id=?", (self.job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (self.job_id,))
        conn.commit()

    def test_cancel_endpoint_requires_join_token(self) -> None:
        resp = self.client.post(f"/jobs/{self.job_id}/cancel")
        self.assertEqual(resp.status_code, 403)

    def test_cancel_endpoint_marks_job_failed_with_reason(self) -> None:
        resp = self.client.post(
            f"/jobs/{self.job_id}/cancel",
            headers={"X-Join-Token": "test-join-token"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body.get("status"), "failed")
        self.assertEqual(body.get("status_reason"), "canceled_by_operator")

        detail = self.client.get(f"/jobs/{self.job_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.get_json()
        self.assertEqual(str(payload.get("status", "")).lower(), "failed")
        self.assertEqual(payload.get("status_reason"), "canceled_by_operator")


class StaleRunningReconcileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._nodes_backup = copy.deepcopy(app_module.NODES)
        self._timeout_backup = app_module.STUCK_RUNNING_TIMEOUT_SECONDS
        app_module.STUCK_RUNNING_TIMEOUT_SECONDS = 60
        self.offline_node = "node-offline-reconcile"
        app_module.NODES[self.offline_node] = {
            "role": "creator",
            "last_seen_unix": app_module.unix_now() - (app_module.ONLINE_THRESHOLD + 120),
            "supports": ["image"],
            "models": [SDXL_MODEL],
            "pipelines": ["sdxl"],
            "current_task": {"task_id": "placeholder"},
        }
        self.job_id = f"job-stale-{int(time.time() * 1000)}"
        payload = json.dumps({"prompt": "stale job", "loras": [{"name": "incase_style", "weight": 0.5}]})
        now = time.time()
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, assigned_at)
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (self.job_id, VALID_WALLET, SDXL_MODEL, payload, "IMAGE_GEN", 10.0, self.offline_node, now - 600, now - 600),
        )
        conn.commit()

    def tearDown(self) -> None:
        app_module.STUCK_RUNNING_TIMEOUT_SECONDS = self._timeout_backup
        app_module.NODES.clear()
        app_module.NODES.update(self._nodes_backup)
        conn = app_module.get_db()
        conn.execute("DELETE FROM rewards WHERE task_id=?", (self.job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (self.job_id,))
        conn.commit()

    def test_reconcile_marks_stale_running_job_failed(self) -> None:
        reconciled = app_module._reconcile_stale_running_jobs()
        self.assertGreaterEqual(reconciled, 1)

        detail = self.client.get(f"/jobs/{self.job_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.get_json()
        self.assertEqual(str(payload.get("status", "")).lower(), "failed")
        self.assertEqual(payload.get("status_reason"), "node_offline_timeout")


class JobObservabilityFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self.job_id = f"job-observe-{int(time.time() * 1000)}"
        payload = json.dumps(
            {
                "prompt": "portrait",
                "loras": [{"name": "incase_style", "weight": 0.74}],
                "requested_loras": [{"name": "incase_style", "weight": 0.74}],
                "applied_loras": [{"name": "incase_style", "requested_weight": 0.74, "applied_weight": 0.74}],
                "status_reason": "success",
            }
        )
        now = time.time()
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, assigned_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, 'success', ?, ?, ?, ?)
            """,
            (self.job_id, VALID_WALLET, SDXL_MODEL, payload, "IMAGE_GEN", 10.0, "node-observe", now - 20, now - 10, now - 1),
        )
        conn.commit()

    def tearDown(self) -> None:
        conn = app_module.get_db()
        conn.execute("DELETE FROM rewards WHERE task_id=?", (self.job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (self.job_id,))
        conn.commit()

    def test_job_detail_exposes_lora_and_status_reason_fields(self) -> None:
        resp = self.client.get(f"/jobs/{self.job_id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body.get("status_reason"), "success")
        self.assertEqual(body.get("lora_summary"), "incase_style:0.74")
        self.assertEqual(body.get("requested_loras")[0]["name"], "incase_style")
        self.assertEqual(body.get("applied_loras")[0]["name"], "incase_style")

    def test_jobs_recent_exposes_lora_summary(self) -> None:
        resp = self.client.get("/jobs/recent?limit=200")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        feed = body.get("jobs", [])
        match = next((item for item in feed if item.get("job_id") == self.job_id), None)
        self.assertIsNotNone(match)
        self.assertEqual(match.get("lora_summary"), "incase_style:0.74")


class ResultsConflictReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self.job_id = f"job-conflict-{int(time.time() * 1000)}"
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
            """,
            (self.job_id, VALID_WALLET, SDXL_MODEL, json.dumps({"prompt": "conflict"}), "IMAGE_GEN", 10.0, time.time()),
        )
        conn.commit()

    def tearDown(self) -> None:
        conn = app_module.get_db()
        conn.execute("DELETE FROM rewards WHERE task_id=?", (self.job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (self.job_id,))
        conn.commit()

    def test_results_conflict_includes_reason(self) -> None:
        resp = self.client.post(
            "/results",
            json={
                "node_id": "node-test-conflict",
                "task_id": self.job_id,
                "status": "failed",
                "metrics": {"status": "failed", "error": "test conflict"},
                "utilization": 0,
            },
        )
        self.assertEqual(resp.status_code, 409)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "conflict")
        self.assertIn("reason", body)

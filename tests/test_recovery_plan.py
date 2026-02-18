"""Regression tests for dashboard telemetry, capacity checks, and scheduling."""

from __future__ import annotations

import copy
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
                    "source_image_url": "https://example.com/source.png",
                },
            )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body.get("error"), "no_capacity")
        self.assertEqual(body.get("task_type"), "FACE_SWAP")
        self.assertEqual(body.get("model"), SDXL_MODEL)


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


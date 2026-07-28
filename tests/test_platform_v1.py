"""Contract tests for the owner-first durable job API."""

from __future__ import annotations

import copy
import io
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module
import job_helpers
import platform_v1


WALLET = "0x1111111111111111111111111111111111111111"
IMAGE_MODEL = "test_sdxl"
VIDEO_MODEL = "test_ltx23_wangp"


class VideoSpecTests(unittest.TestCase):
    def test_fast_landscape_spec_is_exact_and_bounded(self) -> None:
        spec = platform_v1.resolve_video_spec(
            {
                "preset": "fast_upscaled",
                "aspect_ratio": "16:9",
                "duration_seconds": 5,
                "seed": 42,
            },
            model="ltx_video_distilled",
            backend="diffusers",
        )
        self.assertEqual(spec["timeout_seconds"], 3600)
        self.assertEqual(spec["parameters"]["seed"], 42)
        self.assertEqual(spec["parameters"]["width"], 832)
        self.assertEqual(spec["parameters"]["height"], 480)
        self.assertEqual(spec["parameters"]["delivery_width"], 1280)
        self.assertEqual(spec["parameters"]["delivery_height"], 720)
        self.assertEqual((spec["parameters"]["frames"] - 1) % 8, 0)

    def test_unsupported_duration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_duration"):
            platform_v1.resolve_video_spec(
                {"duration_seconds": 4}, model="ltx_video_distilled"
            )


class ImageDefaultsRegressionTests(unittest.TestCase):
    def test_manifest_tuning_does_not_replace_create_quality_profile(self) -> None:
        original_manifest = copy.deepcopy(app_module.MANIFEST_MODELS)
        original_stats = copy.deepcopy(app_module.MODEL_STATS)
        original_weights = copy.deepcopy(app_module.MODEL_WEIGHTS)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_path = Path(temp_dir) / "registry.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "name": "image_one",
                                    "pipeline": "sdxl",
                                    "task_type": "IMAGE_GEN",
                                    "steps": 18,
                                    "guidance": 4.0,
                                    "width": 512,
                                    "height": 512,
                                    "sampler": "weaker_sampler",
                                    "negative_prompt_default": "manifest negative",
                                },
                                {
                                    "name": "image_two",
                                    "pipeline": "sdxl",
                                    "task_type": "IMAGE_GEN",
                                    "steps": 22,
                                    "guidance": 5.0,
                                },
                                {
                                    "name": "video_one",
                                    "pipeline": "ltx23_wangp",
                                    "task_type": "LTX_VIDEO_GEN",
                                    "video_defaults": {"steps": 8, "guidance": 1.0},
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                with patch.object(app_module, "MANIFEST_FILE", manifest_path):
                    app_module.load_manifest()

                for model_name in ("image_one", "image_two"):
                    model = app_module.MANIFEST_MODELS[model_name]
                    resolved, sources = app_module.resolve_image_defaults(
                        model, {}, app_module.RUNTIME_PROFILE
                    )
                    self.assertEqual(
                        resolved,
                        {"steps": 32, "guidance": 6.5, "width": 832, "height": 1216},
                    )
                    self.assertEqual(set(sources.values()), {"profile"})

                self.assertEqual(
                    app_module.MANIFEST_MODELS["video_one"]["video_defaults"],
                    {"steps": 8, "guidance": 1.0},
                )
        finally:
            app_module.MANIFEST_MODELS.clear()
            app_module.MANIFEST_MODELS.update(original_manifest)
            app_module.MODEL_STATS.clear()
            app_module.MODEL_STATS.update(original_stats)
            app_module.MODEL_WEIGHTS.clear()
            app_module.MODEL_WEIGHTS.update(original_weights)


class PlatformApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.originals = {
            "DB_PATH": app_module.DB_PATH,
            "STATIC_DIR": app_module.STATIC_DIR,
            "OUTPUTS_DIR": app_module.OUTPUTS_DIR,
            "ASSETS_DIR": app_module.ASSETS_DIR,
            "REGISTRY_FILE": app_module.REGISTRY_FILE,
            "DB_CONN": app_module.DB_CONN,
            "OWNER_API_TOKEN": app_module.OWNER_API_TOKEN,
            "NODE_API_TOKEN": app_module.NODE_API_TOKEN,
            "VIDEO_V2_ENABLED": app_module.VIDEO_V2_ENABLED,
        }
        self.nodes = copy.deepcopy(app_module.NODES)
        self.manifest = copy.deepcopy(app_module.MANIFEST_MODELS)
        self.tasks = copy.deepcopy(app_module.TASKS)

        app_module.DB_PATH = root / "ledger.db"
        app_module.STATIC_DIR = root / "static"
        app_module.OUTPUTS_DIR = app_module.STATIC_DIR / "outputs"
        app_module.ASSETS_DIR = app_module.STATIC_DIR / "assets"
        app_module.REGISTRY_FILE = root / "nodes.json"
        app_module.DB_CONN = None
        app_module.OWNER_API_TOKEN = "owner-test"
        app_module.NODE_API_TOKEN = "node-test"
        app_module.VIDEO_V2_ENABLED = True
        app_module.NODES.clear()
        app_module.TASKS.clear()
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(
            {
                IMAGE_MODEL: {
                    "name": IMAGE_MODEL,
                    "pipeline": "sdxl",
                    "task_type": "IMAGE_GEN",
                    "reward_weight": 1.0,
                    "defaults": {"steps": 28, "guidance": 5.5, "width": 768, "height": 1024},
                },
                VIDEO_MODEL: {
                    "name": VIDEO_MODEL,
                    "pipeline": "ltx23_wangp",
                    "task_type": "LTX_VIDEO_GEN",
                    "reward_weight": 1.0,
                    "model_family": "ltx23_wangp",
                    "model_version": "2.3-distilled-1.1",
                    "license_status": "research_owner_only",
                    "capabilities": ["image_to_video"],
                },
            }
        )
        for directory in (app_module.STATIC_DIR, app_module.OUTPUTS_DIR, app_module.ASSETS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        with app_module.app.app_context():
            app_module.init_db()
        self.client = app_module.app.test_client()
        self.owner_headers = {"Authorization": "Bearer owner-test"}
        self.node_headers = {"X-HavnAI-Token": "node-test"}
        self.refresh_patch = patch.object(app_module, "refresh_manifest", return_value=None)
        self.refresh_patch.start()

    def tearDown(self) -> None:
        self.refresh_patch.stop()
        if app_module.DB_CONN is not None:
            app_module.DB_CONN.close()
        for key, value in self.originals.items():
            setattr(app_module, key, value)
        app_module.NODES.clear()
        app_module.NODES.update(self.nodes)
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self.manifest)
        app_module.TASKS.clear()
        app_module.TASKS.update(self.tasks)
        self.temp.cleanup()

    def _create_image_job(self) -> str:
        response = self.client.post(
            "/v1/jobs",
            json={"type": "image", "model": IMAGE_MODEL, "prompt": "ten word test prompt", "wallet": WALLET, "seed": 7},
            headers=self.owner_headers,
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        return str(response.get_json()["id"])

    def _register_node(self) -> None:
        app_module.NODES["node-test"] = {
            "role": "creator",
            "last_seen_unix": time.time(),
            "supports": ["image", "ltx_video"],
            "models": [IMAGE_MODEL, VIDEO_MODEL],
            "pipelines": ["sdxl", "ltx23_wangp"],
            "capabilities": {
                IMAGE_MODEL: {
                    "files_present": True,
                    "pipeline": "sdxl",
                    "capabilities": ["text_to_image"],
                },
                VIDEO_MODEL: {
                    "files_present": True,
                    "pipeline": "ltx23_wangp",
                    "model_version": "2.3-distilled-1.1",
                    "capabilities": ["image_to_video"],
                    "available_modes": ["distilled"],
                },
            },
        }

    def _claim(self, job_id: str) -> dict:
        self._register_node()
        response = self.client.get(
            "/tasks/creator?node_id=node-test", headers=self.node_headers
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        task = response.get_json()["tasks"][0]
        self.assertEqual(task["task_id"], job_id)
        self.assertTrue(task["attempt_id"].startswith("attempt-"))
        return task

    def test_auth_is_required(self) -> None:
        self.assertEqual(self.client.get("/v1/capabilities").status_code, 401)
        self.assertEqual(self.client.get("/v1/jobs").status_code, 401)
        self.assertEqual(self.client.get("/tasks/creator?node_id=x").status_code, 401)
        self.assertEqual(self.client.post("/results", json={}).status_code, 401)

    def test_missing_role_tokens_fail_closed(self) -> None:
        with patch.object(app_module, "OWNER_API_TOKEN", ""), patch.object(
            app_module, "NODE_API_TOKEN", ""
        ):
            self.assertEqual(self.client.get("/v1/capabilities").status_code, 401)
            self.assertEqual(self.client.get("/tasks/creator?node_id=x").status_code, 401)

    def test_capabilities_require_an_online_runtime_probe(self) -> None:
        unavailable = self.client.get("/v1/capabilities", headers=self.owner_headers)
        self.assertEqual(unavailable.status_code, 200)
        self.assertFalse(unavailable.get_json()["video_v2_available"])

        self._register_node()
        available = self.client.get("/v1/capabilities", headers=self.owner_headers)
        payload = available.get_json()
        video = next(model for model in payload["models"] if model["id"] == VIDEO_MODEL)
        self.assertTrue(payload["video_v2_available"])
        self.assertTrue(video["available"])
        self.assertEqual(video["verified_nodes"], ["node-test"])
        self.assertEqual(video["capabilities"], ["image_to_video"])

    def test_video_job_list_includes_legacy_video_jobs(self) -> None:
        with app_module.app.app_context():
            job_id = job_helpers.enqueue_job(
                WALLET,
                VIDEO_MODEL,
                "LTX_VIDEO_GEN",
                json.dumps({"prompt": "legacy video task"}),
                1.0,
            )
            app_module.get_db().execute(
                "UPDATE jobs SET status='running', stage='generation', progress=42 WHERE id=?",
                (job_id,),
            )
            app_module.get_db().commit()

        response = self.client.get(
            "/v1/jobs?type=image_to_video&limit=10", headers=self.owner_headers
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["count"], 1)
        self.assertEqual(response.get_json()["jobs"][0]["id"], job_id)
        self.assertEqual(response.get_json()["jobs"][0]["progress"], 42)

    def test_asset_and_video_job_preserve_resolved_spec(self) -> None:
        upload = self.client.post(
            "/v1/assets",
            data={"kind": "image", "file": (io.BytesIO(b"image-bytes"), "source.png")},
            headers=self.owner_headers,
            content_type="multipart/form-data",
        )
        self.assertEqual(upload.status_code, 201, upload.get_json())
        asset_id = upload.get_json()["id"]
        response = self.client.post(
            "/v1/jobs",
            json={
                "type": "image_to_video",
                "model": VIDEO_MODEL,
                "prompt": "subtle camera move",
                "source_asset_id": asset_id,
                "preset": "fast_upscaled",
                "aspect_ratio": "16:9",
                "duration_seconds": 3,
                "seed": 99,
            },
            headers=self.owner_headers,
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        recent = self.client.get(
            "/v1/jobs?type=image_to_video&limit=10", headers=self.owner_headers
        )
        self.assertEqual(recent.status_code, 200, recent.get_json())
        self.assertEqual(recent.get_json()["count"], 1)
        self.assertEqual(recent.get_json()["jobs"][0]["id"], response.get_json()["id"])
        spec = response.get_json()["resolved_spec"]
        self.assertEqual(spec["parameters"]["seed"], 99)
        self.assertEqual(spec["parameters"]["delivery_width"], 1280)
        task = self._claim(str(response.get_json()["id"]))
        self.assertEqual(task["type"], "LTX_VIDEO_GEN")
        self.assertEqual(task["timeout"], 3600)
        self.assertEqual(task["source_asset_id"], asset_id)

    def test_lease_upload_completion_and_stale_attempt_rejection(self) -> None:
        job_id = self._create_image_job()
        task = self._claim(job_id)
        attempt_id = task["attempt_id"]

        progress = self.client.post(
            f"/v1/node/jobs/{job_id}/progress",
            json={"node_id": "node-test", "attempt_id": attempt_id, "progress": 35, "stage": "generation"},
            headers=self.node_headers,
        )
        self.assertEqual(progress.status_code, 200)
        stale = self.client.post(
            f"/v1/node/jobs/{job_id}/progress",
            json={"node_id": "node-test", "attempt_id": "attempt-stale", "progress": 50, "stage": "generation"},
            headers=self.node_headers,
        )
        self.assertEqual(stale.status_code, 409)
        stale_result = self.client.post(
            "/results",
            json={
                "node_id": "node-test",
                "task_id": job_id,
                "attempt_id": "attempt-stale",
                "status": "success",
                "metrics": {},
            },
            headers=self.node_headers,
        )
        self.assertEqual(stale_result.status_code, 409)

        artifact = self.client.post(
            f"/v1/node/jobs/{job_id}/artifacts",
            data={
                "node_id": "node-test",
                "attempt_id": attempt_id,
                "kind": "image",
                "metadata": json.dumps({"seed": 7}),
                "file": (io.BytesIO(b"png-result"), "result.png"),
            },
            headers=self.node_headers,
            content_type="multipart/form-data",
        )
        self.assertEqual(artifact.status_code, 201, artifact.get_json())
        completed = self.client.post(
            "/results",
            json={
                "node_id": "node-test",
                "task_id": job_id,
                "attempt_id": attempt_id,
                "status": "success",
                "metrics": {
                    "inference_time_ms": 12,
                    "resolved_render_spec": {
                        "schema_version": 1,
                        "parameters": {"seed": 7, "steps": 28},
                        "output": {"sha256": artifact.get_json()["sha256"]},
                    },
                },
            },
            headers=self.node_headers,
        )
        self.assertEqual(completed.status_code, 200, completed.get_json())
        detail = self.client.get(f"/v1/jobs/{job_id}", headers=self.owner_headers)
        self.assertEqual(detail.get_json()["status"], "succeeded")
        self.assertEqual(detail.get_json()["resolved_spec"]["parameters"]["steps"], 28)
        self.assertEqual(detail.get_json()["artifacts"][0]["sha256"], artifact.get_json()["sha256"])

    def test_cancel_signals_active_attempt(self) -> None:
        job_id = self._create_image_job()
        task = self._claim(job_id)
        response = self.client.post(
            f"/v1/jobs/{job_id}/cancel", headers=self.owner_headers
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["status"], "cancelling")
        control = self.client.get(
            f"/v1/node/jobs/{job_id}/control?node_id=node-test&attempt_id={task['attempt_id']}",
            headers=self.node_headers,
        )
        self.assertEqual(control.status_code, 200)
        self.assertTrue(control.get_json()["cancel_requested"])


if __name__ == "__main__":
    unittest.main()

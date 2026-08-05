"""Tests for coordinator-hosted model artifact delivery.

Two things must hold. Only authenticated nodes may pull weights we host, and
the endpoint must never serve a file outside the configured storage root - a
mistyped manifest path is a misconfiguration, not permission to read the disk.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

NODE_TOKEN = "test-node-token"


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._storage = tempfile.TemporaryDirectory()
        storage = Path(cls._storage.name)
        (storage / "hosted-model.safetensors").write_bytes(b"W" * 4096)

        cls._outside = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        cls._outside.write(b"NOT YOURS")
        cls._outside.close()

        os.environ["HAVNAI_NODE_TOKEN"] = NODE_TOKEN
        os.environ["HAVNAI_MODEL_STORAGE_DIR"] = str(storage)

        import app as app_module

        cls.app_module = importlib.reload(app_module)
        cls.app_module.MANIFEST_MODELS = {
            "hosted-model": {
                "name": "hosted-model",
                "path": str(storage / "hosted-model.safetensors"),
                "source": {"kind": "coordinator", "filename": "hosted-model.safetensors"},
            },
            "escapee": {
                "name": "escapee",
                "path": cls._outside.name,
                "source": {"kind": "coordinator", "filename": "escapee.safetensors"},
            },
            "cdn-model": {
                "name": "cdn-model",
                "path": "",
                "source": {
                    "kind": "hf",
                    "repo_id": "vendor/model",
                    "filename": "cdn-model.safetensors",
                },
            },
        }
        # The manifest is injected directly; do not let a reload clobber it.
        cls.app_module.refresh_manifest = lambda: None
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._storage.cleanup()
        os.unlink(cls._outside.name)

    @property
    def auth(self) -> dict:
        return {"Authorization": f"Bearer {NODE_TOKEN}"}

    def test_unauthenticated_requests_are_rejected(self) -> None:
        self.assertEqual(self.client.get("/models/download/hosted-model").status_code, 401)

    def test_wrong_token_is_rejected(self) -> None:
        response = self.client.get(
            "/models/download/hosted-model", headers={"Authorization": "Bearer wrong"}
        )
        self.assertEqual(response.status_code, 401)

    def test_authenticated_node_receives_the_artifact(self) -> None:
        response = self.client.get("/models/download/hosted-model", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 4096)
        self.assertEqual(response.headers.get("Accept-Ranges"), "bytes")

    def test_range_requests_resume_a_partial_transfer(self) -> None:
        response = self.client.get(
            "/models/download/hosted-model",
            headers={**self.auth, "Range": "bytes=100-199"},
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.data), 100)

    def test_artifact_outside_the_storage_root_is_refused(self) -> None:
        # The manifest points at a real, readable file - containment is what
        # stops it being served, not the file being absent.
        self.assertTrue(Path(self._outside.name).is_file())
        response = self.client.get("/models/download/escapee", headers=self.auth)
        self.assertEqual(response.status_code, 404)

    def test_hugging_face_sourced_models_are_not_served_by_us(self) -> None:
        response = self.client.get("/models/download/cdn-model", headers=self.auth)
        self.assertEqual(response.status_code, 404)

    def test_unknown_model_is_not_found(self) -> None:
        self.assertEqual(
            self.client.get("/models/download/nonexistent", headers=self.auth).status_code, 404
        )

    def test_path_traversal_is_refused(self) -> None:
        response = self.client.get(
            "/models/download/../../../etc/passwd", headers=self.auth
        )
        self.assertIn(response.status_code, (400, 404))


class PublicModelSourceTests(unittest.TestCase):
    """The manifest's internal storage paths must never reach a node."""

    @classmethod
    def setUpClass(cls) -> None:
        import app as app_module

        cls.app_module = app_module

    def test_storage_paths_are_not_exposed(self) -> None:
        source = self.app_module._public_model_source(
            {"name": "secret", "path": "/mnt/d/havnai-storage/models/creator/secret.safetensors"}
        )
        self.assertEqual(source["kind"], "coordinator")
        self.assertEqual(source["filename"], "secret.safetensors")
        self.assertNotIn("path", source)
        for value in source.values():
            self.assertNotIn("/mnt/", str(value))

    def test_entries_without_a_path_default_to_operator_supplied(self) -> None:
        source = self.app_module._public_model_source({"name": "byo", "path": ""})
        self.assertEqual(source["kind"], "operator")

    def test_hugging_face_source_carries_repo_details(self) -> None:
        source = self.app_module._public_model_source(
            {
                "name": "ltx",
                "path": "",
                "source": {
                    "kind": "hf",
                    "repo_id": "Lightricks/LTX-Video",
                    "filename": "ltx.safetensors",
                },
            }
        )
        self.assertEqual(source["repo_id"], "Lightricks/LTX-Video")
        self.assertEqual(source["revision"], "main")


if __name__ == "__main__":
    unittest.main()

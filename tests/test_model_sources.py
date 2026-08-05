"""Tests for node-side model acquisition.

The invariant that matters most: a node must never end up with a truncated or
mismatched checkpoint sitting where a valid one belongs. A corrupt checkpoint
loads as garbage weights and produces silently bad output rather than an error.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from client.model_sources import (
    ModelDownloadError,
    ModelRequirement,
    ensure_model,
    models_dir,
    requirements_from_manifest,
    verify,
)


def _requirement(destination: Path, **overrides) -> ModelRequirement:
    defaults = dict(
        name="test-model",
        kind="coordinator",
        destination=destination,
        filename=destination.name,
    )
    defaults.update(overrides)
    return ModelRequirement(**defaults)  # type: ignore[arg-type]


class RequirementPlanningTests(unittest.TestCase):
    def test_source_kinds_map_to_download_capability(self) -> None:
        home = Path("/tmp/havnai-plan-test")
        manifest = [
            {"name": "open-model", "pipeline": "sdxl", "source": {"kind": "hf"}},
            {"name": "ours", "pipeline": "sdxl", "source": {"kind": "coordinator"}},
            {"name": "byo", "pipeline": "sdxl", "source": {"kind": "operator"}},
        ]
        plan = {item.name: item for item in requirements_from_manifest(manifest, home)}

        self.assertTrue(plan["open-model"].downloadable)
        self.assertTrue(plan["ours"].downloadable)
        self.assertFalse(plan["byo"].downloadable)

    def test_filename_defaults_to_the_model_name(self) -> None:
        plan = requirements_from_manifest(
            [{"name": "juggernautXL", "pipeline": "sdxl"}], Path("/tmp/havnai-plan-test")
        )
        self.assertEqual(plan[0].destination.name, "juggernautXL.safetensors")

    def test_pipeline_filter_excludes_unrunnable_models(self) -> None:
        manifest = [
            {"name": "image-model", "pipeline": "sdxl"},
            {"name": "video-model", "pipeline": "ltx2"},
        ]
        plan = requirements_from_manifest(
            manifest, Path("/tmp/havnai-plan-test"), pipelines=["sdxl"]
        )
        self.assertEqual([item.name for item in plan], ["image-model"])

    def test_duplicate_entries_are_collapsed(self) -> None:
        manifest = [
            {"name": "same", "pipeline": "sdxl"},
            {"name": "SAME", "pipeline": "sdxl"},
        ]
        plan = requirements_from_manifest(manifest, Path("/tmp/havnai-plan-test"))
        self.assertEqual(len(plan), 1)

    def test_model_dir_override_is_honoured(self) -> None:
        with patch.dict("os.environ", {"HAVNAI_MODEL_DIR": "/custom/models"}):
            self.assertEqual(models_dir(Path("/home/x/.havnai")), Path("/custom/models"))


class VerificationTests(unittest.TestCase):
    def test_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            target.write_bytes(b"actual contents")
            requirement = _requirement(target, sha256="0" * 64)
            self.assertFalse(verify(requirement))

    def test_matching_digest_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            payload = b"actual contents"
            target.write_bytes(payload)
            requirement = _requirement(
                target, sha256=hashlib.sha256(payload).hexdigest()
            )
            self.assertTrue(verify(requirement))

    def test_absent_digest_accepts_on_presence(self) -> None:
        # We would rather serve a model we cannot checksum than refuse to start.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            target.write_bytes(b"weights")
            self.assertTrue(verify(_requirement(target)))

    def test_missing_file_never_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(verify(_requirement(Path(tmp) / "absent.safetensors")))


class EnsureModelTests(unittest.TestCase):
    def test_valid_existing_model_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            target.write_bytes(b"weights")
            result = ensure_model(_requirement(target))
            self.assertEqual(result.status, "present")
            self.assertTrue(result.ok)

    def test_operator_supplied_model_reports_what_to_do(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "licensed.safetensors"
            result = ensure_model(
                _requirement(target, kind="operator", notes="Copy it in from your archive.")
            )
            self.assertEqual(result.status, "skipped")
            self.assertIn("Copy it in", result.detail)
            self.assertFalse(result.ok)

    def test_operator_model_without_notes_still_names_the_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "licensed.safetensors"
            result = ensure_model(_requirement(target, kind="operator"))
            self.assertIn("licensed.safetensors", result.detail)
            self.assertIn(tmp, result.detail)

    def test_corrupt_existing_model_is_replaced_not_trusted(self) -> None:
        payload = b"good weights"
        digest = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            target.write_bytes(b"corrupt")
            requirement = _requirement(target, sha256=digest)

            def fake_download(req, **_kwargs):
                part = req.destination.with_suffix(req.destination.suffix + ".part")
                part.write_bytes(payload)
                part.replace(req.destination)

            with patch("client.model_sources._download_coordinator", fake_download):
                result = ensure_model(requirement)

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(target.read_bytes(), payload)

    def test_checksum_failure_does_not_leave_a_bad_file_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            requirement = _requirement(target, sha256="a" * 64)

            def bad_download(req, **_kwargs):
                raise ModelDownloadError(f"{req.name}: checksum mismatch")

            with patch("client.model_sources._download_coordinator", bad_download):
                result = ensure_model(requirement)

            self.assertEqual(result.status, "failed")
            self.assertIn("checksum", result.detail)
            # The destination must stay empty rather than holding partial data.
            self.assertFalse(target.exists())

    def test_transient_failures_are_retried(self) -> None:
        attempts = {"count": 0}

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"

            def flaky(req, **_kwargs):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise OSError("connection reset")
                req.destination.write_bytes(b"weights")

            with patch("client.model_sources._download_coordinator", flaky), patch(
                "client.model_sources.time.sleep"
            ):
                result = ensure_model(_requirement(target), retries=3)

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(attempts["count"], 3)

    def test_auth_failures_are_not_retried(self) -> None:
        attempts = {"count": 0}

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"

            def unauthorized(req, **_kwargs):
                attempts["count"] += 1
                raise ModelDownloadError("coordinator rejected the join token")

            with patch("client.model_sources._download_coordinator", unauthorized), patch(
                "client.model_sources.time.sleep"
            ):
                result = ensure_model(_requirement(target), retries=3)

            self.assertEqual(result.status, "failed")
            # Retrying a rejected token just delays the real error.
            self.assertEqual(attempts["count"], 1)


if __name__ == "__main__":
    unittest.main()

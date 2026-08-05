"""Tests for node preflight diagnostics.

The doctor's job is to stop a node advertising a capability it cannot service.
These tests pin the behaviour that decides that: a failure gating a capability
must mark it not-ready, and a core failure must sink everything.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from client import doctor
from client.doctor import FAIL, OK, WARN, CheckResult, Report


class ReportLogicTests(unittest.TestCase):
    def test_warnings_do_not_block_a_capability(self) -> None:
        report = Report()
        report.add(CheckResult("a", "A", OK, feature="core"))
        report.add(CheckResult("b", "B", WARN, feature="image"))
        self.assertTrue(report.feature_ready("image"))
        self.assertIn("image", report.capabilities)

    def test_a_failure_blocks_only_its_own_capability(self) -> None:
        report = Report()
        report.add(CheckResult("core", "Core", OK, feature="core"))
        report.add(CheckResult("img", "Image", OK, feature="image"))
        report.add(CheckResult("face", "Face", FAIL, feature="face_swap"))

        self.assertTrue(report.feature_ready("image"))
        self.assertFalse(report.feature_ready("face_swap"))
        self.assertEqual(report.capabilities, ["image", "video"])

    def test_a_core_failure_sinks_every_capability(self) -> None:
        report = Report()
        report.add(CheckResult("core", "Core deps", FAIL, feature="core"))
        report.add(CheckResult("img", "Image", OK, feature="image"))

        self.assertFalse(report.feature_ready("image"))
        self.assertEqual(report.capabilities, [])
        self.assertFalse(report.healthy)

    def test_serialised_report_is_json_safe(self) -> None:
        report = Report()
        report.add(CheckResult("x", "X", WARN, feature="video", detail="d", remedy="r"))
        payload = report.to_dict()

        # The desktop app parses this; it must survive a round trip.
        decoded = json.loads(json.dumps(payload))
        self.assertIn("capabilities", decoded)
        self.assertIn("features", decoded)
        self.assertEqual(decoded["checks"][0]["remedy"], "r")


class RuntimeModuleCheckTests(unittest.TestCase):
    """The check that catches a partial install."""

    def _run_against(self, root: Path) -> Report:
        report = Report()
        with patch.object(doctor, "_runtime_root", return_value=root):
            doctor.check_runtime_modules(report)
        return report

    def _status(self, report: Report, check_id: str) -> str:
        return next(check.status for check in report.checks if check.id == check_id)

    def test_complete_runtime_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "client/pipeline_stable_diffusion_xl_instantid.py",
                "client/pipeline_stable_diffusion_xl_instantid_inpaint.py",
                "client/ip_adapter/__init__.py",
                "client/ip_adapter/resampler.py",
                "client/ip_adapter/attention_processor.py",
                "client/ip_adapter/utils.py",
                "engines/__init__.py",
                "engines/ltx_video/runner.py",
                "engines/ltx_video/config.py",
                "engines/ltx2/ltx2_runner.py",
                "engines/animatediff/animatediff_runner.py",
                "engines/wangp/runner.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            report = self._run_against(root)
            self.assertEqual(self._status(report, "runtime_face_swap"), OK)
            self.assertEqual(self._status(report, "runtime_video"), OK)

    def test_legacy_partial_install_is_caught(self) -> None:
        # Exactly what the old installer produced: client.py and nothing else.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client").mkdir(parents=True)
            (root / "client" / "client.py").write_text("", encoding="utf-8")

            report = self._run_against(root)
            self.assertEqual(self._status(report, "runtime_face_swap"), FAIL)
            self.assertEqual(self._status(report, "runtime_video"), FAIL)
            self.assertFalse(report.feature_ready("face_swap"))
            self.assertFalse(report.feature_ready("video"))

    def test_failures_tell_the_operator_how_to_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run_against(Path(tmp))
            for check in report.checks:
                if check.status == FAIL:
                    self.assertTrue(check.remedy, f"{check.id} has no remedy")
                    self.assertIn("installer", check.remedy.lower())


class OutputsCheckTests(unittest.TestCase):
    def test_unwritable_outputs_directory_is_a_blocking_failure(self) -> None:
        report = Report()
        with patch.object(doctor, "_havnai_home", return_value=Path("/proc/nonexistent")):
            doctor.check_outputs(report)
        self.assertEqual(report.checks[0].status, FAIL)


class ReportRenderingTests(unittest.TestCase):
    def test_formatted_report_names_blocking_issues(self) -> None:
        report = Report()
        report.add(CheckResult("deps", "Core dependencies", FAIL, feature="core", detail="missing torch"))
        text = doctor.format_report(report)

        self.assertIn("Core dependencies", text)
        self.assertIn("Blocking issues", text)
        self.assertIn("nothing yet", text)

    def test_ready_capabilities_are_listed(self) -> None:
        report = Report()
        report.add(CheckResult("core", "Core", OK, feature="core"))
        text = doctor.format_report(report)
        self.assertIn("image, face_swap, video", text)


if __name__ == "__main__":
    unittest.main()

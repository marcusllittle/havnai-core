"""Portable tests; Apple's tools are mocked and no credentials are used."""

import contextlib
import io
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import macos_release as release


TEAM = "TESTTEAM01"
SUBMISSION = "12345678-1234-1234-1234-123456789abc"
ENV = {
    "APPLE_CERTIFICATE": "not-a-real-certificate",
    "APPLE_CERTIFICATE_PASSWORD": "test-export-password",
    "APPLE_SIGNING_IDENTITY": f"Developer ID Application: Test User ({TEAM})",
    "APPLE_ID": "test@example.invalid",
    "APPLE_PASSWORD": "not-a-real-secret",
    "APPLE_TEAM_ID": TEAM,
}


class MacReleaseTest(unittest.TestCase):
    def test_preflight_accepts_developer_id_for_correct_team(self):
        self.assertEqual(release.validate_environment(release.SIGNING + release.NOTARY, ENV), TEAM)

    def test_preflight_rejects_missing_secrets_without_echoing_values(self):
        env = {**ENV, "APPLE_PASSWORD": ""}
        with self.assertRaisesRegex(ValueError, "APPLE_PASSWORD") as err:
            release.validate_environment(release.SIGNING + release.NOTARY, env)
        self.assertNotIn(ENV["APPLE_CERTIFICATE"], str(err.exception))

    def test_preflight_rejects_dev_certificate_and_wrong_team(self):
        for identity in ("Apple Development: Test User (TESTTEAM01)", "Developer ID Application: Test User (OTHERTEAM1)"):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "Developer ID Application"):
                release.validate_environment(release.SIGNING + release.NOTARY, {**ENV, "APPLE_SIGNING_IDENTITY": identity})

    def test_find_dmg_requires_single_exact_target_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = "aarch64-apple-darwin"
            bundles = root / "src-tauri" / "target" / target / "release" / "bundle" / "dmg"
            bundles.mkdir(parents=True)
            wrong = bundles / "HavnAI Node_0.2.0_x64.dmg"
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                release.find_dmg(root, target)
            wrong.unlink()
            arm = bundles / "HavnAI Node_0.2.0_aarch64.dmg"
            arm.touch()
            self.assertEqual(release.find_dmg(root, target), arm)
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                release.find_dmg(root, target)

    def test_payload_check_rejects_mislabeled_intel_dmg_and_detaches(self):
        def mock_run(*args, **kwargs):
            if args[:2] == ("hdiutil", "attach"):
                mount = Path(args[-2])
                contents = mount / "HavnAI Node.app" / "Contents"
                contents.mkdir(parents=True)
                with (contents / "Info.plist").open("wb") as handle:
                    plistlib.dump({"CFBundleIdentifier": "io.joinhavn.node", "CFBundleExecutable": "havnai-node-desktop"}, handle)
            if args[0] == "lipo":
                return "arm64\n"

        with patch.object(release, "run", side_effect=mock_run) as runner:
            with self.assertRaisesRegex(ValueError, "wrong app architecture"):
                release.verify_payload(Path("installer_x64.dmg"), "x86_64-apple-darwin", "test requirement")
        self.assertEqual(runner.call_args.args[:2], ("hdiutil", "detach"))

    def test_outer_dmg_must_be_accepted_before_stapling_and_assessment(self):
        accepted = json.dumps({"id": SUBMISSION, "status": "Accepted"})
        dmg = Path("installer.dmg")
        with patch.object(release, "run", return_value=accepted) as runner, contextlib.redirect_stdout(io.StringIO()):
            release.notarize(dmg, ENV)
        calls = [c.args[:3] for c in runner.call_args_list]
        self.assertEqual(calls, [
            ("xcrun", "notarytool", "submit"),
            ("xcrun", "stapler", "staple"),
            ("xcrun", "stapler", "validate"),
            ("spctl", "--assess", "--type"),
        ])
        self.assertTrue(all(c.args[-1] == str(dmg) for c in runner.call_args_list[1:]))

    def test_notary_rejection_prevents_any_staple(self):
        response = json.dumps({"id": SUBMISSION, "status": "Invalid"})
        with patch.object(release, "run", return_value=response) as runner:
            with self.assertRaisesRegex(ValueError, SUBMISSION):
                release.notarize(Path("installer.dmg"), ENV)
        runner.assert_called_once()

    def test_subprocess_failure_does_not_disclose_password_argument(self):
        args = ("xcrun", "notarytool", "--password", ENV["APPLE_PASSWORD"])
        error = subprocess.CalledProcessError(1, args, stderr=ENV["APPLE_PASSWORD"])
        with patch.object(release.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(ValueError, "xcrun failed") as err:
                release.run(*args, capture_output=True)
        self.assertNotIn(ENV["APPLE_PASSWORD"], str(err.exception))


if __name__ == "__main__":
    unittest.main()

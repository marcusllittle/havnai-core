"""Fail-closed checks for the macOS desktop release runners."""

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import tempfile

TARGETS = {
    "aarch64-apple-darwin": ("aarch64", "arm64"),
    "x86_64-apple-darwin": ("x64", "x86_64"),
}
SIGNING = ("APPLE_CERTIFICATE", "APPLE_CERTIFICATE_PASSWORD", "APPLE_SIGNING_IDENTITY")
NOTARY = ("APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID")
DESKTOP = Path(__file__).resolve().parents[1]


def validate_environment(names, env):
    """Report missing setting names only; never print their values."""
    missing = [name for name in names if not env.get(name, "").strip()]
    if missing:
        raise ValueError("Configure GitHub Actions secrets: " + ", ".join(missing))
    team = env["APPLE_TEAM_ID"]
    if not re.fullmatch(r"[A-Z0-9]{10}", team):
        raise ValueError("APPLE_TEAM_ID must be a ten-character Apple team ID")
    if "APPLE_SIGNING_IDENTITY" in names:
        identity = env["APPLE_SIGNING_IDENTITY"]
        if not (identity.startswith("Developer ID Application: ") and identity.endswith(f" ({team})")):
            raise ValueError("APPLE_SIGNING_IDENTITY must be a Developer ID Application for APPLE_TEAM_ID")
    return team


def run(*args, **kwargs):
    # Apple credentials can appear in notarytool arguments. Do not surface
    # exceptions that embed those arguments, even when the Apple call fails.
    try:
        return subprocess.run(args, check=True, text=True, **kwargs).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{args[0]} failed (exit {exc.returncode})") from None


def find_dmg(desktop, target):
    suffix, _ = TARGETS[target]
    folder = desktop / "src-tauri" / "target" / target / "release" / "bundle" / "dmg"
    matches = list(folder.glob("*.dmg"))
    if len(matches) != 1 or not matches[0].name.endswith(f"_{suffix}.dmg"):
        raise ValueError(f"Expected exactly one matching _{suffix}.dmg in the current {target} build")
    return matches[0]


def verify_payload(dmg, target, requirement):
    run("codesign", "--verify", "--strict", "-R", requirement, str(dmg))
    with tempfile.TemporaryDirectory(prefix="havnai-dmg-") as tmp:
        mount = Path(tmp) / "mount"
        mount.mkdir()
        run("hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount), str(dmg))
        try:
            app = mount / "HavnAI Node.app"
            run("codesign", "--verify", "--deep", "--strict", str(app))
            run("codesign", "--verify", "--strict", "-R", requirement, str(app))
            with (app / "Contents" / "Info.plist").open("rb") as handle:
                plist = plistlib.load(handle)
            if plist.get("CFBundleIdentifier") != "io.joinhavn.node":
                raise ValueError("Unexpected macOS bundle identifier")
            executable = plist.get("CFBundleExecutable")
            if not executable or Path(executable).name != executable:
                raise ValueError("Invalid macOS app executable name")
            binary = app / "Contents" / "MacOS" / executable
            arch = run("lipo", "-archs", str(binary), capture_output=True).split()
            if arch != [TARGETS[target][1]]:
                raise ValueError(f"DMG contains the wrong app architecture for {target}")
        finally:
            run("hdiutil", "detach", str(mount))


def notarize(dmg, env):
    # Notarize only the outer signed DMG, then verify that exact file.
    # Capture subprocess streams so no authentication errors can echo secrets.
    response = run(
        "xcrun", "notarytool", "submit", str(dmg),
        "--apple-id", env["APPLE_ID"], "--password", env["APPLE_PASSWORD"],
        "--team-id", env["APPLE_TEAM_ID"], "--wait", "--timeout", "45m",
        "--output-format", "json", capture_output=True,
    )
    result = json.loads(response)
    submission = result.get("id", "")
    if not isinstance(submission, str) or not re.fullmatch(r"[a-fA-F0-9-]{36}", submission):
        raise ValueError("Apple notary service did not return a submission ID")
    if result.get("status") != "Accepted":
        raise ValueError(f"Apple did not accept notarization; retrieve notary log for {submission}")
    print(f"Apple notarization Accepted: {submission}")
    run("xcrun", "stapler", "staple", str(dmg))
    run("xcrun", "stapler", "validate", str(dmg))
    run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature",
        "--verbose=2", str(dmg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("preflight")
    release = modes.add_parser("notarize")
    release.add_argument("target", choices=TARGETS)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            validate_environment(SIGNING + NOTARY, os.environ)
            print("macOS signing and notarization settings supplied")
            return
        team = validate_environment(NOTARY, os.environ)
        requirement = (
            "anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.13] exists "
            f'and certificate leaf[subject.OU] = "{team}"'
        )
        dmg = find_dmg(DESKTOP, args.target)
        verify_payload(dmg, args.target, requirement)
        notarize(dmg, os.environ)
        print(f"Verified release installer: {dmg.name}")
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        parser.exit(1, f"macOS release verification failed: {exc}\n")


if __name__ == "__main__":
    main()

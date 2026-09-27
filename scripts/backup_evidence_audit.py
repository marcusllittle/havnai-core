#!/usr/bin/env python3
"""Validate HAVN-44 backup/restore evidence packets.

This script does not perform a restore. It audits redacted JSON evidence that was
already produced by backup, restore, and media recovery drills, and it requires
either remote/offsite proof or an explicit dated waiver.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _check_backup_manifest(path: str) -> tuple[bool, list[str], dict[str, Any]]:
    data = _load_json(path)
    failures: list[str] = []
    if data.get("integrity_check") != "ok":
        failures.append("integrity_check is not ok")
    if int(data.get("backup_size_bytes") or 0) <= 0:
        failures.append("backup_size_bytes is missing or zero")
    if data.get("local_mode_octal") != "0o600":
        failures.append("local_mode_octal is not 0o600")
    if int(data.get("local_retention_count") or 0) < 1:
        failures.append("local_retention_count is missing or zero")
    if not data.get("backup_sha256"):
        failures.append("backup_sha256 is missing")
    return not failures, failures, data


def _check_restore_report(path: str) -> tuple[bool, list[str], dict[str, Any]]:
    data = _load_json(path)
    failures: list[str] = []
    if data.get("verified") is not True:
        failures.append("verified is not true")
    integrity = data.get("integrity") or data.get("integrity_check")
    if integrity != "ok":
        failures.append("integrity is not ok")
    fk = data.get("foreign_key_violations")
    if fk not in (0, [], None):
        failures.append("foreign_key_violations is not zero/empty")
    table_count = data.get("table_count")
    tables = data.get("tables") or data.get("table_counts")
    if table_count is None and isinstance(tables, dict):
        table_count = len(tables)
    if int(table_count or 0) <= 0:
        failures.append("table_count/tables is missing or empty")
    return not failures, failures, data


def _check_media_report(path: str) -> tuple[bool, list[str], dict[str, Any]]:
    data = _load_json(path)
    failures: list[str] = []
    if data.get("verified") is not True and data.get("passed") is not True:
        failures.append("verified/passed is not true")
    text = json.dumps(data, sort_keys=True)
    if "sha256" not in text and "hash" not in text:
        failures.append("no checksum/hash field found")
    return not failures, failures, data


def audit_backup_evidence(
    *,
    backup_manifest: str,
    restore_report: str,
    media_reports: list[str],
    remote_waiver: str | None,
    require_remote: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    ok, failures, manifest = _check_backup_manifest(backup_manifest)
    checks.append({"name": "backup_manifest", "ok": ok, "path": backup_manifest, "failures": failures})

    ok, failures, restore = _check_restore_report(restore_report)
    checks.append({"name": "restore_report", "ok": ok, "path": restore_report, "failures": failures})

    for path in media_reports:
        ok, failures, _ = _check_media_report(path)
        checks.append({"name": "media_report", "ok": ok, "path": path, "failures": failures})

    remote = manifest.get("remote") if isinstance(manifest, dict) else None
    remote_configured = bool(isinstance(remote, dict) and remote.get("configured"))
    if require_remote and not remote_configured and not remote_waiver:
        blockers.append("remote/offsite backup is not configured and no dated waiver was supplied")
    if remote_waiver:
        waiver = _load_json(remote_waiver)
        if not waiver.get("approved_by") or not waiver.get("expires_at") or not waiver.get("mitigation"):
            blockers.append("remote waiver must include approved_by, expires_at, and mitigation")

    return {
        "schema_version": "havnai.backup-evidence-audit.v1",
        "passed": all(check["ok"] for check in checks) and not blockers,
        "checks": checks,
        "summary": {
            "backup_created_at": manifest.get("created_at"),
            "backup_size_bytes": manifest.get("backup_size_bytes"),
            "local_retention_count": manifest.get("local_retention_count"),
            "remote_configured": remote_configured,
            "restore_verified": restore.get("verified"),
            "media_report_count": len(media_reports),
        },
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-manifest", required=True)
    parser.add_argument("--restore-report", required=True)
    parser.add_argument("--media-report", action="append", default=[])
    parser.add_argument("--media-report-glob", action="append", default=[])
    parser.add_argument("--remote-waiver")
    parser.add_argument("--no-require-remote", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    media_reports = list(args.media_report)
    for pattern in args.media_report_glob:
        media_reports.extend(sorted(glob.glob(pattern)))

    report = audit_backup_evidence(
        backup_manifest=args.backup_manifest,
        restore_report=args.restore_report,
        media_reports=media_reports,
        remote_waiver=args.remote_waiver,
        require_remote=not args.no_require_remote,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"passed={report['passed']}")
        for check in report["checks"]:
            print(f"{'ok' if check['ok'] else 'FAIL'} {check['name']} {check['path']}")
        for blocker in report["blockers"]:
            print(f"BLOCKER {blocker}", file=sys.stderr)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

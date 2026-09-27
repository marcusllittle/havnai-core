#!/usr/bin/env python3
"""Build a redacted backup-retention evidence report for HAVN-44.

This script is read-only. It inspects local backup files and an optional
operator-supplied remote listing JSON file, then reports whether the evidence is
strong enough for launch or whether a remote-backup waiver is still required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def redact_path(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    return f"sha256:{hash_text(value)}"


def backup_timestamp(path: Path) -> str | None:
    name = path.name
    prefix = "ledger-"
    suffix = ".sqlite.gz"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stamp = name[len(prefix):-len(suffix)]
    try:
        parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def local_backups(backup_dir: Path) -> list[dict[str, Any]]:
    if not backup_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(backup_dir.glob("ledger-*.sqlite.gz")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append({
            "name": path.name,
            "path_hash": redact_path(str(path.resolve())),
            "timestamp": backup_timestamp(path),
            "size_bytes": stat.st_size,
        })
    return sorted(rows, key=lambda item: str(item.get("timestamp") or ""), reverse=True)


def remote_backups(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"configured": False, "entries": [], "error": "remote listing not supplied"}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("remote listing must be a list or an object with an entries list")
    normalized = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or Path(str(raw.get("path") or "")).name)
        normalized.append({
            "name": name,
            "path_hash": redact_path(str(raw.get("path") or name)),
            "timestamp": raw.get("timestamp") or backup_timestamp(Path(name)),
            "size_bytes": int(raw.get("size_bytes") or raw.get("size") or 0),
        })
    return {
        "configured": True,
        "target_hash": redact_path(str(data.get("target") or "")) if isinstance(data, dict) else "",
        "entries": sorted(normalized, key=lambda item: str(item.get("timestamp") or ""), reverse=True),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    local = local_backups(Path(args.backup_dir))
    remote = remote_backups(Path(args.remote_listing) if args.remote_listing else None)
    remote_entries = remote.get("entries") or []
    newest_local = local[0] if local else None
    newest_remote = remote_entries[0] if remote_entries else None
    remote_verified = bool(
        remote.get("configured")
        and remote_entries
        and args.remote_retention_days >= 30
        and args.encryption_owner.strip()
        and args.access_group.strip()
        and args.restore_read_test.strip()
        and args.alert_owner.strip()
    )
    waiver_present = bool(args.waiver_id.strip())
    passed = bool(local) and (remote_verified or waiver_present)
    missing: list[str] = []
    if not local:
        missing.append("local_backup_listing")
    if not remote_verified and not waiver_present:
        missing.append("remote_backup_evidence_or_waiver")
    if remote.get("configured") and not remote_entries:
        missing.append("remote_backup_listing_entries")
    return {
        "schema": "havn-44-backup-evidence-audit.v1",
        "generated_at": utc_now(),
        "passed": passed,
        "missing": missing,
        "local": {
            "backup_dir_hash": redact_path(str(Path(args.backup_dir).resolve())),
            "count": len(local),
            "newest": newest_local,
            "retention_policy_count": 7,
        },
        "remote": {
            "configured": bool(remote.get("configured")),
            "target_hash": remote.get("target_hash", ""),
            "count": len(remote_entries),
            "newest": newest_remote,
            "retention_days": args.remote_retention_days,
            "verified": remote_verified,
            "encryption_owner_present": bool(args.encryption_owner.strip()),
            "access_group_present": bool(args.access_group.strip()),
            "restore_read_test_present": bool(args.restore_read_test.strip()),
            "alert_owner_present": bool(args.alert_owner.strip()),
        },
        "waiver": {
            "present": waiver_present,
            "id": args.waiver_id.strip(),
            "expires": args.waiver_expires.strip(),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", default=os.environ.get("HAVNAI_BACKUP_DIR", "/var/lib/havnai/backups"))
    parser.add_argument("--remote-listing", default="")
    parser.add_argument("--remote-retention-days", type=int, default=0)
    parser.add_argument("--encryption-owner", default="")
    parser.add_argument("--access-group", default="")
    parser.add_argument("--restore-read-test", default="")
    parser.add_argument("--alert-owner", default="")
    parser.add_argument("--waiver-id", default="")
    parser.add_argument("--waiver-expires", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = collect(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

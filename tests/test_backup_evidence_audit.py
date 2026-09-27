from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backup_evidence_audit as audit  # type: ignore


def write_backup(path: Path, name: str = "ledger-20260926T225900Z.sqlite.gz") -> Path:
    path.mkdir()
    backup = path / name
    backup.write_bytes(b"backup")
    return backup


def test_local_backup_without_remote_or_waiver_is_not_launch_ready(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    write_backup(backup_dir)

    report = audit.collect(audit.parse_args(["--backup-dir", str(backup_dir)]))

    assert report["passed"] is False
    assert report["local"]["count"] == 1
    assert report["local"]["newest"]["timestamp"] == "2026-09-26T22:59:00Z"
    assert "remote_backup_evidence_or_waiver" in report["missing"]
    assert str(backup_dir) not in json.dumps(report)


def test_full_remote_evidence_passes_with_redacted_paths(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    write_backup(backup_dir)
    remote_listing = tmp_path / "remote.json"
    remote_listing.write_text(json.dumps({
        "target": "backup-host:/private/havnai/backups",
        "entries": [
            {"path": "/private/havnai/backups/ledger-20260926T225900Z.sqlite.gz", "size_bytes": 42},
        ],
    }), encoding="utf-8")

    report = audit.collect(audit.parse_args([
        "--backup-dir", str(backup_dir),
        "--remote-listing", str(remote_listing),
        "--remote-retention-days", "30",
        "--encryption-owner", "platform-ops",
        "--access-group", "havnai-backup-operators",
        "--restore-read-test", "operator-read-20260926",
        "--alert-owner", "on-call",
    ]))

    encoded = json.dumps(report)
    assert report["passed"] is True
    assert report["remote"]["verified"] is True
    assert report["remote"]["count"] == 1
    assert "backup-host" not in encoded
    assert "/private/havnai" not in encoded


def test_waiver_can_satisfy_remote_gap_but_not_missing_local_backup(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    missing_local = audit.collect(audit.parse_args([
        "--backup-dir", str(empty_dir),
        "--waiver-id", "HAVN-44-waiver-20260926",
        "--waiver-expires", "2026-10-10",
    ]))

    assert missing_local["passed"] is False
    assert "local_backup_listing" in missing_local["missing"]

    backup_dir = tmp_path / "backups"
    write_backup(backup_dir)
    report = audit.collect(audit.parse_args([
        "--backup-dir", str(backup_dir),
        "--waiver-id", "HAVN-44-waiver-20260926",
        "--waiver-expires", "2026-10-10",
    ]))

    assert report["passed"] is True
    assert report["waiver"] == {
        "present": True,
        "id": "HAVN-44-waiver-20260926",
        "expires": "2026-10-10",
    }

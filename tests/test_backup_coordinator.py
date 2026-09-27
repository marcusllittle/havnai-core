import gzip
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import backup_coordinator


def _database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE accounts(id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO accounts VALUES ('acct_one')")
        conn.commit()
    finally:
        conn.close()


def test_backup_writes_owner_only_gzip_and_manifest(tmp_path, monkeypatch):
    source = tmp_path / "ledger.db"
    backups = tmp_path / "backups"
    report = tmp_path / "report.json"
    _database(source)
    monkeypatch.setenv("HAVNAI_DB_PATH", str(source))
    monkeypatch.setenv("HAVNAI_BACKUP_DIR", str(backups))
    monkeypatch.setenv("HAVNAI_BACKUP_REPORT", str(report))

    assert backup_coordinator.main() == 0

    [archive] = backups.glob("ledger-*.sqlite.gz")
    [manifest_path] = backups.glob("ledger-*.manifest.json")
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    with gzip.open(archive, "rb") as handle:
        assert handle.read(16)
    manifest = json.loads(manifest_path.read_text())
    assert manifest == json.loads(report.read_text())
    assert manifest["schema_version"] == "havnai.coordinator-backup.v1"
    assert manifest["integrity_check"] == "ok"
    assert manifest["backup"] == str(archive)
    assert manifest["backup_size_bytes"] == archive.stat().st_size
    assert manifest["local_mode_octal"] == "0o600"
    assert manifest["remote"] == {"configured": False}


def test_backup_remote_copies_archive_and_manifest_without_secret_key(tmp_path, monkeypatch):
    source = tmp_path / "ledger.db"
    backups = tmp_path / "backups"
    _database(source)
    monkeypatch.setenv("HAVNAI_DB_PATH", str(source))
    monkeypatch.setenv("HAVNAI_BACKUP_DIR", str(backups))
    monkeypatch.setenv("HAVNAI_BACKUP_REMOTE", "backup.example:/srv/havnai")
    monkeypatch.setenv("HAVNAI_BACKUP_SSH_KEY", "/secret/key")
    monkeypatch.setenv("HAVNAI_BACKUP_REMOTE_RETENTION_DAYS", "45")
    calls = []

    def fake_run(args, check):
        calls.append((list(args), check))

    monkeypatch.setattr(backup_coordinator.subprocess, "run", fake_run)

    assert backup_coordinator.main() == 0

    [manifest_path] = backups.glob("ledger-*.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["remote"] == {"configured": True, "host": "backup.example", "path": "/srv/havnai", "retention_days": 45}
    flat = "\n".join(" ".join(args) for args, _check in calls)
    assert "/secret/key" in flat
    assert "backup.example mkdir -p /srv/havnai" in flat
    assert str(manifest_path) in flat
    assert "+45" in flat
    assert all(check is True for _args, check in calls)


def test_backup_rejects_invalid_remote(tmp_path, monkeypatch):
    source = tmp_path / "ledger.db"
    _database(source)
    monkeypatch.setenv("HAVNAI_DB_PATH", str(source))
    monkeypatch.setenv("HAVNAI_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("HAVNAI_BACKUP_REMOTE", "missing-path")

    with pytest.raises(RuntimeError, match="host:path"):
        backup_coordinator.main()

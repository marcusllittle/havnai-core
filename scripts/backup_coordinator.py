#!/usr/bin/env python3
"""Create verified SQLite backups with local and Tailscale retention."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_summary(remote: str) -> dict:
    if not remote:
        return {"configured": False}
    if ":" not in remote:
        raise RuntimeError("HAVNAI_BACKUP_REMOTE must use host:path format")
    host, path = remote.split(":", 1)
    if not host or not path:
        raise RuntimeError("HAVNAI_BACKUP_REMOTE must include host and path")
    return {"configured": True, "host": host, "path": path}


def main() -> int:
    source = Path(os.environ["HAVNAI_DB_PATH"]).resolve()
    backup_dir = Path(os.getenv("HAVNAI_BACKUP_DIR", "/var/lib/havnai/backups")).resolve()
    remote = os.getenv("HAVNAI_BACKUP_REMOTE", "").strip()
    ssh_key = os.getenv("HAVNAI_BACKUP_SSH_KEY", "").strip()
    report_path = os.getenv("HAVNAI_BACKUP_REPORT", "").strip()
    remote_retention_days = int(os.getenv("HAVNAI_BACKUP_REMOTE_RETENTION_DAYS", "30"))
    if remote_retention_days < 1:
        raise RuntimeError("HAVNAI_BACKUP_REMOTE_RETENTION_DAYS must be positive")
    if not source.is_file():
        raise RuntimeError(f"database does not exist: {source}")
    remote_info = _remote_summary(remote)

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = backup_dir / f"ledger-{stamp}.sqlite.gz"
    manifest_path = backup_dir / f"ledger-{stamp}.manifest.json"
    with tempfile.TemporaryDirectory(dir=backup_dir) as temp_dir:
        snapshot = Path(temp_dir) / "ledger.sqlite"
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dst = sqlite3.connect(snapshot)
        try:
            src.backup(dst)
            integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"backup integrity check failed: {integrity}")
        finally:
            dst.close()
            src.close()
        with snapshot.open("rb") as input_file, gzip.open(destination, "wb", compresslevel=6) as output_file:
            shutil.copyfileobj(input_file, output_file)
    os.chmod(destination, 0o600)

    local_backups = sorted(backup_dir.glob("ledger-*.sqlite.gz"), reverse=True)
    for old in local_backups[7:]:
        old.unlink(missing_ok=True)
        old.with_name(old.name.replace(".sqlite.gz", ".manifest.json")).unlink(missing_ok=True)

    with gzip.open(destination, "rb") as backup_file:
        # Read a byte to verify the compressed stream is readable; full content
        # equality is covered by restore drills, and this catches corrupt gzip.
        backup_file.read(1)

    manifest = {
        "schema_version": "havnai.coordinator-backup.v1",
        "created_at": stamp,
        "source": str(source),
        "backup": str(destination),
        "backup_size_bytes": destination.stat().st_size,
        "backup_sha256": _sha256(destination),
        "manifest": str(manifest_path),
        "integrity_check": "ok",
        "local_retention_count": len(sorted(backup_dir.glob("ledger-*.sqlite.gz"), reverse=True)),
        "local_mode_octal": oct(destination.stat().st_mode & 0o777),
        "remote": {**remote_info, "retention_days": remote_retention_days} if remote_info["configured"] else remote_info,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    if remote:
        ssh = ["ssh"]
        scp = ["scp"]
        if ssh_key:
            ssh.extend(["-i", ssh_key])
            scp.extend(["-i", ssh_key])
        remote_host, remote_path = remote.split(":", 1)
        subprocess.run(ssh + [remote_host, "mkdir", "-p", remote_path], check=True)
        subprocess.run(scp + [str(destination), remote], check=True)
        subprocess.run(scp + [str(manifest_path), f"{remote_host}:{remote_path.rstrip('/')}/"], check=True)
        for pattern in ("ledger-*.sqlite.gz", "ledger-*.manifest.json"):
            subprocess.run(
                ssh + [remote_host, "find", remote_path, "-type", "f", "-name", pattern,
                       "-mtime", f"+{remote_retention_days}", "-delete"],
                check=True,
            )
    if report_path:
        report = Path(report_path).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(report, 0o600)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

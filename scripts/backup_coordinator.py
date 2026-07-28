#!/usr/bin/env python3
"""Create verified SQLite backups with local and Tailscale retention."""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> int:
    source = Path(os.environ["HAVNAI_DB_PATH"]).resolve()
    backup_dir = Path(os.getenv("HAVNAI_BACKUP_DIR", "/var/lib/havnai/backups")).resolve()
    remote = os.getenv("HAVNAI_BACKUP_REMOTE", "").strip()
    ssh_key = os.getenv("HAVNAI_BACKUP_SSH_KEY", "").strip()
    if not source.is_file():
        raise RuntimeError(f"database does not exist: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = backup_dir / f"ledger-{stamp}.sqlite.gz"
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

    local_backups = sorted(backup_dir.glob("ledger-*.sqlite.gz"), reverse=True)
    for old in local_backups[7:]:
        old.unlink(missing_ok=True)

    if remote:
        ssh = ["ssh"]
        scp = ["scp"]
        if ssh_key:
            ssh.extend(["-i", ssh_key])
            scp.extend(["-i", ssh_key])
        remote_host, remote_path = remote.split(":", 1)
        subprocess.run(ssh + [remote_host, "mkdir", "-p", remote_path], check=True)
        subprocess.run(scp + [str(destination), remote], check=True)
        subprocess.run(
            ssh
            + [
                remote_host,
                "find",
                remote_path,
                "-type",
                "f",
                "-name",
                "ledger-*.sqlite.gz",
                "-mtime",
                "+30",
                "-delete",
            ],
            check=True,
        )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

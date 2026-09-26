#!/usr/bin/env python3
"""Verify a SQLite backup/restore in a new private directory, never in-place."""
from __future__ import annotations

import argparse
from contextlib import closing
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inspect_database(path: Path) -> dict:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        integrity = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise RuntimeError("Database integrity check failed")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Database has {len(violations)} foreign-key violations")
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        counts = {name: conn.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                  for name in tables}
        return {"integrity": "ok", "foreign_key_violations": 0, "table_counts": counts}


def run_drill(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Source must be an existing SQLite database")
    output = output.resolve()
    # A new directory prevents replacing databases, previous evidence or symlinks.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    snapshot, archive, restored = (output / name for name in
                                    ("snapshot.sqlite3", "snapshot.sqlite3.gz", "restored.sqlite3"))
    # The backup API includes committed WAL contents and takes a consistent view
    # without stopping the coordinator or checkpointing/writing its source.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst)
    os.chmod(snapshot, 0o600)
    before = inspect_database(snapshot)
    with snapshot.open("rb") as input_file, gzip.open(archive, "xb") as output_file:
        shutil.copyfileobj(input_file, output_file)
    os.chmod(archive, 0o600)
    with gzip.open(archive, "rb") as input_file, restored.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file)
    os.chmod(restored, 0o600)
    after = inspect_database(restored)
    snapshot_hash, restored_hash = digest(snapshot), digest(restored)
    if snapshot_hash != restored_hash or before != after:
        raise RuntimeError("Restored database does not match the backup")
    report = {"format_version": 1, "verified": True, "snapshot_sha256": snapshot_hash,
              "restored_sha256": restored_hash, "archive_sha256": digest(archive), **after,
              "scope": "SQLite only; media files and external payment/provider state are not backed up"}
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    os.chmod(output / "report.json", 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory in an existing private parent")
    args = parser.parse_args()
    report = run_drill(args.source, args.output)
    print(json.dumps({"verified": report["verified"], "table_count": len(report["table_counts"]),
                      "snapshot_sha256": report["snapshot_sha256"], "report": str(args.output / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit and optionally remove job_attempts rows whose parents are gone."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from typing import Any


def _connect(database: Path, *, readonly: bool) -> sqlite3.Connection:
    database = database.resolve(strict=True)
    uri = database.as_uri() + ("?mode=ro" if readonly else "?mode=rw")
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def find_orphan_attempts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT a.*
        FROM job_attempts a
        LEFT JOIN jobs j ON j.id = a.job_id
        LEFT JOIN job_settlement s ON s.job_id = a.job_id
        WHERE j.id IS NULL AND s.job_id IS NULL
        ORDER BY a.id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _foreign_key_violations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()]


def run(database: Path, report_path: Path, *, apply: bool = False) -> dict[str, Any]:
    report_path = report_path.resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(database, readonly=not apply)) as conn:
        before_violations = _foreign_key_violations(conn)
        orphans = find_orphan_attempts(conn)
        deleted = 0
        after_violations = before_violations
        if apply and orphans:
            ids = [int(row["id"]) for row in orphans]
            with conn:
                conn.executemany("DELETE FROM job_attempts WHERE id = ?", [(value,) for value in ids])
            deleted = len(ids)
            after_violations = _foreign_key_violations(conn)
        elif apply:
            after_violations = _foreign_key_violations(conn)

    report = {
        "format_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "database": str(database.resolve()),
        "mode": "apply" if apply else "audit",
        "orphan_attempt_count": len(orphans),
        "deleted_attempt_count": deleted,
        "orphan_attempts": orphans,
        "foreign_key_violations_before": before_violations,
        "foreign_key_violations_after": after_violations,
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, default=str)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="New JSON report path")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete orphan job_attempts rows after exporting them to the report.",
    )
    args = parser.parse_args()
    report = run(args.database, args.report, apply=args.apply)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "orphan_attempt_count": report["orphan_attempt_count"],
                "deleted_attempt_count": report["deleted_attempt_count"],
                "foreign_key_violations_before": len(report["foreign_key_violations_before"]),
                "foreign_key_violations_after": len(report["foreign_key_violations_after"]),
                "report": str(args.report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

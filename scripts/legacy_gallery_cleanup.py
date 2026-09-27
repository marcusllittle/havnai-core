#!/usr/bin/env python3
"""Audit or delist legacy wallet-era public gallery rows.

This is an operator helper for launch hardening. It does not delete rows and it
does not touch account-owned marketplace listings.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


SCHEMA = "havn-72-legacy-gallery-cleanup.v1"


@dataclass
class Candidate:
    id: int
    job_id: str
    title_hash: str
    asset_type: str
    model: str
    created_at: float
    updated_at: float


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def find_candidates(conn: sqlite3.Connection, only_ids: list[int] | None = None) -> list[Candidate]:
    params: list[Any] = []
    id_filter = ""
    if only_ids:
        placeholders = ",".join("?" for _ in only_ids)
        id_filter = f" AND gl.id IN ({placeholders})"
        params.extend(only_ids)

    rows = conn.execute(
        f"""
        SELECT gl.id, gl.job_id, gl.title, gl.asset_type, gl.model, gl.created_at, gl.updated_at
        FROM gallery_listings gl
        WHERE gl.listed = 1
          AND gl.sold = 0
          AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.id = gl.job_id
              AND j.owner_account_id IS NOT NULL
          )
          {id_filter}
        ORDER BY gl.id
        """,
        params,
    ).fetchall()

    return [
        Candidate(
            id=int(row["id"]),
            job_id=str(row["job_id"]),
            title_hash=_hash(str(row["title"] or "")),
            asset_type=str(row["asset_type"] or ""),
            model=str(row["model"] or ""),
            created_at=float(row["created_at"] or 0),
            updated_at=float(row["updated_at"] or 0),
        )
        for row in rows
    ]


def apply_delist(conn: sqlite3.Connection, candidates: list[Candidate], now: float) -> int:
    if not candidates:
        return 0
    ids = [candidate.id for candidate in candidates]
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"""
        UPDATE gallery_listings
        SET listed = 0, updated_at = ?
        WHERE id IN ({placeholders})
          AND listed = 1
          AND sold = 0
          AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.id = gallery_listings.job_id
              AND j.owner_account_id IS NOT NULL
          )
        """,
        [now, *ids],
    )
    conn.commit()
    return int(cursor.rowcount)


def build_report(
    *,
    db_path: str,
    candidates: list[Candidate],
    applied: bool,
    delisted_count: int,
    include_job_ids: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "apply" if applied else "audit",
        "db_path": str(Path(db_path).expanduser()),
        "candidate_count": len(candidates),
        "delisted_count": delisted_count,
        "account_owned_excluded": True,
        "deleted_rows": 0,
        "candidates": [],
    }
    for candidate in candidates:
        item: dict[str, Any] = {
            "id": candidate.id,
            "title_hash": candidate.title_hash,
            "asset_type": candidate.asset_type,
            "model": candidate.model,
            "created_at": candidate.created_at,
            "updated_at": candidate.updated_at,
        }
        if include_job_ids:
            item["job_id"] = candidate.job_id
        else:
            item["job_id_hash"] = _hash(candidate.job_id)
        report["candidates"].append(item)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=os.getenv("HAVNAI_DB_PATH"), help="SQLite DB path, or HAVNAI_DB_PATH.")
    parser.add_argument("--apply", action="store_true", help="Delist matching rows. Default is audit-only.")
    parser.add_argument("--include-job-ids", action="store_true", help="Include job ids in output for operator follow-up.")
    parser.add_argument("--only-id", action="append", type=int, default=[], help="Limit cleanup to a listing id. Repeatable.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.db_path:
        print("missing --db-path or HAVNAI_DB_PATH", file=sys.stderr)
        return 2
    if not Path(args.db_path).exists():
        print(f"database not found: {args.db_path}", file=sys.stderr)
        return 2

    with connect(args.db_path) as conn:
        candidates = find_candidates(conn, args.only_id or None)
        delisted = apply_delist(conn, candidates, time.time()) if args.apply else 0
        report = build_report(
            db_path=args.db_path,
            candidates=candidates,
            applied=args.apply,
            delisted_count=delisted,
            include_job_ids=args.include_job_ids,
        )

    if not args.json:
        action = "delisted" if args.apply else "would delist"
        print(f"{action} {report['candidate_count']} legacy public gallery rows")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Account-owned durable jobs: additive schema and atomic reservation/queue writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from decimal import Decimal, InvalidOperation

import account_ledger


def initialize(conn: sqlite3.Connection) -> None:
    for table, additions in {
        "jobs": ["creator_account_id", "owner_account_id"],
        "assets": ["owner_account_id"],
    }.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name in additions:
            if name not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT REFERENCES accounts(id)")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS jobs_account_history ON jobs(owner_account_id, timestamp DESC);
        CREATE TABLE IF NOT EXISTS account_job_requests (
            account_id TEXT NOT NULL REFERENCES accounts(id),
            request_key TEXT NOT NULL, payload_hash TEXT NOT NULL,
            job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
            PRIMARY KEY(account_id,request_key)
        );
    """)


def cost_units(cost, batch_size=1) -> int:
    try:
        units = Decimal(str(cost)) * account_ledger.SCALE * batch_size
        if not units.is_finite() or units <= 0 or units != units.to_integral_value() or units > account_ledger.MAX_UNITS:
            raise ValueError("invalid_generation_price")
        return int(units)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("invalid_generation_price") from exc


def enqueue(conn: sqlite3.Connection, account_id: str, *, request_key: str, request_payload: dict,
            model: str, task_type: str, settings: dict, resolved_spec: dict, weight: float,
            units: int) -> str:
    if not request_key or len(request_key) > 128:
        raise ValueError("idempotency_key_required")
    digest = hashlib.sha256(json.dumps(request_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        previous = conn.execute("SELECT payload_hash,job_id FROM account_job_requests WHERE account_id=? AND request_key=?",
                                (account_id, request_key)).fetchone()
        if previous:
            if previous[0] != digest:
                raise account_ledger.LedgerError("idempotency_conflict")
            return str(previous[1])
        # Recheck source ownership under the same lock used for enqueuing.
        for field in ("source_asset_id", "audio_asset_id", "reference_asset_id"):
            asset_id = request_payload.get(field)
            if asset_id and not conn.execute("SELECT 1 FROM assets WHERE id=? AND owner_account_id=?",
                                              (asset_id, account_id)).fetchone():
                raise ValueError("asset_not_found")
        now = time.time()
        job_id = "job-" + uuid.uuid4().hex
        account_ledger.reserve_in_transaction(conn, account_id, units, job_id=job_id)
        conn.execute("""INSERT INTO jobs
            (id,wallet,creator_account_id,owner_account_id,model,data,task_type,weight,status,
             timestamp,progress,stage,updated_at,resolved_spec)
            VALUES (?,'',?,?,?,?,?,?,'queued',?,0,'queued',?,?)""",
            (job_id, account_id, account_id, model, json.dumps(settings), task_type, weight,
             now, now, json.dumps(resolved_spec)))
        conn.execute("INSERT INTO account_job_requests VALUES (?,?,?,?)", (account_id, request_key, digest, job_id))
        return job_id


def settle_if_account_job(conn: sqlite3.Connection, job_id: str, status: str) -> None:
    # Legacy isolated callers may not yet have account tables. No migration on reads.
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_credit_reservations'").fetchone():
        return
    row = conn.execute("SELECT account_id FROM account_credit_reservations WHERE job_id=?", (job_id,)).fetchone()
    if row:
        account_ledger.finish_in_transaction(conn, row[0], job_id=job_id,
            succeeded=status.lower() in {"success", "succeeded", "completed", "done"})

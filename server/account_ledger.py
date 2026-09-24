"""Integer, transactional account-credit ledger. 1000 units = one credit.

Operations ending in _in_transaction require a caller-owned write transaction so
job/payment changes and credit mutations can commit atomically. No float inputs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

SCALE = 1000
MAX_UNITS = 10**15


class LedgerError(ValueError):
    pass


def initialize(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        raise RuntimeError("ledger initialization requires an idle connection")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_credit_balances (
            account_id TEXT PRIMARY KEY REFERENCES accounts(id),
            settled_units INTEGER NOT NULL DEFAULT 0 CHECK(typeof(settled_units)='integer'),
            reserved_units INTEGER NOT NULL DEFAULT 0 CHECK(typeof(reserved_units)='integer' AND reserved_units>=0),
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account_credit_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            operation TEXT NOT NULL, operation_key TEXT NOT NULL,
            payload_hash TEXT NOT NULL, resource_id TEXT NOT NULL,
            settled_delta INTEGER NOT NULL CHECK(typeof(settled_delta)='integer'),
            reserved_delta INTEGER NOT NULL CHECK(typeof(reserved_delta)='integer'),
            settled_after INTEGER NOT NULL, reserved_after INTEGER NOT NULL,
            actor TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,
            UNIQUE(account_id, operation_key)
        );
        CREATE TABLE IF NOT EXISTS account_credit_reservations (
            job_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            units INTEGER NOT NULL CHECK(typeof(units)='integer' AND units>0),
            state TEXT NOT NULL CHECK(state IN ('reserved','captured','released')),
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS account_ledger_no_update
            BEFORE UPDATE ON account_credit_ledger BEGIN
                SELECT RAISE(ABORT, 'credit ledger is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_ledger_no_delete
            BEFORE DELETE ON account_credit_ledger BEGIN
                SELECT RAISE(ABORT, 'credit ledger is append-only'); END;
    """)


def _units(units: int) -> None:
    if type(units) is not int or not 0 < units <= MAX_UNITS:
        raise LedgerError("invalid_credit_units")


def balance(conn: sqlite3.Connection, account_id: str) -> dict:
    row = conn.execute("SELECT settled_units,reserved_units FROM account_credit_balances WHERE account_id=?",
                       (account_id,)).fetchone()
    settled, reserved = row if row else (0, 0)
    return {"scale": SCALE, "settled_units": settled, "reserved_units": reserved,
            "available_units": max(0, settled - reserved), "debt_units": max(0, -settled)}


def _apply(conn: sqlite3.Connection, account_id: str, *, operation: str, key: str,
           resource: str, settled_delta: int, reserved_delta: int, actor: str,
           reason: str, require_available: int = 0) -> dict:
    if not conn.in_transaction:
        raise RuntimeError("ledger mutation requires a caller-owned transaction")
    if not key or len(key) > 255 or not resource or not actor or not reason:
        raise LedgerError("invalid_ledger_metadata")
    body = [operation, resource, settled_delta, reserved_delta, actor, reason]
    digest = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
    existing = conn.execute("SELECT id,payload_hash,settled_after,reserved_after FROM account_credit_ledger WHERE account_id=? AND operation_key=?",
                            (account_id, key)).fetchone()
    if existing:
        if existing[1] != digest:
            raise LedgerError("idempotency_conflict")
        return {"entry_id": existing[0], "settled_units": existing[2], "reserved_units": existing[3], "replayed": True}
    now = time.time()
    conn.execute("INSERT INTO account_credit_balances(account_id,updated_at) VALUES (?,?) ON CONFLICT(account_id) DO NOTHING",
                 (account_id, now))
    changed = conn.execute("""UPDATE account_credit_balances
        SET settled_units=settled_units+?,reserved_units=reserved_units+?,updated_at=?
        WHERE account_id=? AND (?=0 OR settled_units-reserved_units>=?)
        AND reserved_units+? >= 0
        AND settled_units+? BETWEEN ? AND ? AND reserved_units+? <= ?""",
        (settled_delta, reserved_delta, now, account_id, require_available, require_available,
         reserved_delta, settled_delta, -MAX_UNITS, MAX_UNITS, reserved_delta, MAX_UNITS))
    if changed.rowcount != 1:
        raise LedgerError("insufficient_credits" if require_available else "ledger_limit_exceeded")
    state = balance(conn, account_id)
    cursor = conn.execute("""INSERT INTO account_credit_ledger
        (account_id,operation,operation_key,payload_hash,resource_id,settled_delta,reserved_delta,
         settled_after,reserved_after,actor,reason,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (account_id, operation, key, digest, resource, settled_delta, reserved_delta,
         state["settled_units"], state["reserved_units"], actor, reason, now))
    return {"entry_id": cursor.lastrowid, "settled_units": state["settled_units"],
            "reserved_units": state["reserved_units"], "replayed": False}


def fund_in_transaction(conn: sqlite3.Connection, account_id: str, units: int, *,
                        payment_id: str, actor: str = "stripe") -> dict:
    _units(units)
    return _apply(conn, account_id, operation="fund", key=f"fund:{payment_id}", resource=payment_id,
                  settled_delta=units, reserved_delta=0, actor=actor, reason="paid_credit_purchase")


def reverse_funding_in_transaction(conn: sqlite3.Connection, account_id: str, units: int, *,
                                   payment_id: str, adjustment_id: str, reason: str) -> dict:
    _units(units)
    if reason not in {"refund", "chargeback"}:
        raise LedgerError("invalid_adjustment_reason")
    # Stable key catches replay before cumulative limit validation.
    key = f"adjust:{adjustment_id}"
    previous = conn.execute("SELECT 1 FROM account_credit_ledger WHERE account_id=? AND operation_key=?",
                            (account_id, key)).fetchone()
    if not previous:
        funding = conn.execute("SELECT settled_delta FROM account_credit_ledger WHERE account_id=? AND operation_key=? AND operation='fund'",
                               (account_id, f"fund:{payment_id}")).fetchone()
        reversed_units = conn.execute("SELECT COALESCE(-SUM(settled_delta),0) FROM account_credit_ledger WHERE account_id=? AND resource_id=? AND operation='reverse_funding'",
                                      (account_id, payment_id)).fetchone()[0]
        if not funding or reversed_units + units > funding[0]:
            raise LedgerError("funding_adjustment_exceeds_purchase")
    return _apply(conn, account_id, operation="reverse_funding", key=key, resource=payment_id,
                  settled_delta=-units, reserved_delta=0, actor="stripe", reason=reason)


def reserve_in_transaction(conn: sqlite3.Connection, account_id: str, units: int, *, job_id: str) -> dict:
    _units(units)
    if not conn.in_transaction:
        raise RuntimeError("reservation requires a caller-owned transaction")
    existing = conn.execute("SELECT account_id,units FROM account_credit_reservations WHERE job_id=?", (job_id,)).fetchone()
    if existing and tuple(existing) != (account_id, units):
        raise LedgerError("idempotency_conflict")
    result = _apply(conn, account_id, operation="reserve", key=f"reserve:{job_id}", resource=job_id,
                    settled_delta=0, reserved_delta=units, actor=account_id, reason="generation_reservation",
                    require_available=units)
    if not existing:
        now = time.time()
        conn.execute("INSERT INTO account_credit_reservations VALUES (?,?,?,'reserved',?,?)",
                     (job_id, account_id, units, now, now))
    return result


def finish_in_transaction(conn: sqlite3.Connection, account_id: str, *, job_id: str, succeeded: bool) -> dict:
    if not conn.in_transaction:
        raise RuntimeError("reservation settlement requires a caller-owned transaction")
    row = conn.execute("SELECT account_id,units,state FROM account_credit_reservations WHERE job_id=?", (job_id,)).fetchone()
    if not row or row[0] != account_id:
        raise LedgerError("reservation_not_found")
    state = "captured" if succeeded else "released"
    if row[2] not in {"reserved", state}:
        raise LedgerError("reservation_already_finalized")
    result = _apply(conn, account_id, operation="capture" if succeeded else "release",
                    key=f"finish:{job_id}", resource=job_id, settled_delta=-row[1] if succeeded else 0,
                    reserved_delta=-row[1], actor="coordinator", reason="generation_succeeded" if succeeded else "generation_failed_or_cancelled")
    conn.execute("UPDATE account_credit_reservations SET state=?,updated_at=? WHERE job_id=? AND state='reserved'",
                 (state, time.time(), job_id))
    return result

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
        CREATE TABLE IF NOT EXISTS account_credit_sales (
            sale_id TEXT PRIMARY KEY,
            buyer_account_id TEXT NOT NULL REFERENCES accounts(id),
            seller_account_id TEXT NOT NULL REFERENCES accounts(id),
            units INTEGER NOT NULL CHECK(typeof(units)='integer' AND units>0),
            debit_entry_id INTEGER NOT NULL REFERENCES account_credit_ledger(id),
            credit_entry_id INTEGER NOT NULL REFERENCES account_credit_ledger(id),
            created_at REAL NOT NULL,
            CHECK(buyer_account_id<>seller_account_id)
        );
        CREATE TRIGGER IF NOT EXISTS account_ledger_no_update
            BEFORE UPDATE ON account_credit_ledger BEGIN
                SELECT RAISE(ABORT, 'credit ledger is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_ledger_no_delete
            BEFORE DELETE ON account_credit_ledger BEGIN
                SELECT RAISE(ABORT, 'credit ledger is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_sales_no_update
            BEFORE UPDATE ON account_credit_sales BEGIN
                SELECT RAISE(ABORT, 'sale receipts are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_sales_no_delete
            BEFORE DELETE ON account_credit_sales BEGIN
                SELECT RAISE(ABORT, 'sale receipts are append-only'); END;
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


def settle_sale_in_transaction(conn: sqlite3.Connection, buyer_account_id: str,
                               seller_account_id: str, units: int, *, sale_id: str) -> dict:
    """Pair a marketplace debit and seller credit with a durable sale receipt.

    The caller must validate the listing and commit ownership in this same write
    transaction. This is an internal settlement primitive, not a transfer API.
    A savepoint prevents a caller that catches an error from retaining half a
    transfer. A globally unique sale ID binds both participants and the price.
    """
    if not conn.in_transaction:
        raise RuntimeError("sale settlement requires a caller-owned transaction")
    _units(units)
    if not isinstance(sale_id, str) or not sale_id.strip() or len(sale_id) > 200:
        raise LedgerError("invalid_sale_id")
    if buyer_account_id == seller_account_id:
        raise LedgerError("cannot_buy_own_listing")
    existing = conn.execute("""SELECT buyer_account_id,seller_account_id,units,
        debit_entry_id,credit_entry_id FROM account_credit_sales WHERE sale_id=?""",
        (sale_id,)).fetchone()
    if existing:
        if tuple(existing[:3]) != (buyer_account_id, seller_account_id, units):
            raise LedgerError("idempotency_conflict")
        return {"sale_id": sale_id, "debit_entry_id": existing[3],
                "credit_entry_id": existing[4], "replayed": True}
    for account_id in (buyer_account_id, seller_account_id):
        row = conn.execute("SELECT status FROM accounts WHERE id=?", (account_id,)).fetchone()
        if not row or row[0] != "active":
            raise LedgerError("sale_account_unavailable")
    conn.execute("SAVEPOINT account_sale_settlement")
    try:
        debit = _apply(conn, buyer_account_id, operation="marketplace_purchase",
                       key=f"sale_debit:{sale_id}", resource=sale_id, settled_delta=-units,
                       reserved_delta=0, actor=buyer_account_id, reason="marketplace_purchase",
                       require_available=units)
        credit = _apply(conn, seller_account_id, operation="marketplace_sale",
                        key=f"sale_credit:{sale_id}", resource=sale_id, settled_delta=units,
                        reserved_delta=0, actor=buyer_account_id, reason="marketplace_sale")
        conn.execute("""INSERT INTO account_credit_sales
            (sale_id,buyer_account_id,seller_account_id,units,debit_entry_id,credit_entry_id,created_at)
            VALUES (?,?,?,?,?,?,?)""", (sale_id, buyer_account_id, seller_account_id, units,
                                       debit["entry_id"], credit["entry_id"], time.time()))
    except Exception:
        conn.execute("ROLLBACK TO account_sale_settlement")
        conn.execute("RELEASE account_sale_settlement")
        raise
    conn.execute("RELEASE account_sale_settlement")
    return {"sale_id": sale_id, "debit_entry_id": debit["entry_id"],
            "credit_entry_id": credit["entry_id"], "replayed": False}


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


def adjust_payment_in_transaction(conn: sqlite3.Connection, account_id: str, *, payment_id: str,
                                  target_units: int, adjustment_id: str) -> dict:
    """Reconcile the retained credit grant, including restoration of won disputes.

    Target is a provider-verified purchase balance, not the user's spendable balance.
    The caller commits its receipt adjustment in the same write transaction.
    """
    if not conn.in_transaction:
        raise RuntimeError("payment adjustment requires a caller-owned transaction")
    funding = conn.execute("SELECT settled_delta FROM account_credit_ledger WHERE account_id=? AND operation_key=? AND operation='fund'",
                           (account_id, f"fund:{payment_id}")).fetchone()
    if not funding or type(target_units) is not int or not 0 <= target_units <= funding[0]:
        raise LedgerError("invalid_payment_target")
    key = f"payment_adjust:{adjustment_id}"
    reason = f"payment_retained_units:{target_units}"
    existing = conn.execute("SELECT settled_delta,reason FROM account_credit_ledger WHERE account_id=? AND operation_key=?",
                            (account_id, key)).fetchone()
    if existing and existing[1] != reason:
        raise LedgerError("idempotency_conflict")
    retained = conn.execute("""SELECT COALESCE(SUM(settled_delta),0) FROM account_credit_ledger
        WHERE account_id=? AND resource_id=? AND operation IN ('fund','reverse_funding','payment_adjust')""",
        (account_id, payment_id)).fetchone()[0]
    delta = existing[0] if existing else target_units - retained
    return _apply(conn, account_id, operation="payment_adjust", key=key, resource=payment_id,
                  settled_delta=delta, reserved_delta=0, actor="stripe", reason=reason)


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

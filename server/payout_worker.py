"""On-chain payout worker for HavnAI node rewards.

Background daemon that drains the node_payouts table — sending real
ERC-20 HAI transfers for any payout still in 'pending' status.

When CHAIN_RPC_URL / HAVNAI_PAYER_KEY are not set the worker runs in
simulation mode: payouts are marked 'simulated' rather than 'confirmed',
matching the existing simulated_hai asset type used during development.

Injection: call start(get_db, log_event) after app.py startup, same
pattern as stale_job_recovery and health modules.

Env vars:
  PAYOUT_INTERVAL_SECONDS   How often to check for pending payouts (default 120)
  PAYOUT_BATCH_SIZE          Max payouts to process per tick (default 20)
  PAYOUT_MIN_AMOUNT          Skip payouts below this amount (default 0.01)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PAYOUT_INTERVAL_SECONDS = int(os.getenv("PAYOUT_INTERVAL_SECONDS", "120"))
PAYOUT_BATCH_SIZE = int(os.getenv("PAYOUT_BATCH_SIZE", "20"))
PAYOUT_MIN_AMOUNT = float(os.getenv("PAYOUT_MIN_AMOUNT", "0.01"))

get_db: Callable
log_event: Callable

_started = False
_lock = threading.Lock()


def start(get_db_fn: Callable, log_event_fn: Callable) -> None:
    global get_db, log_event, _started
    with _lock:
        if _started:
            return
        get_db = get_db_fn
        log_event = log_event_fn
        t = threading.Thread(target=_run_loop, name="payout-worker", daemon=True)
        t.start()
        _started = True
        logger.info(
            "payout_worker: started (interval=%ss batch=%s min_amount=%s)",
            PAYOUT_INTERVAL_SECONDS, PAYOUT_BATCH_SIZE, PAYOUT_MIN_AMOUNT,
        )


def _run_loop() -> None:
    while True:
        try:
            _process_batch()
        except Exception as exc:
            logger.error("payout_worker: unhandled error: %s", exc, exc_info=True)
        time.sleep(PAYOUT_INTERVAL_SECONDS)


def _process_batch() -> int:
    """Process up to PAYOUT_BATCH_SIZE pending payouts. Returns count processed."""
    from server import chain  # lazy — avoids import order issues at startup

    db = get_db()
    rows = db.execute(
        """
        SELECT id, node_id, job_id, reward_amount
          FROM node_payouts
         WHERE status = 'pending' AND reward_amount >= ?
         ORDER BY created_at ASC
         LIMIT ?
        """,
        (PAYOUT_MIN_AMOUNT, PAYOUT_BATCH_SIZE),
    ).fetchall()

    if not rows:
        return 0

    on_chain = chain.is_connected() and bool(os.getenv("HAVNAI_PAYER_KEY", "").strip())
    processed = 0

    for row in rows:
        payout_id = row["id"]
        node_id = row["node_id"]
        job_id = row["job_id"]
        amount = float(row["reward_amount"])

        try:
            if on_chain:
                tx_hash = chain.send_hai(node_id, amount)
                if tx_hash:
                    _mark(db, payout_id, "confirmed", tx_hash)
                    log_event("Node payout confirmed", node_id=node_id, job_id=job_id,
                              amount=amount, tx_hash=tx_hash)
                else:
                    continue  # send_hai logged the error; retry next tick
            else:
                _mark(db, payout_id, "simulated", None)
                log_event("Node payout simulated", node_id=node_id, job_id=job_id, amount=amount)

            processed += 1
        except Exception as exc:
            logger.error("payout_worker: payout %s failed: %s", payout_id, exc)

    if processed:
        logger.info("payout_worker: processed %d payouts (on_chain=%s)", processed, on_chain)

    return processed


def _mark(db, payout_id: int, status: str, tx_hash: Optional[str]) -> None:
    db.execute(
        "UPDATE node_payouts SET status = ?, tx_hash = ?, updated_at = ? WHERE id = ?",
        (status, tx_hash, time.time(), payout_id),
    )
    db.commit()


def get_payout_stats() -> Dict[str, Any]:
    """Payout queue summary for health / admin endpoints."""
    from server import chain
    db = get_db()
    row = db.execute(
        """
        SELECT
            COUNT(CASE WHEN status = 'pending'   THEN 1 END) AS pending,
            COUNT(CASE WHEN status = 'confirmed' THEN 1 END) AS confirmed,
            COUNT(CASE WHEN status = 'simulated' THEN 1 END) AS simulated,
            COALESCE(SUM(CASE WHEN status = 'pending'   THEN reward_amount END), 0) AS pending_amount,
            COALESCE(SUM(CASE WHEN status = 'confirmed' THEN reward_amount END), 0) AS confirmed_amount
          FROM node_payouts
        """
    ).fetchone()
    return {
        "pending": row["pending"],
        "confirmed": row["confirmed"],
        "simulated": row["simulated"],
        "pending_amount": round(float(row["pending_amount"]), 6),
        "confirmed_amount": round(float(row["confirmed_amount"]), 6),
        "on_chain_active": chain.is_connected() and bool(os.getenv("HAVNAI_PAYER_KEY", "").strip()),
    }

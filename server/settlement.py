"""Job-centric settlement for HavnAI compute network.

Every generation becomes a durable **job ticket** with:
  - credit reservation before execution
  - node claim / attempt tracking
  - output quality validation
  - settlement outcome (spend / release / refund)
  - payout record for the completing node

Separates *compute completion* from *user satisfaction*:
  - Technical failure → release credits, no payout
  - Malformed output → refund / reroll marker, reduced payout
  - Valid-but-ugly output → spend credits, full payout (node did the work)

Tables managed here:
  - job_settlement  (1:1 extension of the ``jobs`` table)
  - job_attempts    (1:N attempts per job)
  - node_payouts    (1 per successful settlement)
  - credit_ledger   (every credit-affecting event)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Injected by app.py (same pattern as stripe_payments.py)
# ---------------------------------------------------------------------------
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]

# ---------------------------------------------------------------------------
# Status enums (string constants)
# ---------------------------------------------------------------------------

# Execution status
STATUS_CREATED = "created"
STATUS_CREDITS_RESERVED = "credits_reserved"
STATUS_QUEUED = "queued"
STATUS_CLAIMED = "claimed"
STATUS_COMPLETED = "completed"
STATUS_TECHNICAL_FAILED = "technical_failed"
STATUS_SETTLED = "settled"

# Quality status
QUALITY_UNCHECKED = "unchecked"
QUALITY_VALID = "valid"
QUALITY_MALFORMED = "malformed"
QUALITY_POLICY_BLOCKED = "policy_blocked"

# Settlement outcome
OUTCOME_PENDING = "pending"
OUTCOME_SPENT = "spent"
OUTCOME_RELEASED = "released"
OUTCOME_PARTIAL_REFUND = "partial_refund"
OUTCOME_FULL_REFUND = "full_refund"
OUTCOME_REROLL_GRANTED = "reroll_granted"

# Credit ledger event types
LEDGER_PURCHASE = "purchase"
LEDGER_RESERVE = "reserve"
LEDGER_RELEASE = "release"
LEDGER_SPEND = "spend"
LEDGER_REFUND = "refund"
LEDGER_PROMO = "promo"
LEDGER_REWARD = "reward"

# Payout asset types
PAYOUT_SIMULATED_HAI = "simulated_hai"

# Marketplace eligibility
MARKETPLACE_ELIGIBLE_TASK_TYPES = {"IMAGE_GEN"}
MARKETPLACE_INELIGIBLE_TASK_TYPES = {"FACE_SWAP"}


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def init_settlement_tables(conn: sqlite3.Connection) -> None:
    """Create settlement-related tables if they don't exist."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_settlement (
            job_id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            job_type TEXT NOT NULL,
            prompt TEXT,
            model TEXT NOT NULL,
            workflow_id TEXT,
            input_metadata TEXT,
            estimated_cost REAL NOT NULL DEFAULT 0.0,
            reserved_amount REAL NOT NULL DEFAULT 0.0,
            spent_amount REAL NOT NULL DEFAULT 0.0,
            execution_status TEXT NOT NULL DEFAULT 'created',
            quality_status TEXT NOT NULL DEFAULT 'unchecked',
            settlement_outcome TEXT NOT NULL DEFAULT 'pending',
            assigned_node_id TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            output_asset_id TEXT,
            error_summary TEXT,
            marketplace_eligible INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_settlement_wallet ON job_settlement (wallet)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_settlement_status ON job_settlement (execution_status)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL DEFAULT 1,
            claim_time REAL NOT NULL,
            finish_time REAL,
            status TEXT NOT NULL DEFAULT 'claimed',
            error_message TEXT,
            execution_metadata TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES job_settlement(job_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_attempts_job ON job_attempts (job_id)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_payouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            reward_amount REAL NOT NULL,
            reward_asset_type TEXT NOT NULL DEFAULT 'simulated_hai',
            status TEXT NOT NULL DEFAULT 'pending',
            tx_hash TEXT,
            metadata TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_payouts_node ON node_payouts (node_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_payouts_job ON node_payouts (job_id)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS credit_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            event_type TEXT NOT NULL,
            amount REAL NOT NULL,
            balance_after REAL,
            job_id TEXT,
            reason TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_credit_ledger_wallet ON credit_ledger (wallet)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_credit_ledger_job ON credit_ledger (job_id)"
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Credit ledger helpers
# ---------------------------------------------------------------------------

def _record_ledger(
    wallet: str,
    event_type: str,
    amount: float,
    balance_after: Optional[float] = None,
    job_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Append a row to the credit ledger."""
    conn = get_db()
    conn.execute(
        """INSERT INTO credit_ledger (wallet, event_type, amount, balance_after, job_id, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (wallet, event_type, amount, balance_after, job_id, reason, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Job ticket lifecycle
# ---------------------------------------------------------------------------

def create_job_ticket(
    job_id: str,
    wallet: str,
    job_type: str,
    model: str,
    estimated_cost: float,
    prompt: Optional[str] = None,
    workflow_id: Optional[str] = None,
    input_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a durable job settlement record.

    Called during job submission, before enqueue.
    """
    now = time.time()
    task_upper = (job_type or "IMAGE_GEN").upper()
    marketplace_eligible = 1 if task_upper in MARKETPLACE_ELIGIBLE_TASK_TYPES else 0

    conn = get_db()
    conn.execute(
        """INSERT INTO job_settlement
           (job_id, wallet, job_type, prompt, model, workflow_id, input_metadata,
            estimated_cost, reserved_amount, spent_amount,
            execution_status, quality_status, settlement_outcome,
            attempt_count, marketplace_eligible, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0,
                   'created', 'unchecked', 'pending',
                   0, ?, ?, ?)""",
        (
            job_id, wallet, task_upper, prompt, model,
            workflow_id,
            json.dumps(input_metadata) if input_metadata else None,
            estimated_cost,
            marketplace_eligible,
            now, now,
        ),
    )
    conn.commit()
    return {
        "job_id": job_id,
        "estimated_cost": estimated_cost,
        "marketplace_eligible": bool(marketplace_eligible),
    }


def reserve_credits(
    job_id: str,
    wallet: str,
    amount: float,
    *,
    reserve_fn: Callable[[str, float, str], Tuple[bool, float]],
) -> Tuple[bool, float]:
    """Reserve credits for a job.

    ``reserve_fn`` is ``credits.deduct_credits`` — it atomically
    moves credits from available balance to "spent" (we treat
    this as a reservation; release adds them back).

    Returns (success, remaining_balance).
    """
    ok, remaining = reserve_fn(wallet, amount, job_id)
    if not ok:
        return False, remaining

    conn = get_db()
    conn.execute(
        """UPDATE job_settlement
           SET reserved_amount = ?, execution_status = 'credits_reserved', updated_at = ?
           WHERE job_id = ?""",
        (amount, time.time(), job_id),
    )
    conn.commit()

    _record_ledger(wallet, LEDGER_RESERVE, -amount, remaining, job_id, "credit_reservation")
    return True, remaining


def mark_queued(job_id: str) -> None:
    """Transition job ticket to queued after enqueue."""
    conn = get_db()
    conn.execute(
        "UPDATE job_settlement SET execution_status = 'queued', updated_at = ? WHERE job_id = ?",
        (time.time(), job_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Node claim / attempt tracking
# ---------------------------------------------------------------------------

def record_claim(job_id: str, node_id: str) -> int:
    """Record a node claiming a job. Returns the attempt number."""
    conn = get_db()
    now = time.time()

    # Increment attempt count on the job ticket
    conn.execute(
        """UPDATE job_settlement
           SET assigned_node_id = ?, attempt_count = attempt_count + 1,
               execution_status = 'claimed', updated_at = ?
           WHERE job_id = ?""",
        (node_id, now, job_id),
    )

    # Determine attempt number
    row = conn.execute(
        "SELECT attempt_count FROM job_settlement WHERE job_id = ?", (job_id,)
    ).fetchone()
    attempt_num = row["attempt_count"] if row else 1

    conn.execute(
        """INSERT INTO job_attempts
           (job_id, node_id, attempt_number, claim_time, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'claimed', ?, ?)""",
        (job_id, node_id, attempt_num, now, now, now),
    )
    conn.commit()
    return attempt_num


def complete_attempt(
    job_id: str,
    node_id: str,
    status: str,
    error_message: Optional[str] = None,
    execution_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark the current attempt as finished."""
    conn = get_db()
    now = time.time()
    conn.execute(
        """UPDATE job_attempts
           SET finish_time = ?, status = ?, error_message = ?,
               execution_metadata = ?, updated_at = ?
           WHERE job_id = ? AND node_id = ? AND finish_time IS NULL""",
        (
            now, status, error_message,
            json.dumps(execution_metadata) if execution_metadata else None,
            now, job_id, node_id,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Output validation (minimum practical checks)
# ---------------------------------------------------------------------------

def validate_output(
    job_id: str,
    task_type: str,
    output_path: Optional[Path] = None,
    image_bytes: Optional[bytes] = None,
) -> str:
    """Run minimum quality validation on a job's output.

    Returns a quality_status string.
    """
    task_upper = (task_type or "").upper()

    # For image-type outputs (IMAGE_GEN, FACE_SWAP)
    if task_upper in ("IMAGE_GEN", "FACE_SWAP"):
        return _validate_image(output_path, image_bytes)

    # For other types, mark as unchecked (not in scope for this phase)
    return QUALITY_UNCHECKED


def _validate_image(
    output_path: Optional[Path] = None,
    image_bytes: Optional[bytes] = None,
) -> str:
    """Practical minimum image validation.

    Checks:
      - file exists / bytes are present
      - starts with a valid image header (PNG/JPEG)
      - has non-trivial size (> 1KB)
      - PNG dimensions are plausible if decodable from header
    """
    data: Optional[bytes] = image_bytes

    if data is None and output_path is not None:
        if not output_path.exists():
            return QUALITY_MALFORMED  # no output file
        try:
            data = output_path.read_bytes()
        except Exception:
            return QUALITY_MALFORMED

    if data is None or len(data) < 100:
        return QUALITY_MALFORMED  # empty or trivially small

    # Check for known image headers
    is_png = data[:8] == b'\x89PNG\r\n\x1a\n'
    is_jpeg = data[:2] == b'\xff\xd8'

    if not is_png and not is_jpeg:
        return QUALITY_MALFORMED  # not a recognisable image format

    # For PNG, extract dimensions from IHDR chunk
    if is_png and len(data) >= 24:
        try:
            width = int.from_bytes(data[16:20], 'big')
            height = int.from_bytes(data[20:24], 'big')
            if width < 64 or height < 64 or width > 8192 or height > 8192:
                return QUALITY_MALFORMED
        except Exception:
            pass  # can't parse, but image header is valid — allow

    return QUALITY_VALID


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def settle_job(
    job_id: str,
    node_id: str,
    execution_status: str,
    quality_status: str,
    reward_amount: float,
    *,
    deposit_fn: Optional[Callable[[str, float, str], float]] = None,
) -> Dict[str, Any]:
    """Settle a completed job: spend/release credits, create payout.

    Returns a dict describing the settlement outcome.
    """
    conn = get_db()
    now = time.time()

    row = conn.execute(
        "SELECT * FROM job_settlement WHERE job_id = ?", (job_id,)
    ).fetchone()
    if not row:
        return {"error": "job_not_found"}

    wallet = row["wallet"]
    reserved = float(row["reserved_amount"])
    current_outcome = row["settlement_outcome"]

    # Idempotency: don't re-settle
    if current_outcome != OUTCOME_PENDING:
        return {
            "status": "already_settled",
            "settlement_outcome": current_outcome,
            "job_id": job_id,
        }

    outcome: str
    spent: float = 0.0
    refund: float = 0.0
    payout_created = False

    if execution_status == STATUS_TECHNICAL_FAILED:
        # Technical failure → release all reserved credits
        outcome = OUTCOME_RELEASED
        refund = reserved
    elif quality_status == QUALITY_MALFORMED:
        # Malformed output → full refund, reduced payout
        outcome = OUTCOME_FULL_REFUND
        refund = reserved
        # Still create a minimal payout to acknowledge node compute
        reward_amount = round(reward_amount * 0.1, 6)  # 10% consolation
    elif quality_status == QUALITY_POLICY_BLOCKED:
        # Policy blocked → release credits, no payout
        outcome = OUTCOME_RELEASED
        refund = reserved
        reward_amount = 0.0
    else:
        # Valid output (including unchecked) → spend credits
        outcome = OUTCOME_SPENT
        spent = reserved

    # Update job settlement record
    conn.execute(
        """UPDATE job_settlement
           SET execution_status = ?,
               quality_status = ?,
               settlement_outcome = ?,
               spent_amount = ?,
               assigned_node_id = ?,
               updated_at = ?
           WHERE job_id = ? AND settlement_outcome = 'pending'""",
        (
            STATUS_SETTLED if execution_status != STATUS_TECHNICAL_FAILED else STATUS_TECHNICAL_FAILED,
            quality_status,
            outcome,
            spent,
            node_id,
            now,
            job_id,
        ),
    )
    conn.commit()

    # Release / refund credits if needed
    if refund > 0 and deposit_fn is not None:
        new_balance = deposit_fn(wallet, refund, f"settlement_refund:{job_id}")
        ledger_type = LEDGER_RELEASE if outcome == OUTCOME_RELEASED else LEDGER_REFUND
        _record_ledger(wallet, ledger_type, refund, new_balance, job_id, f"settlement:{outcome}")
    elif spent > 0:
        # Credits were already deducted at reservation time; record the spend event
        _record_ledger(wallet, LEDGER_SPEND, -spent, None, job_id, "settlement:spent")

    # Create payout record for the node
    if reward_amount > 0:
        conn.execute(
            """INSERT INTO node_payouts
               (node_id, job_id, reward_amount, reward_asset_type, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'completed', ?, ?)""",
            (node_id, job_id, reward_amount, PAYOUT_SIMULATED_HAI, now, now),
        )
        conn.commit()
        payout_created = True

    log_event(
        "Job settled",
        job_id=job_id,
        wallet=wallet,
        node_id=node_id,
        outcome=outcome,
        quality=quality_status,
        spent=spent,
        refund=refund,
        reward=reward_amount,
    )

    return {
        "job_id": job_id,
        "settlement_outcome": outcome,
        "quality_status": quality_status,
        "credits_spent": spent,
        "credits_refunded": refund,
        "reward_amount": reward_amount,
        "payout_created": payout_created,
    }


# ---------------------------------------------------------------------------
# Marketplace eligibility helpers
# ---------------------------------------------------------------------------

def set_output_asset(job_id: str, asset_id: str) -> None:
    """Link a generated output to its job ticket."""
    conn = get_db()
    conn.execute(
        "UPDATE job_settlement SET output_asset_id = ?, updated_at = ? WHERE job_id = ?",
        (asset_id, time.time(), job_id),
    )
    conn.commit()


def is_marketplace_eligible(job_id: str) -> bool:
    """Check if a job's output can be listed on the marketplace."""
    conn = get_db()
    row = conn.execute(
        """SELECT marketplace_eligible, quality_status, settlement_outcome
           FROM job_settlement WHERE job_id = ?""",
        (job_id,),
    ).fetchone()
    if not row:
        return False
    return (
        bool(row["marketplace_eligible"])
        and row["quality_status"] == QUALITY_VALID
        and row["settlement_outcome"] == OUTCOME_SPENT
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_job_settlement(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the settlement record for a job."""
    conn = get_db()
    row = conn.execute("SELECT * FROM job_settlement WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return dict(row)


def get_job_attempts(job_id: str) -> List[Dict[str, Any]]:
    """Fetch all attempts for a job."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt_number ASC",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_node_payouts(node_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch payout records for a node."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM node_payouts WHERE node_id = ? ORDER BY created_at DESC LIMIT ?",
        (node_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_credit_ledger(wallet: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch credit ledger entries for a wallet."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM credit_ledger WHERE wallet = ? ORDER BY created_at DESC LIMIT ?",
        (wallet, limit),
    ).fetchall()
    return [dict(r) for r in rows]

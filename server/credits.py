"""Credit system for HavnAI job payments."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
get_model_config: Callable[[str], Optional[Dict[str, Any]]]
CREDITS_ENABLED: bool = False

# Default credit costs per pipeline family.
# Override per-model via "credit_cost" in registry.json.
DEFAULT_CREDIT_COSTS: Dict[str, float] = {
    "sdxl": 1.0,
    "sd15": 0.5,
    "ltx2": 3.0,
    "animatediff": 2.0,
    "face_swap": 1.5,
}


def resolve_credit_cost(model_name: str, task_type: str = "") -> float:
    """Return the credit cost for a job.

    Priority: manifest credit_cost → pipeline default → 1.0 fallback.
    """
    cfg = get_model_config(model_name)
    if cfg:
        explicit = cfg.get("credit_cost")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        pipeline = str(cfg.get("pipeline", "")).lower()
        if pipeline in DEFAULT_CREDIT_COSTS:
            return DEFAULT_CREDIT_COSTS[pipeline]
    # Fall back on task_type for non-manifest models
    tt = (task_type or "").upper()
    if tt == "FACE_SWAP":
        return DEFAULT_CREDIT_COSTS.get("face_swap", 1.5)
    if tt in ("VIDEO_GEN", "ANIMATEDIFF"):
        return DEFAULT_CREDIT_COSTS.get("ltx2", 3.0)
    return 1.0


def get_credit_balance(wallet: str) -> float:
    """Return current credit balance for a wallet."""
    conn = get_db()
    row = conn.execute(
        "SELECT balance FROM credits WHERE wallet=?", (wallet,)
    ).fetchone()
    return float(row["balance"]) if row else 0.0


def deposit_credits(wallet: str, amount: float, reason: str = "") -> float:
    """Add credits to a wallet.  Returns new balance."""
    if amount <= 0:
        return get_credit_balance(wallet)
    conn = get_db()
    conn.execute(
        """
        INSERT INTO credits (wallet, balance, total_deposited, total_spent, updated_at)
        VALUES (?, ?, ?, 0.0, ?)
        ON CONFLICT(wallet) DO UPDATE SET
            balance = balance + excluded.balance,
            total_deposited = total_deposited + excluded.total_deposited,
            updated_at = excluded.updated_at
        """,
        (wallet, amount, amount, time.time()),
    )
    conn.commit()
    new_balance = get_credit_balance(wallet)
    log_event("Credits deposited", wallet=wallet, amount=amount, new_balance=new_balance, reason=reason)
    return new_balance


def deduct_credits(wallet: str, amount: float, job_id: str = "") -> Tuple[bool, float]:
    """Deduct credits from a wallet.  Returns (success, remaining_balance).

    Uses an atomic conditional UPDATE to prevent race conditions —
    the WHERE clause ensures balance never goes negative.
    """
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            """
            UPDATE credits
            SET balance = balance - ?,
                total_spent = total_spent + ?,
                updated_at = ?
            WHERE wallet = ? AND balance >= ?
            """,
            (amount, amount, time.time(), wallet, amount),
        )
        if cur.rowcount == 0:
            conn.rollback()
            # Either wallet doesn't exist or insufficient balance
            row = conn.execute("SELECT balance FROM credits WHERE wallet=?", (wallet,)).fetchone()
            current = float(row["balance"]) if row else 0.0
            return False, current
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    remaining = get_credit_balance(wallet)
    log_event("Credits deducted", wallet=wallet, amount=amount, remaining=remaining, job_id=job_id)
    return True, remaining


def check_and_deduct_credits(
    wallet: str, model_name: str, task_type: str = "", jsonify_func: Optional[Callable] = None
) -> Optional[Any]:
    """If credits are enabled, check balance and deduct.

    Returns a Flask error response if insufficient, or None if OK / credits disabled.
    """
    if not CREDITS_ENABLED:
        return None
    if jsonify_func is None:
        # Import here to avoid circular dependency
        from flask import jsonify as jsonify_func  # type: ignore[assignment]
    cost = resolve_credit_cost(model_name, task_type)
    balance = get_credit_balance(wallet)
    if balance < cost:
        return jsonify_func({
            "error": "insufficient_credits",
            "balance": balance,
            "cost": cost,
            "message": f"This job costs {cost} credits but you only have {balance}.",
        }), 402
    # Deduction happens at enqueue time (not completion) so the slot is reserved.
    ok, remaining = deduct_credits(wallet, cost)
    if not ok:
        return jsonify_func({
            "error": "insufficient_credits",
            "balance": remaining,
            "cost": cost,
        }), 402
    return None



def convert_credits_to_hai(wallet: str, amount: float) -> Tuple[bool, float]:
    """Convert credits to HAI tokens. This is a stub that deducts credits and records equivalent HAI reward.

    Returns a tuple of (success, remaining_credits).
    """
    # Deduct credits from the wallet
    ok, remaining = deduct_credits(wallet, amount)
    if not ok:
        return False, remaining
    # TODO: Integrate with blockchain or reward system to mint/transfer HAI tokens
    # For now, simply log the event
    log_event("Credits converted to HAI", wallet=wallet, amount=amount, remaining_credits=remaining)
    return True, remaining

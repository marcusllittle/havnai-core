"""On-chain integration foundation for HavnAI.

Provides database tables for on-chain state tracking and stub endpoints
for future smart-contract interaction (wallet verification, reward claims).
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
WALLET_REGEX: Any  # re.Pattern


def init_blockchain_tables(conn: sqlite3.Connection) -> None:
    """Create on-chain integration tables if they don't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            creator_wallet TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            config TEXT DEFAULT '{}',
            published INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reward_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            amount REAL NOT NULL,
            tx_hash TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def verify_wallet(wallet: str) -> Dict[str, Any]:
    """Placeholder for signature verification.

    In a future release this will verify an EIP-712 or personal_sign
    signature against the provided wallet address.
    """
    return {
        "wallet": wallet,
        "verified": False,
        "message": "Signature verification not yet implemented. This is a placeholder endpoint.",
    }


def get_claimable_rewards(wallet: str) -> Dict[str, Any]:
    """Return unclaimed reward total for a wallet.

    Sums all rewards that have not yet been associated with a claim.
    """
    conn = get_db()
    # Total earned
    row = conn.execute(
        "SELECT COALESCE(SUM(reward_hai), 0) AS total FROM rewards WHERE wallet = ?",
        (wallet,),
    ).fetchone()
    total_earned = float(row["total"]) if row else 0.0

    # Total already claimed
    claimed_row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM reward_claims WHERE wallet = ? AND status IN ('pending', 'confirmed')",
        (wallet,),
    ).fetchone()
    total_claimed = float(claimed_row["total"]) if claimed_row else 0.0

    claimable = max(0.0, round(total_earned - total_claimed, 6))
    return {
        "wallet": wallet,
        "total_earned": round(total_earned, 6),
        "total_claimed": round(total_claimed, 6),
        "claimable": claimable,
    }


def claim_rewards(wallet: str, amount: Optional[float] = None) -> Dict[str, Any]:
    """Mark rewards as claimed (DB-only for now, no on-chain tx).

    If amount is None, claims the full claimable balance.
    Returns the created claim record.
    """
    info = get_claimable_rewards(wallet)
    claimable = info["claimable"]

    if amount is None:
        amount = claimable
    if amount <= 0:
        return {"error": "nothing_to_claim", "claimable": claimable}
    if amount > claimable:
        return {"error": "insufficient_claimable", "requested": amount, "claimable": claimable}

    conn = get_db()
    now = time.time()
    conn.execute(
        """
        INSERT INTO reward_claims (wallet, amount, tx_hash, status, created_at)
        VALUES (?, ?, NULL, 'pending', ?)
        """,
        (wallet, amount, now),
    )
    conn.commit()
    log_event("Reward claim created", wallet=wallet, amount=amount)

    return {
        "wallet": wallet,
        "claimed": round(amount, 6),
        "status": "pending",
        "message": "Claim recorded. On-chain settlement will be available in a future release.",
    }


def get_claim_history(wallet: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent reward claim records for a wallet."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, amount, tx_hash, status, created_at
        FROM reward_claims
        WHERE wallet = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (wallet, limit),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "amount": float(row["amount"]),
            "tx_hash": row["tx_hash"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]

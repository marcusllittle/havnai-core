"""Validator quorum system for HavnAI job verification.

Provides M-of-N validation quorum where multiple validators independently
verify job results.  Validator selection is weighted random based on
reputation scores.
"""

from __future__ import annotations

import random
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
NODES: Dict[str, Dict[str, Any]]

# Quorum configuration (M-of-N)
QUORUM_M: int = 2   # minimum agreements needed
QUORUM_N: int = 3   # total validators selected per job


def init_validator_tables(conn: sqlite3.Connection) -> None:
    """Create validator-related tables if they don't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validators (
            node_id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            reputation_score REAL NOT NULL DEFAULT 1.0,
            total_validations INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            validator_node_id TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp REAL NOT NULL,
            UNIQUE(job_id, validator_node_id)
        )
        """
    )
    conn.commit()


def register_validator(node_id: str, wallet: str) -> Dict[str, Any]:
    """Register or update a node as a validator."""
    conn = get_db()
    conn.execute(
        """
        INSERT INTO validators (node_id, wallet, reputation_score, total_validations, active)
        VALUES (?, ?, 1.0, 0, 1)
        ON CONFLICT(node_id) DO UPDATE SET
            wallet = excluded.wallet,
            active = 1
        """,
        (node_id, wallet),
    )
    conn.commit()
    log_event("Validator registered", node_id=node_id, wallet=wallet)
    return {"node_id": node_id, "wallet": wallet, "status": "registered"}


def deactivate_validator(node_id: str) -> Dict[str, Any]:
    """Mark a validator as inactive."""
    conn = get_db()
    conn.execute("UPDATE validators SET active = 0 WHERE node_id = ?", (node_id,))
    conn.commit()
    return {"node_id": node_id, "status": "deactivated"}


def get_active_validators() -> List[Dict[str, Any]]:
    """Return all active validators with their reputation scores."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT node_id, wallet, reputation_score, total_validations, active
        FROM validators
        WHERE active = 1
        ORDER BY reputation_score DESC
        """
    ).fetchall()
    return [
        {
            "node_id": row["node_id"],
            "wallet": row["wallet"],
            "reputation_score": float(row["reputation_score"]),
            "total_validations": row["total_validations"],
        }
        for row in rows
    ]


def select_validators(job_id: str, n: Optional[int] = None) -> List[str]:
    """Select N validators using weighted random selection based on reputation.

    Args:
        job_id: The job to select validators for.
        n: Number of validators to select (defaults to QUORUM_N).

    Returns:
        List of selected validator node_ids.
    """
    if n is None:
        n = QUORUM_N
    active = get_active_validators()
    if len(active) <= n:
        return [v["node_id"] for v in active]

    # Weighted random selection based on reputation score
    weights = [max(0.1, v["reputation_score"]) for v in active]
    selected = random.choices(active, weights=weights, k=n)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for v in selected:
        if v["node_id"] not in seen:
            seen.add(v["node_id"])
            unique.append(v["node_id"])

    # If we got fewer than n due to dedup, fill from remaining
    if len(unique) < n:
        remaining = [v for v in active if v["node_id"] not in seen]
        for v in remaining:
            unique.append(v["node_id"])
            if len(unique) >= n:
                break

    return unique[:n]


def submit_validation(
    job_id: str,
    validator_node_id: str,
    result: str,
) -> Dict[str, Any]:
    """Submit a validation result for a job.

    Args:
        job_id: The job being validated.
        validator_node_id: The validator submitting the result.
        result: "approve" or "reject".

    Returns:
        Dict with the validation status and quorum info.
    """
    conn = get_db()
    now = time.time()

    # Record the validation
    try:
        conn.execute(
            """
            INSERT INTO validations (job_id, validator_node_id, result, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id, validator_node_id) DO UPDATE SET
                result = excluded.result,
                timestamp = excluded.timestamp
            """,
            (job_id, validator_node_id, result, now),
        )
        # Update validator stats
        conn.execute(
            """
            UPDATE validators
            SET total_validations = total_validations + 1
            WHERE node_id = ?
            """,
            (validator_node_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    log_event(
        "Validation submitted",
        job_id=job_id,
        validator=validator_node_id,
        result=result,
    )

    # Check quorum
    quorum_result = check_quorum(job_id)
    return {
        "job_id": job_id,
        "validator": validator_node_id,
        "result": result,
        "quorum": quorum_result,
    }


def check_quorum(job_id: str, m: Optional[int] = None) -> Dict[str, Any]:
    """Check if a job has reached quorum.

    Args:
        job_id: The job to check.
        m: Minimum agreements needed (defaults to QUORUM_M).

    Returns:
        Dict with quorum status information.
    """
    if m is None:
        m = QUORUM_M
    conn = get_db()
    rows = conn.execute(
        "SELECT validator_node_id, result FROM validations WHERE job_id = ?",
        (job_id,),
    ).fetchall()

    total = len(rows)
    approvals = sum(1 for r in rows if r["result"] == "approve")
    rejections = sum(1 for r in rows if r["result"] == "reject")

    reached = False
    outcome = "pending"
    if approvals >= m:
        reached = True
        outcome = "approved"
    elif rejections >= m:
        reached = True
        outcome = "rejected"

    return {
        "job_id": job_id,
        "total_validations": total,
        "approvals": approvals,
        "rejections": rejections,
        "quorum_m": m,
        "quorum_n": QUORUM_N,
        "reached": reached,
        "outcome": outcome,
    }


def update_reputation(validator_node_id: str, delta: float) -> None:
    """Adjust a validator's reputation score.

    Reputation is clamped to [0.0, 10.0].
    """
    conn = get_db()
    conn.execute(
        """
        UPDATE validators
        SET reputation_score = MAX(0.0, MIN(10.0, reputation_score + ?))
        WHERE node_id = ?
        """,
        (delta, validator_node_id),
    )
    conn.commit()


def get_validation_history(job_id: str) -> List[Dict[str, Any]]:
    """Return all validations for a given job."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT v.id, v.validator_node_id, v.result, v.timestamp,
               val.wallet, val.reputation_score
        FROM validations v
        LEFT JOIN validators val ON val.node_id = v.validator_node_id
        WHERE v.job_id = ?
        ORDER BY v.timestamp ASC
        """,
        (job_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "validator_node_id": row["validator_node_id"],
            "wallet": row["wallet"],
            "reputation_score": float(row["reputation_score"]) if row["reputation_score"] else None,
            "result": row["result"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]

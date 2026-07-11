"""Job queue management for HavnAI coordinator."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
get_model_config: Callable[[str], Optional[Dict[str, Any]]]
get_dispatch_decision: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None
NODES: Dict[str, Dict[str, Any]]
CREATOR_TASK_TYPE: str = "IMAGE_GEN"


def _image_job_requires_reference_face(raw_data: Any) -> bool:
    if not isinstance(raw_data, str) or not raw_data.strip():
        return False
    try:
        parsed = json.loads(raw_data)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    value = parsed.get("reference_face_url")
    return isinstance(value, str) and bool(value.strip())


def enqueue_job(
    wallet: str,
    model: str,
    task_type: str,
    data: str,
    weight: float,
    invite_code: Optional[str] = None,
) -> str:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    task_type = (task_type or CREATOR_TASK_TYPE).upper()
    conn = get_db()
    conn.execute(
        """
        INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, invite_code)
        VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?, ?)
        """,
        (job_id, wallet, model, data, task_type, float(weight), time.time(), invite_code),
    )
    conn.commit()
    return job_id


def _eligible_job_for_node(
    node_id: str,
    rows: Any,
    *,
    enforce_preference: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return the oldest job compatible with a node from a row iterable."""
    node = NODES.get(node_id, {})
    role = node.get("role", "worker")
    node_supports = {s.lower() for s in node.get("supports", []) if isinstance(s, str)}
    # Legacy nodes do not advertise supports; treat as image-only.
    if not node_supports:
        node_supports = {"image"}
    support_map = {
        CREATOR_TASK_TYPE: "image",
        "VIDEO_GEN": "video",
        "ANIMATEDIFF": "animatediff",
        "FACE_SWAP": "face_swap",
        "LTX_VIDEO_GEN": "ltx_video",
    }
    for row in rows:
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        # Support standard IMAGE_GEN, LTX2 video jobs, LTX-Video 2.3, AnimateDiff, and face swap.
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF", "FACE_SWAP", "LTX_VIDEO_GEN"}:
            continue
        if role != "creator":
            continue
        required_support = support_map.get(task_type, "image")
        if task_type == CREATOR_TASK_TYPE and _image_job_requires_reference_face(row["data"]):
            required_support = "face_swap"
        if node_supports and required_support not in node_supports:
            continue
        model_name = row["model"].lower()
        cfg = get_model_config(model_name)
        if not cfg:
            continue
        node_models = {m.lower() for m in node.get("models", []) if isinstance(m, str)}
        if node_models and model_name not in node_models:
            continue
        required_pipeline = (cfg.get("pipeline") or "sd15").lower()
        node_pipelines = {p.lower() for p in node.get("pipelines", []) if isinstance(p, str)}
        if node_pipelines and required_pipeline not in node_pipelines:
            continue
        job = dict(row)
        if enforce_preference and callable(get_dispatch_decision):
            decision = get_dispatch_decision(node_id, job)
            if not decision.get("allowed", False):
                continue
            job["dispatch_decision"] = decision
        return job
    return None


def fetch_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Inspect the queue without claiming a job.

    New dispatch code should use :func:`claim_next_job_for_node` so selection
    and assignment happen atomically. This function remains for callers that
    only need queue visibility.
    """
    conn = get_db()
    rows = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC").fetchall()
    return _eligible_job_for_node(node_id, rows)


def claim_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Atomically select and claim the oldest compatible queued job.

    ``BEGIN IMMEDIATE`` serializes writers at the database boundary, which
    keeps two coordinator processes from dispatching the same job. The
    conditional UPDATE is an additional guard for databases with different
    transaction semantics.
    """
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC"
        ).fetchall()
        job = _eligible_job_for_node(node_id, rows, enforce_preference=True)
        if not job:
            conn.rollback()
            return None

        assigned_at = time.time()
        lease_seconds = max(30, int(getattr(claim_next_job_for_node, "lease_seconds", 1800)))
        lease_expires_at = assigned_at + lease_seconds
        dispatch_decision = job.pop("dispatch_decision", {})
        cursor = conn.execute(
            """
            UPDATE jobs
            SET status='running', node_id=?, assigned_at=?,
                lease_renewed_at=?, lease_expires_at=?, last_failure_reason=NULL
                , preferred_node_id=?, dispatch_score=?, dispatch_reason=?
            WHERE id=? AND status='queued'
            """,
            (
                node_id,
                assigned_at,
                assigned_at,
                lease_expires_at,
                dispatch_decision.get("preferred_node_id"),
                dispatch_decision.get("score"),
                dispatch_decision.get("reason"),
                job["id"],
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
        job.update({
            "status": "running",
            "node_id": node_id,
            "assigned_at": assigned_at,
            "lease_renewed_at": assigned_at,
            "lease_expires_at": lease_expires_at,
            "preferred_node_id": dispatch_decision.get("preferred_node_id"),
            "dispatch_score": dispatch_decision.get("score"),
            "dispatch_reason": dispatch_decision.get("reason"),
        })
        return job
    except Exception:
        conn.rollback()
        raise


def renew_job_leases(node_id: str, lease_seconds: int = 1800, job_id: Optional[str] = None) -> int:
    """Extend active claims owned by a node and return the renewed count."""
    conn = get_db()
    now = time.time()
    expires_at = now + max(30, int(lease_seconds))
    sql = """
        UPDATE jobs
        SET lease_renewed_at=?, lease_expires_at=?
        WHERE node_id=? AND status='running'
    """
    params: list[Any] = [now, expires_at, node_id]
    if job_id:
        sql += " AND id=?"
        params.append(job_id)
    cursor = conn.execute(sql, params)
    conn.commit()
    return int(cursor.rowcount)


def recover_expired_leases(now: Optional[float] = None, max_retries: int = 3) -> list[Dict[str, Any]]:
    """Requeue expired claims, failing jobs that exhausted their retry budget."""
    conn = get_db()
    current_time = float(now if now is not None else time.time())
    retry_limit = max(0, int(max_retries))
    recovered: list[Dict[str, Any]] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status='running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= ?
            ORDER BY lease_expires_at ASC
            """,
            (current_time,),
        ).fetchall()
        for row in rows:
            job = dict(row)
            retry_count = int(job.get("retry_count") or 0) + 1
            exhausted = retry_count > retry_limit
            next_status = "failed" if exhausted else "queued"
            conn.execute(
                """
                UPDATE jobs
                SET status=?, node_id=NULL, assigned_at=NULL,
                    lease_renewed_at=NULL, lease_expires_at=NULL,
                    retry_count=?, last_failure_reason='lease_expired',
                    completed_at=?
                WHERE id=? AND status='running'
                """,
                (next_status, retry_count, current_time if exhausted else None, job["id"]),
            )
            recovered.append({
                "job_id": job["id"],
                "wallet": job.get("wallet"),
                "model": job.get("model"),
                "task_type": job.get("task_type"),
                "previous_node_id": job.get("node_id"),
                "retry_count": retry_count,
                "status": next_status,
            })
        conn.commit()
        return recovered
    except Exception:
        conn.rollback()
        raise


def assign_job_to_node(job_id: str, node_id: str) -> None:
    conn = get_db()
    conn.execute("UPDATE jobs SET status='running', node_id=?, assigned_at=? WHERE id=?", (node_id, time.time(), job_id))
    conn.commit()


def complete_job(job_id: str, node_id: str, status: str) -> bool:
    """Mark a job complete only if it is still running for the given node."""
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, node_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        current_status = (row["status"] or "").lower() if row else ""
        owner = row["node_id"] if row else None
        if not row or current_status != "running" or (owner and owner != node_id):
            conn.rollback()
            return False
        conn.execute(
            "UPDATE jobs SET status=?, node_id=?, completed_at=? WHERE id=?",
            (status, node_id, time.time(), job_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def complete_job_if_queued(job_id: str, node_id: str, status: str) -> bool:
    """Allow late completion when a running job was reset to queued (e.g., server restart)."""
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, node_id, assigned_at, completed_at FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if not row:
            conn.rollback()
            return False
        current_status = (row["status"] or "").lower()
        owner = row["node_id"]
        assigned_at = row["assigned_at"]
        completed_at = row["completed_at"]
        if current_status != "queued" or completed_at is not None:
            conn.rollback()
            return False
        if owner and owner != node_id:
            conn.rollback()
            return False
        if assigned_at is None:
            conn.rollback()
            return False
        conn.execute(
            "UPDATE jobs SET status=?, node_id=?, completed_at=? WHERE id=?",
            (status, node_id, time.time(), job_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise

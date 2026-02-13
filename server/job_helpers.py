"""Job queue management for HavnAI coordinator."""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
get_model_config: Callable[[str], Optional[Dict[str, Any]]]
NODES: Dict[str, Dict[str, Any]]
CREATOR_TASK_TYPE: str = "IMAGE_GEN"


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


def fetch_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC").fetchall()
    node = NODES.get(node_id, {})
    role = node.get("role", "worker")
    node_supports = {s.lower() for s in node.get("supports", []) if isinstance(s, str)}
    for row in rows:
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        # Support standard IMAGE_GEN, LTX2 video jobs, AnimateDiff video jobs, and face swap.
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF", "FACE_SWAP"}:
            continue
        if role != "creator":
            continue
        if task_type == "ANIMATEDIFF":
            required_support = "animatediff"
        elif task_type == "VIDEO_GEN":
            required_support = "video"
        else:
            required_support = "image"
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
        return dict(row)
    return None


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

"""Job queue management for HavnAI coordinator."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)

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
    # Legacy nodes do not advertise supports; treat as image-only.
    if not node_supports:
        node_supports = {"image"}
    support_map = {
        CREATOR_TASK_TYPE: "image",
        "VIDEO_GEN": "video",
        "ANIMATEDIFF": "animatediff",
        "FACE_SWAP": "face_swap",
    }
    for row in rows:
        job_id_short = (row["id"] or "")[:12]
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        # Support standard IMAGE_GEN, LTX2 video jobs, AnimateDiff video jobs, and face swap.
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF", "FACE_SWAP"}:
            continue
        if role != "creator":
            logger.debug("Job %s skipped: node %s role is '%s', not 'creator'", job_id_short, node_id, role)
            continue
        required_support = support_map.get(task_type, "image")
        if node_supports and required_support not in node_supports:
            logger.debug(
                "Job %s skipped: node %s missing '%s' support (has: %s)",
                job_id_short, node_id, required_support, node_supports,
            )
            continue
        model_name = row["model"].lower()
        cfg = get_model_config(model_name)
        if not cfg:
            logger.warning("Job %s skipped: model '%s' not found in manifest", job_id_short, model_name)
            continue
        node_models = {m.lower() for m in node.get("models", []) if isinstance(m, str)}
        if node_models and model_name not in node_models:
            logger.debug(
                "Job %s skipped: model '%s' not in node %s models (%s)",
                job_id_short, model_name, node_id, node_models,
            )
            continue
        required_pipeline = (cfg.get("pipeline") or "sd15").lower()
        node_pipelines = {p.lower() for p in node.get("pipelines", []) if isinstance(p, str)}
        if node_pipelines and required_pipeline not in node_pipelines:
            logger.debug(
                "Job %s skipped: pipeline '%s' not in node %s pipelines (%s)",
                job_id_short, required_pipeline, node_id, node_pipelines,
            )
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


# ---------------------------------------------------------------------------
# Stale job reaper helpers
# ---------------------------------------------------------------------------

def cancel_job(job_id: str) -> bool:
    """Cancel a queued or running job.  Returns True if the job was cancelled."""
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            conn.rollback()
            return False
        status = (row["status"] or "").lower()
        if status not in ("queued", "running"):
            conn.rollback()
            return False
        conn.execute(
            "UPDATE jobs SET status='cancelled', status_reason='cancelled', completed_at=? WHERE id=?",
            (time.time(), job_id),
        )
        conn.commit()
        logger.info("Job %s cancelled", job_id[:12])
        return True
    except Exception:
        conn.rollback()
        raise


def reset_stuck_running_jobs(
    now: float,
    image_timeout: int,
    video_timeout: int,
    max_retries: int,
    is_node_online_fn: Any,
) -> List[Dict[str, Any]]:
    """Find running jobs whose assigned node is offline and reset them.

    Returns a list of dicts with ``job_id``, ``action`` ('requeued' or 'failed'),
    ``wallet``, and ``reason`` for each affected job.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT id, task_type, node_id, assigned_at, wallet, retry_count FROM jobs WHERE status='running'"
    ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        job_id = row["id"]
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        node_id = row["node_id"]
        assigned_at = row["assigned_at"] or 0.0
        retry_count = row["retry_count"] or 0

        timeout = video_timeout if task_type in ("VIDEO_GEN", "ANIMATEDIFF") else image_timeout
        elapsed = now - assigned_at
        if elapsed < timeout:
            continue

        # Only reap if the assigned node is offline
        node = NODES.get(node_id, {})
        if node and is_node_online_fn(node, now=now):
            continue

        if retry_count >= max_retries:
            conn.execute(
                "UPDATE jobs SET status='failed', status_reason='max_retries_exceeded', completed_at=? WHERE id=?",
                (now, job_id),
            )
            conn.commit()
            logger.warning("Job %s failed: max retries exceeded (%d)", job_id[:12], retry_count)
            results.append({
                "job_id": job_id, "action": "failed",
                "wallet": row["wallet"], "reason": "max_retries_exceeded",
            })
        else:
            conn.execute(
                "UPDATE jobs SET status='queued', node_id=NULL, assigned_at=NULL, "
                "retry_count=?, status_reason='node_offline' WHERE id=?",
                (retry_count + 1, job_id),
            )
            conn.commit()
            logger.info(
                "Job %s requeued (retry %d): node %s offline after %ds",
                job_id[:12], retry_count + 1, (node_id or "?")[:12], int(elapsed),
            )
            results.append({
                "job_id": job_id, "action": "requeued",
                "wallet": row["wallet"], "reason": "node_offline",
            })
    return results


def timeout_stale_queued_jobs(now: float, queued_timeout: int) -> List[Dict[str, Any]]:
    """Fail queued jobs that have been waiting longer than *queued_timeout* seconds.

    Returns a list of dicts with ``job_id``, ``wallet``, and ``reason``.
    """
    cutoff = now - queued_timeout
    conn = get_db()
    rows = conn.execute(
        "SELECT id, wallet FROM jobs WHERE status='queued' AND timestamp < ?",
        (cutoff,),
    ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        job_id = row["id"]
        conn.execute(
            "UPDATE jobs SET status='failed', status_reason='timed_out', completed_at=? WHERE id=?",
            (now, job_id),
        )
        conn.commit()
        logger.info("Job %s timed out after %ds in queue", job_id[:12], queued_timeout)
        results.append({"job_id": job_id, "wallet": row["wallet"], "reason": "timed_out"})
    return results

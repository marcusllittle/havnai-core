"""Job queue management for HavnAI coordinator."""

from __future__ import annotations

import json
import os
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
LEASE_SECONDS = max(30, int(os.getenv("HAVNAI_JOB_LEASE_SECONDS", "90")))


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
        INSERT INTO jobs (
            id, wallet, model, data, task_type, weight, status, node_id,
            timestamp, invite_code, progress, stage, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?, ?, 0, 'queued', ?)
        """,
        (job_id, wallet, model, data, task_type, float(weight), time.time(), invite_code, time.time()),
    )
    conn.commit()
    return job_id


def fetch_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    recover_expired_leases(conn=conn)
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
        "LTX_VIDEO_GEN": "ltx_video",
    }
    for row in rows:
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        # Support image, legacy video, verified LTX-Video, AnimateDiff, and face swap jobs.
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
        return dict(row)
    return None


def assign_job_to_node(job_id: str, node_id: str) -> str:
    """Atomically lease a queued job and return its unique attempt id."""
    conn = get_db()
    now = time.time()
    attempt_id = f"attempt-{uuid.uuid4().hex}"
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE jobs
               SET status='leased', node_id=?, assigned_at=?, attempt_id=?,
                   lease_expires_at=?, progress=0, stage='leased', updated_at=?
             WHERE id=? AND status='queued'
            """,
            (node_id, now, attempt_id, now + LEASE_SECONDS, now, job_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise RuntimeError("job_claim_conflict")
        conn.commit()
        return attempt_id
    except Exception:
        conn.rollback()
        raise


def recover_expired_leases(*, conn: Optional[sqlite3.Connection] = None) -> int:
    """Return abandoned attempts to the queue without touching active leases."""
    db = conn or get_db()
    now = time.time()
    try:
        cursor = db.execute(
            """
            UPDATE jobs
               SET status='queued', node_id=NULL, attempt_id=NULL, assigned_at=NULL,
                   lease_expires_at=NULL, progress=0, stage='queued', updated_at=?
             WHERE status IN ('leased', 'running', 'uploading')
               AND lease_expires_at IS NOT NULL
               AND lease_expires_at < ?
            """,
            (now, now),
        )
    except sqlite3.OperationalError as exc:
        # Compatibility for isolated legacy tests and one-release old databases.
        if "no such column" not in str(exc).lower():
            raise
        return 0
    db.commit()
    return int(cursor.rowcount)


def renew_lease(job_id: str, node_id: str, attempt_id: str) -> bool:
    conn = get_db()
    now = time.time()
    cursor = conn.execute(
        """
        UPDATE jobs
           SET lease_expires_at=?, updated_at=?
         WHERE id=? AND node_id=? AND attempt_id=?
           AND status IN ('leased', 'running', 'uploading')
        """,
        (now + LEASE_SECONDS, now, job_id, node_id, attempt_id),
    )
    conn.commit()
    return cursor.rowcount == 1


def update_job_progress(
    job_id: str,
    node_id: str,
    attempt_id: str,
    *,
    progress: float,
    stage: str,
) -> bool:
    conn = get_db()
    now = time.time()
    normalized_progress = max(0.0, min(100.0, float(progress)))
    normalized_stage = str(stage or "running").strip().lower()[:64] or "running"
    target_status = "uploading" if normalized_stage == "uploading" else "running"
    cursor = conn.execute(
        """
        UPDATE jobs
           SET status=?, progress=?, stage=?, lease_expires_at=?, updated_at=?
         WHERE id=? AND node_id=? AND attempt_id=?
           AND status IN ('leased', 'running', 'uploading')
        """,
        (
            target_status,
            normalized_progress,
            normalized_stage,
            now + LEASE_SECONDS,
            now,
            job_id,
            node_id,
            attempt_id,
        ),
    )
    conn.commit()
    return cursor.rowcount == 1


def control_state(job_id: str, node_id: str, attempt_id: str) -> Optional[str]:
    row = get_db().execute(
        "SELECT status FROM jobs WHERE id=? AND node_id=? AND attempt_id=?",
        (job_id, node_id, attempt_id),
    ).fetchone()
    return str(row["status"] or "") if row else None


def complete_job(
    job_id: str,
    node_id: str,
    status: str,
    attempt_id: Optional[str] = None,
) -> bool:
    """Mark a job complete only if it is still running for the given node."""
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, node_id, attempt_id FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        current_status = (row["status"] or "").lower() if row else ""
        owner = row["node_id"] if row else None
        active_attempt = row["attempt_id"] if row else None
        requested_status = str(status).lower()
        allowed_current = {"leased", "running", "uploading"}
        if requested_status in {"cancelled", "canceled"}:
            allowed_current.add("cancelling")
        if (
            not row
            or current_status not in allowed_current
            or (owner and owner != node_id)
            or (attempt_id and active_attempt != attempt_id)
        ):
            conn.rollback()
            return False
        canonical_status = "succeeded" if requested_status in {"success", "completed", "done"} else requested_status
        if canonical_status == "canceled":
            canonical_status = "cancelled"
        if canonical_status not in {"succeeded", "failed", "cancelled", "expired"}:
            canonical_status = "failed"
        conn.execute(
            """
            UPDATE jobs
               SET status=?, node_id=?, completed_at=?, lease_expires_at=NULL,
                   progress=100, stage=?, updated_at=?
             WHERE id=? AND (? IS NULL OR attempt_id=?)
            """,
            (
                canonical_status,
                node_id,
                time.time(),
                canonical_status,
                time.time(),
                job_id,
                attempt_id,
                attempt_id,
            ),
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

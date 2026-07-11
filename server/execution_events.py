"""Durable, ordered execution timeline for HavnAI jobs."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional


get_db: Callable[[], sqlite3.Connection]


def init_execution_event_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_execution_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            node_id TEXT,
            attempt_number INTEGER,
            message TEXT,
            metadata TEXT,
            stage_latency_ms INTEGER NOT NULL DEFAULT 0,
            total_elapsed_ms INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            UNIQUE(job_id, sequence)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_execution_events_job ON job_execution_events (job_id, sequence)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_execution_events_stage ON job_execution_events (stage, created_at)"
    )
    conn.commit()


def record_event(
    job_id: str,
    stage: str,
    status: str,
    *,
    node_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    created_at: Optional[float] = None,
    dedupe_window_seconds: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Append an event, returning ``None`` when a duplicate is suppressed."""
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise ValueError("job_id is required")
    normalized_stage = str(stage or "UNKNOWN").strip().upper()
    normalized_status = str(status or normalized_stage).strip().upper()
    now = float(created_at if created_at is not None else time.time())
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            """
            SELECT sequence, stage, node_id, created_at
            FROM job_execution_events
            WHERE job_id=?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (normalized_job_id,),
        ).fetchone()
        if (
            previous
            and dedupe_window_seconds > 0
            and str(previous["stage"]) == normalized_stage
            and str(previous["node_id"] or "") == str(node_id or "")
            and (now - float(previous["created_at"])) < dedupe_window_seconds
        ):
            conn.rollback()
            return None

        first = conn.execute(
            "SELECT created_at FROM job_execution_events WHERE job_id=? ORDER BY sequence ASC LIMIT 1",
            (normalized_job_id,),
        ).fetchone()
        sequence = int(previous["sequence"] or 0) + 1 if previous else 1
        previous_at = float(previous["created_at"]) if previous else now
        first_at = float(first["created_at"]) if first else now
        stage_latency_ms = max(0, int(round((now - previous_at) * 1000)))
        total_elapsed_ms = max(0, int(round((now - first_at) * 1000)))
        cursor = conn.execute(
            """
            INSERT INTO job_execution_events (
                job_id, sequence, stage, status, node_id, attempt_number,
                message, metadata, stage_latency_ms, total_elapsed_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_job_id,
                sequence,
                normalized_stage,
                normalized_status,
                node_id,
                attempt_number,
                message,
                json.dumps(metadata, sort_keys=True) if metadata else None,
                stage_latency_ms,
                total_elapsed_ms,
                now,
            ),
        )
        conn.commit()
        return {
            "id": int(cursor.lastrowid),
            "job_id": normalized_job_id,
            "sequence": sequence,
            "stage": normalized_stage,
            "status": normalized_status,
            "node_id": node_id,
            "attempt_number": attempt_number,
            "message": message,
            "metadata": metadata or {},
            "stage_latency_ms": stage_latency_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "created_at": now,
        }
    except Exception:
        conn.rollback()
        raise


def get_timeline(job_id: str) -> Dict[str, Any]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM job_execution_events WHERE job_id=? ORDER BY sequence ASC",
        (job_id,),
    ).fetchall()
    events: List[Dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        try:
            event["metadata"] = json.loads(event.get("metadata") or "{}")
        except Exception:
            event["metadata"] = {}
        events.append(event)
    return {
        "job_id": job_id,
        "events": events,
        "event_count": len(events),
        "current_stage": events[-1]["stage"] if events else None,
        "current_status": events[-1]["status"] if events else None,
        "total_elapsed_ms": events[-1]["total_elapsed_ms"] if events else 0,
    }

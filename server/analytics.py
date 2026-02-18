"""Analytics endpoints for the HavnAI dashboard."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
NODES: Dict[str, Dict[str, Any]]
ONLINE_THRESHOLD: int = 120
CREATOR_TASK_TYPE: str = "IMAGE_GEN"


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def analytics_jobs(days: int = 30, wallet: Optional[str] = None) -> Dict[str, Any]:
    """Job counts grouped by day, model, and task_type.

    Args:
        days: Number of days to look back (default 30).
        wallet: Optional wallet filter.

    Returns:
        Dictionary with ``by_day``, ``by_model``, and ``by_task_type`` breakdowns.
    """
    conn = get_db()
    cutoff = time.time() - (days * 86400)

    # --- By day (with success/failed breakdown) ---
    if wallet:
        day_rows = conn.execute(
            """
            SELECT date(timestamp, 'unixepoch') AS day,
                   COUNT(*) AS count,
                   SUM(CASE WHEN status IN ('completed', 'success') THEN 1 ELSE 0 END) AS success,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
            FROM jobs
            WHERE timestamp >= ? AND wallet = ?
            GROUP BY day ORDER BY day
            """,
            (cutoff, wallet),
        ).fetchall()
    else:
        day_rows = conn.execute(
            """
            SELECT date(timestamp, 'unixepoch') AS day,
                   COUNT(*) AS count,
                   SUM(CASE WHEN status IN ('completed', 'success') THEN 1 ELSE 0 END) AS success,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
            FROM jobs
            WHERE timestamp >= ?
            GROUP BY day ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
    by_day = [{"date": row["day"], "count": row["count"], "success": row["success"], "failed": row["failed"]} for row in day_rows]

    # --- By model ---
    if wallet:
        model_rows = conn.execute(
            """
            SELECT model, COUNT(*) AS count
            FROM jobs
            WHERE timestamp >= ? AND wallet = ?
            GROUP BY model ORDER BY count DESC
            """,
            (cutoff, wallet),
        ).fetchall()
    else:
        model_rows = conn.execute(
            """
            SELECT model, COUNT(*) AS count
            FROM jobs
            WHERE timestamp >= ?
            GROUP BY model ORDER BY count DESC
            """,
            (cutoff,),
        ).fetchall()
    by_model = [{"model": row["model"], "count": row["count"]} for row in model_rows]

    # --- By task_type ---
    if wallet:
        type_rows = conn.execute(
            """
            SELECT task_type, COUNT(*) AS count
            FROM jobs
            WHERE timestamp >= ? AND wallet = ?
            GROUP BY task_type ORDER BY count DESC
            """,
            (cutoff, wallet),
        ).fetchall()
    else:
        type_rows = conn.execute(
            """
            SELECT task_type, COUNT(*) AS count
            FROM jobs
            WHERE timestamp >= ?
            GROUP BY task_type ORDER BY count DESC
            """,
            (cutoff,),
        ).fetchall()
    by_task_type = [{"task_type": row["task_type"], "count": row["count"]} for row in type_rows]

    return {
        "days": by_day,
        "by_model": by_model,
        "by_type": by_task_type,
    }


def analytics_costs(days: int = 30, wallet: Optional[str] = None) -> Dict[str, Any]:
    """Credit spend breakdown by model and time period.

    Args:
        days: Number of days to look back.
        wallet: Optional wallet filter.
    """
    conn = get_db()
    cutoff = time.time() - (days * 86400)

    # Total spent from credits table
    if wallet:
        row = conn.execute(
            "SELECT total_spent FROM credits WHERE wallet = ?", (wallet,)
        ).fetchone()
        total_spent = float(row["total_spent"]) if row else 0.0
    else:
        row = conn.execute("SELECT COALESCE(SUM(total_spent), 0) AS total FROM credits").fetchone()
        total_spent = float(row["total"]) if row else 0.0

    # Per-model job counts in period
    if wallet:
        model_rows = conn.execute(
            """
            SELECT model, COUNT(*) AS job_count
            FROM jobs
            WHERE timestamp >= ? AND wallet = ?
            GROUP BY model ORDER BY job_count DESC
            """,
            (cutoff, wallet),
        ).fetchall()
    else:
        model_rows = conn.execute(
            """
            SELECT model, COUNT(*) AS job_count
            FROM jobs
            WHERE timestamp >= ?
            GROUP BY model ORDER BY job_count DESC
            """,
            (cutoff,),
        ).fetchall()
    # Distribute total_spent proportionally across models by job count
    total_jobs = sum(row["job_count"] for row in model_rows)
    by_model = [
        {
            "model": row["model"],
            "job_count": row["job_count"],
            "total_cost": round(total_spent * row["job_count"] / total_jobs, 6) if total_jobs > 0 else 0.0,
        }
        for row in model_rows
    ]

    return {
        "total_spent": round(total_spent, 6),
        "by_model": by_model,
    }


def analytics_nodes() -> Dict[str, Any]:
    """Node performance metrics: uptime percentage, avg latency, jobs completed."""
    now = time.time()
    node_metrics: List[Dict[str, Any]] = []

    for node_id, info in NODES.items():
        start_time = info.get("start_time", now)
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace("Z", "")).timestamp()
            except Exception:
                start_time = now
        total_uptime = max(1, now - float(start_time))

        last_seen_unix = info.get("last_seen_unix", 0)
        online = (now - last_seen_unix) <= ONLINE_THRESHOLD

        tasks_completed = int(info.get("tasks_completed", 0))
        avg_util = float(info.get("avg_utilization", 0.0))

        # Estimate uptime % — online nodes are 100%, offline proportional to how long ago
        if online:
            uptime_pct = 100.0
        else:
            offline_duration = now - last_seen_unix
            uptime_pct = max(0.0, round(100.0 * (1.0 - offline_duration / total_uptime), 1))

        node_metrics.append({
            "node_id": node_id,
            "node_name": info.get("node_name", node_id),
            "wallet": info.get("wallet"),
            "online": online,
            "uptime_pct": uptime_pct,
            "avg_utilization": avg_util,
            "tasks_completed": tasks_completed,
            "role": info.get("role", "worker"),
        })

    node_metrics.sort(key=lambda n: n["tasks_completed"], reverse=True)
    return {"nodes": node_metrics, "total": len(node_metrics)}


def analytics_rewards(days: int = 30) -> Dict[str, Any]:
    """Reward distribution by node wallet and model."""
    conn = get_db()
    cutoff = time.time() - (days * 86400)

    # By node (using node_id from jobs, falling back to wallet)
    node_rows = conn.execute(
        """
        SELECT COALESCE(j.node_id, r.wallet) AS node_id,
               r.wallet,
               SUM(r.reward_hai) AS total,
               COUNT(*) AS count
        FROM rewards r
        LEFT JOIN jobs j ON j.id = r.task_id
        WHERE r.timestamp >= ?
        GROUP BY node_id
        ORDER BY total DESC
        """,
        (cutoff,),
    ).fetchall()
    by_node = []
    for row in node_rows:
        nid = row["node_id"]
        node_info = NODES.get(nid, {})
        by_node.append({
            "node_id": nid,
            "node_name": node_info.get("node_name", nid),
            "total": round(float(row["total"]), 6),
            "count": row["count"],
        })

    # By model (join with jobs)
    model_rows = conn.execute(
        """
        SELECT j.model, SUM(r.reward_hai) AS total, COUNT(*) AS count
        FROM rewards r
        JOIN jobs j ON j.id = r.task_id
        WHERE r.timestamp >= ?
        GROUP BY j.model
        ORDER BY total DESC
        """,
        (cutoff,),
    ).fetchall()
    by_model = [
        {"model": row["model"], "total": round(float(row["total"]), 6), "count": row["count"]}
        for row in model_rows
    ]

    grand_total = sum(n["total"] for n in by_node)

    return {
        "by_node": by_node,
        "by_model": by_model,
        "total": round(grand_total, 6),
    }


def analytics_overview() -> Dict[str, Any]:
    """Combined summary stats for the dashboard."""
    conn = get_db()
    now = time.time()

    # Total jobs
    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # Jobs today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    jobs_today = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE timestamp >= ?", (today_start,)
    ).fetchone()[0]

    # Completed / failed counts
    completed = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status IN ('completed', 'success')"
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'failed'"
    ).fetchone()[0]

    # Success rate
    finished = completed + failed
    success_rate = round(100.0 * completed / finished, 1) if finished > 0 else 0.0

    # Total rewards
    total_rewards = conn.execute(
        "SELECT COALESCE(SUM(reward_hai), 0) FROM rewards"
    ).fetchone()[0]

    # Active nodes
    online_count = 0
    for info in NODES.values():
        last_seen_unix = info.get("last_seen_unix", 0)
        if (now - last_seen_unix) <= ONLINE_THRESHOLD:
            online_count += 1

    # Unique wallets
    unique_wallets = conn.execute(
        "SELECT COUNT(DISTINCT wallet) FROM jobs"
    ).fetchone()[0]

    # Unique models used
    unique_models = conn.execute(
        "SELECT COUNT(DISTINCT model) FROM jobs"
    ).fetchone()[0]

    # Total credits spent
    total_credits_spent_row = conn.execute(
        "SELECT COALESCE(SUM(total_spent), 0) AS total FROM credits"
    ).fetchone()
    total_credits_spent = float(total_credits_spent_row["total"]) if total_credits_spent_row else 0.0

    return {
        "total_jobs": total_jobs,
        "jobs_today": jobs_today,
        "completed": completed,
        "failed": failed,
        "success_rate": success_rate,
        "total_rewards": round(float(total_rewards), 6),
        "active_nodes": online_count,
        "online_nodes": online_count,
        "total_nodes": len(NODES),
        "unique_wallets": unique_wallets,
        "unique_models": unique_models,
        "total_credits_spent": round(total_credits_spent, 6),
    }

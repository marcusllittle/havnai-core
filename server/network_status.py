"""Public network telemetry from durable jobs and coordinator node snapshots.

Read endpoints never recover leases, change jobs, or request chain transactions.
Unavailable scheduler history is represented as null rather than invented counts.
"""
from collections import Counter
from datetime import datetime, timezone
import math
import time

import platform_v1




def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 3)


def snapshot(db, nodes, *, version, lease_seconds, receipt_count, receipt_batches, now=None):
    now = time.time() if now is None else now
    generated = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")
    online = [node for node in nodes if node.get("online")]
    creators = [node for node in online if node.get("role") == "creator"]
    capacity = Counter()
    for node in creators:
        capacity.update(set(node.get("supported_job_types") or []))
    states = Counter()
    for row in db.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"):
        states[platform_v1.canonical_job_state(row["status"])] += row["count"]
    active_rows = db.execute("""SELECT id, model, task_type, node_id, assigned_at, lease_expires_at
        FROM jobs WHERE status IN ('leased','assigned','running','uploading','cancelling')
        ORDER BY assigned_at, id""").fetchall()
    busy_ids = {row["node_id"] for row in active_rows if row["node_id"]}
    active = [{
        "job_id": row["id"], "model": row["model"], "task_type": row["task_type"],
        "node_id": row["node_id"], "assigned_at": row["assigned_at"],
        "lease_expires_at": row["lease_expires_at"],
        "lease_remaining_seconds": max(0, number(row["lease_expires_at"]) - now),
        "at_risk": row["lease_expires_at"] is None or number(row["lease_expires_at"]) - now <= min(30, lease_seconds / 3),
        "retry_count": None, "dispatch_score": None, "dispatch_reason": None,
    } for row in active_rows]
    oldest = db.execute("SELECT MIN(timestamp) FROM jobs WHERE status IN ('queued','pending')").fetchone()[0]
    samples = db.execute("""SELECT timestamp, assigned_at, completed_at FROM jobs
        WHERE completed_at >= ? AND assigned_at IS NOT NULL
        AND completed_at >= assigned_at AND assigned_at >= timestamp""", (now - 86400,)).fetchall()
    queue_times = [row["assigned_at"] - row["timestamp"] for row in samples]
    run_times = [row["completed_at"] - row["assigned_at"] for row in samples]
    queue = {"queued": states["queued"], "running": len(active), "completed": states["succeeded"],
             "failed": states["failed"] + states["expired"]}
    expired = sum(1 for row in active_rows if row["lease_expires_at"] is not None and row["lease_expires_at"] < now)
    at_risk = sum(claim["at_risk"] for claim in active)
    alerts = []
    if not creators:
        alerts.append({"severity": "critical" if queue["queued"] else "warning", "code": "no_online_creators",
                       "message": "No creator nodes are online to execute generation jobs."})
    if at_risk:
        alerts.append({"severity": "warning", "code": "claims_at_risk",
                       "message": f"{at_risk} execution leases are missing, expired, or near expiry."})
    health = "critical" if any(item["severity"] == "critical" for item in alerts) else "degraded" if alerts else "healthy"
    summary = {
        "schema_version": "network-summary.v1", "generated_at": generated,
        "coordinator": {"status": health, "version": version},
        "nodes": {"total": len(nodes), "online": len(online), "offline": len(nodes) - len(online),
                  "operators_online": len({str(n.get("wallet")).lower() for n in online if n.get("wallet")})},
        "capacity": {"by_job_type": dict(capacity),
                     "total_vram_mb": sum(max(0, number((n.get("gpu") or {}).get("memory_total_mb", (n.get("gpu") or {}).get("memory_total")))) for n in creators),
                     "average_gpu_utilization": sum(number(n.get("utilization")) for n in creators) / len(creators) if creators else 0},
        "queue": queue,
        "recovery": {"lease_seconds": lease_seconds, "max_retries": None, "jobs_retried": None, "expired_claims": expired},
        "scheduler": {"strategy": "fifo-capability-matching", "preference_grace_seconds": 0,
                      "signals": ["role", "supports", "model", "pipeline"]},
    }
    control = {
        "schema_version": "network-control-plane.v1", "generated_at": generated,
        "health": {"status": health, "alerts": alerts},
        "nodes": {"tracked": len(nodes), "online": len(online), "offline": len(nodes) - len(online),
                  "ready": sum(n["node_id"] not in busy_ids for n in creators),
                  "busy": sum(n["node_id"] in busy_ids for n in creators)},
        "queue": {**queue, "oldest_wait_seconds": max(0, now - oldest) if oldest is not None else 0},
        "latency_24h": {"sample_size": len(samples), "queue_p50_seconds": percentile(queue_times, .5),
                        "queue_p95_seconds": percentile(queue_times, .95), "run_p50_seconds": percentile(run_times, .5),
                        "run_p95_seconds": percentile(run_times, .95)},
        "claims": {"at_risk": at_risk, "active_count": len(active), "active": active[:200]},
        "scheduler_24h": {"strategy": "fifo-capability-matching", "decisions": {}, "preferred": None, "fallback": None},
        "receipts": {"unbatched": receipt_count, "batch_size": 100, "recent_batches": receipt_batches},
    }
    return summary, control

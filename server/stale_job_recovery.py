"""Background daemon thread for continuous stale job recovery and node dropout detection.

Stale job recovery
------------------
Jobs that remain in 'running' status beyond HAVNAI_STALE_JOB_TIMEOUT_SECONDS
(default: 600s / 10 minutes) are reset to 'queued' so another node can pick
them up. This closes the gap that the existing startup reset (line ~966 in
app.py) leaves open: a node that crashes *after* startup won't stall jobs until
the next coordinator restart.

Node dropout detection
----------------------
When a node's last heartbeat exceeds HAVNAI_ONLINE_THRESHOLD seconds, an alert
is fired via server.alerting (Slack webhook or custom endpoint). Per-node
cooldown prevents alert floods if the node stays offline.

Startup
-------
Call ``stale_job_recovery.start(get_db, NODES, log_event)`` once from app.py
after the Flask app and DB are initialised. The thread is a daemon so it stops
with the process automatically.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

STALE_JOB_TIMEOUT_SECONDS = int(os.getenv("HAVNAI_STALE_JOB_TIMEOUT_SECONDS", "600"))
RECOVERY_INTERVAL_SECONDS = int(os.getenv("HAVNAI_RECOVERY_INTERVAL_SECONDS", "60"))
NODE_ONLINE_THRESHOLD = int(os.getenv("HAVNAI_ONLINE_THRESHOLD", "120"))
NODE_ALERT_COOLDOWN = int(os.getenv("HAVNAI_NODE_ALERT_COOLDOWN_SECONDS", "300"))

_logger = logging.getLogger("havnai.recovery")

# Injected at startup via start()
_get_db: Optional[Callable] = None
_NODES: Optional[Dict[str, Dict[str, Any]]] = None
_log_event: Optional[Callable] = None

_node_alert_times: Dict[str, float] = {}
_started = False
_lock = threading.Lock()


def _recover_stale_jobs() -> int:
    """Reset jobs stuck in 'running' beyond the stale timeout. Returns reset count."""
    assert _get_db is not None
    try:
        conn = _get_db()
        cutoff = time.time() - STALE_JOB_TIMEOUT_SECONDS
        rows = conn.execute(
            "SELECT id, node_id, assigned_at FROM jobs WHERE status='running' AND assigned_at < ?",
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0
        for row in rows:
            conn.execute(
                "UPDATE jobs SET status='queued', node_id=NULL WHERE id=?",
                (row["id"],),
            )
        conn.commit()
        for row in rows:
            if _log_event:
                _log_event(
                    "Stale job requeued by recovery thread",
                    level="warning",
                    job_id=row["id"],
                    node_id=row["node_id"],
                    stuck_seconds=round(time.time() - (row["assigned_at"] or 0)),
                )
        return len(rows)
    except Exception as exc:
        _logger.error("Stale job recovery error: %s", exc)
        return 0


def _check_node_dropouts() -> None:
    """Fire alerts for nodes that have exceeded the heartbeat threshold."""
    if _NODES is None:
        return
    try:
        import alerting
    except ImportError:
        return

    now = time.time()
    for node_id, node in list(_NODES.items()):
        last_seen = node.get("last_seen_unix") or node.get("last_seen", 0)
        if isinstance(last_seen, str):
            continue
        seconds_offline = now - float(last_seen or 0)
        if seconds_offline < NODE_ONLINE_THRESHOLD:
            # Node recovered — reset alert state
            _node_alert_times.pop(node_id, None)
            continue
        last_alert = _node_alert_times.get(node_id, 0)
        if now - last_alert < NODE_ALERT_COOLDOWN:
            continue
        _node_alert_times[node_id] = now
        alerting.node_offline(node_id=node_id, last_seen_seconds_ago=seconds_offline)
        if _log_event:
            _log_event(
                "Node dropout detected",
                level="warning",
                node_id=node_id,
                offline_seconds=round(seconds_offline),
            )


def _run_loop() -> None:
    while True:
        try:
            recovered = _recover_stale_jobs()
            if recovered:
                try:
                    import alerting
                    alerting.stale_jobs_requeued(count=recovered)
                except ImportError:
                    pass
            _check_node_dropouts()
        except Exception as exc:
            _logger.error("Recovery loop unhandled error: %s", exc)
        time.sleep(RECOVERY_INTERVAL_SECONDS)


def start(
    get_db: Callable,
    nodes: Dict[str, Dict[str, Any]],
    log_event: Callable,
) -> None:
    """Start the background recovery thread. Safe to call only once.

    Typical call site in app.py (after init_db() and app startup)::

        import stale_job_recovery
        stale_job_recovery.start(get_db, NODES, log_event)
    """
    global _get_db, _NODES, _log_event, _started
    with _lock:
        if _started:
            return
        _get_db = get_db
        _NODES = nodes
        _log_event = log_event
        _started = True

    t = threading.Thread(target=_run_loop, name="stale-job-recovery", daemon=True)
    t.start()
    _logger.info(
        "Stale job recovery thread started (check_interval=%ds, stale_timeout=%ds)",
        RECOVERY_INTERVAL_SECONDS,
        STALE_JOB_TIMEOUT_SECONDS,
    )

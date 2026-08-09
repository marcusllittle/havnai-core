"""Structured health check for the HavnAI coordinator.

Provides a ``check()`` function that returns a status dict and HTTP status code
suitable for load-balancer and uptime-monitor health checks.

Wiring (add to app.py after _inject_module_dependencies()):

    import health as health_module
    health_module.start(get_db, NODES, ONLINE_THRESHOLD, startup_time=time.time())

    @app.route("/health")
    def healthz():
        payload, status = health_module.check()
        return jsonify(payload), status

The existing /health route in app.py can delegate to health_module.check()
rather than replacing it outright.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

_logger = logging.getLogger("havnai.health")

# Injected at startup via start()
_get_db: Optional[Callable] = None
_NODES: Optional[Dict[str, Any]] = None
_online_threshold: int = 120
_startup_time: float = time.time()


def start(
    get_db: Callable,
    nodes: Dict[str, Any],
    online_threshold: int = 120,
    startup_time: Optional[float] = None,
) -> None:
    """Inject dependencies.  Call once from app.py startup."""
    global _get_db, _NODES, _online_threshold, _startup_time
    _get_db = get_db
    _NODES = nodes
    _online_threshold = online_threshold
    _startup_time = startup_time if startup_time is not None else time.time()


def _db_ok() -> bool:
    if _get_db is None:
        return False
    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        return True
    except Exception as exc:
        _logger.warning("Health: DB check failed: %s", exc)
        return False


def _active_node_count() -> int:
    if not _NODES:
        return 0
    cutoff = time.time() - _online_threshold
    return sum(
        1
        for node in _NODES.values()
        if isinstance(node.get("last_seen_unix"), (int, float))
        and node["last_seen_unix"] >= cutoff
    )


def _queue_depth() -> int:
    if _get_db is None:
        return 0
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return -1


def check() -> Tuple[Dict[str, Any], int]:
    """Return (payload_dict, http_status_code).

    HTTP 200 = healthy; HTTP 503 = database unreachable.
    """
    db_healthy = _db_ok()
    active_nodes = _active_node_count()
    queue = _queue_depth()
    uptime_seconds = round(time.time() - _startup_time)

    payload: Dict[str, Any] = {
        "status": "ok" if db_healthy else "degraded",
        "db_ok": db_healthy,
        "active_nodes": active_nodes,
        "queue_depth": queue,
        "uptime_seconds": uptime_seconds,
    }
    http_status = 200 if db_healthy else 503
    return payload, http_status

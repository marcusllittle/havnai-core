"""Fire-and-forget webhook alerting for critical operational events.

Set HAVNAI_ALERT_WEBHOOK_URL to any Slack-compatible incoming webhook URL to
enable. Alerts are sent in background daemon threads so they never block request
handling. Delivery failures are logged as warnings and silently dropped.

Typed helpers
-------------
    alerting.node_offline(node_id, last_seen_seconds_ago)
    alerting.job_failure_spike(failure_rate, window_minutes, failed, total)
    alerting.coordinator_started(version)
    alerting.stale_jobs_requeued(count)
    alerting.custom(alert_type, text, **extra)
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

ALERT_WEBHOOK_URL: Optional[str] = os.getenv("HAVNAI_ALERT_WEBHOOK_URL", "").strip() or None
ALERT_TIMEOUT = int(os.getenv("HAVNAI_ALERT_TIMEOUT_SECONDS", "5"))
ALERT_ENABLED = bool(ALERT_WEBHOOK_URL)

_logger = logging.getLogger("havnai.alerting")


def _send(payload: Dict[str, Any]) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        resp = requests.post(
            ALERT_WEBHOOK_URL,
            json=payload,
            timeout=ALERT_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except Exception as exc:
        _logger.warning("Alert delivery failed: %s", exc)


def _fire(payload: Dict[str, Any]) -> None:
    """Dispatch alert in a daemon thread — never blocks the caller."""
    if not ALERT_ENABLED:
        return
    threading.Thread(target=_send, args=(payload,), daemon=True).start()


def _base(alert_type: str, **extra: Any) -> Dict[str, Any]:
    return {
        "source": "havnai-core",
        "alert_type": alert_type,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **extra,
    }


# ---------------------------------------------------------------------------
# Typed alert helpers
# ---------------------------------------------------------------------------

def node_offline(node_id: str, last_seen_seconds_ago: float, **extra: Any) -> None:
    """Alert when a node exceeds the heartbeat threshold."""
    payload = _base(
        "node_offline",
        node_id=node_id,
        last_seen_seconds_ago=round(last_seen_seconds_ago, 1),
        **extra,
    )
    payload["text"] = (
        f":warning: *HavnAI*: Node `{node_id}` went offline "
        f"({round(last_seen_seconds_ago)}s since last heartbeat)"
    )
    _fire(payload)


def job_failure_spike(
    failure_rate: float,
    window_minutes: int,
    failed: int,
    total: int,
    **extra: Any,
) -> None:
    """Alert when job failure rate exceeds an acceptable threshold."""
    payload = _base(
        "job_failure_spike",
        failure_rate=round(failure_rate, 4),
        window_minutes=window_minutes,
        failed_jobs=failed,
        total_jobs=total,
        **extra,
    )
    payload["text"] = (
        f":red_circle: *HavnAI*: Job failure spike — "
        f"{failed}/{total} failed in the last {window_minutes}m "
        f"({failure_rate:.1%})"
    )
    _fire(payload)


def coordinator_started(version: str = "unknown", **extra: Any) -> None:
    """Alert on coordinator process start (helps detect unexpected restarts)."""
    payload = _base("coordinator_start", version=version, **extra)
    payload["text"] = f":rocket: *HavnAI*: Coordinator started (version: `{version}`)"
    _fire(payload)


def stale_jobs_requeued(count: int, **extra: Any) -> None:
    """Alert when stuck jobs are automatically requeued."""
    if count == 0:
        return
    payload = _base("stale_jobs_requeued", count=count, **extra)
    payload["text"] = f":recycle: *HavnAI*: {count} stale job(s) requeued (node likely crashed)"
    _fire(payload)


def custom(alert_type: str, text: str, **extra: Any) -> None:
    """Send a one-off alert with arbitrary payload."""
    payload = _base(alert_type, text=text, **extra)
    _fire(payload)

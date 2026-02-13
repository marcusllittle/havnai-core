"""Invite code and quota management for HavnAI."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from flask import Request

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
request: Request
jsonify: Callable
INVITE_CONFIG_PATH: Path
INVITE_GATING: bool = False
LOGGER: logging.Logger


def load_invite_config() -> Dict[str, Dict[str, Any]]:
    if not INVITE_CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(INVITE_CONFIG_PATH.read_text())
    except Exception as exc:
        LOGGER.error("Failed to parse invite config: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    config: Dict[str, Dict[str, Any]] = {}
    for code, entry in raw.items():
        if not isinstance(code, str) or not isinstance(entry, dict):
            continue
        config[code] = entry
    return config


def invite_gating_enabled(invite_config: Dict[str, Dict[str, Any]]) -> bool:
    if INVITE_GATING:
        return True
    return bool(invite_config)


def extract_invite_code(payload: Dict[str, Any]) -> str:
    header_code = request.headers.get("X-INVITE-CODE", "").strip()
    body_code = str(payload.get("invite_code") or "").strip()
    return header_code or body_code


def invite_error_response() -> Any:
    return jsonify({"error": "invite_required", "message": "Invite code required."}), 403


def quota_payload(
    max_daily: int,
    used_today: int,
    max_concurrent: int,
    used_concurrent: int,
    reset_at: str,
) -> Dict[str, Any]:
    return {
        "max_daily": max_daily,
        "used_today": used_today,
        "max_concurrent": max_concurrent,
        "used_concurrent": used_concurrent,
        "reset_at": reset_at,
    }


def quota_limit_response(
    max_daily: int,
    used_today: int,
    max_concurrent: int,
    used_concurrent: int,
    reset_at: str,
) -> Any:
    payload = {
        "error": "rate_limited",
        "message": "Invite quota exceeded.",
    }
    payload.update(quota_payload(max_daily, used_today, max_concurrent, used_concurrent, reset_at))
    return jsonify(payload), 429


def resolve_invite_limits(
    invite_code: str, invite_config: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    entry = invite_config.get(invite_code)
    if not entry:
        return None
    if entry.get("enabled") is False:
        return None
    return entry


def compute_invite_usage(invite_code: str) -> Dict[str, Any]:
    conn = get_db()
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_ts = midnight.timestamp()
    reset_at = (midnight + timedelta(days=1)).isoformat().replace("+00:00", "Z")

    statuses = ("queued", "running", "assigned", "uploading")
    concurrent = conn.execute(
        f"""
        SELECT COUNT(*) FROM jobs
        WHERE invite_code=?
          AND status IN ({",".join("?" for _ in statuses)})
        """,
        (invite_code, *statuses),
    ).fetchone()[0]
    used_today = conn.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE invite_code=?
          AND timestamp >= ?
        """,
        (invite_code, midnight_ts),
    ).fetchone()[0]
    return {
        "used_today": int(used_today or 0),
        "used_concurrent": int(concurrent or 0),
        "reset_at": reset_at,
    }


def enforce_invite_quota(
    payload: Dict[str, Any]
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Any]]:
    invite_config = load_invite_config()
    if not invite_gating_enabled(invite_config):
        return None, None, None

    invite_code = extract_invite_code(payload)
    if not invite_code:
        return None, None, invite_error_response()

    limits = resolve_invite_limits(invite_code, invite_config)
    if not limits:
        return None, None, invite_error_response()

    usage = compute_invite_usage(invite_code)
    max_daily = int(limits.get("max_daily") or 0)
    max_concurrent = int(limits.get("max_concurrent") or 0)
    used_today = usage["used_today"]
    used_concurrent = usage["used_concurrent"]
    reset_at = usage["reset_at"]

    if max_daily > 0 and used_today >= max_daily:
        return invite_code, usage, quota_limit_response(
            max_daily, used_today, max_concurrent, used_concurrent, reset_at
        )
    if max_concurrent > 0 and used_concurrent >= max_concurrent:
        return invite_code, usage, quota_limit_response(
            max_daily, used_today, max_concurrent, used_concurrent, reset_at
        )
    return invite_code, usage, None


def enforce_invite_limits(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    invite_code, _usage, error = enforce_invite_quota(payload)
    return invite_code, error

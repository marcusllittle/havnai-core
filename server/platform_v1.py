"""Durable job, asset, and render-contract helpers for the v1 API."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Mapping, Optional


CANONICAL_JOB_STATES = {
    "queued",
    "leased",
    "running",
    "uploading",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
    "expired",
}

FINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "expired"}

LEGACY_STATE_MAP = {
    "pending": "queued",
    "assigned": "leased",
    "success": "succeeded",
    "completed": "succeeded",
    "done": "succeeded",
    "error": "failed",
    "canceled": "cancelled",
}

VIDEO_PRESETS: Dict[str, Dict[str, Dict[str, int | float | str]]] = {
    "fast_upscaled": {
        "16:9": {"width": 832, "height": 480, "delivery_width": 1280, "delivery_height": 720},
        "9:16": {"width": 480, "height": 832, "delivery_width": 720, "delivery_height": 1280},
    },
    "native_quality": {
        "16:9": {"width": 1280, "height": 704, "delivery_width": 1280, "delivery_height": 720},
        "9:16": {"width": 704, "height": 1280, "delivery_width": 720, "delivery_height": 1280},
    },
}

VIDEO_DURATIONS = {3, 5, 8}
VIDEO_TIMEOUT_SECONDS = 3600


def canonical_job_state(value: Any) -> str:
    state = str(value or "queued").strip().lower()
    state = LEGACY_STATE_MAP.get(state, state)
    return state if state in CANONICAL_JOB_STATES else "failed"


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply coordinator-safe SQLite settings to every connection."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    """Apply additive schema changes; safe to run during every startup."""
    configure_connection(conn)
    columns = _column_names(conn, "jobs")
    additions = {
        "attempt_id": "TEXT",
        "lease_expires_at": "REAL",
        "progress": "REAL NOT NULL DEFAULT 0",
        "stage": "TEXT NOT NULL DEFAULT 'queued'",
        "resolved_spec": "TEXT",
        "error_code": "TEXT",
        "updated_at": "REAL",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            kind TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            attempt_id TEXT,
            kind TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_job_created ON artifacts(job_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_lease_expiry ON jobs(status, lease_expires_at)"
    )
    conn.execute(
        "UPDATE jobs SET updated_at=COALESCE(updated_at, timestamp), stage=COALESCE(NULLIF(stage, ''), status)"
    )
    now = time.time()
    conn.execute(
        """
        UPDATE jobs
           SET status='queued', node_id=NULL, attempt_id=NULL, assigned_at=NULL,
               lease_expires_at=NULL, progress=0, stage='queued', updated_at=?
         WHERE status IN ('leased', 'running', 'uploading')
           AND (attempt_id IS NULL OR lease_expires_at IS NULL OR lease_expires_at < ?)
        """,
        (now, now),
    )
    conn.commit()


def tokens_match(expected: str, supplied: Optional[str]) -> bool:
    if not expected:
        return False
    return bool(supplied) and secrets.compare_digest(expected, str(supplied))


def new_attempt_id() -> str:
    return f"attempt-{uuid.uuid4().hex}"


def new_asset_id() -> str:
    return f"asset-{uuid.uuid4().hex}"


def new_artifact_id() -> str:
    return f"artifact-{uuid.uuid4().hex}"


def safe_filename(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip(".")
    return cleaned[:160] or fallback


def atomic_stream_write(stream: BinaryIO, destination: Path, max_bytes: int) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    total = 0
    try:
        with temp_path.open("wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("file_too_large")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return total, digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def resolve_video_spec(payload: Mapping[str, Any], *, model: str, backend: str = "auto") -> Dict[str, Any]:
    preset = str(payload.get("preset") or "fast_upscaled").strip().lower()
    aspect = str(payload.get("aspect_ratio") or payload.get("aspect") or "9:16").strip()
    if preset not in VIDEO_PRESETS:
        raise ValueError("invalid_preset")
    if aspect not in {"9:16", "16:9"}:
        raise ValueError("invalid_aspect_ratio")
    try:
        duration = int(payload.get("duration_seconds") or payload.get("duration") or 5)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_duration") from exc
    if duration not in VIDEO_DURATIONS:
        raise ValueError("invalid_duration")

    dimensions = VIDEO_PRESETS[preset][aspect]
    fps = max(8, min(24, int(payload.get("fps") or 16)))
    requested_frames = int(payload.get("frames") or (duration * fps + 1))
    # LTX/SANA video pipelines commonly require 8n+1 frame counts.
    frames = max(9, ((requested_frames - 1) // 8) * 8 + 1)
    seed = payload.get("seed")
    if seed is None or int(seed) < 0:
        seed = secrets.randbelow(2**31)

    return {
        "schema_version": 1,
        "task_type": "image_to_video",
        "model": {"id": model},
        "engine": {"backend": backend},
        "preset": preset,
        "aspect_ratio": aspect,
        "duration_seconds": duration,
        "timeout_seconds": VIDEO_TIMEOUT_SECONDS,
        "parameters": {
            "seed": int(seed),
            "width": int(dimensions["width"]),
            "height": int(dimensions["height"]),
            "delivery_width": int(dimensions["delivery_width"]),
            "delivery_height": int(dimensions["delivery_height"]),
            "frames": frames,
            "fps": fps,
            "steps": int(payload.get("steps") or (8 if preset == "fast_upscaled" else 12)),
            "guidance": float(payload.get("guidance") or (1.0 if "distilled" in model.lower() else 3.0)),
            "motion_strength": float(payload.get("motion_strength") or 0.65),
        },
    }


def redact_mapping(values: Mapping[str, Any], secret_keys: Iterable[str] = ()) -> Dict[str, Any]:
    blocked = {key.lower() for key in secret_keys} | {"token", "password", "secret", "api_key"}
    return {
        key: ("[redacted]" if key.lower() in blocked else value)
        for key, value in values.items()
    }


def failure_category(error: Any) -> str:
    message = str(error or "").lower()
    categories = (
        ("cancel", "cancelled"),
        ("timeout", "timeout"),
        ("out of memory", "gpu_oom"),
        ("cuda", "cuda"),
        ("cudnn", "cuda_runtime"),
        ("disk", "storage"),
        ("space", "storage"),
        ("upload", "upload"),
        ("asset", "asset"),
        ("model", "model"),
        ("authentication", "authentication"),
        ("unauthorized", "authentication"),
    )
    return next((category for needle, category in categories if needle in message), "runtime")

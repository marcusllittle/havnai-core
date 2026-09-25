"""Durable job, asset, and render-contract helpers for the v1 API."""

from __future__ import annotations

import hashlib
import json
import math
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
MUSIC_DURATION_MIN = 10
MUSIC_DURATION_MAX = 600
MUSIC_TIMEOUT_SECONDS = 900
MUSIC_BATCH_MAX = 8

#: Studio-facing mode -> ACE-Step engine task_type.
MUSIC_MODES: Dict[str, str] = {
    "create": "text2music",
    "remix": "cover",
    "remix_loose": "cover-nofsq",
    "repaint": "repaint",
    "extract": "extract",
    "layer": "lego",
    "arrange": "complete",
}

#: Reverse lookup so callers may pass the engine name directly.
MUSIC_TASK_TO_MODE: Dict[str, str] = {task: mode for mode, task in MUSIC_MODES.items()}

#: Modes that need a source recording to work from.
MUSIC_MODES_NEEDING_SOURCE = frozenset({"remix", "remix_loose", "repaint", "extract", "layer", "arrange"})

#: Modes that operate on exactly one named stem.
MUSIC_MODES_NEEDING_TRACK = frozenset({"extract", "layer"})

#: Capability flag each mode maps onto in the model registry.
MUSIC_MODE_CAPABILITY: Dict[str, str] = {
    "create": "text_to_music",
    "remix": "cover",
    "remix_loose": "cover",
    "repaint": "repaint",
    "extract": "extract",
    "layer": "lego",
    "arrange": "complete",
}

MUSIC_TRACK_NAMES = (
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
)

MUSIC_AUDIO_FORMATS = ("mp3", "flac", "wav", "wav32", "opus", "aac")
MUSIC_REPAINT_MODES = ("conservative", "balanced", "aggressive")
MUSIC_INFER_METHODS = ("ode", "sde")


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


def _video_number(payload, key, default, low, high, *, integer=False, error=None):
    raw = payload.get(key, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = float(raw)
        if isinstance(raw, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError()
        if integer and not value.is_integer():
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error or f"invalid_video_{key}") from exc
    return int(value) if integer else value


def resolve_video_spec(payload: Mapping[str, Any], *, model: str, backend: str = "auto") -> Dict[str, Any]:
    preset = str(payload.get("preset") or "fast_upscaled").strip().lower()
    aspect = str(payload.get("aspect_ratio") or payload.get("aspect") or "9:16").strip()
    if preset not in VIDEO_PRESETS:
        raise ValueError("invalid_preset")
    if aspect not in {"9:16", "16:9"}:
        raise ValueError("invalid_aspect_ratio")
    duration = _video_number({"duration": payload.get("duration_seconds", payload.get("duration", 5))},
                             "duration", 5, 3, 8, integer=True, error="invalid_duration")
    if duration not in VIDEO_DURATIONS:
        raise ValueError("invalid_duration")

    dimensions = dict(VIDEO_PRESETS[preset][aspect])
    custom_dimensions = payload.get("width") is not None or payload.get("height") is not None
    if custom_dimensions:
        if payload.get("width") is None or payload.get("height") is None:
            raise ValueError("invalid_video_dimensions")
        for key in ("width", "height"):
            dimensions[key] = _video_number(payload, key, dimensions[key], 256, 1280, integer=True)
            if dimensions[key] % 32:
                raise ValueError("invalid_video_dimensions")
            dimensions[f"delivery_{key}"] = dimensions[key]
    fps = _video_number(payload, "fps", 16, 8, 30, integer=True)
    requested_frames = _video_number(payload, "frames", duration * fps + 1, 9, 257, integer=True)
    # LTX/SANA video pipelines commonly require 8n+1 frame counts.
    frames = max(9, ((requested_frames - 1) // 8) * 8 + 1)
    seed = _video_number(payload, "seed", -1, -1, 2**32 - 1, integer=True, error="invalid_seed")
    if seed < 0:
        seed = secrets.randbelow(2**31)
    steps = _video_number(payload, "steps", 8 if preset == "fast_upscaled" else 12, 1, 150, integer=True)
    guidance = _video_number(payload, "guidance", 1.0 if "distilled" in model.lower() else 3.0, 0, 20)
    motion = _video_number(payload, "motion_strength", 0.65, 0, 1)
    strength = _video_number(payload, "strength", 1.0, 0, 1)

    return {
        "schema_version": 1,
        "task_type": "image_to_video",
        "model": {"id": model},
        "engine": {"backend": backend},
        "preset": "custom" if custom_dimensions else preset,
        "aspect_ratio": f"{dimensions['width']}:{dimensions['height']}" if custom_dimensions else aspect,
        "duration_seconds": (frames - 1) / fps,
        "timeout_seconds": VIDEO_TIMEOUT_SECONDS,
        "parameters": {
            "seed": int(seed),
            "width": int(dimensions["width"]),
            "height": int(dimensions["height"]),
            "delivery_width": int(dimensions["delivery_width"]),
            "delivery_height": int(dimensions["delivery_height"]),
            "frames": frames,
            "fps": fps,
            "steps": steps,
            "guidance": guidance,
            "motion_strength": motion,
            **({"strength": strength} if payload.get("strength") is not None else {}),
        },
    }


def _music_number(payload: Mapping[str, Any], key: str, error: str, *, cast=float, low=None, high=None, default=None):
    raw = payload.get(key)
    if raw in (None, ""):
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if low is not None and value < low:
        raise ValueError(error)
    if high is not None and value > high:
        raise ValueError(error)
    return value


def resolve_music_mode(payload: Mapping[str, Any]) -> str:
    """Normalise a studio mode or a raw engine task_type into a studio mode."""
    raw = str(payload.get("mode") or payload.get("task_type") or "create").strip().lower()
    if raw in MUSIC_MODES:
        return raw
    if raw in MUSIC_TASK_TO_MODE:
        return MUSIC_TASK_TO_MODE[raw]
    if raw in {"text_to_music", "text2music"}:
        return "create"
    raise ValueError("invalid_mode")


def resolve_music_spec(
    payload: Mapping[str, Any],
    *,
    model: str,
    capabilities: Iterable[str] = (),
    defaults: Mapping[str, Any] | None = None,
    max_batch_size: int | None = None,
    timeout_seconds: int | None = None,
    engine_model: str = "acestep-v15-turbo",
) -> Dict[str, Any]:
    mode = resolve_music_mode(payload)
    declared = {str(value) for value in capabilities or ()}
    required = MUSIC_MODE_CAPABILITY[mode]
    if declared and required not in declared:
        raise ValueError(f"mode_unsupported_by_model:{mode}")

    base = dict(defaults or {})

    prompt = str(payload.get("prompt") or "").strip()
    # extract/layer/arrange describe a stem, not a whole song, so an empty prompt is fine there.
    if not prompt and mode in {"create", "remix", "remix_loose", "repaint"}:
        raise ValueError("invalid_prompt")
    if len(prompt) > 4000:
        raise ValueError("invalid_prompt")

    style = str(payload.get("style") or "").strip()
    if len(style) > 500:
        raise ValueError("invalid_style")
    lyrics = str(payload.get("lyrics") or "")
    if len(lyrics) > 12000:
        raise ValueError("invalid_lyrics")
    instrumental = payload.get("instrumental", False)
    if not isinstance(instrumental, bool):
        raise ValueError("invalid_instrumental")

    duration = _music_number(
        payload, "duration", "invalid_duration",
        cast=float, low=MUSIC_DURATION_MIN, high=MUSIC_DURATION_MAX, default=None,
    )
    if duration is None:
        duration = _music_number(
            payload, "duration_seconds", "invalid_duration",
            cast=float, low=MUSIC_DURATION_MIN, high=MUSIC_DURATION_MAX, default=60.0,
        )

    bpm = _music_number(payload, "bpm", "invalid_bpm", cast=int, low=30, high=300)
    key = str(payload.get("key") or "").strip()
    if len(key) > 64:
        raise ValueError("invalid_key")
    time_signature = str(payload.get("time_signature") or "").strip()
    if len(time_signature) > 16:
        raise ValueError("invalid_time_signature")
    vocal_language = str(payload.get("vocal_language") or "").strip()
    if len(vocal_language) > 32:
        raise ValueError("invalid_vocal_language")
    seed = _music_number(payload, "seed", "invalid_seed", cast=int, low=0, high=2**31 - 1)

    ceiling = max(1, min(MUSIC_BATCH_MAX, int(max_batch_size or MUSIC_BATCH_MAX)))
    # Reject nonsense outright, but quietly clamp a legitimate request down to
    # whatever this checkpoint can actually serve rather than failing the job.
    batch_size = _music_number(
        payload, "batch_size", "invalid_batch_size",
        cast=int, low=1, high=MUSIC_BATCH_MAX, default=int(base.get("batch_size") or 1),
    )
    batch_size = max(1, min(ceiling, int(batch_size)))

    steps = _music_number(
        payload, "inference_steps", "invalid_inference_steps",
        cast=int, low=1, high=200, default=base.get("inference_steps"),
    )
    guidance = _music_number(
        payload, "guidance_scale", "invalid_guidance_scale",
        cast=float, low=0.0, high=30.0, default=base.get("guidance_scale"),
    )
    shift = _music_number(
        payload, "shift", "invalid_shift",
        cast=float, low=0.1, high=10.0, default=base.get("shift"),
    )
    infer_method = str(payload.get("infer_method") or base.get("infer_method") or "").strip().lower()
    if infer_method and infer_method not in MUSIC_INFER_METHODS:
        raise ValueError("invalid_infer_method")

    audio_format = str(payload.get("audio_format") or base.get("audio_format") or "mp3").strip().lower()
    if audio_format not in MUSIC_AUDIO_FORMATS:
        raise ValueError("invalid_audio_format")

    parameters: Dict[str, Any] = {
        "mode": mode,
        "task_type": MUSIC_MODES[mode],
        "prompt": prompt,
        "style": style,
        "lyrics": "" if instrumental else lyrics,
        "instrumental": instrumental,
        "duration": duration,
        "bpm": bpm,
        "key": key,
        "time_signature": time_signature or None,
        "vocal_language": vocal_language or None,
        "seed": seed,
        "batch_size": batch_size,
        "audio_format": audio_format,
        "inference_steps": steps,
        "guidance_scale": guidance,
        "shift": shift,
        "infer_method": infer_method or None,
    }

    # --- source / reference conditioning ---------------------------------
    source_asset_id = str(payload.get("audio_asset_id") or payload.get("source_audio_asset_id") or "").strip()
    reference_asset_id = str(payload.get("reference_asset_id") or "").strip()
    if mode in MUSIC_MODES_NEEDING_SOURCE and not source_asset_id:
        raise ValueError("source_audio_required")
    parameters["audio_asset_id"] = source_asset_id or None
    parameters["reference_asset_id"] = reference_asset_id or None

    # --- per-mode extras --------------------------------------------------
    if mode in {"remix", "remix_loose"}:
        parameters["audio_cover_strength"] = _music_number(
            payload, "audio_cover_strength", "invalid_cover_strength",
            cast=float, low=0.0, high=1.0, default=0.6,
        )
        parameters["cover_noise_strength"] = _music_number(
            payload, "cover_noise_strength", "invalid_cover_noise",
            cast=float, low=0.0, high=1.0, default=0.0,
        )

    if mode == "repaint":
        start = _music_number(
            payload, "repainting_start", "invalid_repaint_range",
            cast=float, low=0.0, high=MUSIC_DURATION_MAX, default=0.0,
        )
        end = _music_number(
            payload, "repainting_end", "invalid_repaint_range",
            cast=float, low=-1.0, high=MUSIC_DURATION_MAX, default=-1.0,
        )
        if end != -1.0 and end <= start:
            raise ValueError("invalid_repaint_range")
        repaint_mode = str(payload.get("repaint_mode") or "balanced").strip().lower()
        if repaint_mode not in MUSIC_REPAINT_MODES:
            raise ValueError("invalid_repaint_mode")
        parameters.update({
            "repainting_start": start,
            "repainting_end": end,
            "repaint_mode": repaint_mode,
            "repaint_strength": _music_number(
                payload, "repaint_strength", "invalid_repaint_strength",
                cast=float, low=0.0, high=1.0, default=0.5,
            ),
            "repaint_wav_crossfade_sec": _music_number(
                payload, "repaint_wav_crossfade_sec", "invalid_repaint_crossfade",
                cast=float, low=0.0, high=5.0, default=0.0,
            ),
        })

    if mode in MUSIC_MODES_NEEDING_TRACK:
        track = str(payload.get("track_name") or "").strip().lower()
        if track not in MUSIC_TRACK_NAMES:
            raise ValueError("invalid_track_name")
        parameters["track_name"] = track

    if mode == "arrange":
        raw_classes = payload.get("track_classes") or []
        if not isinstance(raw_classes, (list, tuple)):
            raise ValueError("invalid_track_classes")
        tracks = []
        for value in raw_classes:
            name = str(value).strip().lower()
            if name not in MUSIC_TRACK_NAMES:
                raise ValueError("invalid_track_classes")
            if name not in tracks:
                tracks.append(name)
        if not tracks:
            raise ValueError("invalid_track_classes")
        parameters["track_classes"] = tracks

    global_caption = str(payload.get("global_caption") or "").strip()
    if len(global_caption) > 4000:
        raise ValueError("invalid_global_caption")
    if global_caption:
        parameters["global_caption"] = global_caption

    timeout = int(timeout_seconds or MUSIC_TIMEOUT_SECONDS)
    # CFG-guided checkpoints and long batches occupy the node far longer.
    if batch_size > 1:
        timeout = int(timeout * min(2.5, 1 + 0.4 * (batch_size - 1)))

    return {
        "schema_version": 1,
        "task_type": "text_to_music",
        "mode": mode,
        "model": {"id": model},
        "engine": {"provider": "ace_step", "model": engine_model, "task_type": MUSIC_MODES[mode]},
        "timeout_seconds": timeout,
        "parameters": {
            key_: value
            for key_, value in parameters.items()
            # Keep booleans and an intentionally empty prompt; drop unset optionals.
            if value is not None and (value != "" or key_ == "prompt")
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

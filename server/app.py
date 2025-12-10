from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import abort, Flask, jsonify, request, send_file, send_from_directory, g, has_app_context
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(os.getenv("HAVNAI_STATIC_DIR", Path(__file__).resolve().parent.parent / "static"))
LEGACY_STATIC_DIR = Path(os.getenv("HAVNAI_INSTALLER_DIR", Path(__file__).resolve().parent / "static"))
MANIFESTS_DIR = Path(os.getenv("HAVNAI_MANIFEST_DIR", Path(__file__).resolve().parent / "manifests"))
MANIFEST_FILE = Path(os.getenv("HAVNAI_MANIFEST_FILE", MANIFESTS_DIR / "registry.json"))
LOGS_DIR = Path(os.getenv("HAVNAI_LOG_DIR", Path(__file__).resolve().parent / "logs"))
OUTPUTS_DIR = Path(os.getenv("HAVNAI_OUTPUTS_DIR", STATIC_DIR / "outputs"))
REGISTRY_FILE = BASE_DIR / "nodes.json"
DB_PATH = BASE_DIR / "db" / "ledger.db"
CLIENT_PATH = BASE_DIR / "client" / "client.py"
CLIENT_REGISTRY = BASE_DIR / "client" / "registry.py"
CLIENT_REQUIREMENTS = BASE_DIR / "client" / "requirements-node.txt"
VERSION_FILE = BASE_DIR / "VERSION"

CREATOR_TASK_TYPE = "IMAGE_GEN"

# Ensure local packages (e.g., havnai.video_engine) are importable when running server/app.py directly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from havnai.video_engine.gguf_wan2_2 import VideoEngine, VideoJobRequest

# Reward weights bootstrap – enriched at runtime via registration
MODEL_WEIGHTS: Dict[str, float] = {
    "triomerge_v10": 12.0,
    "unstablepornhwa_beta": 12.0,
}

MODEL_STATS: Dict[str, Dict[str, float]] = {}
MANIFEST_MODELS: Dict[str, Dict[str, Any]] = {}
EVENT_LOGS: deque = deque(maxlen=200)
NODES: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()
DB_CONN: Optional[sqlite3.Connection] = None
RATE_LIMIT_BUCKETS: Dict[str, deque] = defaultdict(deque)

ONLINE_THRESHOLD = 120  # seconds before a node is considered offline
WALLET_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

SERVER_JOIN_TOKEN = os.getenv("SERVER_JOIN_TOKEN", "").strip()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
RESET_ON_STARTUP = os.getenv("RESET_ON_STARTUP", "").strip()

# Reward configuration (can be overridden via environment variables)
REWARD_CONFIG: Dict[str, float] = {
    "baseline_runtime": float(os.getenv("REWARD_BASELINE_RUNTIME", "8.0")),
    "sdxl_factor": float(os.getenv("REWARD_SDXL_FACTOR", "1.5")),
    "sd15_factor": float(os.getenv("REWARD_SD15_FACTOR", "1.0")),
    "anime_factor": float(os.getenv("REWARD_ANIME_FACTOR", "0.7")),
    "base_reward": float(os.getenv("REWARD_BASE_HAI", "0.05")),
}

# Slight global positive bias to help realism skin detail.
GLOBAL_POSITIVE_SUFFIX = "(ultra-realistic 8k:1.05), (detailed skin pores:1.03)"

# Global negative prompt to discourage common artifacts across all models.
GLOBAL_NEGATIVE_PROMPT = ", ".join(
    [
        "FastNegativeV2",
        "liquid fingers",
        "interlocked fingers",
        "bad teeth",
        "oversaturated",
        "scan artifact",
        "doll skin",
        "exaggerated features",
        "unnatural pose",
        "disembodied limb",
        "oversized head",
        "webbed fingers",
        "bad ears",
        "bad chin",
        "bad mouth",
        "bad eyes",
        "bad digit",
        "bad shadow",
        "bad color",
        "bad lighting",
        "bad crop",
        "bad aspect ratio",
        "bad vector",
        "bad lineart",
        "bad perspective",
        "bad shading",
        "bad sketch",
        "bad trace",
        "bad typesetting",
        "color error",
        "color mismatch",
        "dirty art",
        "dirty scan",
        "dubious anatomy",
        "exaggerated limbs",
        "flat colors",
        "gradient background",
        "heavily pixelated",
        "high noise",
        "image noise",
        "moire pattern",
        "motion blur",
        "muddy colors",
        "overcompressed",
        "poor lineart",
        "scanned with errors",
        "scan errors",
        "very low quality",
        "visible pixels",
        "aliasing",
        "anatomy error",
        "anatomy mistake",
        "broken anatomy",
        "broken pose",
        "camera aberration",
        "chromatic aberration",
        "clashing styles",
        "compression artifacts",
        "corrupted",
        "cribbed from",
        "downsampling",
        "faded lines",
        "filter abuse",
        "grainy",
        "noise",
        "noisy background",
        "pixelation",
        "pixels",
        "poor quality",
        "amateur",
        "amateur drawing",
        "bad art",
        "bad coloring",
        "bad composition",
        "bad contrast",
        "bad drawing",
        "bad image",
        "bad photoshop",
        "bad pose",
        "bad proportions",
        "beginner",
        "black and white",
        "deformed",
        "disfigured",
        "displeasing",
        "distorted",
        "distorted proportions",
        "drawing",
        "duplicate",
        "early",
        "exaggerated pose",
        "gross proportions",
        "malformed limbs",
        "missing arm",
        "missing leg",
        "extra arm",
        "extra leg",
        "fused fingers",
        "too many fingers",
        "extra fingers",
        "mutated hands",
        "blurry",
        "out of frame",
        "contortionist",
        "contorted limbs",
        "disproportionate",
        "twisted posture",
        "disconnected",
        "warped",
        "misshapen",
        "out of scale",
        "score_6",
        "score_5",
        "score_4",
        "tan",
        "piercing",
        "3d",
        "render",
        "cgi",
        "doll",
        "cartoon",
        "illustration",
        "painting",
        "digital art",
        "anime",
        "fake",
        "3d modeling",
        "old",
        "asymmetrical features",
        "unrealistic proportions",
        "mutated",
        "unnatural textures",
        "twisted limbs",
        "malformed hands",
        "bad hands",
        "bad fingers",
        "split image",
        "amputee",
        "mutation",
        "missing fingers",
        "extra digit",
        "fewer digits",
        "cropped",
        "worst quality",
        "low quality",
        "normal quality",
        "jpeg artifacts",
        "signature",
        "watermark",
        "username",
        "artist name",
        "ugly",
        "symbol",
        "hieroglyph",
        "six fingers per hand",
        "four fingers per hand",
        "disfigured hand",
        "monochrome",
        "missing limb",
        "linked limb",
        "connected limb",
        "interconnected limb",
        "broken finger",
        "broken hand",
        "broken wrist",
        "broken leg",
        "split limbs",
        "no thumb",
        "missing hand",
        "missing arms",
        "missing legs",
        "fused digit",
        "missing digit",
        "extra knee",
        "extra elbow",
        "storyboard",
        "split arms",
        "split hands",
        "split fingers",
        "twisted fingers",
        "disfigured butt",
        "deformed hands",
        "blurred faces",
        "irregular face",
        "irrregular body shape",
        "ugly eyes",
        "squint",
        "tiling",
        "poorly drawn hands",
        "poorly drawn feet",
        "poorly drawn face",
        "poorly framed",
        "body out of frame",
        "cut off",
        "draft",
        "grainy",
        "oversaturated",
        "teeth",
        "closed eyes",
        "weird neck",
        "long neck",
        "long body",
        "disgusting",
        "childish",
        "mutilated",
        "mangled",
        "surreal",
        "fuse",
        "off-center",
        "text",
        "logo",
        "letterbox",
        "bokeh",
        "multiple views",
        "multiple panels",
        "extra hands",
        "extra limbs",
        "mutated fingers",
        "detached arm",
        "liquid hand",
        "inverted hand",
        "oversized head",
        "three hands",
        "three legs",
        "bad arms",
        "three crus",
        "extra crus",
        "fused crus",
        "worst feet",
        "three feet",
        "fused feet",
        "fused thigh",
        "three thigh",
        "extra thigh",
        "worst thigh",
        "ugly fingers",
        "horn",
        "realistic photo",
        "extra eyes",
        "huge eyes",
        "2girl",
        "2boy",
        "dehydrated",
        "morbid",
        "mutation text",
        "score_6, score_5, score_4",
        "3d, render, cgi, doll, cartoon, illustration, painting, digital art, anime, fake, 3d modeling, old, bad anatomy, bad proportions, asymmetrical features, disfigured, deformed, malformed, unrealistic proportions, mutated, unnatural textures, fused fingers, extra limbs, extra fingers, distorted, twisted limbs, malformed hands, bad hands, bad fingers, bad eyes, bad teeth, blurry",
        "split image, out of frame, amputee, mutated, mutation, deformed, severed, dismembered, corpse, photograph, poorly drawn, bad anatomy, blur, blurry, lowres, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, ugly, symbol, hieroglyph, extra fingers, six fingers per hand, four fingers per hand, disfigured hand, monochrome, missing limb, disembodied limb, linked limb, connected limb, interconnected limb, broken finger, broken hand, broken wrist, broken leg, split limbs, no thumb, missing hand, missing arms, missing legs, fused finger, fused digit, missing digit, bad digit, extra knee, extra elbow, storyboard, split arms, split hands, split fingers, twisted fingers, disfigured butt",
        "deformed hands, watermark, text, deformed fingers, blurred faces, irregular face, irrregular body shape, ugly eyes, deformed face, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
        "cartoon, black and white photo, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal",
        "bad anatomy, fuze, ugly eyes, imperfections, bad finger, off-center, interlocked fingers, text, logo, watermark, signature, letterbox, bokeh, blurry, multiple views, multiple panels, missing limbs, missing fingers, deformed, cropped, extra hands, extra fingers, too many fingers, fused fingers, bad arm, monochrome, distorted arm, extra arms, malformed hands, poorly drawn hands, mutated fingers, extra limbs, poorly drawn face, artist name, fused arms, extra legs, missing leg, disembodied leg, detached arm, liquid hand, inverted hand, disembodied limb, oversized head",
        "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs",
        "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, out of frame double, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, 2boy, amputation, disconnected limbs",
        "mutation, deformed, deformed iris, duplicate, morbid, mutilated, disfigured, poorly drawn hand, poorly drawn face, bad proportions, gross proportions, extra limbs, cloned face, long neck, malformed limbs, missing arm, missing leg, extra arm, extra leg, fused fingers, too many fingers, extra fingers, mutated hands, blurry, bad anatomy, out of frame, contortionist, contorted limbs, exaggerated features, disproportionate, twisted posture, unnatural pose, disconnected, disproportionate, warped, misshapen, out of scale",
        "3 or 4 ears, never BUT ONE EAR, blurry, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, mangled teeth, weird teeth, poorly drawn eyes, blurry eyes, tan skin, oversaturated, teeth, poorly drawn, ugly, closed eyes, 3D, weird neck, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused finger",
        "amateur, amateur drawing, bad anatomy, bad art, bad aspect ratio, bad color, bad coloring, bad composition, bad contrast, bad crop, bad drawing, bad image, bad lighting, bad lineart, bad perspective, bad photoshop, bad pose, bad proportions, bad shading, bad sketch, bad trace, bad vector, beginner, black and white, broken anatomy, broken pose, clashing styles, color error, color issues, color mismatch, deformed, dirty art, disfigured, displeasing, distorted, distorted proportions, drawing, dubious anatomy, duplicate, early, exaggerated limbs, exaggerated pose, flat colors, gro",
        "aliasing, anatomy error, anatomy mistake, artifact, artifacts, broken anatomy, broken pose, camera aberration, chromatic aberration, clashing styles, cloned face, color banding, color error, color issues, color mismatch, compression artifacts, corrupted, cribbed from, cropped, deformed, dirty scan, disfigured, distorted, downsampling, draft, dubious anatomy, emoji, error, exaggerated limbs, exaggerated pose, extra arms, extra digits, extra fingers, extra legs, extra limbs, faded lines, filter abuse, fused fingers, gradient background, grainy, heavily compressed, heavily pixelated, high noise",
        "ai-generated, artifact, artifacts, bad quality, bad scan, blurred, blurry, compressed, compression artifacts, corrupted, dirty art scan, dirty scan, dithering, downsampling, faded lines, frameborder, grainy, heavily compressed, heavily pixelated, high noise, image noise, low dpi, low fidelity, low resolution, lowres, moire pattern, moiré pattern, motion blur, muddy colors, noise, noisy background, overcompressed, pixelation, pixels, poor quality, poor lineart, scanned with errors, scan artifact, scan errors, very low quality, visible pixels BREAK amateur, amateur drawing, bad anatomy, bad art",
        "ugly, old, fat, slight, disgusting, Unflattering, Distorted, Poorly lit, Blurry, Grainy, Overexposed, Underexposed, Cluttered background, Distracti",
        "bad anatomy, bad proportions, blurry, cloned face, cropped, deformed, dehydrated, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, low-res, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation,text, signature, ugly, username, watermark, poorly drawn hands, worst quality",
        "extra fingers, fused fingers, long necks, missing arms, mutated hands, malformed limbs, bad anatomy",
        "mutations, merged features, gross proportions",
        "extra wings, disproportionate body parts, random glowing elements, plastic texture, over-saturated colors, deformed legs, asymmetry, misplaced shadows, wrong perspective, poorly blended details",
        "deformed, asymmetrical, extra eyes, extra limbs, worst quality, blurry, unnatural skin, bad anatomy",
        "deformed, asymmetrical, extra fingers, extra limbs, fused fingers, unrealistic proportions, bad anatomy",
        "deformed, asymmetrical, extra eyes, blurry, distorted, bad anatomy, disproportional, unrealistic skin",
        "blurry, pixelated, noisy, low resolution, JPEG artifacts, lack of focus, grainy textures, over-sharpened edges",
        "extra fingers, fused fingers, long necks, missing arms, mutated hands, malformed limbs, bad anatomy",
        "mutations, merged features, gross proportions",
        "extra breasts",
        "missing breasts",
        "deformed breasts",
        "plastic breasts",
        "fake breasts",
        "gravity-defying breasts",
        "extra nipples",
        "cloned face",
        "duplicate body parts",
        "censored",
        "clothing on nude body",
        "smooth featureless crotch",
        "dildo clipping through body",
        "merged holes",
        "plastic skin",
        "dry skin",
        "flat chest",
        "single snake",
        "few snakes",
        "bald spots",
        "horror monster",
    ]
)

# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def resolve_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR)
            .decode()
            .strip()
        )
    except Exception:
        return "dev"


APP_VERSION = resolve_version()

# ---------------------------------------------------------------------------
# Flask application & CORS
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
if CORS_ORIGINS and CORS_ORIGINS != "*":
    origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
    CORS(app, resources={r"/*": {"origins": origins}})
else:
    CORS(app)

app.config["APP_VERSION"] = APP_VERSION

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("havnai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(LOGS_DIR / "havnai.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(JSONFormatter())
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def log_event(message: str, level: str = "info", **extra: Any) -> None:
    LOGGER.log(getattr(logging, level.upper(), logging.INFO), message, extra=extra if extra else None)
    EVENT_LOGS.append({"timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "level": level, "message": message})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unix_now() -> float:
    return time.time()


def parse_timestamp(value: Any) -> float:
    if value is None:
        return unix_now()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "")).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return unix_now()
    return unix_now()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def resolve_weight(model_name: str, default: float = 1.0) -> float:
    return float(MODEL_WEIGHTS.get(model_name, default))


def check_join_token() -> bool:
    if not SERVER_JOIN_TOKEN:
        return True
    header_token = request.headers.get("X-Join-Token", "")
    query_token = request.args.get("token", "")
    provided = header_token or query_token
    if provided != SERVER_JOIN_TOKEN:
        return False
    return True


def rate_limit(key: str, limit: int, per_seconds: int = 60) -> bool:
    now = unix_now()
    window_start = now - per_seconds
    bucket = RATE_LIMIT_BUCKETS.setdefault(key, deque())
    while bucket and bucket[0] < window_start:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    for directory in (STATIC_DIR, LEGACY_STATIC_DIR, MANIFESTS_DIR, LOGS_DIR, OUTPUTS_DIR, BASE_DIR / "db"):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

# ---------------------------------------------------------------------------
# WAN 2.2 GGUF video worker
# ---------------------------------------------------------------------------

VIDEO_ENGINE = VideoEngine()
VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("WAN_W2V_WORKERS", "1")) or 1)
VIDEO_JOBS: Dict[str, Dict[str, Any]] = {}
VIDEO_JOB_LOCK = threading.Lock()


def submit_video_job(job_request: VideoJobRequest) -> str:
    """Queue a WAN GGUF video generation job."""

    job_id = job_request.job_id or f"wan-video-{uuid.uuid4().hex[:8]}"
    job_request.job_id = job_id
    mode = "i2v" if job_request.init_image_b64 else "t2v"
    record = {
        "job_id": job_id,
        "status": "queued",
        "prompt": job_request.prompt,
        "negative_prompt": job_request.negative_prompt,
        "mode": mode,
        "loras": job_request.lora_list,
        "motion_type": job_request.motion_type,
        "duration": job_request.duration,
        "fps": job_request.fps,
        "created_at": iso_now(),
        "frames_dir": str(VIDEO_ENGINE.registry.paths.frames_dir),
        "videos_dir": str(VIDEO_ENGINE.registry.paths.videos_dir),
    }
    with VIDEO_JOB_LOCK:
        VIDEO_JOBS[job_id] = record
    VIDEO_EXECUTOR.submit(_run_video_job, job_id, job_request)
    return job_id


def _run_video_job(job_id: str, job_request: VideoJobRequest) -> None:
    with VIDEO_JOB_LOCK:
        if job_id in VIDEO_JOBS:
            VIDEO_JOBS[job_id]["status"] = "running"
            VIDEO_JOBS[job_id]["started_at"] = iso_now()
    result = VIDEO_ENGINE.generate(job_request)
    with VIDEO_JOB_LOCK:
        payload = VIDEO_JOBS.get(job_id, {})
        payload["status"] = result.status
        payload["completed_at"] = iso_now()
        payload["video_path"] = result.video_path
        payload["frame_paths"] = result.frame_paths
        payload["error"] = result.error
        payload["metadata"] = result.metadata
        VIDEO_JOBS[job_id] = payload


def get_video_job(job_id: str) -> Optional[Dict[str, Any]]:
    with VIDEO_JOB_LOCK:
        job = VIDEO_JOBS.get(job_id)
        return dict(job) if job else None

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    # Use a per-request connection when inside a Flask app context to avoid
    # concurrent cursor reuse across threads. Fallback to a module-level
    # connection during module/init when no app context is present.
    if has_app_context():
        conn = getattr(g, "db_conn", None)
        if conn is None:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            setattr(g, "db_conn", conn)
        return conn
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        DB_CONN.row_factory = sqlite3.Row
    return DB_CONN

@app.teardown_appcontext
def _close_db_conn(exception: Optional[BaseException]) -> None:
    conn = getattr(g, "db_conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            try:
                delattr(g, "db_conn")
            except Exception:
                pass


def init_db() -> None:
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            model TEXT NOT NULL,
            data TEXT,
            task_type TEXT NOT NULL,
            weight REAL NOT NULL,
            status TEXT NOT NULL,
            node_id TEXT,
            timestamp REAL NOT NULL,
            assigned_at REAL,
            completed_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            reward_hai REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS node_wallets (
            node_id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            node_name TEXT,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("UPDATE jobs SET status='queued', node_id=NULL WHERE status='running'")
    conn.commit()


init_db()

# Optional: clear database and in-memory state on startup for a fresh dashboard
if RESET_ON_STARTUP:
    try:
        # Define below in Admin utilities; forward declare via function placeholder pattern
        pass
    except Exception:
        # If the helper is not yet defined, we'll perform the reset after definitions
        pass

# ---------------------------------------------------------------------------
# Manifest bootstrap
# ---------------------------------------------------------------------------


def load_manifest() -> None:
    global MANIFEST_MODELS
    MANIFEST_MODELS = {}
    if not MANIFEST_FILE.exists():
        LOGGER.warning("Manifest file missing at %s", MANIFEST_FILE)
        return
    try:
        raw = json.loads(MANIFEST_FILE.read_text())
    except Exception as exc:
        LOGGER.error("Failed to parse manifest: %s", exc)
        return
    models = raw.get("models", []) if isinstance(raw, dict) else []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        pipeline = str(entry.get("pipeline", "")).strip().lower()
        path_str = str(entry.get("path", "")).strip()
        artifact_type = str(entry.get("type", "")).strip() or "checkpoint"
        weight = float(entry.get("weight", MODEL_WEIGHTS.get(key, 10.0)))
        MODEL_WEIGHTS[key] = weight
        entry_data = {
            "name": name,
            "pipeline": pipeline or "sd15",
            "path": path_str,
            "type": artifact_type,
            "tags": entry.get("tags", []),
            "reward_weight": weight,
            "task_type": entry.get("task_type", CREATOR_TASK_TYPE),
            "strengths": entry.get("strengths"),
            "weaknesses": entry.get("weaknesses"),
        }
        MANIFEST_MODELS[key] = entry_data
        MODEL_STATS.setdefault(key, {"count": 0.0, "total_time": 0.0})
    stale_stats = set(MODEL_STATS.keys()) - set(MANIFEST_MODELS.keys())
    for name in stale_stats:
        MODEL_STATS.pop(name, None)


load_manifest()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def refresh_manifest() -> None:
    load_manifest()

# ---------------------------------------------------------------------------
# Node persistence (nodes.json + wallet bindings)
# ---------------------------------------------------------------------------


def load_nodes() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return {}
    with REGISTRY_FILE.open() as f:
        data = json.load(f)
    now = unix_now()
    for node in data.values():
        node.setdefault("os", "unknown")
        node.setdefault("gpu", {})
        node.setdefault("pipelines", [])
        node.setdefault("models", [])
        node.setdefault("rewards", 0.0)
        node.setdefault("utilization", 0.0)
        node.setdefault("avg_utilization", node.get("utilization", 0.0))
        node.setdefault("tasks_completed", 0)
        node.setdefault("current_task", None)
        node.setdefault("last_result", {})
        node.setdefault("reward_history", [])
        node.setdefault("last_reward", 0.0)
        node.setdefault("start_time", now)
        node.setdefault("last_seen", iso_now())
        node.setdefault("role", node.get("role", "worker"))
        node.setdefault("node_name", node.get("node_name") or node.get("node_id"))
        node.setdefault("wallet", node.get("wallet"))
        node["last_seen_unix"] = parse_timestamp(node.get("last_seen"))
    return data


def save_nodes() -> None:
    payload = {}
    for node_id, info in NODES.items():
        serial = dict(info)
        serial.pop("last_seen_unix", None)
        payload[node_id] = serial
    with REGISTRY_FILE.open("w") as f:
        json.dump(payload, f, indent=2)


NODES = load_nodes()


def load_node_wallets() -> None:
    conn = get_db()
    rows = conn.execute("SELECT node_id, wallet, node_name FROM node_wallets").fetchall()
    for row in rows:
        node = NODES.setdefault(row["node_id"], {
            "os": "unknown",
            "gpu": {},
            "rewards": 0.0,
            "utilization": 0.0,
            "avg_utilization": 0.0,
            "tasks_completed": 0,
            "current_task": None,
            "last_result": {},
            "reward_history": [],
            "last_reward": 0.0,
            "start_time": unix_now(),
            "role": "worker",
            "last_seen": iso_now(),
        })
        node["wallet"] = row["wallet"]
        if row["node_name"]:
            node["node_name"] = row["node_name"]


load_node_wallets()
log_event(f"Telemetry online with {len(NODES)} cached node(s).", version=APP_VERSION)

# ---------------------------------------------------------------------------
# Job + reward helpers
# ---------------------------------------------------------------------------


def enqueue_job(wallet: str, model: str, task_type: str, data: str, weight: float) -> str:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    task_type = (task_type or CREATOR_TASK_TYPE).upper()
    conn = get_db()
    conn.execute(
        """
        INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
        """,
        (job_id, wallet, model, data, task_type, float(weight), unix_now()),
    )
    conn.commit()
    return job_id


def compute_reward(
    model_name: str,
    pipeline: str,
    metrics: Dict[str, Any],
    status: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute dynamic reward for a completed job.

    Formula:
        reward = base_reward * weight_factor * compute_cost_factor
                  * runtime_factor * success_factor
    """

    try:
        base_reward = float(REWARD_CONFIG.get("base_reward", 0.05))
        # Weight factor based on manifest/model weight
        model_weight = resolve_weight(model_name, 10.0)
        weight_factor = model_weight / 10.0

        # Compute-cost factor based on pipeline family
        pipeline_norm = (pipeline or "sd15").lower()
        if pipeline_norm == "sdxl":
            compute_cost_factor = float(REWARD_CONFIG.get("sdxl_factor", 1.5))
        elif pipeline_norm == "sd15":
            compute_cost_factor = float(REWARD_CONFIG.get("sd15_factor", 1.0))
        elif pipeline_norm in {"anime", "cartoon"}:
            compute_cost_factor = float(REWARD_CONFIG.get("anime_factor", 0.7))
        else:
            compute_cost_factor = 1.0

        # Runtime factor from actual runtime vs baseline
        baseline_runtime = float(REWARD_CONFIG.get("baseline_runtime", 8.0)) or 8.0
        runtime_sec = 0.0
        inf_ms = metrics.get("inference_time_ms")
        if isinstance(inf_ms, (int, float)) and inf_ms > 0:
            runtime_sec = float(inf_ms) / 1000.0
        else:
            dur = metrics.get("duration")
            if isinstance(dur, (int, float)) and dur > 0:
                runtime_sec = float(dur)
        runtime_sec = max(0.0, runtime_sec)
        runtime_factor = max(1.0, runtime_sec / baseline_runtime) if baseline_runtime > 0 else 1.0

        # Success / failure factor
        status_norm = (status or "").lower()
        success_factor = 1.0 if status_norm == "success" else 0.0

        reward = base_reward * weight_factor * compute_cost_factor * runtime_factor * success_factor
        reward = round(float(reward), 6)

        factors = {
            "base_reward": base_reward,
            "weight_factor": weight_factor,
            "compute_cost_factor": compute_cost_factor,
            "runtime_factor": runtime_factor,
            "success_factor": success_factor,
            "runtime_seconds": runtime_sec,
            "model_weight": model_weight,
            "pipeline": pipeline_norm,
            # TODO: future quality verification boost
            # "quality_factor": 1.0,
        }
        return reward, factors
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Reward computation failed for %s: %s", model_name, exc)
        return 0.0, {
            "base_reward": float(REWARD_CONFIG.get("base_reward", 0.05)),
            "weight_factor": 0.0,
            "compute_cost_factor": 1.0,
            "runtime_factor": 1.0,
            "success_factor": 0.0,
            "runtime_seconds": 0.0,
            "error": str(exc),
        }


def fetch_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC").fetchall()
    node = NODES.get(node_id, {})
    role = node.get("role", "worker")
    node_supports = {s.lower() for s in node.get("supports", []) if isinstance(s, str)}
    for row in rows:
        task_type = (row["task_type"] or CREATOR_TASK_TYPE).upper()
        # Support standard IMAGE_GEN, WAN video jobs, and AnimateDiff video jobs.
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF"}:
            continue
        if role != "creator":
            continue
        required_support = "image"
        if task_type in {"VIDEO_GEN", "ANIMATEDIFF"}:
            required_support = "animatediff" if task_type == "ANIMATEDIFF" else "image"
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


def assign_job_to_node(job_id: str, node_id: str) -> None:
    conn = get_db()
    conn.execute("UPDATE jobs SET status='running', node_id=?, assigned_at=? WHERE id=?", (node_id, unix_now(), job_id))
    conn.commit()


def complete_job(job_id: str, status: str) -> None:
    conn = get_db()
    conn.execute("UPDATE jobs SET status=?, completed_at=? WHERE id=?", (status, unix_now(), job_id))
    conn.commit()


def record_reward(wallet: Optional[str], task_id: str, reward: float) -> None:
    if not wallet:
        return
    conn = get_db()
    conn.execute(
        """
        INSERT INTO rewards (wallet, task_id, reward_hai, timestamp)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            wallet=excluded.wallet,
            reward_hai=excluded.reward_hai,
            timestamp=excluded.timestamp
        """,
        (wallet, task_id, reward, unix_now()),
    )
    conn.commit()


def get_job_summary(limit: int = 50) -> Dict[str, Any]:
    conn = get_db()
    queued = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='queued' AND UPPER(task_type)=?",
        (CREATOR_TASK_TYPE,),
    ).fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='running' AND UPPER(task_type)=?",
        (CREATOR_TASK_TYPE,),
    ).fetchone()[0]
    total_distributed = conn.execute("SELECT COALESCE(SUM(reward_hai),0) FROM rewards").fetchone()[0]
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    # Count jobs that have finished today (either legacy 'completed' or
    # newer 'success' status values).
    completed_today = conn.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE status IN ('completed', 'success')
          AND completed_at IS NOT NULL
          AND completed_at >= ?
          AND UPPER(task_type)=?
        """,
        (today_start, CREATOR_TASK_TYPE),
    ).fetchone()[0]
    # Avoid binding LIMIT to sidestep driver quirks under concurrency; limit is internal and cast to int.
    limit_int = int(limit)
    rows = conn.execute(
        f"""
        SELECT jobs.id, jobs.wallet, jobs.model, jobs.task_type, jobs.status, jobs.weight,
               jobs.completed_at, jobs.timestamp, rewards.reward_hai
        FROM jobs
        LEFT JOIN rewards ON rewards.task_id = jobs.id
        WHERE UPPER(jobs.task_type)=?
        ORDER BY jobs.timestamp DESC
        LIMIT {limit_int}
        """,
        (CREATOR_TASK_TYPE,),
    ).fetchall()
    feed = []
    for row in rows:
        completed_at = row["completed_at"]
        completed_iso = (
            datetime.fromtimestamp(completed_at, timezone.utc).isoformat().replace("+00:00", "Z")
            if completed_at
            else None
        )
        image_filename = f"{row['id']}.png"
        image_path = OUTPUTS_DIR / image_filename
        has_image = image_path.exists()
        image_url = f"/static/outputs/{image_filename}" if has_image else None
        output_path = str(image_path) if has_image else None
        timestamp_value = row["timestamp"]
        submitted_iso = (
            datetime.fromtimestamp(timestamp_value, timezone.utc).isoformat().replace("+00:00", "Z")
            if timestamp_value
            else None
        )
        reward_value = float(row["reward_hai"] or 0.0)
        feed.append(
            {
                "job_id": row["id"],
                "wallet": row["wallet"],
                "model": row["model"],
                "task_type": row["task_type"],
                "status": (row["status"] or "").upper(),
                "weight": float(row["weight"] or MODEL_WEIGHTS.get(row["model"], 1.0)),
                "reward": round(reward_value, 6),
                "reward_hai": round(reward_value, 6),
                "completed_at": completed_iso,
                "submitted_at": submitted_iso,
                "image_url": image_url,
                "output_path": output_path,
            }
        )
    return {
        "queued_jobs": queued,
        "active_jobs": active,
        "total_distributed": round(total_distributed or 0.0, 6),
        "jobs_completed_today": completed_today,
        "feed": feed,
    }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    model_name = (model_name or "").lower()
    entry = MANIFEST_MODELS.get(model_name)
    if not entry:
        return None
    return dict(entry)


def build_models_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for name, meta in MANIFEST_MODELS.items():
        catalog.append(
            {
                "model": name,
                "weight": resolve_weight(name, meta.get("reward_weight", 5.0)),
                "source": "manifest",
                "nodes": [],
                "tags": meta.get("tags", []),
                "size": 0,
            }
        )
    return catalog


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def pending_tasks_for_node(node_id: str) -> List[Dict[str, Any]]:
    relevant_status = {"pending", "assigned"}
    tasks = []
    for task in TASKS.values():
        if task.get("assigned_to") != node_id:
            continue
        if task.get("status") not in relevant_status:
            continue
        task_type = (task.get("task_type") or CREATOR_TASK_TYPE).upper()
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF"}:
            continue
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Leaderboard helpers
# ---------------------------------------------------------------------------


def leaderboard_rows(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_db()
    all_time_rows = conn.execute(
        "SELECT wallet, SUM(reward_hai) AS total, COUNT(*) AS jobs FROM rewards GROUP BY wallet"
    ).fetchall()
    totals = {row["wallet"]: float(row["total"] or 0) for row in all_time_rows}
    job_counts = {row["wallet"]: row["jobs"] for row in all_time_rows}

    cutoff = unix_now() - 86400
    last24_rows = conn.execute(
        "SELECT wallet, SUM(reward_hai) AS total FROM rewards WHERE timestamp >= ? GROUP BY wallet",
        (cutoff,),
    ).fetchall()
    last24 = {row["wallet"]: float(row["total"] or 0) for row in last24_rows}

    wallet_nodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows = conn.execute("SELECT wallet, node_id, node_name FROM node_wallets").fetchall()
    for row in rows:
        node_id = row["node_id"]
        node = NODES.get(node_id, {})
        wallet_nodes[row["wallet"]].append(
            {
                "node_id": node_id,
                "node_name": row["node_name"] or node.get("node_name") or node_id,
                "role": node.get("role", "worker"),
            }
        )

    leaderboard = []
    for wallet, total in totals.items():
        nodes = wallet_nodes.get(wallet, [])
        creator = any(node.get("role") == "creator" for node in nodes)
        leaderboard.append(
            {
                "wallet": wallet,
                "wallet_short": wallet[:6] + "…" + wallet[-4:] if wallet else "—",
                "all_time": round(total, 6),
                "jobs": job_counts.get(wallet, 0),
                "rewards_24h": round(last24.get(wallet, 0.0), 6),
                "nodes": nodes,
                "creator": creator,
            }
        )
    leaderboard.sort(key=lambda row: row["all_time"], reverse=True)
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx
        if idx > limit:
            break
    return leaderboard[:limit]


# ---------------------------------------------------------------------------
# Routes – public pages
# ---------------------------------------------------------------------------


@app.route("/health")
def health() -> Any:
    job_summary = get_job_summary()
    return jsonify(
        {
            "status": "ok",
            "nodes": len(NODES),
            "queue_depth": job_summary["queued_jobs"],
            "version": APP_VERSION,
        }
    )


@app.route("/join")
def join_page() -> Any:
    host = request.host_url.rstrip("/")
    token_hint = f" --token {SERVER_JOIN_TOKEN}" if SERVER_JOIN_TOKEN else ""
    install_cmd = f"curl -fsSL {host}/installers/install-node.sh | bash -s -- --server {host}{token_hint}"
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Join HavnAI Grid</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#03121b; color:#e6f6ff; padding:2.5rem; }}
        code {{ background:#07202d; padding:0.25rem 0.4rem; border-radius:4px; }}
        pre {{ background:#07202d; padding:1rem; border-radius:8px; overflow:auto; }}
        a {{ color:#5fe4ff; }}
        h1 {{ color:#5fe4ff; }}
        ul {{ line-height:1.6; }}
      </style>
    </head>
    <body>
      <h1>Join the HavnAI GPU Grid</h1>
      <p>Run the installer on your GPU machine:</p>
      <pre><code>{install_cmd}</code></pre>
      <h2>Prerequisites</h2>
      <ul>
        <li>64-bit Linux (Ubuntu/Debian/RHEL) or macOS (12+)</li>
        <li>Python 3.10 or newer, curl, and a GPU driver/runtime</li>
        <li>$HAI wallet address (EVM compatible)</li>
      </ul>
      <h2>Optional – WAN I2V Video Support</h2>
      <ul>
        <li><code>ffmpeg</code> installed and on <code>PATH</code> (e.g. <code>sudo apt-get install -y ffmpeg</code> on Debian/Ubuntu)</li>
        <li>WAN I2V safetensor checkpoint(s) downloaded to the paths referenced by the coordinator manifest (for example <code>/mnt/d/havnai-storage/models/video/wan-i2v/wan_2.2_lightning.safetensors</code>)</li>
        <li><code>CREATOR_MODE=true</code> in <code>~/.havnai/.env</code> if you want this node to accept WAN video jobs</li>
      </ul>
      <h2>What happens next?</h2>
      <ol>
        <li>Installer prepares <code>~/.havnai</code>, Python venv, and the node binary</li>
        <li>Configure your wallet inside <code>~/.havnai/.env</code></li>
        <li>Enable the service (systemd or launchd)</li>
        <li>Monitor progress via <a href=\"/dashboard\">dashboard</a> and <a href=\"/network/leaderboard\">leaderboard</a></li>
      </ol>
      <p>Need the join token or help? Contact the grid operator.</p>
    </body>
    </html>
    """
    return html


@app.route("/network/leaderboard")
def leaderboard() -> Any:
    data = leaderboard_rows()
    if request.args.get("format") == "json":
        return jsonify({"leaderboard": data})

    rows_html = "".join(
        f"<tr><td>{row['rank']}</td><td>{', '.join(node['node_name'] for node in row['nodes']) or '—'}</td>"
        f"<td>{row['wallet_short']}</td><td>{row['jobs']}</td><td>{row['rewards_24h']:.6f}</td>"
        f"<td>{row['all_time']:.6f}</td><td>{'✅' if row['creator'] else ''}</td></tr>"
        for row in data
    )
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>HavnAI Leaderboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#03121b; color:#e6f6ff; padding:2rem; }}
        table {{ width:100%; border-collapse:collapse; margin-top:1.5rem; }}
        th, td {{ padding:0.75rem 1rem; border-bottom:1px solid rgba(95,228,255,0.15); }}
        th {{ text-transform:uppercase; font-size:0.75rem; letter-spacing:0.08em; color:#5fe4ff; }}
        a {{ color:#5fe4ff; }}
      </style>
    </head>
    <body>
      <h1>HavnAI Network Leaderboard</h1>
      <p><a href=\"/dashboard\">Back to dashboard</a> · <a href=\"/join\">Join the grid</a></p>
      <table>
        <thead><tr><th>Rank</th><th>Node Name(s)</th><th>Wallet</th><th>Jobs</th><th>24h Rewards</th><th>All-Time</th><th>Creator</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# Client asset helpers
# ---------------------------------------------------------------------------


@app.route("/client/download")
def client_download() -> Any:
    return send_file(CLIENT_PATH, as_attachment=True, download_name="havnai_client.py")


@app.route("/client/registry.py")
def client_registry_module() -> Any:
    if not CLIENT_REGISTRY.exists():
        return jsonify({"error": "registry module not available"}), 404
    return send_file(CLIENT_REGISTRY, as_attachment=True, download_name="registry.py")


@app.route("/client/requirements")
def client_requirements() -> Any:
    if not CLIENT_REQUIREMENTS.exists():
        return jsonify({"error": "requirements not available"}), 404
    return send_file(CLIENT_REQUIREMENTS, as_attachment=True, download_name="requirements-node.txt")


@app.route("/client/version")
def client_version() -> Any:
    return APP_VERSION, 200, {"Content-Type": "text/plain"}


@app.route("/favicon.ico")
def favicon() -> Any:
    """Serve favicon if present; otherwise return empty 204 to avoid 404 spam."""
    icon_path = STATIC_DIR / "favicon.ico"
    if icon_path.exists():
        return send_from_directory(STATIC_DIR, "favicon.ico", mimetype="image/x-icon")
    return ("", 204)


# ---------------------------------------------------------------------------
# API routes – manifest catalog only
# ---------------------------------------------------------------------------


@app.route("/models/list")
def list_models() -> Any:
    load_manifest()
    return jsonify({"models": list(MANIFEST_MODELS.values())})


@app.route("/installers/<path:filename>")
def installer_assets(filename: str) -> Any:
    safe_path = (LEGACY_STATIC_DIR / filename).resolve()
    try:
        safe_path.relative_to(LEGACY_STATIC_DIR.resolve())
    except ValueError:
        abort(404)
    if not safe_path.exists() or not safe_path.is_file():
        abort(404)
    mimetype = "text/x-shellscript" if safe_path.suffix == ".sh" else None
    return send_from_directory(LEGACY_STATIC_DIR, filename, mimetype=mimetype)


# ---------------------------------------------------------------------------
# API routes – jobs & nodes
# ---------------------------------------------------------------------------


@app.route("/submit-job", methods=["POST"])
def submit_job() -> Any:
    if not rate_limit(f"submit-job:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    payload = request.get_json() or {}
    wallet = str(payload.get("wallet", "")).strip()
    model_name_raw = str(payload.get("model", "")).strip()
    model_name = model_name_raw.lower()
    weight = payload.get("weight")

    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    # Allow "auto" model selection based on manifest weights
    if not model_name or model_name in {"auto", "auto_image", "auto-image"}:
        load_manifest()
        # Candidate models: creator IMAGE_GEN only
        candidates = [
            meta
            for key, meta in MANIFEST_MODELS.items()
            if (meta.get("task_type") or CREATOR_TASK_TYPE).upper() == CREATOR_TASK_TYPE
        ]
        if not candidates:
            return jsonify({"error": "no_creator_models"}), 400
        names = [meta["name"] for meta in candidates]
        weights = [resolve_weight(meta["name"].lower(), meta.get("reward_weight", 10.0)) for meta in candidates]
        chosen = random.choices(names, weights=weights, k=1)[0]
        model_name_raw = chosen
        model_name = chosen.lower()

    if not model_name:
        return jsonify({"error": "missing model"}), 400

    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400

    if weight is None:
        weight = cfg.get("reward_weight", resolve_weight(model_name, 10.0))

    # Special-case WAN I2V video jobs so we preserve structured settings
    is_wan_i2v = model_name == "wan-i2v" or str(cfg.get("pipeline", "")).lower() == "wan-i2v"

    # Special-case AnimateDiff video jobs with rich structured payload
    is_animatediff = model_name == "animatediff" or str(cfg.get("pipeline", "")).lower() == "animatediff"

    if is_wan_i2v:
        # Persist all WAN-specific controls inside the job data blob as JSON
        prompt_text = str(payload.get("prompt") or "")
        settings: Dict[str, Any] = {
            "prompt": prompt_text,
            "init_image": payload.get("init_image") or None,
            "steps_high": int(payload.get("steps_high", 2)),
            "steps_low": int(payload.get("steps_low", 2)),
            "sampler": str(payload.get("sampler") or "euler"),
            "cfg": float(payload.get("cfg", 1.0)),
            "num_frames": int(payload.get("num_frames", 32)),
            "fps": int(payload.get("fps", 24)),
            "resolution": str(payload.get("resolution") or "720x512"),
        }
        job_data = json.dumps(settings)
        task_type = "VIDEO_GEN"
    elif is_animatediff:
        prompt_text = str(payload.get("prompt") or "")
        negative_prompt = str(payload.get("negative_prompt") or "")
        # Core AnimateDiff controls – validated/coerced into safe ranges
        try:
            frames = int(payload.get("frames", 16))
        except (TypeError, ValueError):
            frames = 16
        try:
            fps = int(payload.get("fps", 8))
        except (TypeError, ValueError):
            fps = 8
        frames = max(1, min(frames, 64))
        fps = max(1, min(fps, 60))

        motion = str(payload.get("motion") or "").strip().lower() or "zoom-in"
        base_model = str(payload.get("base_model") or "realisticVision")
        width = int(payload.get("width", 512) or 512)
        height = int(payload.get("height", 512) or 512)
        # Clamp to supported grid
        if width not in {512, 768}:
            width = 512
        if height not in {512, 768}:
            height = 512

        seed = payload.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        lora_strength = payload.get("lora_strength")
        try:
            lora_strength = float(lora_strength) if lora_strength is not None else None
        except (TypeError, ValueError):
            lora_strength = None
        init_image = payload.get("init_image") or None
        scheduler = str(payload.get("scheduler") or "DDIM").upper()
        if scheduler not in {"DDIM", "EULER", "HEUN"}:
            scheduler = "DDIM"

        settings = {
            "prompt": prompt_text,
            "negative_prompt": negative_prompt,
            "frames": frames,
            "fps": fps,
            "motion": motion,
            "base_model": base_model,
            "width": width,
            "height": height,
            "seed": seed,
            "lora_strength": lora_strength,
            "init_image": init_image,
            "scheduler": scheduler,
        }
        job_data = json.dumps(settings)
        task_type = "ANIMATEDIFF"
    else:
        prompt_text = str(payload.get("prompt") or payload.get("data") or "")
        if prompt_text:
            prompt_text = f"{prompt_text}, {GLOBAL_POSITIVE_SUFFIX}"
        else:
            prompt_text = GLOBAL_POSITIVE_SUFFIX
        negative_prompt = str(payload.get("negative_prompt") or "")
        job_settings: Dict[str, Any] = {"prompt": prompt_text}
        if negative_prompt:
            job_settings["negative_prompt"] = negative_prompt
        pose_image = payload.get("pose_image") or payload.get("pose_image_b64") or ""
        pose_image_path = payload.get("pose_image_path") or ""
        if pose_image:
            job_settings["pose_image_b64"] = str(pose_image)
        if pose_image_path:
            job_settings["pose_image_path"] = str(pose_image_path)

        # Per-model defaults from manifest (steps/guidance/size/sampler/negative)
        if cfg:
            if cfg.get("steps") is not None:
                job_settings["steps"] = cfg["steps"]
            if cfg.get("guidance") is not None:
                job_settings["guidance"] = cfg["guidance"]
            if cfg.get("width") is not None:
                job_settings["width"] = cfg["width"]
            if cfg.get("height") is not None:
                job_settings["height"] = cfg["height"]
            if cfg.get("sampler"):
                job_settings["sampler"] = cfg["sampler"]
            if not negative_prompt:
                base_negative = str(cfg.get("negative_prompt_default") or "").strip()
                combined_negative = ", ".join(filter(None, [base_negative, GLOBAL_NEGATIVE_PROMPT]))
                if combined_negative:
                    job_settings["negative_prompt"] = combined_negative

        job_data = json.dumps(job_settings)
        task_type = CREATOR_TASK_TYPE

    with LOCK:
        job_id = enqueue_job(wallet, cfg.get("name", model_name), task_type, job_data, float(weight))
    log_event("Public job queued", wallet=wallet, model=model_name, job_id=job_id)
    return jsonify({"status": "queued", "job_id": job_id}), 200


@app.route("/generate-video", methods=["POST"])
def generate_video() -> Any:
    """Trigger a WAN 2.2 GGUF T2V/I2V job via background worker."""

    data = request.get_json() or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    negative_prompt = str(data.get("negative_prompt", "")).strip()
    motion_type = str(data.get("motion_type", "high")).strip().lower() or "high"
    lora_list = data.get("lora_list") or []
    if not isinstance(lora_list, list):
        return jsonify({"error": "lora_list must be an array"}), 400

    duration = float(data.get("duration", 4.0) or 4.0)
    fps = int(data.get("fps", 24) or 24)
    fps = max(1, min(fps, 60))
    width = int(data.get("width", 720) or 720)
    height = int(data.get("height", 512) or 512)

    job_request = VideoJobRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        motion_type=motion_type,
        lora_list=[str(item) for item in lora_list],
        init_image_b64=data.get("init_image"),
        duration=duration,
        fps=fps,
        width=width,
        height=height,
    )
    job_id = submit_video_job(job_request)
    job = get_video_job(job_id) or {}
    response = {
        "job_id": job_id,
        "status": job.get("status", "queued"),
        "mode": job.get("mode"),
        "fps": fps,
        "duration": duration,
        "frames_dir": job.get("frames_dir"),
        "videos_dir": job.get("videos_dir"),
        "engine": VIDEO_ENGINE.describe(),
    }
    return jsonify(response), 202


@app.route("/generate-video/<job_id>", methods=["GET"])
def video_job_status(job_id: str) -> Any:
    job = get_video_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job), 200


@app.route("/register", methods=["POST"])
def register() -> Any:
    if not rate_limit(f"register:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json() or {}
    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with LOCK:
        node = NODES.get(node_id)
        if not node:
            node = {
                "os": data.get("os", "unknown"),
                "gpu": data.get("gpu", {}),
                "rewards": 0.0,
                "avg_utilization": 0.0,
                "utilization": data.get("utilization", 0.0),
                "tasks_completed": 0,
                "current_task": None,
                "last_result": {},
                "reward_history": [],
                "last_reward": 0.0,
                "start_time": data.get("start_time", unix_now()),
                "role": data.get("role", "worker"),
                "node_name": data.get("node_name") or node_id,
            }
            NODES[node_id] = node
            log_event("Node registered", node_id=node_id, role=node["role"], version=data.get("version"))

        node["os"] = data.get("os", node.get("os", "unknown"))
        node["gpu"] = data.get("gpu", node.get("gpu", {}))
        node["role"] = data.get("role", node.get("role", "worker"))
        node["node_name"] = data.get("node_name") or node.get("node_name", node_id)
        node["version"] = data.get("version", "dev")
        node["pipelines"] = data.get("pipelines", node.get("pipelines", []))
        node["models"] = data.get("models", node.get("models", []))
        node["supports"] = data.get("supports", node.get("supports", []))
        util = data.get("gpu", {}).get("utilization") if isinstance(data.get("gpu"), dict) else data.get("utilization")
        util = float(util or node.get("utilization", 0.0))
        node["utilization"] = util
        samples = node.setdefault("util_samples", [])
        samples.append(util)
        if len(samples) > 60:
            samples.pop(0)
        node["avg_utilization"] = round(sum(samples) / len(samples), 2) if samples else util
        node["last_seen"] = iso_now()
    node["last_seen_unix"] = unix_now()
    save_nodes()

    return jsonify({"status": "ok", "node": node_id}), 200


@app.route("/register_models", methods=["POST"])
def register_models() -> Any:
    """No-op endpoint to accept local model manifests from nodes."""

    payload = request.get_json() or {}
    node_id = payload.get("node_id")
    models = payload.get("models") or []
    count = len(models) if isinstance(models, list) else 0
    log_event("Models registered (noop)", node_id=node_id, count=count)
    return jsonify({"status": "ok", "registered": count}), 200


@app.route("/link-wallet", methods=["POST"])
def link_wallet() -> Any:
    if not rate_limit(f"link-wallet:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json() or {}
    node_id = data.get("node_id")
    wallet = data.get("wallet")
    node_name = data.get("node_name", node_id)
    if not node_id or not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid payload"}), 400

    with LOCK:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO node_wallets (node_id, wallet, node_name, updated_at) VALUES (?, ?, ?, ?)",
            (node_id, wallet, node_name, unix_now()),
        )
        conn.commit()
        node = NODES.setdefault(node_id, {
            "os": "unknown",
            "gpu": {},
            "rewards": 0.0,
            "utilization": 0.0,
            "avg_utilization": 0.0,
            "tasks_completed": 0,
            "current_task": None,
            "last_result": {},
            "reward_history": [],
            "last_reward": 0.0,
            "start_time": unix_now(),
            "role": "worker",
        })
        node["wallet"] = wallet
        node["node_name"] = node_name or node.get("node_name", node_id)
        save_nodes()
    log_event("Wallet linked", node_id=node_id, wallet=wallet)
    return jsonify({"status": "linked"}), 200


@app.route("/tasks/creator", methods=["GET"])
def get_creator_tasks() -> Any:
    node_id = request.args.get("node_id")
    if not node_id:
        return jsonify({"tasks": []}), 200

    with LOCK:
        node_info = NODES.get(node_id)
        if not node_info:
            return jsonify({"tasks": []}), 200

        pending = pending_tasks_for_node(node_id)
        if not pending:
            job = fetch_next_job_for_node(node_id)
            if job:
                cfg = get_model_config(job["model"])
                if cfg:
                    # Decode prompt/negative_prompt for standard IMAGE_GEN jobs stored as JSON
                    raw_data = job.get("data")
                    prompt_text = raw_data or ""
                    negative_prompt = ""
                    try:
                        parsed = json.loads(raw_data) if isinstance(raw_data, str) else None
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        prompt_text = str(parsed.get("prompt") or "")
                        negative_prompt = str(parsed.get("negative_prompt") or "")
                    # Always send plain prompt text to the node (avoid passing raw JSON)
                    prompt_for_node = prompt_text

                    # Assign under global lock to avoid multiple nodes claiming the same job
                    assign_job_to_node(job["id"], node_id)
                    log_event("Job claimed by node", job_id=job["id"], node_id=node_id)
                    reward_weight = float(job["weight"] or cfg.get("reward_weight", resolve_weight(job["model"], 10.0)))
                    job_task_type = (job.get("task_type") or CREATOR_TASK_TYPE).upper()
                    pending = [
                        {
                            "task_id": job["id"],
                            "task_type": job_task_type,
                            "model_name": job["model"],
                            "model_path": cfg.get("path", ""),
                            "pipeline": cfg.get("pipeline", "sd15"),
                            "input_shape": cfg.get("input_shape", []),
                            "reward_weight": reward_weight,
                            "status": "pending",
                            "wallet": job["wallet"],
                            "assigned_to": node_id,
                            "job_id": job["id"],
                            "data": raw_data,
                            "prompt": prompt_for_node,
                            "negative_prompt": negative_prompt,
                        }
                    ]
                    node_info["current_task"] = {
                        "task_id": job["id"],
                        "model_name": job["model"],
                        "status": "pending",
                            "task_type": pending[0]["task_type"],
                            "weight": pending[0]["reward_weight"],
                        }
                    save_nodes()
                else:
                    complete_job(job["id"], "failed")

        response_tasks = []
        for task in pending:
            if task["status"] == "pending":
                task["status"] = "assigned"
                task["assigned_at"] = unix_now()
            TASKS[task["task_id"]] = dict(task)
            pipeline = str(task.get("pipeline", "sd15")).lower()
            task_payload = {
                "task_id": task["task_id"],
                "type": task.get("task_type", CREATOR_TASK_TYPE),
                "model_name": task["model_name"],
                "model_path": task.get("model_path", ""),
                "pipeline": pipeline,
                "input_shape": task.get("input_shape", []),
                "reward_weight": task.get("reward_weight", 1.0),
                "wallet": task.get("wallet"),
                "prompt": task.get("prompt") or task.get("data", ""),
                "negative_prompt": task.get("negative_prompt") or "",
            }
            # If this is a WAN I2V video job, attempt to expose structured settings to the node
            if task_payload["type"].upper() == "VIDEO_GEN":
                try:
                    raw = task.get("data") or ""
                    settings = json.loads(raw) if isinstance(raw, str) else {}
                except Exception:
                    settings = {}
                if isinstance(settings, dict):
                    for key in ("num_frames", "fps", "steps_high", "steps_low", "cfg", "sampler", "resolution", "init_image"):
                        if key in settings:
                            task_payload[key] = settings[key]
            # If this is an AnimateDiff job, surface rich controls directly on the task payload
            if task_payload["type"].upper() == "ANIMATEDIFF":
                try:
                    raw_ad = task.get("data") or ""
                    ad_settings = json.loads(raw_ad) if isinstance(raw_ad, str) else {}
                except Exception:
                    ad_settings = {}
                if isinstance(ad_settings, dict):
                    for key in (
                        "prompt",
                        "negative_prompt",
                        "frames",
                        "fps",
                        "motion",
                        "base_model",
                        "width",
                        "height",
                        "seed",
                        "lora_strength",
                        "init_image",
                        "scheduler",
                    ):
                        if key in ad_settings and ad_settings[key] is not None:
                            task_payload[key] = ad_settings[key]
            response_tasks.append(task_payload)
    return jsonify({"tasks": response_tasks}), 200


@app.route("/tasks", methods=["GET"])
def tasks_alias() -> Any:
    return get_creator_tasks()


@app.route("/tasks/ai", methods=["GET"])
def tasks_ai_alias() -> Any:
    # Backward/alternate compatibility for clients polling /tasks/ai
    return get_creator_tasks()


@app.route("/results", methods=["POST"])
def submit_results() -> Any:
    data = request.get_json() or {}
    node_id = data.get("node_id")
    task_id = data.get("task_id")
    status = data.get("status", "unknown")
    metrics = data.get("metrics", {})
    utilization = data.get("utilization")
    image_b64 = data.get("image_b64")
    video_b64 = data.get("video_b64")

    if not node_id or not task_id:
        return jsonify({"error": "missing node_id or task_id"}), 400

    with LOCK:
        task = TASKS.get(task_id)
        if not task:
            job = get_job(task_id)
            if not job:
                return jsonify({"error": "task not found"}), 404
            task = {
                "task_id": job["id"],
                "task_type": job.get("task_type", CREATOR_TASK_TYPE),
                "model_name": job["model"],
                "reward_weight": job.get("weight", resolve_weight(job["model"], 10.0)),
                "wallet": job.get("wallet"),
                "prompt": job.get("data"),
            }

        task["status"] = status
        task["completed_at"] = unix_now()
        task["result"] = metrics

        node = NODES.get(node_id)
        reward = 0.0
        wallet = task.get("wallet")
        model_name = task.get("model_name", "creator-model")
        task_type = task.get("task_type", CREATOR_TASK_TYPE)
        # Determine pipeline family for reward computation
        cfg = get_model_config(model_name)
        pipeline = cfg.get("pipeline", "sd15") if cfg else "sd15"

        # Compute dynamic reward and factors
        reward, reward_factors = compute_reward(model_name, pipeline, metrics, status)

        # Validate job ownership/status in a transaction before completing
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            job_row = conn.execute("SELECT * FROM jobs WHERE id=?", (task_id,)).fetchone()
            if not job_row:
                conn.rollback()
                return jsonify({"error": "task not found"}), 404
            current_status = (job_row["status"] or "").lower()
            owner = job_row["node_id"]
            if current_status != "running" or (owner and owner != node_id):
                conn.rollback()
                log_event(
                    "Results rejected (job not running/owned by node)",
                    level="warning",
                    job_id=task_id,
                    node_id=node_id,
                    status=current_status,
                    owner=owner,
                )
                return jsonify({"error": "conflict"}), 409

            log_event("Results accepted for job (running)", job_id=task_id, node_id=node_id)

            if node:
                node["rewards"] = round(node.get("rewards", 0.0) + reward, 6)
                node["last_reward"] = reward
                history = node.setdefault("reward_history", [])
                history.append(
                    {
                        "reward": reward,
                        "task_id": task_id,
                        "timestamp": iso_now(),
                        "factors": reward_factors,
                    }
                )
                if len(history) > 20:
                    history.pop(0)
                node["last_seen"] = iso_now()
                node["last_seen_unix"] = unix_now()
                # If an image or video was uploaded, persist to outputs dir
                image_url = None
                video_url = None
                if image_b64:
                    try:
                        import base64
                        img_bytes = base64.b64decode(image_b64)
                        out_path = OUTPUTS_DIR / f"{task_id}.png"
                        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                        with out_path.open("wb") as fh:
                            fh.write(img_bytes)
                        image_url = f"/static/outputs/{task_id}.png"
                    except Exception:
                        pass
                if video_b64:
                    try:
                        import base64
                        video_bytes = base64.b64decode(video_b64)
                        videos_dir = OUTPUTS_DIR / "videos"
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_path = videos_dir / f"{task_id}.mp4"
                        with video_path.open("wb") as fh:
                            fh.write(video_bytes)
                        # Stored at: STATIC_DIR / outputs / videos / <job_id>.mp4
                        video_url = f"/static/outputs/videos/{task_id}.mp4"
                    except Exception:
                        pass

                node["last_result"] = {
                    "task_id": task_id,
                    "status": status,
                    "metrics": metrics,
                    "model_name": model_name,
                    "reward": reward,
                    "reward_factors": reward_factors,
                    "wallet": wallet,
                    "task_type": task_type,
                    "image_url": image_url,
                    "video_url": video_url,
                }
                if status == "success":
                    node["tasks_completed"] = node.get("tasks_completed", 0) + 1
                node["current_task"] = None
                if utilization is not None:
                    try:
                        util_val = float(utilization)
                        node["utilization"] = util_val
                        samples = node.setdefault("util_samples", [])
                        samples.append(util_val)
                        if len(samples) > 60:
                            samples.pop(0)
                        node["avg_utilization"] = round(sum(samples) / len(samples), 2)
                    except (TypeError, ValueError):
                        pass
                save_nodes()

            job = dict(job_row)
            # Persist reward info into the job's data JSON for later inspection.
            raw_data = job.get("data")
            try:
                payload = json.loads(raw_data) if isinstance(raw_data, str) else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                payload["reward"] = reward
                payload["reward_factors"] = reward_factors
                payload["reward_status"] = status
                try:
                    conn.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(payload), task_id))
                except Exception:
                    conn.rollback()
                    raise
            conn.execute("UPDATE jobs SET status=?, completed_at=? WHERE id=?", (status, unix_now(), task_id))
            conn.commit()
            wallet = wallet or job.get("wallet")
        except Exception:
            conn.rollback()
            raise

        record_reward(wallet, task_id, reward)

        stats = MODEL_STATS.setdefault(model_name, {"count": 0.0, "total_time": 0.0})
        if status == "success":
            inference_time = float(metrics.get("inference_time_ms", 0))
            if inference_time > 0:
                stats["count"] += 1
                stats["total_time"] += inference_time

        TASKS.pop(task_id, None)
    # Return a small payload including image and video URLs if saved
    resp_payload = {"status": "ok", "task_id": task_id, "reward": reward}
    if (OUTPUTS_DIR / f"{task_id}.png").exists():
        resp_payload["image_url"] = f"/static/outputs/{task_id}.png"
    videos_dir = OUTPUTS_DIR / "videos"
    if (videos_dir / f"{task_id}.mp4").exists():
        resp_payload["video_url"] = f"/static/outputs/videos/{task_id}.mp4"
    return jsonify(resp_payload), 200


@app.route("/disconnect", methods=["POST"])
def disconnect_node() -> Any:
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with LOCK:
        # Remove from in-memory registry
        existed = node_id in NODES
        if existed:
            NODES.pop(node_id, None)
            save_nodes()

        # Clean DB state for this node
        conn = get_db()
        try:
            # Release any running jobs back to queue
            conn.execute("UPDATE jobs SET status='queued', node_id=NULL, assigned_at=NULL WHERE node_id=? AND status='running'", (node_id,))
            # Remove node-specific metadata
            conn.execute("DELETE FROM node_wallets WHERE node_id=?", (node_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    log_event("Node disconnected", node_id=node_id)
    return jsonify({"status": "disconnected", "node": node_id, "existed": existed}), 200

    log_event(
        "Task completed",
        node_id=node_id,
        task_id=task_id,
        model=model_name,
        reward=reward,
        status=status,
    )
    return jsonify({"status": "received", "task": task_id, "reward": reward}), 200


@app.route("/rewards", methods=["GET"])
def rewards_endpoint() -> Any:
    job_summary = get_job_summary()
    with LOCK:
        rewards = {node_id: info.get("rewards", 0.0) for node_id, info in NODES.items()}
    return jsonify({"rewards": rewards, "total": job_summary["total_distributed"]})


@app.route("/jobs/recent", methods=["GET"])
def jobs_recent() -> Any:
    """Return recent creator jobs with reward information."""

    try:
        limit = int(request.args.get("limit", 25))
    except Exception:
        limit = 25
    summary = get_job_summary(limit=limit)
    return jsonify({"jobs": summary.get("feed", []), "summary": summary})


@app.route("/jobs/<job_id>", methods=["GET"])
def job_detail(job_id: str) -> Any:
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "job_not_found", "job_id": job_id}), 404
    conn = get_db()
    reward_row = conn.execute(
        "SELECT reward_hai, timestamp FROM rewards WHERE task_id=?",
        (job_id,),
    ).fetchone()
    reward_value = float(reward_row["reward_hai"]) if reward_row else 0.0
    reward_ts = reward_row["timestamp"] if reward_row else None

    raw_data = job.get("data")
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else {}
    except Exception:
        payload = {}
    reward_factors = payload.get("reward_factors") if isinstance(payload, dict) else None

    return jsonify(
        {
            "id": job.get("id"),
            "wallet": job.get("wallet"),
            "model": job.get("model"),
            "task_type": job.get("task_type"),
            "status": job.get("status"),
            "weight": job.get("weight"),
            "node_id": job.get("node_id"),
            "timestamp": job.get("timestamp"),
            "completed_at": job.get("completed_at"),
            "reward": reward_value,
            "reward_timestamp": reward_ts,
            "reward_factors": reward_factors,
            "data": payload,
        }
    )


@app.route("/nodes/<node_id>", methods=["GET"])
def node_detail(node_id: str) -> Any:
    with LOCK:
        node = NODES.get(node_id)
        if not node:
            return jsonify({"error": "node_not_found", "node_id": node_id}), 404
        payload = dict(node)
        payload["node_id"] = node_id
    return jsonify(payload)


@app.route("/api/models/stats", methods=["GET"])
def models_stats() -> Any:
    """
    Lightweight stats endpoint used by the public landing page.

    Returns:
        {
          "active_nodes": int,
          "jobs_completed_24h": int,
          "success_rate": float,  # 0-100
          "top_model": str | null
        }
    """

    # Active nodes = nodes seen within ONLINE_THRESHOLD seconds
    with LOCK:
        now = unix_now()
        active_nodes = 0
        for info in NODES.values():
            last_seen_unix = info.get("last_seen_unix", parse_timestamp(info.get("last_seen")))
            if (now - last_seen_unix) <= ONLINE_THRESHOLD:
                active_nodes += 1

    # Creator jobs + rewards summary
    job_summary = get_job_summary()
    jobs_24h = int(job_summary.get("jobs_completed_today", 0) or 0)

    # Success rate: completed / total for creator jobs.
    conn = get_db()
    # Total finished jobs (exclude queued/running)
    total_row = conn.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE UPPER(task_type)=?
          AND status NOT IN ('queued', 'running')
        """,
        (CREATOR_TASK_TYPE,),
    ).fetchone()
    # Successful jobs (legacy 'completed' or newer 'success')
    ok_row = conn.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE UPPER(task_type)=?
          AND status IN ('completed', 'success')
        """,
        (CREATOR_TASK_TYPE,),
    ).fetchone()
    total_jobs = int(total_row[0]) if total_row else 0
    ok_jobs = int(ok_row[0]) if ok_row else 0
    success_rate = round(100.0 * ok_jobs / total_jobs, 1) if total_jobs else 0.0

    # Top model by count of successful completions
    with LOCK:
        if MODEL_STATS:
            top_model = max(MODEL_STATS.items(), key=lambda kv: kv[1].get("count", 0.0))[0]
        else:
            top_model = None

    return jsonify(
        {
            "active_nodes": active_nodes,
            "jobs_completed_24h": jobs_24h,
            "success_rate": success_rate,
            "top_model": top_model,
        }
    )


@app.route("/models/stats", methods=["GET"])
def models_stats_legacy() -> Any:
    """Backward-compatible alias used by older frontends."""

    return models_stats()


# ---------------------------------------------------------------------------
# Health and admin utilities
# ---------------------------------------------------------------------------


def _reset_db_and_memory() -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM rewards")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM node_wallets")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    with LOCK:
        NODES.clear()
        TASKS.clear()
        EVENT_LOGS.clear()
        MODEL_STATS.clear()
        RATE_LIMIT_BUCKETS.clear()


@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    """Lightweight health check that exercises DB and summary concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Basic DB touch
    conn = get_db()
    try:
        conn.execute("SELECT 1").fetchone()
    except Exception as e:  # pragma: no cover - best-effort
        return jsonify({"ok": False, "error": f"db: {type(e).__name__}: {e}"}), 500

    # Concurrency exercise for get_job_summary using separate app contexts
    results = {"ok": True, "concurrency": {"attempts": 0, "success": 0, "errors": []}}

    def run_summary() -> bool:
        try:
            with app.app_context():
                _ = get_job_summary(3)
            return True
        except Exception as e:  # pragma: no cover - diagnostic
            results["concurrency"]["errors"].append(f"{type(e).__name__}: {e}")
            return False

    attempts = int(request.args.get("n", 5))
    attempts = max(1, min(attempts, 16))
    results["concurrency"]["attempts"] = attempts
    success = 0
    with ThreadPoolExecutor(max_workers=attempts) as pool:
        futures = [pool.submit(run_summary) for _ in range(attempts)]
        for fut in as_completed(futures):
            if fut.result():
                success += 1
    results["concurrency"]["success"] = success
    results["ok"] = success == attempts
    status = 200 if results["ok"] else 500
    return jsonify(results), status


@app.route("/admin/reset", methods=["POST"])
def admin_reset() -> Any:
    """Clear database tables and in-memory state to start fresh.

    If SERVER_JOIN_TOKEN is set, require it via header `X-Join-Token` or `?token=`.
    """
    if not check_join_token():
        abort(403)

    _reset_db_and_memory()

    log_event("Admin reset: cleared DB and memory state", level="info")
    return jsonify({"ok": True, "message": "reset complete"})

# Perform startup reset after all functions are defined
if RESET_ON_STARTUP:
    try:
        _reset_db_and_memory()
        log_event("Startup reset: cleared DB and memory state", level="info")
    except Exception as e:
        LOGGER.exception("Startup reset failed: %s", e)


@app.route("/logs", methods=["GET"])
def logs_endpoint() -> Any:
    with LOCK:
        entries = list(EVENT_LOGS)[-50:]
    return jsonify({"logs": entries})


@app.route("/feed", methods=["GET"])
def feed_catalog() -> Any:
    return jsonify({"models": build_models_catalog()})


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str) -> Any:
    """
    Return a simple JSON payload describing where the result artifact lives.

    For video jobs (WAN I2V or AnimateDiff), this exposes the MP4 download URL
    under ``/static/outputs/videos/<job_id>.mp4`` if it exists.
    """
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    image_path = OUTPUTS_DIR / f"{job_id}.png"
    videos_dir = OUTPUTS_DIR / "videos"
    video_path = videos_dir / f"{job_id}.mp4"

    payload: Dict[str, Any] = {"job_id": job_id}
    if image_path.exists():
        payload["image_url"] = f"/static/outputs/{job_id}.png"
    if video_path.exists():
        payload["video_url"] = f"/static/outputs/videos/{job_id}.mp4"

    if "image_url" not in payload and "video_url" not in payload:
        return jsonify({"error": "result_not_found", "job_id": job_id}), 404
    return jsonify(payload), 200


@app.route("/nodes", methods=["GET"])
def nodes_endpoint() -> Any:
    with LOCK:
        now = unix_now()
        payload = []
        total_util = 0.0
        total_rewards = 0.0
        online_count = 0
        for node_id, info in NODES.items():
            last_seen_unix = info.get("last_seen_unix", parse_timestamp(info.get("last_seen")))
            online = (now - last_seen_unix) <= ONLINE_THRESHOLD
            if online:
                online_count += 1
            avg_util = float(info.get("avg_utilization", info.get("utilization", 0.0)))
            total_util += avg_util
            rewards = float(info.get("rewards", 0.0))
            total_rewards += rewards
            start_time = parse_timestamp(info.get("start_time"))
            uptime_seconds = max(0, int(now - start_time))
            # Normalize possibly-null fields from node telemetry
            last_result = info.get("last_result") or {}
            current_task = info.get("current_task") or {}
            model_name = (last_result.get("model_name") or current_task.get("model_name"))
            inference_time = last_result.get("metrics", {}).get("inference_time_ms")
            task_type = (last_result.get("task_type") or current_task.get("task_type") or CREATOR_TASK_TYPE)
            weight = (
                last_result.get("metrics", {}).get("reward_weight")
                or current_task.get("weight")
                or MODEL_WEIGHTS.get((model_name or "triomerge_v10").lower(), 10.0)
            )
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = resolve_weight(model_name or "triomerge_v10", 10.0)
            payload.append(
                {
                    "node_id": node_id,
                    "node_name": info.get("node_name", node_id),
                    "role": info.get("role", "worker"),
                    "wallet": info.get("wallet"),
                    "task_type": task_type,
                    "model_name": model_name,
                    "model_weight": weight,
                    "inference_time_ms": inference_time,
                    "gpu_utilization": info.get("utilization", 0.0),
                    "avg_utilization": avg_util,
                    "rewards": rewards,
                    "last_reward": info.get("last_reward", 0.0),
                    "last_seen": info.get("last_seen"),
                    "uptime_human": format_duration(uptime_seconds),
                    "pipelines": info.get("pipelines", []),
                    "models": info.get("models", []),
                    "status": "online" if online else "offline",
                    "last_result": last_result,
                }
            )
        total_nodes = len(payload)
        summary = {
            "timestamp": iso_now(),
            "total_nodes": total_nodes,
            "online_nodes": online_count,
            "offline_nodes": total_nodes - online_count,
            "avg_utilization": round(total_util / total_nodes, 2) if total_nodes else 0.0,
            "total_rewards": round(total_rewards, 6),
        }
        job_summary = get_job_summary()
        models_catalog = build_models_catalog()
    summary["tasks_backlog"] = job_summary["queued_jobs"]
    summary["jobs_completed_today"] = job_summary.get("jobs_completed_today", 0)
    summary["total_rewarded"] = job_summary.get("total_distributed", 0.0)
    return jsonify(
        {
            "nodes": payload,
            "summary": summary,
            "job_summary": job_summary,
            "models_catalog": models_catalog,
        }
    )


@app.route("/dashboard")
def dashboard() -> Any:
    return send_from_directory(STATIC_DIR, "dashboard.html")


@app.route("/")
def root() -> Any:
    return dashboard()


if __name__ == "__main__":
    host = os.getenv("SERVER_BIND", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "5001"))
    app.run(host=host, port=port)

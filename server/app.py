from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import sqlite3
import shutil
import subprocess
import sys
import threading
import time
import uuid
import random
import requests
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import abort, Flask, jsonify, request, send_file, send_from_directory, g, has_app_context
from flask_cors import CORS

# Import our local modules
import safety
import credits
import invite
import rewards
import job_helpers

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
LORA_STORAGE_DIR = Path(os.getenv("HAVNAI_LORA_STORAGE_DIR", "/mnt/d/havnai-storage/models/loras"))

CREATOR_TASK_TYPE = "IMAGE_GEN"

SUPPORTED_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}

# Ensure local packages (e.g., havnai.*) are importable when running server/app.py directly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
JOB_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

SERVER_JOIN_TOKEN = os.getenv("SERVER_JOIN_TOKEN", "").strip()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
RESET_ON_STARTUP = os.getenv("RESET_ON_STARTUP", "").strip()
INVITE_CONFIG_PATH = Path(
    os.getenv("HAVNAI_INVITE_CONFIG", str(BASE_DIR / "server" / "invites.json"))
)
INVITE_GATING = os.getenv("HAVNAI_INVITE_GATING", "").strip().lower() in {"1", "true", "yes"}
CREDITS_ENABLED = os.getenv("HAVNAI_CREDITS_ENABLED", "").strip().lower() in {"1", "true", "yes"}

# Reward configuration (can be overridden via environment variables)
REWARD_CONFIG: Dict[str, float] = {
    "baseline_runtime": float(os.getenv("REWARD_BASELINE_RUNTIME", "8.0")),
    "sdxl_factor": float(os.getenv("REWARD_SDXL_FACTOR", "1.5")),
    "sd15_factor": float(os.getenv("REWARD_SD15_FACTOR", "1.0")),
    "anime_factor": float(os.getenv("REWARD_ANIME_FACTOR", "0.7")),
    "ltx2_factor": float(os.getenv("REWARD_LTX2_FACTOR", "2.0")),
    "base_reward": float(os.getenv("REWARD_BASE_HAI", "0.05")),
}

# Slight global positive bias to help realism skin detail.
GLOBAL_POSITIVE_SUFFIX = (
    "(ultra-realistic 8k:1.05), "
    "(detailed skin pores:1.03), "
    "focused eyes, clear pupils, natural gaze, "
    "well formed hands, five fingers on each hand, "
    "natural teeth, realistic mouth structure"
)

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
        "lazy eye",
        "crossed eyes",
        "missing fingers",
        "fused fingers",
        "extra fingers",
        "malformed teeth",
        "extra teeth",
        "bad teeth",
        "open mouth deformity",
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
# Dependency injection for modules
# ---------------------------------------------------------------------------

def _inject_module_dependencies() -> None:
    """Inject app dependencies into extracted modules."""
    # Invite module
    invite.get_db = get_db  # type: ignore[attr-defined]
    invite.request = request  # type: ignore[attr-defined]
    invite.jsonify = jsonify  # type: ignore[attr-defined]
    invite.INVITE_CONFIG_PATH = INVITE_CONFIG_PATH  # type: ignore[attr-defined]
    invite.INVITE_GATING = INVITE_GATING  # type: ignore[attr-defined]
    invite.LOGGER = LOGGER  # type: ignore[attr-defined]

    # Credits module
    credits.get_db = get_db  # type: ignore[attr-defined]
    credits.log_event = log_event  # type: ignore[attr-defined]
    credits.CREDITS_ENABLED = CREDITS_ENABLED  # type: ignore[attr-defined]
    # get_model_config will be injected after it's defined

    # Rewards module
    rewards.get_db = get_db  # type: ignore[attr-defined]
    rewards.LOGGER = LOGGER  # type: ignore[attr-defined]
    rewards.MODEL_WEIGHTS = MODEL_WEIGHTS  # type: ignore[attr-defined]
    rewards.REWARD_CONFIG = REWARD_CONFIG  # type: ignore[attr-defined]

    # Job helpers module
    job_helpers.get_db = get_db  # type: ignore[attr-defined]
    job_helpers.NODES = NODES  # type: ignore[attr-defined]
    job_helpers.CREATOR_TASK_TYPE = CREATOR_TASK_TYPE  # type: ignore[attr-defined]
    # get_model_config will be injected after it's defined

    # Stripe payments module
    stripe_payments.get_db = get_db  # type: ignore[attr-defined]
    stripe_payments.log_event = log_event  # type: ignore[attr-defined]
    stripe_payments.deposit_credits = credits.deposit_credits  # type: ignore[attr-defined]


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


# Invite management moved to server/invite.py


# Safety filtering moved to server/safety.py


def _download_lora_asset(lora: Dict[str, Any], dest_dir: Path) -> Path:
    filename = str(lora.get("filename") or lora.get("name") or "").strip()
    if not filename:
        raise RuntimeError("lora filename missing")
    local_path = dest_dir / filename
    if local_path.exists():
        return local_path
    url = str(lora.get("url") or "").strip()
    if not url:
        raise RuntimeError(f"lora url missing for {filename}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".part")
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "HavnAI/1.0"})
            resp.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(local_path)
            return local_path
        except Exception as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            log_event(
                "LoRA download failed",
                level="warning",
                filename=filename,
                url=url,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt == 0:
                time.sleep(1)
    log_event(
        "LoRA download failed after retry",
        level="error",
        filename=filename,
        url=url,
        error=str(last_error),
    )
    raise RuntimeError(f"LoRA download failed: {last_error}")


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    for directory in (STATIC_DIR, LEGACY_STATIC_DIR, MANIFESTS_DIR, LOGS_DIR, OUTPUTS_DIR, BASE_DIR / "db"):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

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
            completed_at REAL,
            invite_code TEXT
        )
        """
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "invite_code" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN invite_code TEXT")
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS credits (
            wallet TEXT PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 0.0,
            total_deposited REAL NOT NULL DEFAULT 0.0,
            total_spent REAL NOT NULL DEFAULT 0.0,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("UPDATE jobs SET status='queued', node_id=NULL WHERE status='running'")
    conn.commit()


init_db()
stripe_payments.init_stripe_tables(get_db())

# Inject dependencies into extracted modules (initial injection)
_inject_module_dependencies()

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
        node.setdefault("loras", [])
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


# Keep the original dict object so injected module references stay valid.
NODES.clear()
NODES.update(load_nodes())


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
            "loras": [],
        })
        node["wallet"] = row["wallet"]
        if row["node_name"]:
            node["node_name"] = row["node_name"]


load_node_wallets()
log_event(f"Telemetry online with {len(NODES)} cached node(s).", version=APP_VERSION)

# ---------------------------------------------------------------------------
# Job + reward helpers
# ---------------------------------------------------------------------------


# Job queue and reward management moved to server/job_helpers.py and server/rewards.py


# ---------------------------------------------------------------------------
# Credits helpers (moved to server/credits.py)
# ---------------------------------------------------------------------------


def get_job_summary(limit: int = 50) -> Dict[str, Any]:
    conn = get_db()
    summary_types = (CREATOR_TASK_TYPE, "FACE_SWAP")
    queued = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='queued' AND UPPER(task_type) IN (?, ?)",
        summary_types,
    ).fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='running' AND UPPER(task_type) IN (?, ?)",
        summary_types,
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
          AND UPPER(task_type) IN (?, ?)
        """,
        (today_start, *summary_types),
    ).fetchone()[0]
    # Avoid binding LIMIT to sidestep driver quirks under concurrency; limit is internal and cast to int.
    limit_int = int(limit)
    rows = conn.execute(
        f"""
        SELECT jobs.id, jobs.wallet, jobs.model, jobs.task_type, jobs.status, jobs.weight,
               jobs.completed_at, jobs.timestamp, rewards.reward_hai
        FROM jobs
        LEFT JOIN rewards ON rewards.task_id = jobs.id
        WHERE UPPER(jobs.task_type) IN (?, ?)
        ORDER BY jobs.timestamp DESC
        LIMIT {limit_int}
        """,
        summary_types,
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
        videos_dir = OUTPUTS_DIR / "videos"
        video_path = videos_dir / f"{row['id']}.mp4"
        has_video = video_path.exists()
        video_url = f"/static/outputs/videos/{row['id']}.mp4" if has_video else None
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
                "video_url": video_url,
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


# Inject get_model_config into modules that need it
credits.get_model_config = get_model_config  # type: ignore[attr-defined]
job_helpers.get_model_config = get_model_config  # type: ignore[attr-defined]


def build_models_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for name, meta in MANIFEST_MODELS.items():
        catalog.append(
            {
                "model": name,
                "weight": rewards.resolve_weight(name, meta.get("reward_weight", 5.0)),
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
        if task_type not in {CREATOR_TASK_TYPE, "VIDEO_GEN", "ANIMATEDIFF", "FACE_SWAP"}:
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
      <p>If you have a join token, append <code>--token &lt;TOKEN&gt;</code>. To prefill a wallet, add <code>--wallet 0x...</code>.</p>
      <h2>Prerequisites</h2>
      <ul>
        <li>64-bit Linux (Ubuntu/Debian/RHEL) or macOS 12+</li>
        <li>Python 3.10+, curl, and a GPU driver/runtime (NVIDIA + CUDA recommended)</li>
        <li>12 GB+ NVIDIA GPU recommended for creator/video workloads (CPU nodes still supported)</li>
        <li>EVM wallet address (simulated rewards in Alpha)</li>
      </ul>
      <h2>What happens next?</h2>
      <ol>
        <li>Installer prepares <code>~/.havnai</code>, Python venv, and the node client</li>
        <li>Configure <code>WALLET</code> / <code>JOIN_TOKEN</code> inside <code>~/.havnai/.env</code> (or pass flags)</li>
        <li>Start the service (systemd or launchd) or run directly</li>
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


def _normalize_lora_catalog(raw_loras: Any) -> List[Dict[str, Any]]:
    if not raw_loras or not isinstance(raw_loras, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in raw_loras:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            filename = str(entry.get("filename") or "").strip()
            label = str(entry.get("label") or "").strip()
            if not name and filename:
                name = Path(filename).stem or filename
            if not name:
                continue
            item: Dict[str, Any] = {"name": name}
            if filename:
                item["filename"] = filename
            if label:
                item["label"] = label
            normalized.append(item)
            continue
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                normalized.append({"name": name})
    return normalized


@app.route("/loras/list")
def list_loras() -> Any:
    loras: List[Dict[str, Any]] = []
    nodes_with_loras: List[str] = []
    with LOCK:
        for node_id, node in NODES.items():
            node_loras = _normalize_lora_catalog(node.get("loras"))
            if node_loras:
                nodes_with_loras.append(node_id)
                loras.extend(node_loras)
    if loras:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for entry in loras:
            key = str(entry.get("filename") or entry.get("name") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        deduped.sort(key=lambda item: str(item.get("label") or item.get("name") or "").lower())
        return jsonify({"loras": deduped, "nodes": nodes_with_loras, "source": "nodes"})
    path = LORA_STORAGE_DIR
    if path.exists():
        for entry in sorted(path.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in SUPPORTED_LORA_EXTS:
                continue
            loras.append({"name": entry.stem, "filename": entry.name})
    loras.sort(key=lambda item: str(item.get("name") or "").lower())
    return jsonify({"loras": loras, "path": str(path), "source": "local"})


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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, value))


def _normalize_loras(raw_loras: Any) -> List[Dict[str, Any]]:
    if not raw_loras or not isinstance(raw_loras, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in raw_loras:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            item: Dict[str, Any] = {"name": name}
            weight = entry.get("weight")
            if weight is not None:
                try:
                    item["weight"] = float(weight)
                except (TypeError, ValueError):
                    pass
            normalized.append(item)
            continue
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                normalized.append({"name": name})
    return normalized


def build_faceswap_settings(payload: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    base_image_url = str(
        payload.get("base_image_url")
        or payload.get("base_image_b64")
        or payload.get("base_image")
        or payload.get("base_image_path")
        or payload.get("image")
        or payload.get("image_b64")
        or ""
    ).strip()
    face_source_url = str(
        payload.get("face_source_url")
        or payload.get("face_source_b64")
        or payload.get("face_source")
        or payload.get("face_image")
        or payload.get("face_image_b64")
        or payload.get("face_source_path")
        or ""
    ).strip()
    strength = _coerce_float(payload.get("strength", 0.8), 0.8)
    num_steps = _coerce_int(payload.get("num_steps", payload.get("steps", 20)), 20)
    strength = max(0.0, min(1.0, strength))
    num_steps = _clamp(num_steps, 5, 60)
    settings: Dict[str, Any] = {
        "prompt": prompt_text,
        "base_image_url": base_image_url,
        "face_source_url": face_source_url,
        "strength": strength,
        "num_steps": num_steps,
    }
    seed = payload.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    if seed is not None:
        settings["seed"] = seed
    return settings


@app.route("/submit-job", methods=["POST"])
def submit_job() -> Any:
    if not rate_limit(f"submit-job:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    payload = request.get_json() or {}
    invite_code, invite_error = invite.enforce_invite_limits(payload)
    if invite_error:
        return invite_error
    prompt_raw = str(payload.get("prompt") or payload.get("data") or "")
    negative_raw = str(payload.get("negative_prompt") or "")
    block_reason = safety.check_safety(prompt_raw, negative_raw)
    if block_reason:
        return jsonify({"error": "prompt_blocked", "reason": block_reason}), 400
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
        weights = [rewards.resolve_weight(meta["name"].lower(), meta.get("reward_weight", 10.0)) for meta in candidates]
        chosen = random.choices(names, weights=weights, k=1)[0]
        model_name_raw = chosen
        model_name = chosen.lower()

    if not model_name:
        return jsonify({"error": "missing model"}), 400

    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400

    if weight is None:
        weight = cfg.get("reward_weight", rewards.resolve_weight(model_name, 10.0))

    # Special-case LTX2 video jobs with structured payload
    pipeline_name = str(cfg.get("pipeline", "")).lower()
    is_ltx2 = model_name == "ltx2" or pipeline_name == "ltx2"

    # Special-case AnimateDiff video jobs with rich structured payload
    is_animatediff = model_name == "animatediff" or pipeline_name == "animatediff"

    if is_ltx2:
        prompt_text = str(payload.get("prompt") or "").strip()
        if not prompt_text:
            return jsonify({"error": "missing prompt"}), 400
        negative_prompt = str(payload.get("negative_prompt") or "").strip()

        seed = payload.get("seed")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return jsonify({"error": "missing seed"}), 400

        def _clamp_int(value: Any, default: int, min_v: int, max_v: int) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = default
            return max(min_v, min(parsed, max_v))

        def _clamp_float(value: Any, default: float, min_v: float, max_v: float) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = default
            return max(min_v, min(parsed, max_v))

        init_image = payload.get("init_image") or payload.get("init_image_url") or payload.get("init_image_b64")
        settings = {
            "prompt": prompt_text,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": _clamp_int(payload.get("steps", 25), 25, 1, 50),
            "guidance": _clamp_float(payload.get("guidance", 6.0), 6.0, 0.0, 12.0),
            "width": _clamp_int(payload.get("width", 512), 512, 256, 768),
            "height": _clamp_int(payload.get("height", 512), 512, 256, 768),
            "frames": _clamp_int(payload.get("frames", 16), 16, 1, 16),
            "fps": _clamp_int(payload.get("fps", 8), 8, 1, 12),
        }
        if init_image:
            settings["init_image"] = init_image
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
        strength = payload.get("strength")
        try:
            strength = float(strength) if strength is not None else None
        except (TypeError, ValueError):
            strength = None
        init_image = (
            payload.get("init_image")
            or payload.get("init_image_url")
            or payload.get("init_image_b64")
            or None
        )
        extend_raw = payload.get("extend_chunks")
        if extend_raw is None:
            extend_raw = payload.get("extend") or payload.get("extend_video")
        try:
            extend_chunks = int(extend_raw) if extend_raw is not None else 0
        except (TypeError, ValueError):
            extend_chunks = 1 if bool(extend_raw) else 0
        extend_chunks = max(0, min(extend_chunks, 6))
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
            "strength": strength,
            "extend_remaining": extend_chunks,
            "extend_total": extend_chunks,
        }
        job_data = json.dumps(settings)
        task_type = "ANIMATEDIFF"
    else:
        prompt_text = str(payload.get("prompt") or payload.get("data") or "")
        if prompt_text:
            prompt_text = f"{prompt_text}, {GLOBAL_POSITIVE_SUFFIX}"
        else:
            prompt_text = GLOBAL_POSITIVE_SUFFIX
        negative_prompt = str(payload.get("negative_prompt") or "").strip()
        job_settings: Dict[str, Any] = {"prompt": prompt_text}
        loras = _normalize_loras(payload.get("loras") or payload.get("lora_list") or [])
        if loras:
            job_settings["loras"] = loras
        base_negative = str(cfg.get("negative_prompt_default") or "").strip() if cfg else ""
        combined_negative = ", ".join(
            filter(None, [negative_prompt or base_negative, GLOBAL_NEGATIVE_PROMPT])
        )
        if combined_negative:
            job_settings["negative_prompt"] = combined_negative
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

        overrides: Dict[str, Any] = {}
        if payload.get("steps") is not None:
            overrides["steps"] = _coerce_int(payload.get("steps"), int(job_settings.get("steps", 20)))
        if payload.get("guidance") is not None:
            overrides["guidance"] = _coerce_float(
                payload.get("guidance"),
                float(job_settings.get("guidance", 7.0)),
            )
        if payload.get("width") is not None:
            overrides["width"] = _coerce_int(payload.get("width"), int(job_settings.get("width", 512)))
        if payload.get("height") is not None:
            overrides["height"] = _coerce_int(payload.get("height"), int(job_settings.get("height", 512)))
        sampler_override = payload.get("sampler")
        if sampler_override is not None and str(sampler_override).strip():
            overrides["sampler"] = str(sampler_override).strip()
        seed = payload.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        if seed is not None:
            overrides["seed"] = seed
        if overrides:
            job_settings["overrides"] = overrides

        job_data = json.dumps(job_settings)
        task_type = CREATOR_TASK_TYPE

    # Credit gate — only active when HAVNAI_CREDITS_ENABLED=true
    credit_err = credits.check_and_deduct_credits(wallet, cfg.get("name", model_name), task_type)
    if credit_err:
        return credit_err

    with LOCK:
        job_id = job_helpers.enqueue_job(
            wallet,
            cfg.get("name", model_name),
            task_type,
            job_data,
            float(weight),
            invite_code,
        )
    log_event("Public job queued", wallet=wallet, model=model_name, job_id=job_id)
    return jsonify({"status": "queued", "job_id": job_id}), 200


@app.route("/generate-video", methods=["POST"])
def generate_video_job() -> Any:
    if not rate_limit(f"generate-video:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    payload = request.get_json() or {}
    invite_code, invite_error = invite.enforce_invite_limits(payload)
    if invite_error:
        return invite_error

    prompt_text = str(payload.get("prompt") or "").strip()
    if not prompt_text:
        return jsonify({"error": "missing prompt"}), 400

    wallet = str(payload.get("wallet", "")).strip()
    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400

    model_name_raw = str(payload.get("model") or "ltx2").strip()
    model_name = model_name_raw.lower()
    load_manifest()
    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400

    weight = payload.get("weight")
    if weight is None:
        weight = cfg.get("reward_weight", rewards.resolve_weight(model_name, 10.0))

    seed = payload.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    fps = _clamp(_coerce_int(payload.get("fps", 8), 8), 1, 8)
    duration = payload.get("duration")
    frames = payload.get("frames")
    if frames is None and duration is not None:
        try:
            frames = int(float(duration) * fps)
        except (TypeError, ValueError):
            frames = None
    frames = _clamp(_coerce_int(frames or 48, 48), 1, 48)

    steps = _clamp(_coerce_int(payload.get("steps", 25), 25), 1, 50)
    guidance = _coerce_float(payload.get("guidance", 6.0), 6.0)
    guidance = max(0.0, min(guidance, 12.0))
    width = _clamp(_coerce_int(payload.get("width", 512), 512), 256, 512)
    height = _clamp(_coerce_int(payload.get("height", 512), 512), 256, 512)

    settings: Dict[str, Any] = {
        "prompt": prompt_text,
        "negative_prompt": str(payload.get("negative_prompt") or "").strip(),
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
    }
    motion_type = payload.get("motion_type")
    if motion_type:
        settings["motion_type"] = motion_type
    lora_list = payload.get("lora_list")
    if lora_list:
        settings["lora_list"] = lora_list
    init_image = payload.get("init_image")
    if init_image:
        settings["init_image"] = init_image

    job_data = json.dumps(settings)

    # Credit gate — only active when HAVNAI_CREDITS_ENABLED=true
    credit_err = credits.check_and_deduct_credits(wallet, cfg.get("name", model_name), "VIDEO_GEN")
    if credit_err:
        return credit_err

    with LOCK:
        job_id = job_helpers.enqueue_job(
            wallet,
            cfg.get("name", model_name),
            "VIDEO_GEN",
            job_data,
            float(weight),
            invite_code,
        )
    log_event("Video job queued", wallet=wallet, model=model_name, job_id=job_id)
    return jsonify({"status": "queued", "job_id": job_id}), 200


@app.route("/submit-faceswap-job", methods=["POST"])
def submit_faceswap_job_endpoint() -> Any:
    if not rate_limit(f"submit-faceswap-job:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    payload = request.get_json() or {}
    invite_code, invite_error = invite.enforce_invite_limits(payload)
    if invite_error:
        return invite_error
    wallet = str(payload.get("wallet", "")).strip()
    model_name_raw = str(payload.get("model") or "epicrealismxl_vxviicrystalclear").strip()
    model_name = model_name_raw.lower()
    prompt_text = str(payload.get("prompt") or "").strip()
    negative_raw = str(payload.get("negative_prompt") or "")
    block_reason = safety.check_safety(prompt_text, negative_raw)
    if block_reason:
        return jsonify({"error": "prompt_blocked", "reason": block_reason}), 400

    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400

    settings = build_faceswap_settings(payload, prompt_text)
    if not settings.get("base_image_url") or not settings.get("face_source_url"):
        return jsonify({"error": "base_image and face_source are required"}), 400

    load_manifest()
    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400
    pipeline_norm = str(cfg.get("pipeline", "")).lower()
    if "sdxl" not in pipeline_norm:
        return jsonify({"error": "faceswap requires an SDXL base model"}), 400

    weight = payload.get("weight")
    if weight is None:
        weight = cfg.get("reward_weight", rewards.resolve_weight(model_name, 10.0))

    job_data = json.dumps(settings)

    # Credit gate — only active when HAVNAI_CREDITS_ENABLED=true
    credit_err = credits.check_and_deduct_credits(wallet, cfg.get("name", model_name), "FACE_SWAP")
    if credit_err:
        return credit_err

    with LOCK:
        job_id = job_helpers.enqueue_job(
            wallet,
            cfg.get("name", model_name),
            "FACE_SWAP",
            job_data,
            float(weight),
            invite_code,
        )
    log_event("Face-swap job queued", wallet=wallet, model=model_name, job_id=job_id)
    return jsonify({"status": "queued", "job_id": job_id}), 200


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
                "loras": [],
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
        if "loras" in data:
            node["loras"] = _normalize_lora_catalog(data.get("loras"))
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
            job = job_helpers.fetch_next_job_for_node(node_id)
            if job:
                cfg = get_model_config(job["model"])
                if cfg:
                    model_spec: Optional[Dict[str, Any]] = None
                    if isinstance(cfg, dict):
                        model_spec = {"name": job["model"], "pipeline": cfg.get("pipeline")}
                        if cfg.get("lora") is not None:
                            model_spec["lora"] = cfg.get("lora")
                    # Decode prompt/negative_prompt for standard IMAGE_GEN jobs stored as JSON
                    raw_data = job.get("data")
                    prompt_text = raw_data or ""
                    negative_prompt = ""
                    loras: List[Dict[str, Any]] = []
                    image_settings: Dict[str, Any] = {}
                    try:
                        parsed = json.loads(raw_data) if isinstance(raw_data, str) else None
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        prompt_text = str(parsed.get("prompt") or "")
                        negative_prompt = str(parsed.get("negative_prompt") or "")
                        parsed_loras = parsed.get("loras")
                        if isinstance(parsed_loras, list):
                            loras = parsed_loras
                        overrides = parsed.get("overrides")
                        if isinstance(overrides, dict):
                            for key in ("steps", "guidance", "width", "height", "sampler", "seed"):
                                if key in overrides and overrides[key] is not None:
                                    image_settings[key] = overrides[key]
                        # Pass init_image and strength for img2img
                        init_img = parsed.get("init_image") or parsed.get("init_image_b64") or ""
                        if init_img:
                            image_settings["init_image"] = init_img
                        init_strength = parsed.get("strength")
                        if init_strength is not None:
                            image_settings["strength"] = init_strength
                    # Always send plain prompt text to the node (avoid passing raw JSON)
                    prompt_for_node = prompt_text

                    # Assign under global lock to avoid multiple nodes claiming the same job
                    job_helpers.assign_job_to_node(job["id"], node_id)
                    log_event("Job claimed by node", job_id=job["id"], node_id=node_id)
                    reward_weight = float(job["weight"] or cfg.get("reward_weight", rewards.resolve_weight(job["model"], 10.0)))
                    job_task_type = (job.get("task_type") or CREATOR_TASK_TYPE).upper()
                    pending_entry = {
                        "task_id": job["id"],
                        "task_type": job_task_type,
                        "model_name": job["model"],
                        "model": model_spec,
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
                        "loras": loras,
                        "queued_at": job.get("timestamp"),
                    }
                    if image_settings:
                        pending_entry.update(image_settings)
                    pending = [pending_entry]
                    node_info["current_task"] = {
                        "task_id": job["id"],
                        "model_name": job["model"],
                        "status": "pending",
                        "task_type": pending[0]["task_type"],
                        "weight": pending[0]["reward_weight"],
                    }
                    save_nodes()
                else:
                    job_helpers.complete_job(job["id"], node_id, "failed")

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
                "model": task.get("model"),
                "model_path": task.get("model_path", ""),
                "pipeline": pipeline,
                "input_shape": task.get("input_shape", []),
                "reward_weight": task.get("reward_weight", 1.0),
                "wallet": task.get("wallet"),
                "prompt": task.get("prompt") or task.get("data", ""),
                "negative_prompt": task.get("negative_prompt") or "",
                "queued_at": task.get("queued_at"),
                "assigned_at": task.get("assigned_at"),
            }
            for key in ("steps", "guidance", "width", "height", "sampler", "seed", "init_image", "strength"):
                if key in task and task[key] is not None:
                    task_payload[key] = task[key]
            if task.get("loras"):
                task_payload["loras"] = task.get("loras")
            # If this is an LTX2 job, surface structured controls directly on the task payload
            if task_payload["type"].upper() == "VIDEO_GEN":
                try:
                    raw_ltx2 = task.get("data") or ""
                    ltx2_settings = json.loads(raw_ltx2) if isinstance(raw_ltx2, str) else {}
                except Exception:
                    ltx2_settings = {}
                if isinstance(ltx2_settings, dict):
                    for key in ("prompt", "negative_prompt", "seed", "steps", "guidance", "width", "height", "frames", "fps", "init_image"):
                        if key in ltx2_settings and ltx2_settings[key] is not None:
                            task_payload[key] = ltx2_settings[key]
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
                        "strength",
                        "scheduler",
                    ):
                        if key in ad_settings and ad_settings[key] is not None:
                            task_payload[key] = ad_settings[key]
            if task_payload["type"].upper() == "FACE_SWAP":
                try:
                    raw_fs = task.get("data") or ""
                    fs_settings = json.loads(raw_fs) if isinstance(raw_fs, str) else {}
                except Exception:
                    fs_settings = {}
                if isinstance(fs_settings, dict):
                    for key in (
                        "prompt",
                        "negative_prompt",
                        "base_image_url",
                        "face_source_url",
                        "strength",
                        "num_steps",
                        "seed",
                    ):
                        if key in fs_settings and fs_settings[key] is not None:
                            task_payload[key] = fs_settings[key]
            response_tasks.append(task_payload)
    return jsonify({"tasks": response_tasks}), 200


@app.route("/tasks", methods=["GET"])
def tasks_alias() -> Any:
    return get_creator_tasks()


@app.route("/tasks/ai", methods=["GET"])
def tasks_ai_alias() -> Any:
    # Backward/alternate compatibility for clients polling /tasks/ai
    return get_creator_tasks()


def _extract_last_frame(video_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return False
    return output_path.exists()


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
                "reward_weight": job.get("weight", rewards.resolve_weight(job["model"], 10.0)),
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
        reward, reward_factors = rewards.compute_reward(model_name, pipeline, metrics, status)

        # Ensure job is still owned/running before completing
        if not job_helpers.complete_job(task_id, node_id, status):
            if job_helpers.complete_job_if_queued(task_id, node_id, status):
                log_event(
                    "Accepted late results for queued job",
                    level="warning",
                    job_id=task_id,
                    node_id=node_id,
                )
            else:
                log_event(
                    "Results rejected (job not running/owned by node)",
                    level="warning",
                    job_id=task_id,
                    node_id=node_id,
                )
                TASKS.pop(task_id, None)
                return jsonify({"error": "conflict"}), 409

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
            video_path_saved: Optional[Path] = None
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
                    video_path_saved = video_path
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

        # Persist reward info into the job's data JSON for later inspection.
        job = get_job(task_id)
        if job:
            queued_at = job.get("timestamp")
            assigned_at = job.get("assigned_at")
            completed_at = job.get("completed_at")
            queue_ms = None
            run_ms = None
            total_ms = None
            if queued_at and assigned_at:
                queue_ms = int((assigned_at - queued_at) * 1000)
            if assigned_at and completed_at:
                run_ms = int((completed_at - assigned_at) * 1000)
            if queued_at and completed_at:
                total_ms = int((completed_at - queued_at) * 1000)
            log_event(
                "Task timing",
                job_id=task_id,
                node_id=node_id,
                queue_ms=queue_ms,
                run_ms=run_ms,
                total_ms=total_ms,
                status=status,
            )
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
                    conn = get_db()
                    conn.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(payload), task_id))
                    conn.commit()
                except Exception:
                    conn.rollback()
            wallet = wallet or job.get("wallet")
            task_type = task.get("task_type", CREATOR_TASK_TYPE)
            if (
                status == "success"
                and str(task_type).upper() == "ANIMATEDIFF"
                and isinstance(payload, dict)
            ):
                try:
                    remaining = int(payload.get("extend_remaining") or 0)
                except (TypeError, ValueError):
                    remaining = 0
                if remaining > 0:
                    videos_dir = OUTPUTS_DIR / "videos"
                    video_path = video_path_saved or (videos_dir / f"{task_id}.mp4")
                    if video_path.exists():
                        last_frame_path = videos_dir / f"{task_id}_last.png"
                        if _extract_last_frame(video_path, last_frame_path):
                            next_payload = dict(payload)
                            next_payload["extend_remaining"] = remaining - 1
                            next_payload["extend_parent"] = task_id
                            next_payload["init_image"] = str(last_frame_path)
                            next_payload["init_image_path"] = str(last_frame_path)
                            try:
                                invite_code = job.get("invite_code")
                            except Exception:
                                invite_code = None
                            weight_value = float(job.get("weight") or rewards.resolve_weight(model_name, 10.0))
                            next_job_id = job_helpers.enqueue_job(
                                wallet,
                                model_name,
                                "ANIMATEDIFF",
                                json.dumps(next_payload),
                                weight_value,
                                invite_code,
                            )
                            log_event(
                                "Auto-extended job queued",
                                job_id=next_job_id,
                                parent_job_id=task_id,
                            )
                        else:
                            log_event(
                                "Auto-extend skipped (last frame extract failed)",
                                job_id=task_id,
                            )
                    else:
                        log_event(
                            "Auto-extend skipped (video missing)",
                            job_id=task_id,
                        )

        rewards.record_reward(wallet, task_id, reward)

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


# ---------------------------------------------------------------------------
# Credits endpoints
# ---------------------------------------------------------------------------


@app.route("/credits/balance", methods=["GET"])
def credits_balance() -> Any:
    """Return credit balance for a wallet."""
    wallet = request.args.get("wallet", "").strip()
    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    conn = get_db()
    row = conn.execute(
        "SELECT balance, total_deposited, total_spent, updated_at FROM credits WHERE wallet=?",
        (wallet,),
    ).fetchone()
    if not row:
        return jsonify({
            "wallet": wallet,
            "balance": 0.0,
            "total_deposited": 0.0,
            "total_spent": 0.0,
            "credits_enabled": CREDITS_ENABLED,
        })
    return jsonify({
        "wallet": wallet,
        "balance": float(row["balance"]),
        "total_deposited": float(row["total_deposited"]),
        "total_spent": float(row["total_spent"]),
        "updated_at": row["updated_at"],
        "credits_enabled": CREDITS_ENABLED,
    })


@app.route("/credits/deposit", methods=["POST"])
def credits_deposit() -> Any:
    """Deposit credits to a wallet.

    This is an admin / webhook endpoint.  In production, this will be
    called by Stripe webhooks or on-chain listeners — never directly by
    end users.  For now it is gated by the join token.
    """
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    wallet = str(data.get("wallet", "")).strip()
    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid amount"}), 400
    if amount <= 0:
        return jsonify({"error": "amount must be positive"}), 400
    reason = str(data.get("reason", "manual")).strip()
    new_balance = credits.deposit_credits(wallet, amount, reason=reason)
    return jsonify({
        "wallet": wallet,
        "deposited": amount,
        "balance": new_balance,
        "reason": reason,
    })


@app.route("/credits/cost", methods=["GET"])
def credits_cost() -> Any:
    """Return the credit cost for a given model, so the frontend can show prices."""
    model = request.args.get("model", "").strip().lower()
    task_type = request.args.get("task_type", "").strip()
    cost = credits.resolve_credit_cost(model, task_type)
    return jsonify({
        "model": model,
        "cost": cost,
        "credits_enabled": CREDITS_ENABLED,
    })


# ---------------------------------------------------------------------------
# Stripe payment endpoints
# ---------------------------------------------------------------------------


@app.route("/payments/packages", methods=["GET"])
def payments_packages() -> Any:
    """Return available credit packages and whether Stripe is enabled."""
    return jsonify({
        "packages": stripe_payments.CREDIT_PACKAGES,
        "stripe_enabled": stripe_payments.STRIPE_ENABLED,
    })


@app.route("/payments/checkout", methods=["POST"])
def payments_checkout() -> Any:
    """Create a Stripe Checkout Session for buying credits."""
    if not stripe_payments.STRIPE_ENABLED:
        return jsonify({"error": "payments_disabled", "message": "Stripe payments are not enabled."}), 503
    data = request.get_json() or {}
    wallet = str(data.get("wallet", "")).strip()
    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    package_id = str(data.get("package_id", "")).strip()
    if not package_id:
        return jsonify({"error": "missing package_id"}), 400
    success_url = str(data.get("success_url", "")).strip()
    cancel_url = str(data.get("cancel_url", "")).strip()
    if not success_url or not cancel_url:
        return jsonify({"error": "missing success_url or cancel_url"}), 400
    try:
        result = stripe_payments.create_checkout_session(wallet, package_id, success_url, cancel_url)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": "invalid_package", "message": str(exc)}), 400
    except Exception as exc:
        log_event("Stripe checkout error", level="error", error=str(exc))
        return jsonify({"error": "checkout_failed", "message": "Could not create checkout session."}), 500


@app.route("/payments/webhook", methods=["POST"])
def payments_webhook() -> Any:
    """Handle Stripe webhook events (payment confirmations)."""
    if not stripe_payments.STRIPE_ENABLED:
        return jsonify({"error": "payments_disabled"}), 503
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        result = stripe_payments.handle_webhook_event(payload, sig_header)
        return jsonify(result)
    except Exception as exc:
        log_event("Stripe webhook error", level="error", error=str(exc))
        return jsonify({"error": "webhook_failed", "message": str(exc)}), 400


@app.route("/payments/history", methods=["GET"])
def payments_history() -> Any:
    """Return payment history for a wallet."""
    wallet = request.args.get("wallet", "").strip()
    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    history = stripe_payments.get_payment_history(wallet)
    return jsonify({"wallet": wallet, "payments": history})


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


@app.route("/quota", methods=["GET"])
def quota_status() -> Any:
    invite_config = invite.load_invite_config()
    if not invite.invite_gating_enabled(invite_config):
        return jsonify(
            invite.quota_payload(0, 0, 0, 0, iso_now())
        )
    invite_code = invite.extract_invite_code({})
    if not invite_code:
        return invite.invite_error_response()
    limits = invite.resolve_invite_limits(invite_code, invite_config)
    if not limits:
        return invite.invite_error_response()

    usage = invite.compute_invite_usage(invite_code)
    max_daily = int(limits.get("max_daily") or 0)
    max_concurrent = int(limits.get("max_concurrent") or 0)
    return jsonify(
        invite.quota_payload(
            max_daily,
            usage["used_today"],
            max_concurrent,
            usage["used_concurrent"],
            usage["reset_at"],
        )
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

    For video jobs (LTX2/AnimateDiff), this exposes the MP4 download URL
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


@app.route("/videos/stitch", methods=["POST"])
def stitch_videos() -> Any:
    payload = request.get_json() or {}
    job_ids = payload.get("job_ids")
    if not isinstance(job_ids, list) or not job_ids:
        return jsonify({"error": "job_ids_required"}), 400
    if shutil.which("ffmpeg") is None:
        return jsonify({"error": "ffmpeg_missing"}), 400

    videos_dir = OUTPUTS_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    input_paths: List[Path] = []
    for raw_id in job_ids:
        if not isinstance(raw_id, str) or not JOB_ID_REGEX.match(raw_id):
            return jsonify({"error": "invalid_job_id"}), 400
        path = videos_dir / f"{raw_id}.mp4"
        if not path.exists():
            return jsonify({"error": "video_not_found", "job_id": raw_id}), 404
        input_paths.append(path)

    output_name = str(payload.get("output_name") or f"stitched_{int(time.time())}.mp4")
    output_name = Path(output_name).name
    output_path = videos_dir / output_name

    concat_path = videos_dir / f"concat_{uuid.uuid4().hex}.txt"
    try:
        with concat_path.open("w", encoding="utf-8") as handle:
            for path in input_paths:
                handle.write(f"file '{path.as_posix()}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return jsonify({"error": "ffmpeg_failed", "detail": proc.stderr.strip()}), 500
    finally:
        try:
            concat_path.unlink()
        except Exception:
            pass

    return jsonify({"video_url": f"/static/outputs/videos/{output_name}"}), 200


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
                weight = rewards.resolve_weight(model_name or "triomerge_v10", 10.0)
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
    port_raw = os.getenv("SERVER_PORT") or os.getenv("PORT") or "5001"
    port = int(port_raw)
    app.run(host=host, port=port)

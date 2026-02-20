"""HavnAI Node Client — Stage 7 public onboarding."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import gc
from logging.handlers import RotatingFileHandler
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
import atexit
import signal
from typing import Any, Dict, List, Optional, Set, Tuple
import base64
from collections import OrderedDict

import requests

from registry import REGISTRY, ModelEntry, ManifestError

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import onnxruntime as ort  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required for HavnAI workloads") from exc

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import diffusers  # type: ignore  # noqa: F401
    try:
        from diffusers import AutoPipelineForText2Image as _AutoPipe  # type: ignore
    except Exception:  # pragma: no cover
        _AutoPipe = None  # type: ignore
    try:
        from diffusers import StableDiffusionPipeline as _SDPipe  # type: ignore
    except Exception:  # pragma: no cover
        _SDPipe = None  # type: ignore
    try:
        from diffusers import StableDiffusionXLPipeline as _SDXLPipe  # type: ignore
    except Exception:  # pragma: no cover
        _SDXLPipe = None  # type: ignore
    try:
        from diffusers import DPMSolverMultistepScheduler as _DPMSolver  # type: ignore
    except Exception:  # pragma: no cover
        _DPMSolver = None  # type: ignore
    try:
        from diffusers import AutoencoderKL as _AutoencoderKL  # type: ignore
    except Exception:  # pragma: no cover
        _AutoencoderKL = None  # type: ignore
except ImportError:  # pragma: no cover
    diffusers = None
    _AutoPipe = None  # type: ignore
    _SDPipe = None  # type: ignore
    _SDXLPipe = None  # type: ignore
    _LattePipe = None  # type: ignore
    _DPMSolver = None  # type: ignore
    _AutoencoderKL = None  # type: ignore
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:  # pragma: no cover
    FaceAnalysis = None  # type: ignore

_FACE_ANALYSIS = None
_FACE_SWAP_PIPE = None
_FACE_SWAP_PIPE_MODEL = ""
_IMAGE_PIPELINE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_IMAGE_PIPELINE_CACHE_LOCK = threading.Lock()

try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover
    hf_hub_download = None  # type: ignore

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

# ---------------------------------------------------------------------------
# Paths & environment bootstrap
# ---------------------------------------------------------------------------

HAVNAI_HOME = Path(os.environ.get("HAVNAI_HOME", Path.home() / ".havnai"))
ENV_PATH = HAVNAI_HOME / ".env"
LOGS_DIR = HAVNAI_HOME / "logs"
OUTPUTS_DIR = HAVNAI_HOME / "outputs"
LORA_DIR = Path(
    os.environ.get("HAVNAI_LORA_DIR")
    or os.environ.get("HAI_LORA_DIR")
    or os.environ.get("LORA_DIR")
    or (HAVNAI_HOME / "loras")
)
INSTANTID_CACHE_DIR = HAVNAI_HOME / "instantid"
VERSION_SEARCH_PATHS = [HAVNAI_HOME / "VERSION", Path(__file__).resolve().parent / "VERSION"]
DEFAULT_LORA_DIR = Path("/mnt/d/havnai-storage/models/loras")
LORA_DIR = Path(
    os.environ.get("HAI_LORA_DIR")
    or os.environ.get("HAVNAI_LORA_DIR")
    or (DEFAULT_LORA_DIR if DEFAULT_LORA_DIR.exists() else HAVNAI_HOME / "loras")
).expanduser()
MAX_LORAS = 5
MODEL_FILE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
SUPPORTED_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}
MODEL_SEARCH_DIR_CANDIDATES = [
    os.environ.get("HAVNAI_MODEL_DIR", ""),
    os.environ.get("HAI_MODEL_DIR", ""),
    os.environ.get("MODEL_DIR", ""),
    str(HAVNAI_HOME / "models" / "creator"),
    "/mnt/d/havnai-storage/models/creator",
]

# When running from the havnai-core repo, ensure sibling packages (engines/*)
# are importable even if current working directory is elsewhere.
CLIENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLIENT_DIR.parent
if (REPO_ROOT / "engines").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAVNAI_HOME.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("havnai-node")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(LOGS_DIR / "node.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(JSONFormatter())
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def log(message: str, prefix: str = "ℹ️", **extra: Any) -> None:
    LOGGER.info(f"{prefix} {message}", extra={"node": socket.gethostname(), **extra})


WAN_I2V_DEFAULTS: Dict[str, Any] = {
    "steps_high": 2,
    "steps_low": 2,
    "cfg": 1.0,
    "sampler": "euler",
    "num_frames": 32,
    "fps": 24,
    "height": 720,
    "width": 512,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_version() -> str:
    for path in VERSION_SEARCH_PATHS:
        if path.exists():
            return path.read_text().strip()
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent)
            .decode()
            .strip()
        )
    except Exception:
        return "dev"


CLIENT_VERSION = load_version()
lock = threading.Lock()


def load_env_file() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    defaults = {
        "SERVER_URL": os.environ.get("SERVER_URL") or os.environ.get("HAVNAI_SERVER") or "http://127.0.0.1:5001",
        "WALLET": env.get("WALLET", "0xYOUR_WALLET_ADDRESS"),
        "CREATOR_MODE": env.get("CREATOR_MODE", "false"),
        "NODE_NAME": env.get("NODE_NAME", socket.gethostname()),
        "JOIN_TOKEN": env.get("JOIN_TOKEN", ""),
    }
    env.update({k: defaults.get(k, v) for k, v in defaults.items()})
    return env


def save_env_file(env: Dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in env.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n")


ENV_VARS = load_env_file()

# Ensure .env exists on disk
save_env_file(ENV_VARS)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = ENV_VARS.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raw = ENV_VARS.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


SERVER_BASE = ENV_VARS.get("SERVER_URL", "http://127.0.0.1:5001").rstrip("/")
JOIN_TOKEN = ENV_VARS.get("JOIN_TOKEN", "").strip()
ROLE = "creator" if ENV_VARS.get("CREATOR_MODE", "false").lower() in {"1", "true", "yes"} else "worker"
NODE_NAME = ENV_VARS.get("NODE_NAME", socket.gethostname())
FAST_PREVIEW = (
    os.environ.get("HAI_FAST_PREVIEW", ENV_VARS.get("HAI_FAST_PREVIEW", "")).lower()
    in {"1", "true", "yes"}
)
LTX2_MODEL_ID = os.environ.get("LTX2_MODEL_ID") or ENV_VARS.get("LTX2_MODEL_ID") or "maxin-cn/Latte-1"
LTX2_MODEL_PATH = os.environ.get("LTX2_MODEL_PATH") or ENV_VARS.get("LTX2_MODEL_PATH", "")
ANIMATEDIFF_ADAPTER_ID = (
    os.environ.get("ANIMATEDIFF_MOTION_ADAPTER")
    or ENV_VARS.get("ANIMATEDIFF_MOTION_ADAPTER")
    or "guoyww/animatediff-motion-adapter-v1-5-2"
)
INSTANTID_ADAPTER_PATH = (
    os.environ.get("HAVNAI_INSTANTID_ADAPTER")
    or os.environ.get("INSTANTID_ADAPTER_PATH")
    or ENV_VARS.get("HAVNAI_INSTANTID_ADAPTER")
    or str(INSTANTID_CACHE_DIR / "ip-adapter.bin")
)
INSTANTID_CONTROLNET_PATH = (
    os.environ.get("HAVNAI_INSTANTID_CONTROLNET")
    or os.environ.get("INSTANTID_CONTROLNET_PATH")
    or ENV_VARS.get("HAVNAI_INSTANTID_CONTROLNET")
    or str(INSTANTID_CACHE_DIR / "ControlNetModel")
)
INSTANTID_CONTROLNET_REPO = (
    os.environ.get("HAVNAI_INSTANTID_CONTROLNET_REPO")
    or ENV_VARS.get("HAVNAI_INSTANTID_CONTROLNET_REPO")
    or "InstantX/InstantID"
)
ENABLE_XFORMERS = _env_flag("HAI_ENABLE_XFORMERS", False)
SAFE_CUDA_KERNELS = _env_flag("HAI_SAFE_CUDA_KERNELS", True)
IMAGE_PIPELINE_CACHE_SIZE = max(0, _env_int("HAI_IMAGE_PIPELINE_CACHE_SIZE", 1))
PRELOAD_IMAGE_ON_STARTUP = _env_flag("HAI_PRELOAD_IMAGE_ON_STARTUP", True)
PRELOAD_IMAGE_MODEL = str(
    os.environ.get("HAI_PRELOAD_IMAGE_MODEL")
    or ENV_VARS.get("HAI_PRELOAD_IMAGE_MODEL")
    or ""
).strip()
ALLOW_REMOTE_LORA_DOWNLOAD = _env_flag("HAI_ALLOW_REMOTE_LORA_DOWNLOAD", False)

REGISTRY.base_url = SERVER_BASE

HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 15
BACKOFF_BASE = 5
MAX_BACKOFF = 60
START_TIME = time.time()

IMAGE_STEPS = int(os.environ.get("HAI_STEPS", "20"))
IMAGE_GUIDANCE = float(os.environ.get("HAI_GUIDANCE", "7.0"))
IMAGE_WIDTH = int(os.environ.get("HAI_WIDTH", "512"))
IMAGE_HEIGHT = int(os.environ.get("HAI_HEIGHT", "512"))

utilization_hint = random.randint(10, 25 if ROLE == "creator" else 15)

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})
if JOIN_TOKEN:
    SESSION.headers["X-Join-Token"] = JOIN_TOKEN

REGISTRY.session = SESSION


def refresh_manifest_with_backoff(reason: str = "startup", max_attempts: Optional[int] = None) -> bool:
    backoff = BACKOFF_BASE
    attempts = 0
    while True:
        attempts += 1
        try:
            REGISTRY.refresh()
            log("Model manifest refreshed", prefix="✅", reason=reason)
            return True
        except Exception as exc:
            log(f"Manifest refresh failed ({reason}): {exc}", prefix="⚠️")
            if max_attempts is not None and attempts >= max_attempts:
                return False
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)


def endpoint(path: str) -> str:
    return f"{SERVER_BASE}{path}"


def ensure_model_entry(model_name: str) -> ModelEntry:
    try:
        return REGISTRY.get(model_name)
    except (KeyError, ManifestError) as exc:
        raise RuntimeError(f"Model '{model_name}' missing from manifest") from exc


def ensure_model_path(entry: ModelEntry) -> Path:
    path_value = str(getattr(entry, "path", "") or "").strip()
    if path_value:
        path = Path(path_value).expanduser()
        if path.exists():
            return path

    fallback = resolve_model_path_from_name(entry.name)
    if fallback is not None:
        try:
            entry.path = str(fallback)
        except Exception:
            pass
        return fallback

    if path_value:
        raise FileNotFoundError(f"Model path missing on node: {path_value}")
    raise FileNotFoundError(f"Model path missing on node for manifest model: {entry.name}")


def resolve_model_path_from_name(model_name: str) -> Optional[Path]:
    target = re.sub(r"[^a-z0-9]+", "", (model_name or "").lower())
    if not target:
        return None

    seen: Set[str] = set()
    search_dirs: List[Path] = []
    for candidate in MODEL_SEARCH_DIR_CANDIDATES:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_dir():
            search_dirs.append(path)

    for base in search_dirs:
        try:
            for child in base.iterdir():
                if not child.is_file():
                    continue
                if child.suffix.lower() not in MODEL_FILE_EXTENSIONS:
                    continue
                stem = re.sub(r"[^a-z0-9]+", "", child.stem.lower())
                if stem == target:
                    return child
        except Exception:
            continue
    return None


def discover_capabilities() -> Dict[str, List[str]]:
    pipelines: set[str] = set()
    models: List[str] = []
    for entry in REGISTRY.list_entries():
        pipeline = str(getattr(entry, "pipeline", "") or "").lower()
        name = str(getattr(entry, "name", "") or "").strip()
        if not name:
            continue
        # LTX2 can be loaded from HF repo id; local path is optional.
        if pipeline == "ltx2":
            pipelines.add(pipeline)
            models.append(name)
            continue
        try:
            ensure_model_path(entry)
        except Exception:
            continue
        pipelines.add(pipeline or "sd15")
        models.append(name)
    return {"pipelines": sorted(pipelines), "models": sorted(models)}


def _has_runner(module_name: str, function_name: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[function_name])
        return hasattr(module, function_name)
    except Exception:
        return False


def _has_sdxl_base_model() -> bool:
    for entry in REGISTRY.list_entries():
        pipeline = str(getattr(entry, "pipeline", "") or "").lower()
        task_type = str(getattr(entry, "task_type", "IMAGE_GEN") or "IMAGE_GEN").upper()
        if task_type != "IMAGE_GEN":
            continue
        if "sdxl" not in pipeline:
            continue
        try:
            ensure_model_path(entry)
            return True
        except Exception:
            continue
    return False


def _has_image_generation_model() -> bool:
    for entry in REGISTRY.list_entries():
        task_type = str(getattr(entry, "task_type", "IMAGE_GEN") or "IMAGE_GEN").upper()
        if task_type != "IMAGE_GEN":
            continue
        try:
            ensure_model_path(entry)
            return True
        except Exception:
            continue
    return False


def _instantid_assets_ready() -> bool:
    adapter_exists = any(path.exists() for path in _candidate_instantid_adapter_paths())
    controlnet_exists = any(path.exists() for path in _candidate_instantid_controlnet_paths())
    if adapter_exists and controlnet_exists:
        return True
    # If local assets are missing, allow runtime download when HF helper is available.
    return hf_hub_download is not None


def discover_supports(capabilities: Dict[str, List[str]]) -> List[str]:
    if ROLE != "creator":
        return []

    supports: List[str] = []
    pipelines = {p.lower() for p in capabilities.get("pipelines", [])}
    models = {m.lower() for m in capabilities.get("models", [])}

    if _has_image_generation_model():
        supports.append("image")

    if "ltx2" in pipelines and "ltx2" in models and _has_runner("engines.ltx2.ltx2_runner", "run_ltx2"):
        supports.append("video")

    if (
        "animatediff" in pipelines
        and "animatediff" in models
        and _has_runner("engines.animatediff.animatediff_runner", "run_animatediff")
    ):
        supports.append("animatediff")

    face_swap_ready = (
        torch is not None
        and diffusers is not None
        and Image is not None
        and np is not None
        and cv2 is not None
        and FaceAnalysis is not None
        and _has_runner(
            "pipeline_stable_diffusion_xl_instantid_inpaint",
            "StableDiffusionXLInstantIDInpaintPipeline",
        )
        and _has_sdxl_base_model()
        and _instantid_assets_ready()
    )
    if face_swap_ready:
        supports.append("face_swap")

    return sorted(set(supports))


# ---------------------------------------------------------------------------
# Wallet handling
# ---------------------------------------------------------------------------


def ensure_wallet() -> str:
    wallet = ENV_VARS.get("WALLET", "").strip()
    if wallet.lower() == "0xyour_wallet_address" or not wallet:
        try:
            wallet = input("Enter your EVM wallet address (0x...): ").strip()
        except KeyboardInterrupt:
            print()
            sys.exit(1)
        if not wallet:
            log("Wallet address required. Exiting.", prefix="🚫")
            sys.exit(1)
        ENV_VARS["WALLET"] = wallet
        save_env_file(ENV_VARS)
    return wallet


WALLET = ensure_wallet()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def link_wallet(wallet: str) -> None:
    payload = {"node_id": NODE_NAME, "wallet": wallet, "node_name": NODE_NAME}
    try:
        resp = SESSION.post(endpoint("/link-wallet"), data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        log("Wallet linked with coordinator.", prefix="✅", wallet=wallet)
    except Exception as exc:
        log(f"Wallet link failed: {exc}", prefix="⚠️")


def read_gpu_stats() -> Dict[str, Any]:
    global utilization_hint
    output = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if output:
        try:
            name, mem_total, mem_used, util = output.strip().split("\n")[0].split(", ")
            utilization_hint = int(util)
            return {
                "gpu_name": name,
                "memory_total": int(mem_total),
                "memory_used": int(mem_used),
                "utilization": int(util),
            }
        except Exception:
            pass
    if psutil and hasattr(psutil, "cpu_percent"):
        utilization_hint = max(5, int(psutil.cpu_percent(interval=0.2)))
    return {
        "gpu_name": "Simulated",
        "memory_total": 0,
        "memory_used": 0,
        "utilization": utilization_hint,
    }


def run_command(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task execution helpers
# ---------------------------------------------------------------------------


def disconnect() -> None:
    try:
        resp = SESSION.post(endpoint("/disconnect"), data=json.dumps({"node_id": NODE_NAME}), timeout=5)
        if resp.status_code == 200:
            log("Disconnected from coordinator.", prefix="�o.")
    except Exception as exc:
        log(f"Disconnect failed: {exc}", prefix="�s��,?")


def _handle_signal(signum, frame):  # type: ignore
    log(f"Received signal {signum}, disconnecting...", prefix="ℹ️")
    disconnect()
    try:
        sys.exit(0)
    except SystemExit:
        pass


def random_input(shape: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy required for inference")
    dims = [max(1, dim) for dim in shape]
    return (np.random.rand(*dims).astype(np.float32) * 2.0) - 1.0


def coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_loras(raw_loras: Any) -> List[Dict[str, Any]]:
    if not raw_loras or not isinstance(raw_loras, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in raw_loras:
        if isinstance(entry, dict):
            filename = str(entry.get("filename") or "").strip()
            name = str(entry.get("name") or filename or "").strip()
            if not name:
                continue
            item: Dict[str, Any] = {"name": name}
            if filename:
                item["filename"] = filename
            url = str(entry.get("url") or "").strip()
            if url:
                item["url"] = url
            strength = entry.get("strength")
            if strength is not None:
                try:
                    item["strength"] = float(strength)
                except (TypeError, ValueError):
                    pass
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


def _detect_lora_pipeline(path: Path) -> str:
    """Detect whether a LoRA file targets SD1.5 or SDXL by inspecting tensor keys.

    For .safetensors files, reads the JSON header (first 8 bytes = uint64 LE
    header size, then header_size bytes of JSON).  SDXL LoRAs contain keys
    referencing the second text encoder (``lora_te2`` / ``text_encoder_2``).
    SD1.5 LoRAs never have these.

    Returns 'sdxl', 'sd15', or '' if detection fails.
    """
    import struct

    suffix = path.suffix.lower()
    if suffix != ".safetensors":
        # .pt / .bin files are harder to inspect safely; skip detection
        return ""
    try:
        with path.open("rb") as fh:
            raw_size = fh.read(8)
            if len(raw_size) < 8:
                return ""
            header_size = struct.unpack("<Q", raw_size)[0]
            # Sanity check — header should be < 10 MB
            if header_size > 10 * 1024 * 1024:
                return ""
            header_bytes = fh.read(header_size)
        header = json.loads(header_bytes)
        keys = set(header.keys())
        # SDXL LoRAs always train on the second text encoder (CLIP-G)
        has_te2 = any(
            "lora_te2" in k or "text_encoder_2" in k or "te2_text_model" in k
            for k in keys
        )
        if has_te2:
            return "sdxl"
        # If it has UNet keys but no te2, it's SD1.5
        has_unet = any("lora_unet" in k or "unet" in k for k in keys)
        if has_unet:
            return "sd15"
    except Exception:
        pass
    return ""


# Cache detected pipelines to avoid re-reading files on every heartbeat
_lora_pipeline_cache: Dict[str, str] = {}


def list_local_loras() -> List[Dict[str, Any]]:
    loras: List[Dict[str, Any]] = []
    try:
        if not LORA_DIR.exists():
            return loras
        for entry in sorted(LORA_DIR.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in SUPPORTED_LORA_EXTS:
                continue
            item: Dict[str, Any] = {"name": entry.stem, "filename": entry.name}
            # Auto-detect pipeline compatibility from file contents
            cache_key = str(entry)
            if cache_key not in _lora_pipeline_cache:
                _lora_pipeline_cache[cache_key] = _detect_lora_pipeline(entry)
            pipeline = _lora_pipeline_cache[cache_key]
            if pipeline:
                item["pipeline"] = pipeline
            loras.append(item)
    except Exception as exc:
        log(f"Failed to scan LoRA directory: {exc}", prefix="⚠️")
    return loras


def _resolve_lora_path(name: str) -> Optional[Path]:
    raw = str(name or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    if candidate.suffix:
        candidates.append(LORA_DIR / raw)
    else:
        for ext in SUPPORTED_LORA_EXTS:
            candidates.append(LORA_DIR / f"{raw}{ext}")
        candidates.append(LORA_DIR / raw)
    for path in candidates:
        if path.exists():
            return path
    if candidate.exists():
        return candidate
    return None


def _download_lora_spec(lora: Dict[str, Any]) -> Path:
    filename = str(lora.get("filename") or lora.get("name") or "").strip()
    if not filename:
        raise RuntimeError("LoRA filename missing")
    local_path = LORA_DIR / filename
    if local_path.exists():
        return local_path
    url = str(lora.get("url") or "").strip()
    if not url:
        raise RuntimeError(f"LoRA url missing for {filename}")
    if not ALLOW_REMOTE_LORA_DOWNLOAD:
        raise RuntimeError(
            "Remote LoRA downloads are disabled (set HAI_ALLOW_REMOTE_LORA_DOWNLOAD=true to enable)"
        )
    LORA_DIR.mkdir(parents=True, exist_ok=True)
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
            log(f"LoRA download failed (attempt {attempt + 1}/2): {exc}", prefix="⚠️")
            if attempt == 0:
                time.sleep(1)
    log(f"LoRA download failed after retry: {last_error}", prefix="⚠️")
    raise RuntimeError(f"LoRA download failed: {last_error}")


_ADAPTER_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe_adapter_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "lora"
    safe = _ADAPTER_SAFE_RE.sub("_", raw).strip("_")
    if not safe:
        safe = "lora"
    if safe != raw:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
        safe = f"{safe}_{digest}"
    return safe


def _check_lora_pipeline_compat(pipe: Any, lora_entry: Dict[str, Any]) -> bool:
    """Check if a LoRA is likely compatible with the loaded pipeline.

    Uses the pipeline class name to determine sd15 vs sdxl and compares
    against an optional ``pipeline`` field on the LoRA entry.  Returns True
    if compatible or if we can't determine (benefit of the doubt).
    """
    lora_pipeline = str(lora_entry.get("pipeline") or "").lower()
    if not lora_pipeline:
        return True  # No metadata — try loading anyway
    pipe_class = type(pipe).__name__.lower()
    pipe_is_xl = "xl" in pipe_class
    lora_is_xl = "sdxl" in lora_pipeline or "xl" in lora_pipeline
    if pipe_is_xl != lora_is_xl:
        log(
            f"LoRA '{lora_entry.get('name', '?')}' pipeline={lora_pipeline} "
            f"incompatible with {'SDXL' if pipe_is_xl else 'SD1.5'} pipe, skipping",
            prefix="⚠️",
        )
        return False
    return True


def _apply_loras_to_pipe(pipe: Any, raw_loras: Any) -> List[str]:
    loras = _normalize_loras(raw_loras)
    if not loras or not hasattr(pipe, "load_lora_weights"):
        return []
    adapter_names: List[str] = []
    adapter_weights: List[float] = []
    loaded_paths: List[str] = []
    for entry in loras:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        # Pipeline compatibility check — skip LoRAs that don't match the model
        if not _check_lora_pipeline_compat(pipe, entry):
            continue
        if entry.get("url") or entry.get("filename"):
            path = _download_lora_spec(entry)
        else:
            path = _resolve_lora_path(name)
        if not path:
            log(f"LoRA not found: {name}", prefix="⚠️")
            continue
        adapter_name = _safe_adapter_name(path.stem)
        last_error: Optional[Exception] = None
        loaded = False
        strength = entry.get("strength")
        weight = entry.get("weight")
        scale = strength if strength is not None else weight
        if scale is None:
            scale = 1.0
        weight_name = str(entry.get("filename") or "").strip()
        if not weight_name and path.is_dir():
            # Try to locate a single weight file in the directory for offline mode.
            candidates = [p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_LORA_EXTS]
            if len(candidates) == 1:
                weight_name = candidates[0].name
            else:
                for ext in SUPPORTED_LORA_EXTS:
                    candidate = path / f"{name}{ext}"
                    if candidate.exists():
                        weight_name = candidate.name
                        break
        try:
            if path.is_file():
                try:
                    pipe.load_lora_weights(
                        str(path.parent),
                        weight_name=path.name,
                        adapter_name=adapter_name,
                        strength=scale,
                    )
                except TypeError:
                    try:
                        pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter_name)
                    except TypeError:
                        pipe.load_lora_weights(str(path.parent), weight_name=path.name)
            else:
                try:
                    if weight_name:
                        pipe.load_lora_weights(
                            str(path),
                            weight_name=weight_name,
                            adapter_name=adapter_name,
                            strength=scale,
                        )
                    else:
                        pipe.load_lora_weights(str(path), adapter_name=adapter_name, strength=scale)
                except TypeError:
                    if weight_name:
                        pipe.load_lora_weights(str(path), weight_name=weight_name, adapter_name=adapter_name)
                    else:
                        pipe.load_lora_weights(str(path), adapter_name=adapter_name)
            loaded = True
        except Exception as exc:
            last_error = exc
        if not loaded:
            try:
                if weight_name:
                    pipe.load_lora_weights(str(path), weight_name=weight_name, adapter_name=adapter_name)
                else:
                    pipe.load_lora_weights(str(path), adapter_name=adapter_name)
                loaded = True
                last_error = None
            except TypeError:
                try:
                    if weight_name:
                        pipe.load_lora_weights(str(path), weight_name=weight_name)
                    else:
                        pipe.load_lora_weights(str(path))
                    loaded = True
                    last_error = None
                except Exception as exc_inner:
                    last_error = exc_inner
            except Exception as exc_inner:
                    last_error = exc_inner
        if not loaded:
            log(f"LoRA load failed: {last_error}", prefix="⚠️")
            continue
        loaded_paths.append(str(path))
        adapter_names.append(adapter_name)
        if scale is None:
            adapter_weights.append(1.0)
        else:
            try:
                adapter_weights.append(float(scale))
            except (TypeError, ValueError):
                adapter_weights.append(1.0)
    if not adapter_names:
        return loaded_paths
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except Exception as exc:
            log(f"LoRA adapter weights failed: {exc}", prefix="⚠️")
    elif hasattr(pipe, "fuse_lora"):
        if len(adapter_names) == 1:
            try:
                pipe.fuse_lora(lora_scale=adapter_weights[0])
            except Exception as exc:
                log(f"LoRA fuse failed: {exc}", prefix="⚠️")
        else:
            log("Multiple LoRAs loaded; per-LoRA weights unsupported on this pipeline", prefix="⚠️")
    return loaded_paths


def _normalize_image_path(text: str) -> str:
    if text.startswith("file://"):
        parsed = urllib.parse.urlparse(text)
        path = urllib.parse.unquote(parsed.path or "")
        if path.startswith("/") and len(path) > 3 and path[2] == ":":
            drive = path[1].lower()
            rest = path[3:].lstrip("/")
            return f"/mnt/{drive}/{rest}"
        return path
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        drive = text[0].lower()
        rest = text[2:].lstrip("\\/")
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text


def _describe_image_source(value: Any) -> str:
    if value is None:
        return "missing"
    if not isinstance(value, str):
        return f"{type(value).__name__}"
    text = value.strip()
    if not text:
        return "empty"
    if text.startswith("http://") or text.startswith("https://"):
        return f"url:{text[:120]}"
    if text.startswith("data:"):
        return f"data-uri ({len(text)} chars)"
    if text.startswith("file://"):
        return "file-url"
    if len(text) > 120:
        return f"string ({len(text)} chars)"
    return f"path:{text}"


def load_image_source_with_error(
    value: Any, target_size: Optional[Tuple[int, int]] = None
) -> Tuple[Optional["Image.Image"], Optional[str]]:
    if Image is None:
        return None, "PIL is not available"
    if not value:
        return None, "missing image source"
    if not isinstance(value, str):
        return None, "image source must be a string"
    text = value.strip()
    if not text:
        return None, "image source is empty"
    img = None
    if text.startswith("http://") or text.startswith("https://"):
        try:
            resp = requests.get(text, timeout=30, headers={"User-Agent": "HavnAI/1.0"})
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        except Exception as exc:
            return None, f"http fetch failed: {exc}"
    else:
        treat_as_b64 = False
        if text.startswith("data:"):
            parts = text.split(",", 1)
            text = parts[1] if len(parts) > 1 else ""
            treat_as_b64 = True
        if not treat_as_b64:
            normalized = _normalize_image_path(text)
            try:
                path = Path(normalized).expanduser()
            except OSError as exc:
                path = None
                path_error = f"invalid path: {exc}"
            else:
                path_error = None
            path_missing = False
            if path and path.exists():
                try:
                    img = Image.open(path)
                except Exception as exc:
                    return None, f"failed to open file {path}: {exc}"
            else:
                if path_error:
                    return None, path_error
                if path:
                    path_missing = True
        if img is None:
            raw = None
            try:
                raw = base64.b64decode(text, validate=True)
            except Exception:
                try:
                    raw = base64.b64decode(text)
                except Exception as exc:
                    if path_missing:
                        return None, f"file not found at {path} (base64 decode failed: {exc})"
                    return None, f"base64 decode failed: {exc}"
            try:
                img = Image.open(io.BytesIO(raw))
            except Exception as exc:
                return None, f"base64 image decode failed: {exc}"
    if img is None:
        return None, "image decode failed"
    img = img.convert("RGB")
    if target_size and target_size[0] > 0 and target_size[1] > 0:
        img = img.resize(target_size, resample=Image.LANCZOS)
    return img, None


def load_image_source(value: Any, target_size: Optional[Tuple[int, int]] = None) -> Optional["Image.Image"]:
    img, _ = load_image_source_with_error(value, target_size)
    return img


def get_face_analysis() -> "FaceAnalysis":
    global _FACE_ANALYSIS
    if _FACE_ANALYSIS is not None:
        return _FACE_ANALYSIS
    if FaceAnalysis is None:
        raise RuntimeError("insightface is required for face swap")
    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if torch is not None and torch.cuda.is_available():
        available_providers = []
        try:
            available_providers = ort.get_available_providers()
        except Exception:
            available_providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
    app = FaceAnalysis(name="antelopev2", root=str(INSTANTID_CACHE_DIR), providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    _FACE_ANALYSIS = app
    return app


def pick_primary_face(face_infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not face_infos:
        return None
    return sorted(
        face_infos,
        key=lambda info: (info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1]),
    )[-1]


def resize_img(
    input_image: "Image.Image",
    max_side: int = 1280,
    min_side: int = 1024,
    size: Optional[Tuple[int, int]] = None,
    pad_to_max_side: bool = False,
    mode: int = Image.BILINEAR if Image is not None else 2,
    base_pixel_number: int = 64,
) -> "Image.Image":
    if np is None:
        raise RuntimeError("NumPy required for face swap preprocessing")
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def prepare_mask_and_pose_control(
    pose_image: "Image.Image",
    face_info: Dict[str, Any],
    padding: int = 60,
    mask_grow: int = 40,
    resize: bool = True,
) -> Tuple[Tuple["Image.Image", "Image.Image", "Image.Image"], Tuple[int, int, int, int]]:
    if np is None or cv2 is None:
        raise RuntimeError("opencv + numpy required for face swap preprocessing")
    if padding < mask_grow:
        raise ValueError("mask_grow cannot be greater than padding")

    from pipeline_stable_diffusion_xl_instantid import draw_kps  # type: ignore

    kps = np.array(face_info["kps"])
    width, height = pose_image.size

    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    m_x1 = max(0, x1 - mask_grow)
    m_y1 = max(0, y1 - mask_grow)
    m_x2 = min(width, x2 + mask_grow)
    m_y2 = min(height, y2 + mask_grow)

    m_x1, m_y1, m_x2, m_y2 = int(m_x1), int(m_y1), int(m_x2), int(m_y2)

    p_x1 = max(0, x1 - padding)
    p_y1 = max(0, y1 - padding)
    p_x2 = min(width, x2 + padding)
    p_y2 = min(height, y2 + padding)

    p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

    mask = np.zeros([height, width, 3])
    mask[m_y1:m_y2, m_x1:m_x2] = 255
    mask = mask[p_y1:p_y2, p_x1:p_x2]
    mask = Image.fromarray(mask.astype(np.uint8))

    image = np.array(pose_image)[p_y1:p_y2, p_x1:p_x2]
    image = Image.fromarray(image.astype(np.uint8))

    original_width, original_height = image.size
    kps -= [p_x1, p_y1]
    if resize:
        mask = resize_img(mask)
        image = resize_img(image)
        new_width, new_height = image.size
        kps *= [new_width / original_width, new_height / original_height]
    control_image = draw_kps(image, kps)

    return (mask, image, control_image), (p_x1, p_y1, original_width, original_height)


def _candidate_instantid_adapter_paths() -> List[Path]:
    candidates = [
        Path(INSTANTID_ADAPTER_PATH).expanduser(),
        INSTANTID_CACHE_DIR / "InstantID" / "ip-adapter.bin",
        INSTANTID_CACHE_DIR / "ip-adapter.bin",
    ]
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_instantid_controlnet_paths() -> List[Path]:
    candidates = [
        Path(INSTANTID_CONTROLNET_PATH).expanduser(),
        INSTANTID_CACHE_DIR / "InstantID" / "ControlNetModel",
        INSTANTID_CACHE_DIR / "ControlNetModel",
    ]
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_instantid_adapter_path() -> Path:
    for adapter_path in _candidate_instantid_adapter_paths():
        if adapter_path.exists():
            return adapter_path
    if hf_hub_download is None:
        raise RuntimeError(
            "InstantID adapter missing locally and huggingface_hub is unavailable."
        )
    try:
        downloaded = hf_hub_download(
            repo_id=INSTANTID_CONTROLNET_REPO,
            filename="ip-adapter.bin",
            cache_dir=str(INSTANTID_CACHE_DIR),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download InstantID adapter: {exc}") from exc
    return Path(downloaded)


def _resolve_instantid_controlnet_ref() -> str:
    for controlnet_path in _candidate_instantid_controlnet_paths():
        if controlnet_path.exists():
            return str(controlnet_path)
    return INSTANTID_CONTROLNET_REPO


def _load_instantid_controlnet(dtype: Any) -> Any:
    """Load InstantID ControlNet from local path or HF repo/subfolder."""
    from diffusers.models import ControlNetModel  # type: ignore

    controlnet_ref = _resolve_instantid_controlnet_ref()
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}

    # InstantX/InstantID stores ControlNet weights in the ControlNetModel subfolder.
    if not Path(str(controlnet_ref)).expanduser().exists() and str(controlnet_ref) == INSTANTID_CONTROLNET_REPO:
        load_kwargs["subfolder"] = "ControlNetModel"

    return ControlNetModel.from_pretrained(controlnet_ref, **load_kwargs)


def _run_ltx2_task(
    task_id: str,
    entry: ModelEntry,
    reward_weight: float,
    task: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, Optional[str]]:
    try:
        from engines.ltx2.ltx2_runner import run_ltx2, video_to_b64 as ltx2_video_to_b64  # type: ignore
    except Exception as exc:
        return (
            {
                "status": "failed",
                "task_type": "video_gen",
                "model_name": entry.name,
                "reward_weight": reward_weight,
                "error": f"LTX2 runtime unavailable: {exc}",
            },
            utilization_hint,
            None,
        )

    model_ref = LTX2_MODEL_PATH or str(getattr(entry, "path", "") or LTX2_MODEL_ID)
    task_payload = dict(task)
    task_payload["task_id"] = task_id
    task_payload["model_name"] = entry.name
    task_payload["reward_weight"] = reward_weight
    task_payload.setdefault("seed", random.randint(0, 2**31 - 1))

    try:
        metrics, util, video_path = run_ltx2(
            task_payload,
            log_fn=lambda message: log(message, prefix="🎬", task_id=task_id),
            outputs_dir=OUTPUTS_DIR,
            read_gpu_stats=read_gpu_stats,
            utilization_hint=utilization_hint,
            model_id=model_ref,
        )
    except Exception as exc:
        return (
            {
                "status": "failed",
                "task_type": "video_gen",
                "model_name": entry.name,
                "reward_weight": reward_weight,
                "error": f"LTX2 execution failed: {exc}",
            },
            utilization_hint,
            None,
        )

    video_b64 = ltx2_video_to_b64(video_path) if video_path else None
    if metrics.get("status") == "success" and not video_b64:
        metrics["status"] = "failed"
        metrics["error"] = "LTX2 produced no output video payload"
    try:
        util_int = int(util)
    except (TypeError, ValueError):
        util_int = utilization_hint
    return metrics, util_int, video_b64


def _run_animatediff_task(
    task_id: str,
    entry: ModelEntry,
    model_path: Path,
    reward_weight: float,
    task: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, Optional[str]]:
    try:
        from engines.animatediff.animatediff_runner import run_animatediff, video_to_b64 as ad_video_to_b64  # type: ignore
    except Exception as exc:
        return (
            {
                "status": "failed",
                "task_type": "animatediff",
                "model_name": entry.name,
                "reward_weight": reward_weight,
                "error": f"AnimateDiff runtime unavailable: {exc}",
            },
            utilization_hint,
            None,
        )

    task_payload = dict(task)
    task_payload["task_id"] = task_id
    task_payload["model_name"] = entry.name
    task_payload["reward_weight"] = reward_weight
    task_payload.setdefault("seed", random.randint(0, 2**31 - 1))

    try:
        metrics, util_info, video_path = run_animatediff(
            task_payload,
            model_id=str(model_path),
            outputs_dir=OUTPUTS_DIR,
            log_fn=lambda message: log(message, prefix="🎞️", task_id=task_id),
            read_gpu_stats=read_gpu_stats,
            utilization_hint={"utilization": utilization_hint},
        )
    except Exception as exc:
        return (
            {
                "status": "failed",
                "task_type": "animatediff",
                "model_name": entry.name,
                "reward_weight": reward_weight,
                "error": f"AnimateDiff execution failed: {exc}",
            },
            utilization_hint,
            None,
        )

    video_b64 = ad_video_to_b64(video_path) if video_path else None
    if metrics.get("status") == "success" and not video_b64:
        metrics["status"] = "failed"
        metrics["error"] = "AnimateDiff produced no output video payload"

    if isinstance(util_info, dict):
        try:
            util_val = float(util_info.get("utilization", utilization_hint))
        except (TypeError, ValueError):
            util_val = float(utilization_hint)
    else:
        try:
            util_val = float(util_info)
        except (TypeError, ValueError):
            util_val = float(utilization_hint)
    return metrics, int(max(util_val, float(utilization_hint))), video_b64


def _run_faceswap_task(
    task_id: str,
    entry: ModelEntry,
    model_path: Path,
    reward_weight: float,
    task: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, Optional[str]]:
    global _FACE_SWAP_PIPE, _FACE_SWAP_PIPE_MODEL

    start_stats = read_gpu_stats()
    end_stats: Dict[str, Any] = {}
    started = time.time()
    status = "success"
    error_msg = ""
    image_b64: Optional[str] = None
    util = utilization_hint

    try:
        if torch is None or diffusers is None:
            raise RuntimeError("diffusers + torch are required for face swap")
        if Image is None or np is None or cv2 is None:
            raise RuntimeError("pillow + numpy + opencv are required for face swap")

        from pipeline_stable_diffusion_xl_instantid_inpaint import (  # type: ignore
            StableDiffusionXLInstantIDInpaintPipeline,
        )

        prompt = str(task.get("prompt") or "").strip() or "portrait, photorealistic, high detail"
        negative_prompt = str(task.get("negative_prompt") or "").strip()
        strength = coerce_float(task.get("strength"), 0.8)
        strength = max(0.0, min(1.0, strength))
        num_steps = coerce_int(task.get("num_steps"), 20)
        num_steps = max(5, min(60, num_steps))
        seed = task.get("seed")
        try:
            seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        except (TypeError, ValueError):
            seed = random.randint(0, 2**31 - 1)

        base_image, base_error = load_image_source_with_error(task.get("base_image_url"))
        if base_image is None:
            raise RuntimeError(f"Failed to load base image: {base_error or 'unknown error'}")
        source_image, source_error = load_image_source_with_error(task.get("face_source_url"))
        if source_image is None:
            raise RuntimeError(f"Failed to load face source: {source_error or 'unknown error'}")

        face_app = get_face_analysis()
        base_faces = face_app.get(cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR))
        source_faces = face_app.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
        base_face = pick_primary_face(base_faces)
        source_face = pick_primary_face(source_faces)
        if base_face is None:
            raise RuntimeError("No face detected in base image")
        if source_face is None:
            raise RuntimeError("No face detected in face source image")

        crop_inputs, crop_region = prepare_mask_and_pose_control(base_image, base_face, padding=70, mask_grow=48)
        mask_image, cropped_image, control_image = crop_inputs
        crop_x, crop_y, crop_w, crop_h = crop_region

        adapter_path = _resolve_instantid_adapter_path()
        controlnet_ref = _resolve_instantid_controlnet_ref()
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_cache_key = f"{model_path}|{controlnet_ref}|{adapter_path}"
        if _FACE_SWAP_PIPE is None or _FACE_SWAP_PIPE_MODEL != pipeline_cache_key:
            controlnet = _load_instantid_controlnet(dtype)
            pipe = StableDiffusionXLInstantIDInpaintPipeline.from_single_file(
                str(model_path),
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipe.load_ip_adapter_instantid(str(adapter_path))
            pipe.set_ip_adapter_scale(0.8)
            pipe.to(device)
            # InstantID adds a custom projection module that is not always moved by
            # generic pipeline .to(...), so force placement to avoid cpu/cuda mismatch.
            if hasattr(pipe, "image_proj_model") and pipe.image_proj_model is not None:
                pipe.image_proj_model.to(device=device, dtype=dtype)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            _FACE_SWAP_PIPE = pipe
            _FACE_SWAP_PIPE_MODEL = pipeline_cache_key

        if hasattr(_FACE_SWAP_PIPE, "image_proj_model") and _FACE_SWAP_PIPE.image_proj_model is not None:
            _FACE_SWAP_PIPE.image_proj_model.to(device=device, dtype=dtype)

        image_embeds: Any = source_face["embedding"]
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds)
        if isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.to(device=device, dtype=dtype)

        generator = torch.Generator(device=device).manual_seed(seed)
        result = _FACE_SWAP_PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=cropped_image,
            mask_image=mask_image,
            control_image=control_image,
            image_embeds=image_embeds,
            num_inference_steps=num_steps,
            guidance_scale=5.0,
            strength=strength,
            controlnet_conditioning_scale=0.8,
            generator=generator,
        )
        swapped_crop = result.images[0]
        swapped_crop = swapped_crop.resize((crop_w, crop_h), resample=Image.LANCZOS)
        composed = base_image.copy()
        composed.paste(swapped_crop, (crop_x, crop_y))

        output_path = OUTPUTS_DIR / f"{task_id}.png"
        composed.save(output_path)
        with output_path.open("rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log(f"Face swap generation failed: {error_msg}", prefix="🚫", task_id=task_id)
    finally:
        end_stats = read_gpu_stats()
        try:
            util = int(max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint))
        except Exception:
            util = utilization_hint

    metrics: Dict[str, Any] = {
        "status": status,
        "task_type": "face_swap",
        "model_name": entry.name,
        "model_path": str(model_path),
        "reward_weight": reward_weight,
        "inference_time_ms": round((time.time() - started) * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
    }
    if status != "success":
        metrics["error"] = error_msg or "face swap error"
    return metrics, util, image_b64


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = str(task.get("type") or "IMAGE_GEN").upper()
    model_name = (task.get("model_name") or "model").lower()
    reward_weight = float(task.get("reward_weight", 1.0))
    input_shape = task.get("input_shape") or []
    prompt = task.get("prompt") or ""
    negative_prompt = task.get("negative_prompt") or ""

    if task_type in {"IMAGE_GEN", "VIDEO_GEN", "ANIMATEDIFF", "FACE_SWAP"} and ROLE != "creator":
        log(f"Skipping creator task {task_id[:8]} — node not in creator mode", prefix="⚠️")
        return

    log(f"Executing {task_type.lower()} task {task_id[:8]} · {model_name}", prefix="🚀")

    image_b64: Optional[str] = None
    video_b64: Optional[str] = None
    metrics: Dict[str, Any]
    util: int = utilization_hint
    entry: Optional[ModelEntry] = None
    model_path: Optional[Path] = None
    try:
        entry = ensure_model_entry(model_name)
        if task_type in {"IMAGE_GEN", "ANIMATEDIFF", "FACE_SWAP"}:
            model_path = ensure_model_path(entry)
    except Exception as exc:
        metrics = {
            "status": "failed",
            "task_type": task_type.lower(),
            "model_name": model_name,
            "reward_weight": reward_weight,
            "error": f"Model resolution failed: {exc}",
        }
    else:
        if task_type == "IMAGE_GEN":
            # Parse structured settings from task if present
            job_settings = None
            prompt_raw = task.get("prompt")
            try:
                task_loras = task.get("loras") if isinstance(task, dict) else None
                if task_loras:
                    if job_settings is None:
                        job_settings = {}
                    job_settings["loras"] = task_loras
                if isinstance(prompt_raw, dict):
                    job_settings = prompt_raw
                    prompt = str(job_settings.get("prompt") or prompt)
                    negative_prompt = str(job_settings.get("negative_prompt") or negative_prompt)
                elif isinstance(prompt_raw, str) and prompt_raw.strip().startswith("{"):
                    job_settings = json.loads(prompt_raw)
                    if isinstance(job_settings, dict):
                        prompt = str(job_settings.get("prompt") or prompt)
                        negative_prompt = str(job_settings.get("negative_prompt") or negative_prompt)
                    else:
                        job_settings = None
                elif isinstance(prompt_raw, str):
                    prompt = prompt_raw
            except Exception as exc:
                log(f"Failed to parse job settings: {exc}", prefix="⚠️", task_id=task_id)
            # Merge coordinator-sent overrides even when prompt is plain text.
            if isinstance(task, dict):
                for key in ("steps", "guidance", "width", "height", "sampler", "seed"):
                    value = task.get(key)
                    if value is None or value == "":
                        continue
                    if job_settings is None:
                        job_settings = {}
                    job_settings.setdefault(key, value)

            metrics, util, image_b64 = run_image_generation(
                task_id,
                entry,
                model_path,
                reward_weight,
                prompt,
                negative_prompt,
                job_settings,
            )
        elif task_type == "VIDEO_GEN":
            metrics, util, video_b64 = _run_ltx2_task(task_id, entry, reward_weight, task)
        elif task_type == "ANIMATEDIFF":
            metrics, util, video_b64 = _run_animatediff_task(task_id, entry, model_path, reward_weight, task)
        elif task_type == "FACE_SWAP":
            metrics, util, image_b64 = _run_faceswap_task(task_id, entry, model_path, reward_weight, task)
        else:
            metrics = {
                "status": "failed",
                "task_type": task_type.lower(),
                "model_name": model_name,
                "reward_weight": reward_weight,
                "error": f"Unsupported task type: {task_type}",
            }

    with lock:
        utilization_hint = util

    payload = {
        "node_id": NODE_NAME,
        "task_id": task_id,
        "status": metrics.pop("status", "success"),
        "metrics": metrics,
        "utilization": utilization_hint,
        "submitted_at": time.time(),
    }
    if image_b64:
        payload["image_b64"] = image_b64
    if video_b64:
        payload["video_b64"] = video_b64

    submit_error: Optional[Exception] = None
    reward: Any = None
    for attempt in range(2):
        try:
            resp = SESSION.post(endpoint("/results"), data=json.dumps(payload), timeout=15)
            resp.raise_for_status()
            reward = resp.json().get("reward")
            submit_error = None
            break
        except Exception as exc:
            submit_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404 and attempt == 0:
                log(
                    f"Result submit returned 404 for {task_id[:8]}; retrying once",
                    prefix="⚠️",
                )
                time.sleep(1.0)
                continue
            break

    if submit_error is None:
        prefix = "✅" if payload["status"] == "success" else "⚠️"
        log(f"Task {task_id[:8]} {payload['status'].upper()} · reward {reward} HAI", prefix=prefix)
    else:
        log(f"Failed to submit result: {submit_error}", prefix="🚫")


def run_ai_inference(entry: ModelEntry, model_path: Path, input_shape: List[int], reward_weight: float) -> (Dict[str, Any], int):
    try:
        ort_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
    except Exception as exc:
        return ({"status": "failed", "error": f"session init: {exc}", "reward_weight": reward_weight}, utilization_hint)

    if np is None:
        return ({"status": "failed", "error": "numpy missing", "reward_weight": reward_weight}, utilization_hint)

    tensor = random_input(input_shape)
    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""
    try:
        ort_session.run(None, {input_name: tensor})
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    metrics = {
        "status": status,
        "model_name": entry.name,
        "model_path": str(model_path),
        "input_shape": input_shape,
        "reward_weight": reward_weight,
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
    }
    if status == "failed":
        metrics["error"] = error_msg or "inference error"
    return metrics, int(util)


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

POSITION_LORA_NAMES = {
    "povdoggy",
    "povreversecowgirl",
    "pscowgirl",
    "missionaryvaginal-v2",
}

ANATOMY_LORA_NAMES = {
    "handv2",
    "detailedperfectionsd1.5",
    "detailedperfectionsd15",
    "detailed perfection sd1.5",
}
ANATOMY_LORA_PREFIXES = ("badanatomy_sdxl_negative_lora",)

STYLE_LORA_NAMES = {
    "perfectionstyle",
    "perfectionstylesd1.5",
    "perfectionstylev2d",
    "skintexturestylesd1.5v1",
    "skintexturestylev3",
    "skintexturestylev5",
    "hyperlora_sdxl",
}

LORA_BASE_TYPE = {
    "povdoggy": "sd15",
    "povreversecowgirl": "sd15",
    "pscowgirl": "sd15",
    "missionaryvaginal-v2": "sd15",
    "handv2": "sdxl",
    "detailedperfectionsd1.5": "sd15",
    "detailedperfectionsd15": "sd15",
    "detailed perfection sd1.5": "sd15",
    "perfectionstylesd1.5": "sd15",
    "skintexturestylesd1.5v1": "sd15",
    "perfectionstyle": "sdxl",
    "perfectionstylev2d": "sdxl",
    "skintexturestylev3": "sdxl",
    "skintexturestylev5": "sdxl",
    "hyperlora_sdxl": "sdxl",
}

ROLE_WEIGHT_RANGES = {
    "position": (0.0, 2.0, 0.6),
    "anatomy": (0.5, 0.8, 0.65),
    "style": (0.3, 0.4, 0.35),
}


def is_sdxl_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    if "sdxl" in name:
        return True
    if "sd1.5" in name or "sd15" in name:
        return False
    if name.endswith("xl") or "xl_" in name or "_xl" in name or "-xl" in name:
        return True
    return "xl" in name


def normalize_lora_name(lora_name: str) -> str:
    base = Path(lora_name).name
    if base.lower().endswith(".safetensors"):
        base = base[:-len(".safetensors")]
    return base.strip().lower()


def infer_lora_base_type(normalized_name: str) -> Optional[str]:
    for prefix in ANATOMY_LORA_PREFIXES:
        if normalized_name.startswith(prefix):
            return "sdxl"
    mapped = LORA_BASE_TYPE.get(normalized_name)
    if mapped:
        return mapped
    if "sdxl" in normalized_name:
        return "sdxl"
    if "sd1.5" in normalized_name or "sd15" in normalized_name or "sd_15" in normalized_name or "sd-15" in normalized_name:
        return "sd15"
    return None


def classify_lora_role(normalized_name: str) -> str:
    if normalized_name in POSITION_LORA_NAMES:
        return "position"
    if normalized_name in ANATOMY_LORA_NAMES:
        return "anatomy"
    for prefix in ANATOMY_LORA_PREFIXES:
        if normalized_name.startswith(prefix):
            return "anatomy"
    if normalized_name in STYLE_LORA_NAMES:
        return "style"
    # Unknown LoRAs are treated conservatively as style to keep weights low.
    return "style"


def clamp_lora_weight(weight: Optional[float], role: str) -> float:
    min_w, max_w, default_w = ROLE_WEIGHT_RANGES.get(role, (0.25, 0.45, 0.35))
    try:
        value = float(weight) if weight is not None else default_w
    except (TypeError, ValueError):
        value = default_w
    if not math.isfinite(value):
        value = default_w
    if role == "position":
        # Respect server-provided position weights without forcing a narrow clamp.
        return value
    return max(min_w, min(max_w, value))


def resolve_lora_path(lora_name: str) -> Optional[Path]:
    candidate = Path(lora_name).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".safetensors")
    fallback = LORA_DIR / candidate.name
    if fallback.exists():
        return fallback
    return None


def _resolve_image_runtime(entry: ModelEntry) -> Tuple[str, Any, bool, str]:
    pipeline_name = (getattr(entry, "pipeline", "") or "sd15").lower()
    is_xl = "sdxl" in pipeline_name or ("xl" in pipeline_name and "sd15" not in pipeline_name)
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if torch is None:
        dtype = None
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype, is_xl, pipeline_name


def _image_pipeline_cache_key(model_path: Path, pipeline_name: str, device: str, dtype: Any) -> str:
    try:
        model_ref = str(model_path.resolve())
    except Exception:
        model_ref = str(model_path)
    dtype_ref = str(dtype)
    return f"{model_ref}|{pipeline_name}|{device}|{dtype_ref}"


def _release_image_pipeline(pipe: Any) -> None:
    if pipe is None:
        return
    try:
        pipe.to("cpu")
    except Exception:
        pass
    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _configure_image_pipeline(pipe: Any, entry: ModelEntry, is_xl: bool, device: str) -> Any:
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        if ENABLE_XFORMERS:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log("xformers memory-efficient attention enabled", prefix="✅")
            except Exception as exc:
                log(f"xformers unavailable, using default attention: {exc}", prefix="ℹ️")
        else:
            log("xformers disabled (set HAI_ENABLE_XFORMERS=1 to enable)", prefix="ℹ️")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if "_DPMSolver" in globals() and _DPMSolver is not None and hasattr(pipe, "scheduler"):
        try:
            pipe.scheduler = _DPMSolver.from_config(pipe.scheduler.config)
        except Exception as exc:
            log(f"DPM scheduler setup failed: {exc}", prefix="⚠️")
    pipe = pipe.to(device)
    if torch is not None and not is_xl and hasattr(pipe, "vae") and pipe.vae is not None:
        try:
            pipe.vae.to(device=device, dtype=torch.float32)
            if hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "force_upcast"):
                pipe.vae.config.force_upcast = True
            log("Upcasted VAE to fp32 for SD1.5 stability", prefix="✅")
        except Exception as exc:
            log(f"VAE upcast failed: {exc}", prefix="⚠️")
    return pipe


def _construct_base_image_pipeline(
    entry: ModelEntry,
    model_path: Path,
    pipeline_name: str,
    dtype: Any,
    is_xl: bool,
    device: str,
) -> Tuple[Any, int]:
    load_t0 = time.time()
    pipe = None
    if pipeline_name in {"sdxl"}:
        if _SDXLPipe is not None:
            try:
                pipe = _SDXLPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
            except Exception as exc:
                log(f"StableDiffusionXLPipeline load failed: {exc}", prefix="⚠️")
        auto_from_single = getattr(_AutoPipe, "from_single_file", None) if _AutoPipe is not None else None
        if pipe is None and callable(auto_from_single):
            try:
                pipe = auto_from_single(str(model_path), torch_dtype=dtype, safety_checker=None)
            except Exception as exc:
                log(f"AutoPipeline load failed: {exc}", prefix="⚠️")
    if pipe is None and _SDPipe is not None:
        pipe = _SDPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
    if pipe is None:
        raise RuntimeError("Failed to construct a text2image pipeline for model.")
    if getattr(entry, "vae_path", "") and pipeline_name != "sdxl":
        vae_path = Path(str(getattr(entry, "vae_path", ""))).expanduser()
        if vae_path.exists() and _AutoencoderKL is not None:
            try:
                custom_vae = _AutoencoderKL.from_single_file(str(vae_path), torch_dtype=dtype)
                pipe.vae = custom_vae
                log(f"Loaded custom VAE for {entry.name}", prefix="✅", vae=str(vae_path))
            except Exception as exc:
                log(f"Custom VAE load failed for {entry.name}: {exc}", prefix="⚠️", vae=str(vae_path))
    pipe = _configure_image_pipeline(pipe, entry, is_xl, device)
    load_ms = int((time.time() - load_t0) * 1000)
    return pipe, load_ms


def _acquire_base_image_pipeline(
    entry: ModelEntry,
    model_path: Path,
    pipeline_name: str,
    dtype: Any,
    is_xl: bool,
    device: str,
) -> Tuple[Any, bool, int]:
    if IMAGE_PIPELINE_CACHE_SIZE <= 0:
        pipe, load_ms = _construct_base_image_pipeline(entry, model_path, pipeline_name, dtype, is_xl, device)
        return pipe, False, load_ms

    cache_key = _image_pipeline_cache_key(model_path, pipeline_name, device, dtype)
    with _IMAGE_PIPELINE_CACHE_LOCK:
        cached_pipe = _IMAGE_PIPELINE_CACHE.pop(cache_key, None)
        if cached_pipe is not None:
            _IMAGE_PIPELINE_CACHE[cache_key] = cached_pipe
            return cached_pipe, True, 0

    pipe, load_ms = _construct_base_image_pipeline(entry, model_path, pipeline_name, dtype, is_xl, device)
    evicted: List[Any] = []
    with _IMAGE_PIPELINE_CACHE_LOCK:
        _IMAGE_PIPELINE_CACHE[cache_key] = pipe
        while len(_IMAGE_PIPELINE_CACHE) > IMAGE_PIPELINE_CACHE_SIZE:
            _, old_pipe = _IMAGE_PIPELINE_CACHE.popitem(last=False)
            evicted.append(old_pipe)
    for old_pipe in evicted:
        _release_image_pipeline(old_pipe)
    return pipe, False, load_ms


def _collect_explicit_loras(requested_raw: Any, entry: ModelEntry, pipeline_name: str) -> List[Tuple[Path, float, str]]:
    requested = requested_raw
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, list):
        return []

    model_hint = f"{entry.name} {pipeline_name}".strip()
    model_is_sdxl = is_sdxl_model(model_hint)
    lora_entries: List[Tuple[Path, float, str]] = []
    seen: Set[str] = set()
    for item in requested:
        if len(lora_entries) >= MAX_LORAS:
            break
        raw_weight: Optional[float] = None
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            raw_weight = item.get("weight")
        else:
            name = str(item or "").strip()
        if not name:
            continue
        normalized = normalize_lora_name(name)
        if normalized in seen:
            continue
        base_type = infer_lora_base_type(normalized)
        if base_type == "sdxl" and not model_is_sdxl:
            continue
        if base_type == "sd15" and model_is_sdxl:
            continue
        lora_path = resolve_lora_path(name)
        if not lora_path:
            continue
        role = classify_lora_role(normalized)
        weight = clamp_lora_weight(raw_weight, role)
        adapter_name = f"lora_{role}_{lora_path.stem}"
        lora_entries.append((lora_path, weight, adapter_name))
        seen.add(normalized)
    return lora_entries


def _apply_explicit_loras(pipe: Any, entry: ModelEntry, lora_entries: List[Tuple[Path, float, str]]) -> Tuple[List[str], int]:
    if not lora_entries:
        return [], 0
    lora_t0 = time.time()
    adapter_names: List[str] = []
    adapter_weights: List[float] = []
    loaded: List[str] = []
    if hasattr(pipe, "load_lora_weights"):
        for lora_path, lora_weight, adapter_name in lora_entries:
            adapter = adapter_name
            try:
                pipe.load_lora_weights(str(lora_path), adapter_name=adapter)
            except TypeError:
                pipe.load_lora_weights(str(lora_path))
            except Exception as exc:
                log(f"LoRA load failed for {entry.name}: {exc}", prefix="⚠️", lora=str(lora_path))
                continue
            adapter_names.append(adapter)
            adapter_weights.append(lora_weight)
            loaded.append(f"{lora_path.name}:{lora_weight:.2f}")
            log(f"Loaded LoRA for {entry.name}", prefix="✅", lora=str(lora_path), weight=lora_weight)
    if adapter_names:
        if hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except TypeError:
                pipe.set_adapters(adapter_names)
        elif hasattr(pipe, "fuse_lora"):
            try:
                pipe.fuse_lora()
            except Exception as exc:
                log(f"LoRA fuse failed for {entry.name}: {exc}", prefix="⚠️")
    else:
        log("Pipeline does not support LoRA loading", prefix="⚠️")
    return loaded, int((time.time() - lora_t0) * 1000)


def run_image_generation(
    task_id: str,
    entry: ModelEntry,
    model_path: Path,
    reward_weight: float,
    prompt: str,
    negative_prompt: str,
    job_settings: Optional[Dict[str, Any]] = None,
) -> (Dict[str, Any], int, Optional[str]):
    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

    image_b64: Optional[str] = None
    loaded_loras: List[str] = []
    pipeline_cache_hit = False
    pipeline_load_ms = 0
    lora_load_ms = 0
    generation_ms = 0
    output_path = OUTPUTS_DIR / f"{task_id}.png"

    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    use_img2img = False
    init_image_raw: Optional[str] = None
    img2img_strength = 0.75
    if job_settings and isinstance(job_settings, dict):
        try:
            width = int(job_settings.get("width", width) or width)
            height = int(job_settings.get("height", height) or height)
        except (TypeError, ValueError):
            pass
        init_image_raw = job_settings.get("init_image") or job_settings.get("init_image_url")
        use_img2img = bool(init_image_raw)
        try:
            img2img_strength = float(job_settings.get("img2img_strength", 0.75) or 0.75)
        except (TypeError, ValueError):
            img2img_strength = 0.75

    try:
        if not FAST_PREVIEW and torch is not None and diffusers is not None:
            device, dtype, is_xl, pipeline_name = _resolve_image_runtime(entry)
            requested_loras = job_settings.get("loras") if isinstance(job_settings, dict) else []
            lora_entries = _collect_explicit_loras(requested_loras, entry, pipeline_name)
            if lora_entries:
                lora_summary = ", ".join(
                    f"{adapter}:{weight:.2f} ({path.name})" for path, weight, adapter in lora_entries
                )
            else:
                lora_summary = "none"
            log(f"Loading LoRAs: {lora_summary}", prefix="🎛️")

            init_pil = None
            if use_img2img and init_image_raw:
                try:
                    from PIL import Image as _PILImage
                    if init_image_raw.startswith("data:"):
                        b64_part = init_image_raw.split(",", 1)[-1]
                        img_bytes = base64.b64decode(b64_part)
                        init_pil = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    elif init_image_raw.startswith(("http://", "https://", "/")):
                        img_resp = requests.get(init_image_raw, timeout=30)
                        img_resp.raise_for_status()
                        init_pil = _PILImage.open(io.BytesIO(img_resp.content)).convert("RGB")
                    elif os.path.isfile(init_image_raw):
                        init_pil = _PILImage.open(init_image_raw).convert("RGB")
                    else:
                        img_bytes = base64.b64decode(init_image_raw)
                        init_pil = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    if init_pil:
                        init_pil = init_pil.resize((width, height))
                        log(f"Init image loaded for img2img (strength={img2img_strength})", prefix="🖼️")
                except Exception as exc:
                    log(f"Failed to load init image, falling back to txt2img: {exc}", prefix="⚠️")
                    init_pil = None

            pipe_mode = "img2img" if init_pil is not None else "txt2img"
            log(f"Preparing {pipe_mode} pipeline…", prefix="ℹ️", device=device)
            transient_pipeline = bool(lora_entries) or IMAGE_PIPELINE_CACHE_SIZE <= 0
            pipe: Optional[Any] = None
            try:
                if transient_pipeline:
                    pipe, pipeline_load_ms = _construct_base_image_pipeline(
                        entry, model_path, pipeline_name, dtype, is_xl, device
                    )
                else:
                    pipe, pipeline_cache_hit, pipeline_load_ms = _acquire_base_image_pipeline(
                        entry, model_path, pipeline_name, dtype, is_xl, device
                    )
                if not pipeline_cache_hit:
                    log(f"Pipeline ready in {pipeline_load_ms}ms", prefix="✅")
                if lora_entries and pipe is not None:
                    loaded_loras, lora_load_ms = _apply_explicit_loras(pipe, entry, lora_entries)
                if pipe is None:
                    raise RuntimeError("Image pipeline is unavailable")

                seed = job_settings.get("seed") if isinstance(job_settings, dict) else None
                try:
                    seed = int(seed) if seed is not None else None
                except (TypeError, ValueError):
                    seed = None
                if seed is None:
                    seed = int(time.time()) & 0x7FFFFFFF
                generator = torch.Generator(device=device).manual_seed(seed)
                pos_text = prompt or "a high quality photo of a golden retriever on a beach at sunset"
                neg_text = negative_prompt or ""
                steps = IMAGE_STEPS
                guidance = IMAGE_GUIDANCE
                img_h = IMAGE_HEIGHT
                img_w = IMAGE_WIDTH
                sampler = None
                if job_settings and isinstance(job_settings, dict):
                    steps = int(job_settings.get("steps", steps) or steps)
                    guidance = float(job_settings.get("guidance", guidance) or guidance)
                    img_h = int(job_settings.get("height", img_h) or img_h)
                    img_w = int(job_settings.get("width", img_w) or img_w)
                    sampler = str(job_settings.get("sampler") or "").strip().lower() or None
                    if job_settings.get("negative_prompt"):
                        neg_text = str(job_settings.get("negative_prompt") or "")
                steps = max(5, min(50, steps))
                guidance = max(1.0, min(15.0, guidance))
                img_h = max(256, min(1536, img_h))
                img_w = max(256, min(1536, img_w))
                if hasattr(pipe, "tokenizer") and hasattr(pipe.tokenizer, "model_max_length"):
                    try:
                        encoded = pipe.tokenizer(
                            pos_text,
                            max_length=pipe.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        pos_text = pipe.tokenizer.batch_decode(
                            encoded.input_ids, skip_special_tokens=True
                        )[0]
                        if neg_text:
                            encoded_neg = pipe.tokenizer(
                                neg_text,
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                            neg_text = pipe.tokenizer.batch_decode(
                                encoded_neg.input_ids, skip_special_tokens=True
                            )[0]
                    except Exception as exc:
                        log(f"Prompt truncation failed: {exc}", prefix="⚠️")
                gen_t0 = time.time()
                if sampler and hasattr(pipe, "scheduler") and _DPMSolver is not None:
                    sampler_norm = sampler.replace("+", "p")
                    if "dpmpp" in sampler_norm:
                        try:
                            pipe.scheduler = _DPMSolver.from_config(pipe.scheduler.config)
                        except Exception as exc:
                            log(f"Sampler switch failed: {exc}", prefix="⚠️", sampler=sampler)
                with torch.inference_mode():
                    result = pipe(
                        pos_text,
                        negative_prompt=neg_text or None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        height=img_h,
                        width=img_w,
                    )
                generation_ms = int((time.time() - gen_t0) * 1000)
                log(f"Generated in {generation_ms}ms", prefix="✅")
                img = result.images[0]
                img.save(output_path)
                with output_path.open("rb") as fh:
                    image_b64 = base64.b64encode(fh.read()).decode("utf-8")
            finally:
                if transient_pipeline and pipe is not None:
                    _release_image_pipeline(pipe)
        elif Image is not None:
            log("Using fast preview placeholder (no SD detected or FAST_PREVIEW enabled)", prefix="ℹ️")
            w = h = 512
            if np is not None:
                yy = np.linspace(0, 255, h, dtype=np.uint8)
                xx = np.linspace(0, 255, w, dtype=np.uint8)
                red = np.tile(xx, (h, 1))
                green = np.tile(yy, (w, 1)).T
                blue = (red[::-1] // 2 + green // 2).astype(np.uint8)
                arr = np.dstack([red, green, blue])
            else:
                arr = None
            img = Image.fromarray(arr) if arr is not None else Image.new("RGB", (w, h), color=(40, 60, 80))
            img.save(output_path)
            with output_path.open("rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode("utf-8")
        else:
            time.sleep(random.uniform(1.2, 2.2))
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log(f"Image generation failed: {error_msg}", prefix="🚫")

    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    util = int(max(util, 70 if ROLE == "creator" else util))
    metrics = {
        "status": status,
        "model_name": entry.name,
        "model_path": str(model_path),
        "reward_weight": reward_weight,
        "task_type": "image_gen",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
        "pipeline_cache_hit": bool(pipeline_cache_hit),
        "pipeline_load_ms": int(pipeline_load_ms),
        "lora_load_ms": int(lora_load_ms),
        "generation_ms": int(generation_ms),
    }
    if loaded_loras:
        metrics["loras"] = loaded_loras
    if status == "failed":
        metrics["error"] = error_msg or "image generation error"
    return metrics, util, image_b64


def _iter_image_entries() -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for entry in REGISTRY.list_entries():
        task_type = str(getattr(entry, "task_type", "IMAGE_GEN") or "IMAGE_GEN").upper()
        if task_type == "IMAGE_GEN":
            entries.append(entry)
    return entries


def _select_preload_image_entry() -> Optional[ModelEntry]:
    if PRELOAD_IMAGE_MODEL:
        try:
            candidate = REGISTRY.get(PRELOAD_IMAGE_MODEL)
        except Exception as exc:
            log(f"Preload model '{PRELOAD_IMAGE_MODEL}' unavailable: {exc}", prefix="⚠️")
        else:
            task_type = str(getattr(candidate, "task_type", "IMAGE_GEN") or "IMAGE_GEN").upper()
            if task_type == "IMAGE_GEN":
                try:
                    ensure_model_path(candidate)
                    return candidate
                except Exception as exc:
                    log(f"Preload model '{PRELOAD_IMAGE_MODEL}' missing locally: {exc}", prefix="⚠️")
            else:
                log(f"Preload model '{PRELOAD_IMAGE_MODEL}' is not an IMAGE_GEN model", prefix="⚠️")
    for entry in _iter_image_entries():
        try:
            ensure_model_path(entry)
            return entry
        except Exception:
            continue
    return None


def preload_image_pipeline_on_startup() -> None:
    if not PRELOAD_IMAGE_ON_STARTUP:
        return
    if FAST_PREVIEW or torch is None or diffusers is None:
        return
    if IMAGE_PIPELINE_CACHE_SIZE <= 0:
        log("Skipping image preload because HAI_IMAGE_PIPELINE_CACHE_SIZE=0", prefix="ℹ️")
        return
    entry = _select_preload_image_entry()
    if entry is None:
        log("No local IMAGE_GEN model available for preload", prefix="⚠️")
        return
    try:
        model_path = ensure_model_path(entry)
        device, dtype, is_xl, pipeline_name = _resolve_image_runtime(entry)
        _, cache_hit, load_ms = _acquire_base_image_pipeline(entry, model_path, pipeline_name, dtype, is_xl, device)
        if cache_hit:
            log(f"Image preload cache warm for {entry.name}", prefix="✅")
        else:
            log(f"Image preload complete for {entry.name} in {load_ms}ms", prefix="✅")
    except Exception as exc:
        log(f"Image preload failed for {entry.name}: {exc}", prefix="⚠️")


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------


def heartbeat_loop() -> None:
    backoff = BACKOFF_BASE
    while True:
        gpu_stats = read_gpu_stats()
        capabilities = discover_capabilities()
        supports = discover_supports(capabilities)
        payload = {
            "node_id": NODE_NAME,
            "os": os.uname().sysname if hasattr(os, "uname") else os.name,
            "gpu": gpu_stats.get("gpu_name", "Simulated"),
            "gpu_stats": gpu_stats,
            "start_time": START_TIME,
            "uptime": time.time() - START_TIME,
            "role": ROLE,
            "version": CLIENT_VERSION,
            "node_name": NODE_NAME,
            "models": capabilities["models"],
            "loras": list_local_loras(),
            "pipelines": capabilities["pipelines"],
            "supports": supports,
        }
        try:
            resp = SESSION.post(endpoint("/register"), data=json.dumps(payload), timeout=5)
            resp.raise_for_status()
            backoff = BACKOFF_BASE
            log(f"Heartbeat OK ({ROLE})", prefix="✅")
            try:
                REGISTRY.refresh()
            except Exception as exc:
                log(f"Manifest refresh failed (heartbeat): {exc}", prefix="⚠️")
        except Exception as exc:
            log(f"Heartbeat failed: {exc}", prefix="⚠️")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        else:
            time.sleep(HEARTBEAT_INTERVAL)


def poll_tasks_loop() -> None:
    backoff = BACKOFF_BASE
    while True:
        try:
            resp = SESSION.get(endpoint("/tasks/creator"), params={"node_id": NODE_NAME}, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            tasks = payload.get("tasks", [])
            if tasks:
                log(f"Received {len(tasks)} task(s)", prefix="📥")
            for task in tasks:
                execute_task(task)
            backoff = BACKOFF_BASE
        except Exception as exc:
            log(f"Task poll failed: {exc}", prefix="⚠️")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        else:
            time.sleep(TASK_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log(f"Node ID: {NODE_NAME} · Role: {ROLE.upper()} · Version: {CLIENT_VERSION}")
    manifest_ready = refresh_manifest_with_backoff("startup", max_attempts=5)
    if not manifest_ready:
        log("Proceeding with degraded manifest state; heartbeat loop will keep retrying.", prefix="⚠️")
    else:
        preload_image_pipeline_on_startup()
    link_wallet(WALLET)
    # Graceful shutdown hooks
    atexit.register(disconnect)
    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    poll_tasks_loop()

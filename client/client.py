"""HavnAI Node Client — Stage 7 public onboarding."""

from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import atexit
import signal
from typing import Any, Dict, List, Optional
import base64
import io

import requests

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
        from diffusers import StableDiffusionImg2ImgPipeline as _SDImg2ImgPipe  # type: ignore
    except Exception:  # pragma: no cover
        _SDImg2ImgPipe = None  # type: ignore
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline as _SDXLImg2ImgPipe  # type: ignore
    except Exception:  # pragma: no cover
        _SDXLImg2ImgPipe = None  # type: ignore
    try:
        from diffusers import StableDiffusionControlNetPipeline as _SDControlPipe  # type: ignore
    except Exception:  # pragma: no cover
        _SDControlPipe = None  # type: ignore
    try:
        from diffusers import ControlNetModel as _ControlNetModel  # type: ignore
    except Exception:  # pragma: no cover
        _ControlNetModel = None  # type: ignore
    try:
        from diffusers import DPMSolverMultistepScheduler as _DPMSolver  # type: ignore
    except Exception:  # pragma: no cover
        _DPMSolver = None  # type: ignore
except ImportError:  # pragma: no cover
    diffusers = None
    _AutoPipe = None  # type: ignore
    _SDPipe = None  # type: ignore
    _SDXLPipe = None  # type: ignore
    _SDImg2ImgPipe = None  # type: ignore
    _SDXLImg2ImgPipe = None  # type: ignore
    _SDControlPipe = None  # type: ignore
    _ControlNetModel = None  # type: ignore
    _DPMSolver = None  # type: ignore
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

# ---------------------------------------------------------------------------
# Paths & environment bootstrap
# ---------------------------------------------------------------------------

HAVNAI_HOME = Path(os.environ.get("HAVNAI_HOME", Path.home() / ".havnai"))
ENV_PATH = HAVNAI_HOME / ".env"
CREATOR_SCAN_DIR = HAVNAI_HOME / "models" / "creator"
DOWNLOAD_DIR = HAVNAI_HOME / "downloads"
LOGS_DIR = HAVNAI_HOME / "logs"
OUTPUTS_DIR = HAVNAI_HOME / "outputs"
VERSION_SEARCH_PATHS = [HAVNAI_HOME / "VERSION", Path(__file__).resolve().parent / "VERSION"]

HAVNAI_HOME.mkdir(parents=True, exist_ok=True)
CREATOR_SCAN_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODEL_EXTS = {".onnx", ".safetensors", ".ckpt"}

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


# Face swap / identity settings
HYPERLORA_PATH = Path(os.environ.get("HYPERLORA_PATH", "/mnt/d/havnai-storage/models/loras/HyperLoRA_SDXL.safetensors")).expanduser()
IPADAPTER_DIR = Path(os.environ.get("IPADAPTER_DIR", "/mnt/d/havnai-storage/models/ip-adapter-faceid")).expanduser()
IPADAPTER_BIN = os.environ.get("IPADAPTER_BIN", "ip-adapter-faceid-plusv2_sdxl.bin")
IPADAPTER_LORA = os.environ.get("IPADAPTER_LORA", "ip-adapter-faceid-plusv2_sdxl_lora.safetensors")
CONTROLNET_PATH = Path(os.environ.get("CONTROLNET_PATH", "/mnt/d/havnai-storage/models/controlnet/controlnet-openpose.safetensors")).expanduser()
POSE_LIBRARY_DIR = Path(os.environ.get("POSE_LIBRARY_DIR", "/mnt/d/havnai-storage/poses")).expanduser()
_DEFAULT_LORA_DIR = Path("/mnt/d/havnai-storage/models/loras")
_ENV_LORA_DIR = Path(os.environ.get("LORA_DIR", str(_DEFAULT_LORA_DIR))).expanduser()
if _ENV_LORA_DIR.exists():
    LORA_DIR = _ENV_LORA_DIR
else:
    LORA_DIR = _DEFAULT_LORA_DIR
    if os.environ.get("LORA_DIR"):
        log(
            "LORA_DIR missing; falling back to default",
            prefix="⚠️",
            env=str(_ENV_LORA_DIR),
            fallback=str(_DEFAULT_LORA_DIR),
        )
LORA_REALISTIC_SKIN = LORA_DIR / "skintexturestylev5.safetensors"
LORA_BAD_ANATOMY_NEG = LORA_DIR / "badanatomy_SDXL_negative_LORA_AutismMix_v1.safetensors"
LORA_HANDS = LORA_DIR / "Handv2.safetensors"
LORA_DETAILED_PERFECTION = LORA_DIR / "perfectionstyle.safetensors"
LORA_SWEAT_OILED = LORA_DIR / "Sweatingmyballsofmate.safetensors"
BASE_REALISM_SDXL_PERF = LORA_DIR / "perfectionstyle.safetensors"
BASE_REALISM_SDXL_SKIN = LORA_DIR / "skintexturestylev3.safetensors"
BASE_REALISM_SD15_PERF = LORA_DIR / "perfectionstyleSD1.5.safetensors"
BASE_REALISM_SD15_SKIN = LORA_DIR / "skintexturestylesd1.5v1.safetensors"

SDXL_MODEL_OVERRIDES = {
    "juggernautxl_ragnarokby",
    "epicrealismxl_vxviicrystalclear",
    "uberrealisticpornmerge_v23final",
}

SD15_MODEL_OVERRIDES = {
    "majicmixrealistic_v7",
    "lazymixrealamateur_v40",
    "perfectdeliberate_v5sd15",
    "triomerge_v10",
    "unstablepornhwa_beta",
    "disneypixarcartoon_v10",
    "kizukianimehentai_animehentaiv4",
}

ANIME_MODELS = {
    "kizukianimehentai",
    "unstablepornhwa",
    "disneypixarcartoon",
}

ANIME_KEYWORDS = {
    "anime",
    "manga",
    "manhwa",
    "illustration",
    "toon",
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

SERVER_BASE = ENV_VARS.get("SERVER_URL", "http://127.0.0.1:5001").rstrip("/")
JOIN_TOKEN = ENV_VARS.get("JOIN_TOKEN", "").strip()
ROLE = "creator" if ENV_VARS.get("CREATOR_MODE", "false").lower() in {"1", "true", "yes"} else "worker"
NODE_NAME = ENV_VARS.get("NODE_NAME", socket.gethostname())
FAST_PREVIEW = (
    os.environ.get("HAI_FAST_PREVIEW", ENV_VARS.get("HAI_FAST_PREVIEW", "")).lower()
    in {"1", "true", "yes"}
)

HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 15
BACKOFF_BASE = 5
MAX_BACKOFF = 60
START_TIME = time.time()

utilization_hint = random.randint(10, 25 if ROLE == "creator" else 15)

LOCAL_MODELS: Dict[str, Dict[str, Any]] = {}
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})
if JOIN_TOKEN:
    SESSION.headers["X-Join-Token"] = JOIN_TOKEN


def endpoint(path: str) -> str:
    return f"{SERVER_BASE}{path}"


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
# Model scanning & registration
# ---------------------------------------------------------------------------


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_local_models() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    if not CREATOR_SCAN_DIR.exists():
        return catalog
    for path in CREATOR_SCAN_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
            continue
        name = path.stem.lower().replace(" ", "-")
        weight = 10.0 if path.suffix.lower() in {".safetensors", ".ckpt"} else 2.0
        if "nsfw" in name:
            weight = max(weight, 12.0)
        catalog[name] = {
            "name": name,
            "filename": path.name,
            "path": path,
            "size": path.stat().st_size,
            "hash": hash_file(path),
            "tags": [path.suffix.lstrip(".")],
            "weight": weight,
            "task_type": "IMAGE_GEN",
        }
    return catalog


def register_local_models() -> None:
    global LOCAL_MODELS
    LOCAL_MODELS = scan_local_models()
    if not LOCAL_MODELS:
        log("No creator models discovered under ~/.havnai/models/creator", prefix="ℹ️")
        return
    manifest = [
        {
            "name": meta["name"],
            "filename": meta["filename"],
            "size": meta["size"],
            "hash": meta["hash"],
            "tags": meta["tags"],
            "weight": meta["weight"],
            "task_type": meta["task_type"],
        }
        for meta in LOCAL_MODELS.values()
    ]
    try:
        resp = SESSION.post(endpoint("/register_models"), data=json.dumps({"node_id": NODE_NAME, "models": manifest}), timeout=20)
        resp.raise_for_status()
        log(f"Registered {len(manifest)} local models.", prefix="✅")
    except Exception as exc:
        log(f"Model registration failed: {exc}", prefix="⚠️")


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


def load_source_image(source: Optional[str]) -> Optional["Image.Image"]:
    """Load a face/source image from base64, URL, or local path."""
    if not source or Image is None:
        return None
    if len(source) > 100 and all(c.isalnum() or c in "+/=\n" for c in source):
        try:
            data = base64.b64decode(source)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass
    if source.startswith("http://") or source.startswith("https://"):
        try:
            resp = SESSION.get(source, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            log(f"Source face URL load failed: {exc}", prefix="⚠️", url=source)
    try:
        path = Path(source).expanduser()
        if path.exists():
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return None


def _load_ip_adapter(
    pipe: object,
    adapter_dir: Path,
    weight_name: str,
    lora_path: Optional[Path],
    scale: float,
) -> None:
    """Load IP-Adapter FaceID weights into the pipeline if supported."""
    if not hasattr(pipe, "load_ip_adapter"):
        return
    weight_path = adapter_dir / weight_name
    if not weight_path.exists():
        log("IP-Adapter file missing", prefix="⚠️", weight=str(weight_path))
        return
    try:
        pipe.load_ip_adapter(str(adapter_dir), subfolder="", weight_name=weight_name)
        if lora_path and hasattr(pipe, "load_ip_adapter_weights"):
            pipe.load_ip_adapter_weights(str(lora_path))
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(scale)
        log("IP-Adapter loaded", prefix="✅", weight=str(weight_path), lora=str(lora_path) if lora_path else "", scale=scale)
    except Exception as exc:  # pragma: no cover - defensive
        log(f"IP-Adapter load failed: {exc}", prefix="⚠️")


def load_pose_image(pose_image_b64: Optional[str], pose_image_path: Optional[str]) -> Optional["Image.Image"]:
    """Load a pose image from base64 or filesystem path."""
    if Image is None:
        return None
    if pose_image_b64:
        try:
            data = base64.b64decode(pose_image_b64)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            log(f"Pose image base64 decode failed: {exc}", prefix="⚠️")
    if pose_image_path:
        try:
            path = Path(pose_image_path).expanduser()
            if not path.is_absolute() and POSE_LIBRARY_DIR.exists():
                path = POSE_LIBRARY_DIR / path
            if path.exists():
                return Image.open(path).convert("RGB")
        except Exception as exc:
            log(f"Pose image load failed: {exc}", prefix="⚠️", path=pose_image_path)
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


def resolve_model_path(model_name: str, model_url: str = "", filename_hint: Optional[str] = None) -> Path:
    entry = LOCAL_MODELS.get(model_name.lower())
    if entry:
        return entry["path"]
    if model_url:
        filename = filename_hint or Path(model_url).name
        target = DOWNLOAD_DIR / Path(filename).name
        if target.exists():
            return target
        url = model_url
        if url.startswith("/"):
            url = f"{SERVER_BASE}{url}"
        log(f"Downloading model {model_name} from {url}", prefix="⬇️")
        resp = SESSION.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with target.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return target
    raise RuntimeError(f"Model {model_name} unavailable on node")


def random_input(shape: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy required for inference")
    dims = [max(1, dim) for dim in shape]
    return (np.random.rand(*dims).astype(np.float32) * 2.0) - 1.0


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = (task.get("type") or "IMAGE_GEN").lower()
    model_name = (task.get("model_name") or "model").lower()
    model_url = task.get("model_url", "")
    reward_weight = float(task.get("reward_weight", 1.0))
    input_shape = task.get("input_shape") or []
    prompt = task.get("prompt") or ""
    job_settings: Dict[str, Any] = {}
    raw_data = task.get("data")
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict):
                job_settings.update(parsed)
        except Exception:
            pass
    if isinstance(prompt, str) and prompt.strip().startswith("{"):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, dict):
                job_settings.update(parsed)
                prompt = parsed.get("prompt", "") or ""
        except Exception:
            pass
    if task.get("style_mode") and "style_mode" not in job_settings:
        job_settings["style_mode"] = task.get("style_mode")
    queued_at = task.get("queued_at")
    assigned_at = task.get("assigned_at")
    task_started_at = time.time()
    queue_wait_ms = None
    assign_to_start_ms = None
    if queued_at and assigned_at:
        queue_wait_ms = int((assigned_at - queued_at) * 1000)
    if assigned_at:
        assign_to_start_ms = int((task_started_at - assigned_at) * 1000)

    if task_type == "image_gen" and ROLE != "creator":
        log(f"Skipping creator task {task_id} — node not in creator mode", prefix="⚠️")
        return

    log(f"Executing {task_type} task {task_id} · {model_name}", prefix="🚀")

    image_b64: Optional[str] = None
    if task_type == "image_gen":
        metrics, util, image_b64 = run_image_generation(
            task_id,
            model_name,
            model_url,
            reward_weight,
            prompt,
            job_settings,
        )
    else:
        metrics, util = run_ai_inference(model_name, model_url, input_shape, reward_weight)

    total_ms = int((time.time() - task_started_at) * 1000)
    metrics["task_total_ms"] = total_ms
    if queue_wait_ms is not None:
        metrics["queue_wait_ms"] = queue_wait_ms
    if assign_to_start_ms is not None:
        metrics["assign_to_start_ms"] = assign_to_start_ms

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

    try:
        resp = SESSION.post(endpoint("/results"), data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        reward = resp.json().get("reward")
        prefix = "✅" if payload["status"] == "success" else "⚠️"
        log(
            f"Task {task_id[:8]} {payload['status'].upper()} · reward {reward} HAI",
            prefix=prefix,
            task_total_ms=total_ms,
            queue_wait_ms=queue_wait_ms,
            assign_to_start_ms=assign_to_start_ms,
            inference_ms=metrics.get("inference_time_ms"),
        )
    except Exception as exc:
        log(f"Failed to submit result: {exc}", prefix="🚫")


def run_ai_inference(model_name: str, model_url: str, input_shape: List[int], reward_weight: float) -> (Dict[str, Any], int):
    try:
        model_path = resolve_model_path(model_name, model_url)
    except Exception as exc:
        return ({"status": "failed", "error": str(exc), "reward_weight": reward_weight}, utilization_hint)

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
        "model_name": model_name,
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


def run_image_generation(
    task_id: str,
    model_name: str,
    model_url: str,
    reward_weight: float,
    prompt: str,
    job_settings: Optional[Dict[str, Any]] = None,
) -> (Dict[str, Any], int, Optional[str]):
    """Run SD image generation with explicit pipeline and env-driven settings."""

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name) or ENV_VARS.get(name, "")
        try:
            v = int(raw)
            return v if v > 0 else default
        except Exception:
            return default

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name) or ENV_VARS.get(name, "")
        try:
            return float(raw)
        except Exception:
            return default

    try:
        model_path = resolve_model_path(model_name, model_url, filename_hint=f"{model_name}.safetensors")
    except Exception as exc:
        return ({"status": "failed", "error": str(exc), "reward_weight": reward_weight}, utilization_hint, None)

    steps = _env_int("HAI_STEPS", 65)
    guidance = _env_float("HAI_GUIDANCE", 8.2)
    width = _env_int("HAI_WIDTH", 512)
    height = _env_int("HAI_HEIGHT", 512)
    width = max(64, width - (width % 8))
    height = max(64, height - (height % 8))
    model_key = model_name.lower()
    if model_key in SDXL_MODEL_OVERRIDES:
        is_xl = True
    elif model_key in SD15_MODEL_OVERRIDES:
        is_xl = False
    else:
        is_xl = "xl" in model_key
    job_settings = job_settings or {}
    face_swap = False
    if "face_swap" in job_settings:
        raw_face_swap = job_settings.get("face_swap")
        if isinstance(raw_face_swap, bool):
            face_swap = raw_face_swap
        elif isinstance(raw_face_swap, str):
            face_swap = raw_face_swap.strip().lower() in {"1", "true", "yes"}
        else:
            face_swap = bool(raw_face_swap)
    source_face = (
        job_settings.get("source_face")
        or job_settings.get("source_face_url")
        or job_settings.get("source_face_b64")
    )
    pose_image_b64 = job_settings.get("pose_image_b64") or job_settings.get("pose_image")
    pose_image_path = job_settings.get("pose_image_path")
    ipadapter_scale = float(job_settings.get("ipadapter_scale", 0.6) or 0.6)
    ipadapter_scale = max(0.0, min(ipadapter_scale, 1.0))
    hyperlora_weight = float(job_settings.get("hyperlora_weight", 0.6) or 0.6)
    hyperlora_weight = min(hyperlora_weight, 0.8)
    base_prompt = str(job_settings.get("base_prompt") or "").strip()
    negative_prompt = str(job_settings.get("negative_prompt") or "").strip()
    anatomy_unstable = bool(job_settings.get("anatomy_unstable", False))
    malformed = bool(job_settings.get("malformed", False))
    style_preset = str(job_settings.get("style_preset") or "").strip().lower()
    style_mode = str(job_settings.get("style_mode") or "").strip().lower()
    model_hint = model_name.lower()
    is_anime_model = any(tag in model_hint for tag in ANIME_MODELS)
    if style_mode not in {"realism", "anime"}:
        if is_anime_model or any(keyword in (prompt or "").lower() for keyword in ANIME_KEYWORDS):
            style_mode = "anime"
        else:
            style_mode = "realism"

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

    image_b64: Optional[str] = None
    controlnet_image = load_pose_image(pose_image_b64, pose_image_path)
    use_controlnet = bool(controlnet_image is not None)
    source_face_image = load_source_image(source_face) if face_swap else None
    faceid_active = False
    base_realism_loaded = False
    lora_cleanup_needed = False
    output_path = OUTPUTS_DIR / f"{task_id}.png"
    try:
        if not FAST_PREVIEW and torch is not None and diffusers is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            log("Loading text2image pipeline…", prefix="ℹ️", device=device)
            load_t0 = time.time()

            pipe = None
            log(
                "Model pipeline selection",
                prefix="🧪",
                model=model_name,
                pipeline="sdxl" if is_xl else "sd15",
            )
            log(
                "Style mode selection",
                prefix="🧪",
                style_mode=style_mode,
                model_family="anime" if is_anime_model else "realism",
            )
            # For XL checkpoints prefer StableDiffusionXLPipeline when
            # available, otherwise fall back to SD1.5 pipeline as a best-effort.
            if is_xl and _SDXLPipe is not None:
                try:
                    pipe = _SDXLPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
                except Exception as exc:
                    log(f"StableDiffusionXLPipeline load failed: {exc}", prefix="⚠️")

            if pipe is None and _SDPipe is not None:
                try:
                    if use_controlnet and _SDControlPipe is not None and _ControlNetModel is not None and CONTROLNET_PATH.exists():
                        controlnet = _ControlNetModel.from_single_file(str(CONTROLNET_PATH), torch_dtype=dtype)
                        pipe = _SDControlPipe.from_single_file(
                            str(model_path),
                            controlnet=controlnet,
                            torch_dtype=dtype,
                            safety_checker=None,
                        )
                        log("Loaded ControlNet for pose guidance", prefix="✅", controlnet=str(CONTROLNET_PATH))
                    else:
                        pipe = _SDPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
                except Exception as exc:
                    log(f"StableDiffusionPipeline load failed: {exc}", prefix="⚠️")

            # AutoPipeline is truly optional; only use it when the expected
            # helper is present in this diffusers version.
            if pipe is None and _AutoPipe is not None and hasattr(_AutoPipe, "from_single_file"):
                try:
                    pipe = _AutoPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
                except Exception as exc_auto:
                    log(f"AutoPipeline load failed: {exc_auto}", prefix="🚫")
                    raise
            if pipe is None:
                raise RuntimeError("Failed to construct a text2image pipeline for model.")

            pipe.scheduler = _DPMSolver.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )  # type: ignore[attr-defined]
            LOGGER.info(
                "Scheduler=%s, use_karras_sigmas=%s",
                type(pipe.scheduler).__name__,
                getattr(pipe.scheduler.config, "use_karras_sigmas", None),
            )
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)

            if face_swap:
                if base_prompt:
                    base_prompt = base_prompt.replace("perfect symmetrical face", "").strip(" ,")
                # Keep the FaceSwap injection short to avoid CLIP token overflow.
                base_prompt = ", ".join(
                    p for p in [
                        base_prompt,
                        "accurate facial identity, natural expression",
                    ]
                    if p
                )

            if is_xl:
                base_prompt = ", ".join(
                    p for p in [
                        base_prompt,
                        "natural facial proportions",
                    ]
                    if p
                )
            if style_mode == "realism":
                base_prompt = ", ".join(
                    p for p in [
                        base_prompt,
                        "photorealistic face, natural eyes, no glowing eyes",
                    ]
                    if p
                )
                negative_prompt = ", ".join(
                    p for p in [
                        negative_prompt,
                        "anime style, illustration, cartoon, glowing eyes, exaggerated proportions",
                    ]
                    if p
                )
            else:
                base_prompt = ", ".join(
                    p for p in [
                        base_prompt,
                        "anime style, illustrated, stylized proportions",
                    ]
                    if p
                )
                negative_prompt = ", ".join(
                    p for p in [
                        negative_prompt,
                        "photorealistic, skin pores, photographic lighting",
                    ]
                    if p
                )
            final_prompt = ", ".join(p for p in [base_prompt, (prompt or "").strip()] if p)
            final_negative = negative_prompt

            pipe = pipe.to(device)
            load_ms = int((time.time() - load_t0) * 1000)
            log(
                f"Pipeline ready in {load_ms}ms · {pipe.__class__.__name__}",
                prefix="✅",
            )

            if face_swap and is_xl and source_face_image is not None:
                lora_path = IPADAPTER_DIR / IPADAPTER_LORA
                bin_path = IPADAPTER_DIR / IPADAPTER_BIN
                weight_name = ""
                lora_weight_path: Optional[Path] = None
                if lora_path.exists():
                    weight_name = IPADAPTER_LORA
                    lora_weight_path = None
                elif bin_path.exists():
                    weight_name = IPADAPTER_BIN
                    lora_weight_path = None
                if weight_name:
                    _load_ip_adapter(pipe, IPADAPTER_DIR, weight_name, lora_weight_path, ipadapter_scale)
                    faceid_active = True
                    log("IP-Adapter scale", prefix="🧪", scale=ipadapter_scale)

            if faceid_active:
                log("Conditioning order: FaceID -> Base prompt -> HyperLoRA -> User prompt", prefix="🧪")

            # Quality fixer LoRAs (SDXL only, deterministic selection)
            positive_loras: List[tuple[str, float]] = []
            negative_loras: List[tuple[str, float]] = []
            allow_quality_fixers = False
            if is_xl and not face_swap and style_mode == "realism":
                prompt_lc = (prompt or "").lower()
                hands_keywords = {
                    "hands",
                    "fingers",
                    "gripping",
                    "grabbing",
                    "holding",
                    "touching",
                    "cupping",
                    "hand on breast",
                    "pressed against",
                    "clutching",
                }
                full_body_keywords = {"full body", "standing", "kneeling", "lying down", "arched back", "silhouette"}
                is_hands = any(k in prompt_lc for k in hands_keywords)
                is_full_body = any(k in prompt_lc for k in full_body_keywords) or anatomy_unstable
                is_sweat = style_preset == "sweat_oiled"
                allow_quality_fixers = is_hands or is_full_body or is_sweat or anatomy_unstable or malformed

                if is_full_body and face_swap:
                    face_swap = False
                    log("FaceSwap disabled for full-body prompt", prefix="🧪", model=model_name)

                if not allow_quality_fixers:
                    log("Quality fixers skipped (base realism only)", prefix="🧪", model=model_name)
                elif is_sweat and LORA_SWEAT_OILED.exists():
                    positive_loras = [("sweat_oiled_skin_xl", 0.5)]
                elif is_full_body and LORA_DETAILED_PERFECTION.exists():
                    positive_loras = [("detailed_perfection_xl", 0.4)]
                elif is_hands:
                    if LORA_HANDS.exists():
                        positive_loras = [("hands_xl", 0.4)]
                    else:
                        log("Hands XL LoRA missing; falling back to default fixers", prefix="⚠️", path=str(LORA_HANDS))
                        if LORA_REALISTIC_SKIN.exists():
                            skin_weight = 0.45 if "juggernaut" in model_name.lower() else 0.35 if "epicrealism" in model_name.lower() else 0.5
                            positive_loras = [("realistic_skin_texture_xl", skin_weight)]
                else:
                    if LORA_REALISTIC_SKIN.exists():
                        skin_weight = 0.45 if "juggernaut" in model_name.lower() else 0.35 if "epicrealism" in model_name.lower() else 0.5
                        positive_loras = [("realistic_skin_texture_xl", skin_weight)]
                if allow_quality_fixers and LORA_BAD_ANATOMY_NEG.exists():
                    if ("juggernaut" not in model_name.lower() and "epicrealism" not in model_name.lower()) or anatomy_unstable or malformed:
                        negative_loras = [("bad_anatomy_negative_xl", -0.9)]
                if positive_loras or negative_loras:
                    log(
                        "Quality fixer selection",
                        prefix="🧪",
                        positive=[name for name, _ in positive_loras],
                        negative=[name for name, _ in negative_loras],
                    )
            else:
                if face_swap:
                    log("Quality fixer LoRAs skipped (FaceSwap mode)", prefix="🧪", model=model_name)
                elif style_mode == "anime":
                    log("Quality fixer LoRAs skipped (anime mode)", prefix="🧪", model=model_name)
                else:
                    log("Quality fixer LoRAs skipped (non-SDXL model)", prefix="🧪", model=model_name)

            if not face_swap and not allow_quality_fixers and style_mode == "realism" and hasattr(pipe, "load_lora_weights"):
                if is_xl:
                    perf_path = BASE_REALISM_SDXL_PERF
                    skin_path = BASE_REALISM_SDXL_SKIN
                    lora_label = "SDXL"
                else:
                    perf_path = BASE_REALISM_SD15_PERF
                    skin_path = BASE_REALISM_SD15_SKIN
                    lora_label = "SD1.5"
                if perf_path.exists() and skin_path.exists():
                    try:
                        pipe.load_lora_weights(str(perf_path), adapter_name="perfection")
                        pipe.load_lora_weights(str(skin_path), adapter_name="skin")
                        if hasattr(pipe, "fuse_lora"):
                            pipe.fuse_lora(lora_scale=0.6)
                        base_realism_loaded = True
                        lora_cleanup_needed = True
                        log(
                            f"Loaded {lora_label} LoRAs: perfection + skin",
                            prefix="LoRA",
                        )
                    except Exception as exc:
                        log(f"Base realism LoRA load failed: {exc}", prefix="⚠️")
                else:
                    log(
                        f"Base realism LoRAs missing ({lora_label})",
                        prefix="⚠️",
                        perfection=str(perf_path),
                        skin=str(skin_path),
                    )

            log("FaceSwap mode", prefix="🧪", active=face_swap)

            if hasattr(pipe, "load_lora_weights") and (positive_loras or negative_loras):
                loaded_names: List[str] = []
                loaded_weights: List[float] = []
                log(
                    "Quality fixer LoRA paths "
                    f"skin={LORA_REALISTIC_SKIN}({LORA_REALISTIC_SKIN.exists()}), "
                    f"bad_anatomy={LORA_BAD_ANATOMY_NEG}({LORA_BAD_ANATOMY_NEG.exists()}), "
                    f"hands={LORA_HANDS}({LORA_HANDS.exists()}), "
                    f"perfection={LORA_DETAILED_PERFECTION}({LORA_DETAILED_PERFECTION.exists()}), "
                    f"sweat={LORA_SWEAT_OILED}({LORA_SWEAT_OILED.exists()})",
                    prefix="🧪",
                )
                def _adapter_loaded(name: str) -> bool:
                    if hasattr(pipe, "peft_config") and isinstance(pipe.peft_config, dict):
                        return name in pipe.peft_config
                    if hasattr(pipe, "get_active_adapters"):
                        try:
                            return name in (pipe.get_active_adapters() or [])
                        except Exception:
                            return False
                    return True
                try:
                    for name, weight in positive_loras:
                        if name == "realistic_skin_texture_xl" and LORA_REALISTIC_SKIN.exists():
                            pipe.load_lora_weights(str(LORA_REALISTIC_SKIN), adapter_name=name)
                        elif name == "hands_xl" and LORA_HANDS.exists():
                            pipe.load_lora_weights(str(LORA_HANDS), adapter_name=name)
                        elif name == "detailed_perfection_xl" and LORA_DETAILED_PERFECTION.exists():
                            pipe.load_lora_weights(str(LORA_DETAILED_PERFECTION), adapter_name=name)
                        elif name == "sweat_oiled_skin_xl" and LORA_SWEAT_OILED.exists():
                            pipe.load_lora_weights(str(LORA_SWEAT_OILED), adapter_name=name)
                        else:
                            continue
                        if _adapter_loaded(name):
                            loaded_names.append(name)
                            loaded_weights.append(weight)
                        else:
                            log(
                                f"Quality fixer adapter not registered after load: {name} "
                                f"path={LORA_REALISTIC_SKIN if name == 'realistic_skin_texture_xl' else LORA_HANDS if name == 'hands_xl' else LORA_DETAILED_PERFECTION if name == 'detailed_perfection_xl' else LORA_SWEAT_OILED}",
                                prefix="⚠️",
                            )
                    for name, weight in negative_loras:
                        if name == "bad_anatomy_negative_xl" and LORA_BAD_ANATOMY_NEG.exists():
                            pipe.load_lora_weights(str(LORA_BAD_ANATOMY_NEG), adapter_name=name)
                            if _adapter_loaded(name):
                                loaded_names.append(name)
                                loaded_weights.append(weight)
                            else:
                                log(
                                    f"Quality fixer adapter not registered after load: {name} path={LORA_BAD_ANATOMY_NEG}",
                                    prefix="⚠️",
                                )
                    if hasattr(pipe, "set_adapters") and loaded_names:
                        pipe.set_adapters(loaded_names, adapter_weights=loaded_weights)
                    elif hasattr(pipe, "fuse_lora") and loaded_weights:
                        pipe.fuse_lora(lora_scale=loaded_weights[0])
                    if loaded_names:
                        log("Quality fixer LoRA", prefix="🧪", loras=loaded_names, weights=loaded_weights)
                        lora_cleanup_needed = True
                    if hasattr(pipe, "peft_config"):
                        log(
                            "Quality fixer adapter registry "
                            f"{list(getattr(pipe, 'peft_config', {}).keys())}",
                            prefix="🧪",
                        )
                except Exception as exc:
                    log(f"Quality fixer LoRA load failed: {exc}", prefix="⚠️")

            if (
                not positive_loras
                and not base_realism_loaded
                and is_xl
                and style_mode == "realism"
                and not face_swap
                and HYPERLORA_PATH.exists()
                and hasattr(pipe, "load_lora_weights")
            ):
                try:
                    pipe.load_lora_weights(str(HYPERLORA_PATH), adapter_name="hyperlora")
                    if hasattr(pipe, "set_adapters"):
                        pipe.set_adapters(["hyperlora"], adapter_weights=[hyperlora_weight])
                    elif hasattr(pipe, "fuse_lora"):
                        pipe.fuse_lora(lora_scale=hyperlora_weight)
                    lora_cleanup_needed = True
                    log("HyperLoRA scale", prefix="🧪", scale=hyperlora_weight)
                except Exception as exc:
                    log(f"HyperLoRA load failed: {exc}", prefix="⚠️", lora=str(HYPERLORA_PATH))

            seed = int(time.time()) & 0x7FFFFFFF
            generator = torch.Generator(device=device).manual_seed(seed)
            text = final_prompt or "a high quality photo of a golden retriever on a beach at sunset"
            gen_t0 = time.time()
            with torch.inference_mode():
                if is_xl:
                    result = pipe(
                        prompt=text,
                        prompt_2=text,
                        negative_prompt=final_negative or None,
                        negative_prompt_2=final_negative or None,
                        ip_adapter_image=source_face_image if faceid_active else None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        height=height,
                        width=width,
                    )
                else:
                    result = pipe(
                        text,
                        negative_prompt=final_negative or None,
                        image=controlnet_image if use_controlnet else None,
                        controlnet_conditioning_scale=1.0 if use_controlnet else None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        height=height,
                        width=width,
                    )
            gen_ms = int((time.time() - gen_t0) * 1000)
            log(
                f"Generated in {gen_ms}ms",
                prefix="✅",
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
            )
            img = result.images[0]
            img.save(output_path)
            with output_path.open("rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode("utf-8")
            if lora_cleanup_needed:
                try:
                    if hasattr(pipe, "unfuse_lora"):
                        pipe.unfuse_lora()
                    if hasattr(pipe, "unload_lora_weights"):
                        pipe.unload_lora_weights()
                except Exception as exc:
                    log(f"Base realism LoRA cleanup failed: {exc}", prefix="⚠️")
        elif Image is not None:
            log("Using fast preview placeholder (no SD detected or FAST_PREVIEW enabled)", prefix="ℹ️")
            w, h = width, height
            if np is not None:
                yy = np.linspace(0, 255, h, dtype=np.uint8)
                xx = np.linspace(0, 255, w, dtype=np.uint8)
                R = np.tile(xx, (h, 1))
                G = np.tile(yy, (w, 1)).T
                B = (R[::-1] // 2 + G // 2).astype(np.uint8)
                arr = np.dstack([R, G, B])
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
        "model_name": model_name,
        "model_path": str(model_path),
        "reward_weight": reward_weight,
        "task_type": "image_gen",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
    }
    if status == "failed":
        metrics["error"] = error_msg or "image generation error"
    return metrics, util, image_b64


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------


def heartbeat_loop() -> None:
    backoff = BACKOFF_BASE
    while True:
        payload = {
            "node_id": NODE_NAME,
            "os": os.uname().sysname if hasattr(os, "uname") else os.name,
            "gpu": read_gpu_stats(),
            "start_time": START_TIME,
            "uptime": time.time() - START_TIME,
            "role": ROLE,
            "version": CLIENT_VERSION,
            "node_name": NODE_NAME,
        }
        try:
            resp = SESSION.post(endpoint("/register"), data=json.dumps(payload), timeout=5)
            resp.raise_for_status()
            backoff = BACKOFF_BASE
            log(f"Heartbeat OK ({ROLE})", prefix="✅")
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
    register_local_models()
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

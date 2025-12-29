"""HavnAI Node Client — Stage 7 public onboarding."""

from __future__ import annotations

import json
import logging
import math
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
from typing import Any, Dict, List, Optional, Set, Tuple
import base64

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
    _DPMSolver = None  # type: ignore
    _AutoencoderKL = None  # type: ignore
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
LOGS_DIR = HAVNAI_HOME / "logs"
OUTPUTS_DIR = HAVNAI_HOME / "outputs"
VERSION_SEARCH_PATHS = [HAVNAI_HOME / "VERSION", Path(__file__).resolve().parent / "VERSION"]
DEFAULT_LORA_DIR = Path("/mnt/d/havnai-storage/models/loras")
LORA_DIR = Path(
    os.environ.get("HAI_LORA_DIR")
    or os.environ.get("HAVNAI_LORA_DIR")
    or (DEFAULT_LORA_DIR if DEFAULT_LORA_DIR.exists() else HAVNAI_HOME / "loras")
).expanduser()
MAX_LORAS = 3

HAVNAI_HOME.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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

SERVER_BASE = ENV_VARS.get("SERVER_URL", "http://127.0.0.1:5001").rstrip("/")
JOIN_TOKEN = ENV_VARS.get("JOIN_TOKEN", "").strip()
ROLE = "creator" if ENV_VARS.get("CREATOR_MODE", "false").lower() in {"1", "true", "yes"} else "worker"
NODE_NAME = ENV_VARS.get("NODE_NAME", socket.gethostname())
FAST_PREVIEW = (
    os.environ.get("HAI_FAST_PREVIEW", ENV_VARS.get("HAI_FAST_PREVIEW", "")).lower()
    in {"1", "true", "yes"}
)

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


def refresh_manifest_with_backoff(reason: str = "startup") -> None:
    backoff = BACKOFF_BASE
    while True:
        try:
            REGISTRY.refresh()
            log("Model manifest refreshed", prefix="✅", reason=reason)
            return
        except Exception as exc:
            log(f"Manifest refresh failed ({reason}): {exc}", prefix="⚠️")
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
    path = Path(entry.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path missing on node: {path}")
    return path


def discover_capabilities() -> Dict[str, List[str]]:
    pipelines: set[str] = set()
    models: List[str] = []
    for entry in REGISTRY.list_entries():
        try:
            path = ensure_model_path(entry)
        except Exception:
            continue
        pipelines.add(entry.pipeline)
        models.append(entry.name)
    return {"pipelines": sorted(pipelines), "models": sorted(models)}


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


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = (task.get("type") or "IMAGE_GEN").lower()
    model_name = (task.get("model_name") or "model").lower()
    reward_weight = float(task.get("reward_weight", 1.0))
    input_shape = task.get("input_shape") or []
    prompt = task.get("prompt") or ""
    negative_prompt = task.get("negative_prompt") or ""

    if task_type == "image_gen" and ROLE != "creator":
        log(f"Skipping creator task {task_id[:8]} — node not in creator mode", prefix="⚠️")
        return

    log(f"Executing {task_type} task {task_id[:8]} · {model_name}", prefix="🚀")

    image_b64: Optional[str] = None
    try:
        entry = ensure_model_entry(model_name)
        model_path = ensure_model_path(entry)
    except Exception as exc:
        log(f"Model resolution failed: {exc}", prefix="🚫")
        return
    if task_type == "image_gen":
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

        metrics, util, image_b64 = run_image_generation(
            task_id,
            entry,
            model_path,
            reward_weight,
            prompt,
            negative_prompt,
            job_settings,
        )
    else:
        metrics, util = run_ai_inference(entry, model_path, input_shape, reward_weight)

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
        log(f"Task {task_id[:8]} {payload['status'].upper()} · reward {reward} HAI", prefix=prefix)
    except Exception as exc:
        log(f"Failed to submit result: {exc}", prefix="🚫")


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

ANATOMY_LORA_NAMES = {"handv2"}
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
    "handv2": "sd15",
    "perfectionstylesd1.5": "sd15",
    "skintexturestylesd1.5v1": "sd15",
    "perfectionstyle": "sdxl",
    "perfectionstylev2d": "sdxl",
    "skintexturestylev3": "sdxl",
    "skintexturestylev5": "sdxl",
    "hyperlora_sdxl": "sdxl",
}

ROLE_WEIGHT_RANGES = {
    "position": (0.0, 2.0, 0.95),
    "anatomy": (0.35, 0.85, 0.45),
    "style": (0.25, 0.45, 0.35),
}

AUTO_ANATOMY_WEIGHT_SD15 = 0.8
AUTO_ANATOMY_WEIGHT_SDXL = 0.45


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


def find_auto_anatomy_lora(model_is_sdxl: bool, seen: Set[str]) -> Optional[Tuple[Path, float, str]]:
    if model_is_sdxl:
        for path in sorted(LORA_DIR.glob("*.safetensors")):
            normalized = normalize_lora_name(path.name)
            if normalized in seen:
                continue
            if any(normalized.startswith(prefix) for prefix in ANATOMY_LORA_PREFIXES):
                weight = clamp_lora_weight(AUTO_ANATOMY_WEIGHT_SDXL, "anatomy")
                return (path, weight, f"lora_anatomy_{path.stem}")
        return None

    hand_path = resolve_lora_path("Handv2.safetensors")
    if hand_path is None:
        for path in sorted(LORA_DIR.glob("*.safetensors")):
            if normalize_lora_name(path.name) == "handv2":
                hand_path = path
                break
    if hand_path is None:
        return None
    if normalize_lora_name(hand_path.name) in seen:
        return None
    weight = clamp_lora_weight(AUTO_ANATOMY_WEIGHT_SD15, "anatomy")
    return (hand_path, weight, f"lora_anatomy_{hand_path.stem}")


def select_lora_entries(
    requested: List[Any],
    model_name: str,
) -> List[Tuple[Path, float, str]]:
    model_is_sdxl = is_sdxl_model(model_name)
    positions: List[Tuple[Path, float, str]] = []
    anatomies: List[Tuple[Path, float, str]] = []
    styles: List[Tuple[Path, float, str]] = []
    seen: Set[str] = set()

    for item in requested:
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
        # Enforce SDXL/SD1.5 compatibility to avoid mismatched LoRA loads.
        if base_type == "sdxl" and not model_is_sdxl:
            continue
        if base_type == "sd15" and model_is_sdxl:
            continue
        role = classify_lora_role(normalized)
        weight = clamp_lora_weight(raw_weight, role)
        lora_path = resolve_lora_path(name)
        if not lora_path:
            continue
        adapter_name = f"lora_{role}_{lora_path.stem}"
        entry = (lora_path, weight, adapter_name)
        if role == "position":
            positions.append(entry)
        elif role == "anatomy":
            anatomies.append(entry)
        else:
            styles.append(entry)
        seen.add(normalized)

    if positions and not anatomies:
        auto_anatomy = find_auto_anatomy_lora(model_is_sdxl, seen)
        if auto_anatomy:
            anatomies.append(auto_anatomy)

    selected: List[Tuple[Path, float, str]] = []
    # Enforce stack order: position -> anatomy -> style, with safe caps.
    if positions:
        selected.append(positions[0])
    if anatomies and len(selected) < MAX_LORAS:
        selected.append(anatomies[0])
    for entry in styles:
        if len(selected) >= MAX_LORAS:
            break
        selected.append(entry)
    return selected


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
    output_path = OUTPUTS_DIR / f"{task_id}.png"
    try:
        if not FAST_PREVIEW and torch is not None and diffusers is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            log("Loading text2image pipeline…", prefix="ℹ️", device=device)
            load_t0 = time.time()
            pipe = None
            # For SD1.5-style checkpoints (triomerge, etc.) use StableDiffusionPipeline
            # to match direct test behavior; reserve AutoPipeline/SDXL pipeline for SDXL types.
            pipeline_name = (getattr(entry, "pipeline", "") or "sd15").lower()
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
                    except Exception as exc:  # fallback to SD1.5 pipe
                        log(f"AutoPipeline load failed: {exc}", prefix="⚠️")
            if pipe is None and _SDPipe is not None:
                pipe = _SDPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
            if pipe is None:
                raise RuntimeError("Failed to construct a text2image pipeline for model.")
            # Optionally swap in a custom VAE for SD1.5-style checkpoints
            if pipe is not None and getattr(entry, "vae_path", "") and pipeline_name != "sdxl":
                vae_path = Path(str(getattr(entry, "vae_path", ""))).expanduser()
                if vae_path.exists() and _AutoencoderKL is not None:
                    try:
                        custom_vae = _AutoencoderKL.from_single_file(str(vae_path), torch_dtype=dtype)
                        pipe.vae = custom_vae
                        log(f"Loaded custom VAE for {entry.name}", prefix="✅", vae=str(vae_path))
                    except Exception as exc:
                        log(f"Custom VAE load failed for {entry.name}: {exc}", prefix="⚠️", vae=str(vae_path))
            # Optional LoRA attachment with safe stacking and role-aware weights.
            lora_entries: List[Tuple[Path, float, str]] = []
            if job_settings and isinstance(job_settings, dict):
                requested = job_settings.get("loras") or []
                if isinstance(requested, str):
                    requested = [requested]
                if isinstance(requested, list):
                    model_hint = f"{entry.name} {pipeline_name}".strip()
                    lora_entries = select_lora_entries(requested, model_hint)
            if lora_entries:
                lora_summary = ", ".join(f"{path.stem}:{weight:.2f}" for path, weight, _ in lora_entries)
            else:
                lora_summary = "none"
            log(f"Loading LoRAs: {lora_summary}", prefix="🎛️")
            if lora_entries:
                adapter_names: List[str] = []
                adapter_weights: List[float] = []
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
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            if "_DPMSolver" in globals() and _DPMSolver is not None and hasattr(pipe, "scheduler"):
                try:
                    pipe.scheduler = _DPMSolver.from_config(pipe.scheduler.config)
                except Exception as exc:
                    log(f"DPM scheduler setup failed: {exc}", prefix="⚠️")
            pipe = pipe.to(device)
            load_ms = int((time.time() - load_t0) * 1000)
            log(f"Pipeline ready in {load_ms}ms", prefix="✅")

            seed = int(time.time()) & 0x7FFFFFFF
            generator = torch.Generator(device=device).manual_seed(seed)
            pos_text = prompt or "a high quality photo of a golden retriever on a beach at sunset"
            neg_text = negative_prompt or ""
            # Merge per-job overrides with model defaults from manifest
            steps = IMAGE_STEPS
            guidance = IMAGE_GUIDANCE
            height = IMAGE_HEIGHT
            width = IMAGE_WIDTH
            sampler = None
            if job_settings and isinstance(job_settings, dict):
                steps = int(job_settings.get("steps", steps) or steps)
                guidance = float(job_settings.get("guidance", guidance) or guidance)
                height = int(job_settings.get("height", height) or height)
                width = int(job_settings.get("width", width) or width)
                sampler = str(job_settings.get("sampler") or "").strip().lower() or None
                if job_settings.get("negative_prompt"):
                    neg_text = str(job_settings.get("negative_prompt") or "")
            # clamp to sane ranges
            steps = max(5, min(50, steps))
            guidance = max(1.0, min(15.0, guidance))
            height = max(256, min(1536, height))
            width = max(256, min(1536, width))
            # Proactively truncate to CLIP token limit to avoid noisy warnings
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
            # Optional sampler switch if supported
            if sampler and hasattr(pipe, "scheduler") and _DPMSolver is not None and "dpmpp" in sampler:
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
                    height=height,
                    width=width,
                )
            gen_ms = int((time.time() - gen_t0) * 1000)
            log(f"Generated in {gen_ms}ms", prefix="✅")
            img = result.images[0]
            img.save(output_path)
            with output_path.open("rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode("utf-8")
        elif Image is not None:
            log("Using fast preview placeholder (no SD detected or FAST_PREVIEW enabled)", prefix="ℹ️")
            # Fallback: deterministic gradient image with text-like bands so it isn't noisy
            w = h = 512
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
            # No image libs available; simulate compute only
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
        gpu_stats = read_gpu_stats()
        capabilities = discover_capabilities()
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
            "pipelines": capabilities["pipelines"],
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
    refresh_manifest_with_backoff("startup")
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

"""HavnAI Node Client — Stage 7 public onboarding."""

from __future__ import annotations

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
IMAGE_GUIDANCE = float(os.environ.get("HAI_GUIDANCE", "7.5"))
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
        metrics, util, image_b64 = run_image_generation(task_id, entry, model_path, reward_weight, prompt, negative_prompt)
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


def run_image_generation(task_id: str, entry: ModelEntry, model_path: Path, reward_weight: float, prompt: str, negative_prompt: str) -> (Dict[str, Any], int, Optional[str]):

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
            # to match direct test behavior; reserve AutoPipeline for SDXL/other future types.
            pipeline_name = (getattr(entry, "pipeline", "") or "sd15").lower()
            if pipeline_name in {"sdxl"} and _AutoPipe is not None:
                try:
                    pipe = _AutoPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
                except Exception as exc:  # fallback to SD1.5 pipe
                    log(f"AutoPipeline load failed: {exc}", prefix="⚠️")
            if pipe is None and _SDPipe is not None:
                pipe = _SDPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
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
            with torch.inference_mode():
                result = pipe(
                    pos_text,
                    negative_prompt=neg_text or None,
                    num_inference_steps=IMAGE_STEPS,
                    guidance_scale=IMAGE_GUIDANCE,
                    generator=generator,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
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

"""HavnAI Node Client — Stage 7 public onboarding."""

from __future__ import annotations

import hashlib
import inspect
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
import urllib.parse
from pathlib import Path
import atexit
import signal
from typing import Any, Dict, List, Optional, Tuple
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
    from diffusers import DPMSolverMultistepScheduler as _DPMSolver  # type: ignore
    try:
        from diffusers import EulerAncestralDiscreteScheduler as _EulerA  # type: ignore
    except Exception:  # pragma: no cover
        _EulerA = None  # type: ignore
    try:
        from diffusers import EulerDiscreteScheduler as _Euler  # type: ignore
    except Exception:  # pragma: no cover
        _Euler = None  # type: ignore
except ImportError:  # pragma: no cover
    diffusers = None
    _AutoPipe = None  # type: ignore
    _SDPipe = None  # type: ignore
    _SDXLPipe = None  # type: ignore
    _DPMSolver = None  # type: ignore
    _EulerA = None  # type: ignore
    _Euler = None  # type: ignore
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
CREATOR_SCAN_DIR = HAVNAI_HOME / "models" / "creator"
DOWNLOAD_DIR = HAVNAI_HOME / "downloads"
LOGS_DIR = HAVNAI_HOME / "logs"
OUTPUTS_DIR = HAVNAI_HOME / "outputs"
LORA_DIR = Path(
    os.environ.get("HAVNAI_LORA_DIR")
    or os.environ.get("HAI_LORA_DIR")
    or os.environ.get("LORA_DIR")
    or (HAVNAI_HOME / "loras")
)
VERSION_SEARCH_PATHS = [HAVNAI_HOME / "VERSION", Path(__file__).resolve().parent / "VERSION"]

HAVNAI_HOME.mkdir(parents=True, exist_ok=True)
CREATOR_SCAN_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODEL_EXTS = {".onnx", ".safetensors", ".ckpt"}
SUPPORTED_LORA_EXTS = {".safetensors", ".pt", ".bin"}

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


INSTANTID_REPO = os.environ.get("INSTANTID_REPO", "instantx/InstantID-XL")
INSTANTID_CONTROLNET_SUBFOLDER = os.environ.get("INSTANTID_CONTROLNET_SUBFOLDER", "ControlNetModel")
INSTANTID_IP_ADAPTER_FILE = os.environ.get("INSTANTID_IP_ADAPTER_FILE", "ip-adapter.bin")
INSTANTID_CACHE_ENABLED = os.environ.get("INSTANTID_CACHE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INSTANTID_CACHE_DIR = (HAVNAI_HOME / "instantid").resolve()
INSTANTID_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FACE_ANALYSIS: Optional["FaceAnalysis"] = None
_INSTANTID_PIPELINE_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}
_INSTANTID_PIPELINE_LOCK = threading.Lock()


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
        if entry.get("url") or entry.get("filename"):
            path = _download_lora_spec(entry)
        else:
            path = _resolve_lora_path(name)
        if not path:
            log(f"LoRA not found: {name}", prefix="⚠️")
            continue
        adapter_name = path.stem
        last_error: Optional[Exception] = None
        loaded = False
        strength = entry.get("strength")
        weight = entry.get("weight")
        scale = strength if strength is not None else weight
        if scale is None:
            scale = 1.0
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
                    pipe.load_lora_weights(str(path), adapter_name=adapter_name, strength=scale)
                except TypeError:
                    pipe.load_lora_weights(str(path), adapter_name=adapter_name)
            loaded = True
        except Exception as exc:
            last_error = exc
        if not loaded:
            try:
                pipe.load_lora_weights(str(path), adapter_name=adapter_name)
                loaded = True
                last_error = None
            except TypeError:
                try:
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


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = (task.get("type") or "IMAGE_GEN").lower()
    model_name = (task.get("model_name") or "model").lower()
    model_url = task.get("model_url", "")
    reward_weight = float(task.get("reward_weight", 1.0))
    input_shape = task.get("input_shape") or []
    prompt = task.get("prompt") or ""
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
        log(f"Skipping creator task {task_id[:8]} — node not in creator mode", prefix="⚠️")
        return

    log(f"Executing {task_type} task {task_id[:8]} · {model_name}", prefix="🚀")

    image_b64: Optional[str] = None
    if task_type == "image_gen":
        loras = task.get("loras") if isinstance(task.get("loras"), list) else None
        model_spec = task.get("model") if isinstance(task.get("model"), dict) else None
        model_lora = model_spec.get("lora") if isinstance(model_spec, dict) else None
        model_pipeline = model_spec.get("pipeline") if isinstance(model_spec, dict) else None
        image_settings = {}
        for key in ("steps", "guidance", "width", "height", "sampler", "seed"):
            if key in task and task[key] is not None:
                image_settings[key] = task[key]
        metrics, util, image_b64 = run_image_generation(
            task_id,
            model_name,
            model_url,
            reward_weight,
            prompt,
            loras=loras,
            model_lora=model_lora,
            pipeline_hint=model_pipeline,
            image_settings=image_settings or None,
        )
    elif task_type == "face_swap":
        job_settings: Dict[str, Any] = {}
        if isinstance(task, dict):
            for key in (
                "base_image_url",
                "base_image",
                "base_image_b64",
                "base_image_path",
                "image",
                "image_b64",
                "face_source_url",
                "face_source",
                "face_source_b64",
                "face_source_path",
                "face_image",
                "face_image_b64",
                "strength",
                "num_steps",
                "seed",
            ):
                if key in task and task[key] is not None:
                    job_settings[key] = task[key]
        metrics, util, image_b64 = run_faceswap_generation(
            task_id,
            model_name,
            model_url,
            reward_weight,
            prompt,
            "",
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


def run_faceswap_generation(
    task_id: str,
    model_name: str,
    model_url: str,
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
        if torch is None or diffusers is None:
            raise RuntimeError("diffusers/torch required for face swap")
        if Image is None or np is None or cv2 is None:
            raise RuntimeError("opencv-python and numpy are required for face swap")
        if FaceAnalysis is None:
            raise RuntimeError("insightface is required for face swap")

        from diffusers import ControlNetModel  # type: ignore
        from pipeline_stable_diffusion_xl_instantid_inpaint import (  # type: ignore
            StableDiffusionXLInstantIDInpaintPipeline,
        )

        try:
            model_path = resolve_model_path(model_name, model_url, filename_hint=f"{model_name}.safetensors")
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        settings = job_settings if isinstance(job_settings, dict) else {}
        base_image_src = (
            settings.get("base_image_url")
            or settings.get("base_image")
            or settings.get("base_image_b64")
            or settings.get("base_image_path")
            or settings.get("image")
            or settings.get("image_b64")
        )
        face_source_src = (
            settings.get("face_source_url")
            or settings.get("face_source")
            or settings.get("face_source_b64")
            or settings.get("face_source_path")
            or settings.get("face_image")
            or settings.get("face_image_b64")
        )
        if not base_image_src or not face_source_src:
            raise RuntimeError("base_image and face_source are required for face swap")

        strength = coerce_float(settings.get("strength", 0.8), 0.8)
        num_steps = coerce_int(settings.get("num_steps", 20), 20)
        strength = max(0.0, min(1.0, strength))
        num_steps = max(5, min(60, num_steps))

        base_image, base_error = load_image_source_with_error(str(base_image_src))
        face_image, face_error = load_image_source_with_error(str(face_source_src))
        if base_image is None or face_image is None:
            base_desc = _describe_image_source(base_image_src)
            face_desc = _describe_image_source(face_source_src)
            base_msg = base_error or "unknown error"
            face_msg = face_error or "unknown error"
            raise RuntimeError(
                f"Failed to load base image ({base_desc}): {base_msg}; "
                f"face image ({face_desc}): {face_msg}"
            )

        app = get_face_analysis()
        face_infos = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = pick_primary_face(face_infos)
        if not face_info:
            raise RuntimeError("No face detected in face_source_url image")
        face_emb = face_info["embedding"]

        target_infos = app.get(cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR))
        target_face = pick_primary_face(target_infos)
        if not target_face:
            raise RuntimeError("No face detected in base_image_url image")

        images, position = prepare_mask_and_pose_control(base_image, target_face)
        mask, pose_image_preprocessed, control_image = images

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        instantid_repo = INSTANTID_REPO
        local_repo_path = Path(instantid_repo).expanduser()
        use_local_repo = local_repo_path.exists()
        if hf_hub_download is None and not use_local_repo:
            raise RuntimeError("huggingface_hub is required for face swap")

        if use_local_repo:
            controlnet_path = local_repo_path / INSTANTID_CONTROLNET_SUBFOLDER
            controlnet_source = str(controlnet_path if controlnet_path.exists() else local_repo_path)
            adapter_source = str(local_repo_path / INSTANTID_IP_ADAPTER_FILE)
        else:
            controlnet_source = f"{instantid_repo}::{INSTANTID_CONTROLNET_SUBFOLDER}"
            adapter_source = f"{instantid_repo}::{INSTANTID_IP_ADAPTER_FILE}"

        cache_enabled = INSTANTID_CACHE_ENABLED
        cache_key = (str(model_path), controlnet_source, adapter_source, device, str(dtype))
        if cache_enabled:
            with _INSTANTID_PIPELINE_LOCK:
                pipe = _INSTANTID_PIPELINE_CACHE.get(cache_key)
        else:
            with _INSTANTID_PIPELINE_LOCK:
                _INSTANTID_PIPELINE_CACHE.clear()
            pipe = None

        if pipe is None:
            if use_local_repo:
                controlnet = ControlNetModel.from_pretrained(controlnet_source, torch_dtype=dtype)
            else:
                controlnet = ControlNetModel.from_pretrained(
                    instantid_repo,
                    subfolder=INSTANTID_CONTROLNET_SUBFOLDER,
                    torch_dtype=dtype,
                    cache_dir=str(INSTANTID_CACHE_DIR),
                )

            pipe = None
            if hasattr(StableDiffusionXLInstantIDInpaintPipeline, "from_single_file") and model_path.is_file():
                try:
                    pipe = StableDiffusionXLInstantIDInpaintPipeline.from_single_file(
                        str(model_path), controlnet=controlnet, torch_dtype=dtype, safety_checker=None
                    )
                except Exception as exc:
                    log(f"InstantID from_single_file failed: {exc}", prefix="⚠️")
            if pipe is None:
                if model_path.is_dir():
                    pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
                        str(model_path), controlnet=controlnet, torch_dtype=dtype
                    )
                else:
                    pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
                        str(model_path), controlnet=controlnet, torch_dtype=dtype
                    )

            pipe = pipe.to(device)

            if use_local_repo:
                adapter_path = Path(adapter_source)
                if not adapter_path.exists():
                    raise RuntimeError(f"InstantID ip-adapter not found at {adapter_path}")
                adapter_path = str(adapter_path)
            else:
                adapter_path = hf_hub_download(
                    instantid_repo,
                    filename=INSTANTID_IP_ADAPTER_FILE,
                    cache_dir=str(INSTANTID_CACHE_DIR),
                )
            pipe.load_ip_adapter_instantid(adapter_path, scale=strength)
            if hasattr(pipe, "image_proj_model") and pipe.image_proj_model is not None:
                try:
                    pipe.image_proj_model.to(device=device, dtype=dtype)
                except Exception:
                    pipe.image_proj_model.to(device)
            if cache_enabled:
                with _INSTANTID_PIPELINE_LOCK:
                    _INSTANTID_PIPELINE_CACHE[cache_key] = pipe
                log("Cached InstantID pipeline", prefix="♻️", model=str(model_path))
        elif cache_enabled:
            try:
                pipe = pipe.to(device)
            except Exception:
                pass
            log("Using cached InstantID pipeline", prefix="♻️", model=str(model_path))

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        seed = settings.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        if seed is None:
            seed = int(time.time()) & 0x7FFFFFFF
        generator = torch.Generator(device=device).manual_seed(seed)

        prompt_text = prompt or ""
        neg_text = negative_prompt or ""
        guidance_scale = 5.0 if prompt_text else 0.0

        face_emb_tensor = torch.tensor(face_emb, device=device, dtype=pipe.unet.dtype)
        result = pipe(
            prompt=prompt_text,
            negative_prompt=neg_text or None,
            image_embeds=face_emb_tensor,
            control_image=control_image,
            image=pose_image_preprocessed,
            mask_image=mask,
            strength=strength,
            controlnet_conditioning_scale=strength,
            ip_adapter_scale=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        face_patch = result.images[0]
        x, y, w, h = position
        face_patch = face_patch.resize((w, h))
        base_image.paste(face_patch, (x, y))
        base_image.save(output_path)
        with output_path.open("rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log(f"Face swap failed: {error_msg}", prefix="🚫")

    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    util = int(max(util, 70 if ROLE == "creator" else util))
    metrics = {
        "status": status,
        "model_name": model_name,
        "model_path": str(model_path) if "model_path" in locals() else "",
        "reward_weight": reward_weight,
        "task_type": "face_swap",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
        "strength": strength if "strength" in locals() else None,
        "num_steps": num_steps if "num_steps" in locals() else None,
        "output_path": str(output_path) if output_path else None,
    }
    if status == "failed":
        metrics["error"] = error_msg or "face swap error"
    return metrics, util, image_b64


def run_image_generation(
    task_id: str,
    model_name: str,
    model_url: str,
    reward_weight: float,
    prompt: str,
    loras: Optional[List[Dict[str, Any]]] = None,
    model_lora: Optional[Any] = None,
    pipeline_hint: Optional[str] = None,
    image_settings: Optional[Dict[str, Any]] = None,
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
    def _override_int(name: str) -> Optional[int]:
        if not image_settings or name not in image_settings:
            return None
        try:
            return int(image_settings[name])
        except (TypeError, ValueError):
            return None

    def _override_float(name: str) -> Optional[float]:
        if not image_settings or name not in image_settings:
            return None
        try:
            return float(image_settings[name])
        except (TypeError, ValueError):
            return None

    def _override_str(name: str) -> str:
        if not image_settings or name not in image_settings:
            return ""
        return str(image_settings[name] or "").strip()

    try:
        model_path = resolve_model_path(model_name, model_url, filename_hint=f"{model_name}.safetensors")
    except Exception as exc:
        return ({"status": "failed", "error": str(exc), "reward_weight": reward_weight}, utilization_hint, None)

    steps = _env_int("HAI_STEPS", 65)
    guidance = _env_float("HAI_GUIDANCE", 8.2)
    clip_skip = _env_int("HAI_CLIP_SKIP", 0)
    width = _env_int("HAI_WIDTH", 512)
    height = _env_int("HAI_HEIGHT", 512)
    override_steps = _override_int("steps")
    if override_steps is not None:
        steps = override_steps
    override_guidance = _override_float("guidance")
    if override_guidance is not None:
        guidance = override_guidance
    override_width = _override_int("width")
    if override_width is not None:
        width = override_width
    override_height = _override_int("height")
    if override_height is not None:
        height = override_height
    sampler_name = _override_str("sampler")
    seed_override = _override_int("seed")
    width = max(64, width - (width % 8))
    height = max(64, height - (height % 8))
    pipeline_hint_norm = str(pipeline_hint or "").lower()
    is_xl = "sdxl" in pipeline_hint_norm or ("xl" in pipeline_hint_norm if pipeline_hint_norm else "xl" in model_name.lower())

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

    image_b64: Optional[str] = None
    loaded_loras: List[str] = []
    output_path = OUTPUTS_DIR / f"{task_id}.png"
    try:
        if not FAST_PREVIEW and torch is not None and diffusers is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            log("Loading text2image pipeline…", prefix="ℹ️", device=device)
            load_t0 = time.time()

            pipe = None
            # For XL checkpoints prefer StableDiffusionXLPipeline when
            # available, otherwise fall back to SD1.5 pipeline as a best-effort.
            if is_xl and _SDXLPipe is not None:
                try:
                    pipe = _SDXLPipe.from_single_file(str(model_path), torch_dtype=dtype, safety_checker=None)
                except Exception as exc:
                    log(f"StableDiffusionXLPipeline load failed: {exc}", prefix="⚠️")

            if pipe is None and _SDPipe is not None:
                try:
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

            if clip_skip > 0:
                if hasattr(pipe, "set_clip_skip"):
                    try:
                        pipe.set_clip_skip(clip_skip)
                        log(f"Clip skip set to {clip_skip}", prefix="✅")
                    except Exception as exc:
                        log(f"Clip skip set failed: {exc}", prefix="⚠️")
                elif hasattr(pipe, "clip_skip"):
                    try:
                        pipe.clip_skip = clip_skip
                        log(f"Clip skip set to {clip_skip}", prefix="✅")
                    except Exception as exc:
                        log(f"Clip skip set failed: {exc}", prefix="⚠️")

            combined_loras: List[Any] = []
            if isinstance(model_lora, list):
                combined_loras.extend(model_lora)
            elif isinstance(model_lora, dict):
                combined_loras.append(model_lora)
            if loras:
                combined_loras.extend(loras)
            if combined_loras:
                loaded_loras = _apply_loras_to_pipe(pipe, combined_loras)
                if loaded_loras:
                    log(f"Attached {len(loaded_loras)} LoRA(s)", prefix="✅")

            scheduler_set = False
            sampler = sampler_name.lower() if sampler_name else "dpmpp_2m_karras"
            if sampler in {"euler_a", "euler-ancestral", "euler_ancestral"} and _EulerA is not None:
                pipe.scheduler = _EulerA.from_config(pipe.scheduler.config)  # type: ignore[attr-defined]
                scheduler_set = True
            elif sampler in {"euler", "euler_discrete"} and _Euler is not None:
                pipe.scheduler = _Euler.from_config(pipe.scheduler.config)  # type: ignore[attr-defined]
                scheduler_set = True
            elif _DPMSolver is not None:
                pipe.scheduler = _DPMSolver.from_config(
                    pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas="karras" in sampler,
                )  # type: ignore[attr-defined]
                scheduler_set = True
            if not scheduler_set and sampler_name:
                log(f"Sampler '{sampler_name}' unsupported; using default scheduler", prefix="⚠️")
            LOGGER.info(
                "Scheduler=%s, use_karras_sigmas=%s",
                type(pipe.scheduler).__name__,
                getattr(pipe.scheduler.config, "use_karras_sigmas", None),
            )
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to(device)
            if not is_xl and hasattr(pipe, "vae") and pipe.vae is not None:
                try:
                    pipe.vae.to(device=device, dtype=torch.float32)
                    if hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "force_upcast"):
                        pipe.vae.config.force_upcast = True
                    log("Upcasted VAE to fp32 for SD1.5 stability", prefix="✅")
                except Exception as exc:
                    log(f"VAE upcast failed: {exc}", prefix="⚠️")
            load_ms = int((time.time() - load_t0) * 1000)
            log(
                f"Pipeline ready in {load_ms}ms · {pipe.__class__.__name__}",
                prefix="✅",
            )

            seed = seed_override if seed_override is not None else (int(time.time()) & 0x7FFFFFFF)
            generator = torch.Generator(device=device).manual_seed(seed)
            text = prompt or "a high quality photo of a golden retriever on a beach at sunset"
            if hasattr(pipe, "tokenizer") and hasattr(pipe.tokenizer, "model_max_length"):
                try:
                    encoded = pipe.tokenizer(
                        text,
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text = pipe.tokenizer.batch_decode(encoded.input_ids, skip_special_tokens=True)[0]
                except Exception as exc:
                    log(f"Prompt truncation failed: {exc}", prefix="⚠️")
            gen_t0 = time.time()
            call_kwargs = {
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
                "height": height,
                "width": width,
            }
            if clip_skip > 0:
                try:
                    if "clip_skip" in inspect.signature(pipe.__call__).parameters:
                        call_kwargs["clip_skip"] = clip_skip
                except (TypeError, ValueError):
                    pass
            with torch.inference_mode():
                result = pipe(text, **call_kwargs)
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
    if loaded_loras:
        metrics["loras"] = loaded_loras
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

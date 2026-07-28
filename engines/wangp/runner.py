"""HavnAI adapter for the isolated WanGP LTX-2.3 runtime."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


DEFAULT_WANGP_ROOT = Path.home() / ".havnai" / "tools" / "Wan2GP"
MODEL_FILENAME = "ltx-2.3-22b-distilled-1.1_diffusion_model_quanto_bf16_int8.safetensors"
REQUIRED_CHECKPOINTS = (
    MODEL_FILENAME,
    "ltx-2.3-22b_audio_vae.safetensors",
    "ltx-2.3-22b_embeddings_connector.safetensors",
    "ltx-2.3-22b_text_embedding_projection.safetensors",
    "ltx-2.3-22b_vae.safetensors",
    "ltx-2.3-22b_vocoder.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)
GEMMA_FILENAME = (
    "gemma-3-12b-it-qat-q4_0-unquantized/"
    "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors"
)
MAX_SOURCE_BYTES = 50 * 1024 * 1024


def _wangp_root() -> Path:
    return Path(os.getenv("HAVNAI_WANGP_ROOT", str(DEFAULT_WANGP_ROOT))).expanduser()


def _wangp_python(root: Path) -> Path:
    configured = os.getenv("HAVNAI_WANGP_PYTHON", "").strip()
    return Path(configured).expanduser() if configured else root / ".venv" / "bin" / "python"


def runtime_probe() -> Tuple[bool, str]:
    """Verify the exact runtime and files required by the registered model."""
    root = _wangp_root()
    python_bin = _wangp_python(root)
    if not (root / "shared" / "api.py").is_file():
        return False, f"WanGP API missing under {root}"
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        return False, f"WanGP Python missing or not executable: {python_bin}"
    missing = [name for name in (*REQUIRED_CHECKPOINTS, GEMMA_FILENAME) if not (root / "ckpts" / name).is_file()]
    if missing:
        return False, f"WanGP checkpoint missing: {missing[0]}"
    return True, "ready"


def _safe_task_id(value: Any) -> str:
    text = "".join(ch for ch in str(value or "video") if ch.isalnum() or ch in {"-", "_"})
    return text[:96] or "video"


def _write_source_bytes(data: bytes, destination: Path) -> Path:
    if not data:
        raise ValueError("Init image is empty")
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Init image exceeds the 50 MB limit")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.load()
            normalized = image.convert("RGB")
            normalized.save(destination, format="PNG", optimize=True)
    except Exception as exc:
        raise ValueError(f"Init image is invalid: {exc}") from exc
    return destination


def _materialize_source(value: Any, destination: Path, base_url: str = "") -> Optional[Path]:
    source = str(value or "").strip()
    if not source:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Init image data URL must be base64 encoded")
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("Init image data URL is not valid base64") from exc
        return _write_source_bytes(data, destination)

    parsed = urllib.parse.urlparse(source)
    local_path = Path(urllib.parse.unquote(parsed.path) if parsed.scheme == "file" else source).expanduser()
    if parsed.scheme in {"", "file"} and local_path.is_file():
        if local_path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("Init image exceeds the 50 MB limit")
        return _write_source_bytes(local_path.read_bytes(), destination)

    if parsed.scheme in {"http", "https"} or source.startswith("/"):
        import requests

        url = source
        if source.startswith("/"):
            if not base_url:
                raise ValueError("Relative init image URL has no coordinator base URL")
            url = f"{base_url.rstrip('/')}/{source.lstrip('/')}"
        response = requests.get(url, timeout=(15, 90), stream=True)
        response.raise_for_status()
        chunks = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            chunks.extend(chunk)
            if len(chunks) > MAX_SOURCE_BYTES:
                raise ValueError("Init image exceeds the 50 MB limit")
        return _write_source_bytes(bytes(chunks), destination)

    raise FileNotFoundError(f"Init image not found: {local_path}")


def _terminate_process(process: subprocess.Popen[Any], grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _video_stream_info(path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (ValueError, json.JSONDecodeError):
        return {}
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "native_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def run_wangp_ltx23(
    task: Dict[str, Any],
    *,
    log_fn: Callable[[str], None],
    outputs_dir: Path,
    read_gpu_stats: Callable[[], Dict[str, Any]],
    utilization_hint: int,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], int, Optional[Path]]:
    """Run one LTX-2.3 job in WanGP's venv and return HavnAI metrics."""
    ready, reason = runtime_probe()
    if not ready:
        raise RuntimeError(reason)

    root = _wangp_root()
    python_bin = _wangp_python(root)
    task_id = _safe_task_id(task.get("task_id"))
    work_dir = outputs_dir / ".wangp-tasks" / task_id
    worker_output_dir = work_dir / "generated"
    request_path = work_dir / "request.json"
    status_path = work_dir / "status.json"
    result_path = work_dir / "result.json"
    log_path = outputs_dir / "logs" / f"wangp_{task_id}.log"
    output_path = outputs_dir / f"video_{task_id}.mp4"
    source_path = _materialize_source(
        task.get("init_image"),
        work_dir / "source.png",
        str(task.get("_server_base") or ""),
    )

    fps = max(8, min(30, int(task.get("fps") or 24)))
    requested_frames = max(9, min(257, int(task.get("frames") or 97)))
    frames = max(9, ((requested_frames - 1) // 8) * 8 + 1)
    width = max(256, int(task.get("width") or 1280))
    height = max(256, int(task.get("height") or 704))
    if (width, height) == (1280, 704):
        resolution = "1280x720"
    elif (width, height) == (704, 1280):
        resolution = "720x1280"
    else:
        resolution = f"{width}x{height}"
    timeout = max(60, min(7200, int(task.get("timeout") or 3600)))
    seed = int(task.get("seed") or 0)
    if seed < 0:
        seed = int(time.time_ns() % (2**31))
    source_strength_raw = task.get("strength")
    if source_strength_raw is None:
        source_strength_raw = task.get("motion_strength")
    if source_strength_raw is None:
        source_strength_raw = 1.0

    request_payload = {
        "wangp_root": str(root),
        "output_dir": str(worker_output_dir),
        "output_path": str(output_path),
        "status_path": str(status_path),
        "result_path": str(result_path),
        "model_type": "ltx2_22B_distilled_1_1",
        "prompt": str(task.get("prompt") or "").strip(),
        "negative_prompt": str(task.get("negative_prompt") or "").strip(),
        "source_image": str(source_path) if source_path else None,
        "resolution": resolution,
        "steps": max(1, min(16, int(task.get("steps") or 8))),
        "guidance": max(0.0, min(10.0, float(task.get("guidance") or 1.0))),
        "frames": frames,
        "fps": fps,
        "duration_seconds": round((frames - 1) / fps, 3),
        "source_strength": max(0.1, min(1.0, float(source_strength_raw))),
        "seed": seed,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    worker_output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request_payload, indent=2) + "\n", encoding="utf-8")

    worker_script = Path(__file__).with_name("worker.py")
    started = time.monotonic()
    last_status = ""
    last_progress = 5.0
    cancel_event = task.get("_cancel_event")
    log_fn(f"Starting WanGP LTX-2.3 worker ({frames} frames at {fps}fps, {resolution})")
    if progress_fn:
        progress_fn(last_progress, "loading_video_model")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(python_bin), str(worker_script), str(request_path)],
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_process(process)
                    return (
                        {
                            "status": "cancelled",
                            "task_type": "video_gen",
                            "model_family": "ltx23_wangp",
                            "error": "cancelled_by_user",
                        },
                        utilization_hint,
                        None,
                    )
                if time.monotonic() - started >= timeout:
                    _terminate_process(process)
                    raise TimeoutError(f"WanGP LTX-2.3 generation exceeded {timeout}s")

                status_text = ""
                try:
                    status_text = status_path.read_text(encoding="utf-8")
                except OSError:
                    pass
                if status_text and status_text != last_status:
                    last_status = status_text
                    update = _read_json(status_path)
                    raw_progress = max(0.0, min(100.0, float(update.get("progress") or 0)))
                    last_progress = max(last_progress, min(92.0, 10.0 + raw_progress * 0.82))
                    phase = str(update.get("phase") or "generating")
                    if progress_fn:
                        progress_fn(last_progress, phase)
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                _terminate_process(process)

    result = _read_json(result_path)
    if process.returncode != 0 or not result.get("success"):
        errors = result.get("errors") if isinstance(result.get("errors"), list) else []
        detail = "; ".join(str(value) for value in errors if value) or f"worker_exit_{process.returncode}"
        raise RuntimeError(detail)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("WanGP reported success but produced no MP4")

    stream_info = _video_stream_info(output_path)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if progress_fn:
        progress_fn(93, "finalizing_video")
    gpu_stats = read_gpu_stats() or {}
    try:
        utilization = int(float(gpu_stats.get("utilization", utilization_hint)))
    except (TypeError, ValueError):
        utilization = utilization_hint
    metrics: Dict[str, Any] = {
        "status": "success",
        "task_type": "video_gen",
        "model_name": str(task.get("model_name") or "ltx23_wangp_distilled"),
        "model_family": "ltx23_wangp",
        "model_version": "2.3-distilled-1.1",
        "backend": "wangp",
        "seed": seed,
        "steps": request_payload["steps"],
        "guidance": request_payload["guidance"],
        "frames": frames,
        "fps": fps,
        "width": stream_info.get("width") or width,
        "height": stream_info.get("height") or height,
        "native_audio": bool(stream_info.get("native_audio")),
        "inference_time_ms": elapsed_ms,
        "generation_ms": elapsed_ms,
        "resolved_prompt": request_payload["prompt"],
        "resolved_negative_prompt": request_payload["negative_prompt"],
    }
    log_fn(f"WanGP LTX-2.3 completed in {elapsed_ms / 1000:.1f}s")
    return metrics, utilization, output_path


def video_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

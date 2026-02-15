"""LTX2 entrypoint for Latte-1 video generation."""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from .ltx2_generator import generate_video_frames

LogFn = Callable[[str], None]

# Hard timeout (seconds) for a single generation.  0 = no limit.
# Can be set per-job via job["timeout"] or globally via env var.
LTX2_JOB_TIMEOUT = int(os.environ.get("LTX2_JOB_TIMEOUT", "300"))


class _JobTimeout(Exception):
    """Raised when a video generation exceeds its time budget."""


def _default_logger(message: str) -> None:
    print(message)


def _clamp_int(
    name: str,
    value: Any,
    default: int,
    min_value: int,
    max_value: int,
    log_fn: LogFn,
) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    clamped = max(min_value, min(parsed, max_value))
    if parsed != clamped:
        log_fn(f"Adjusted {name} from {parsed} to {clamped}")
    return clamped


def _clamp_float(
    name: str,
    value: Any,
    default: float,
    min_value: float,
    max_value: float,
    log_fn: LogFn,
) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    clamped = max(min_value, min(parsed, max_value))
    if parsed != clamped:
        log_fn(f"Adjusted {name} from {parsed} to {clamped}")
    return clamped


def _save_video_frames(frames_np: Any, output_path: Path, fps: int, log_fn: LogFn) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(str(output_path), frames_np, fps=fps)
        return
    if cv2 is None:
        raise RuntimeError("imageio or opencv-python-headless is required to write mp4")
    height, width = frames_np[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    try:
        for frame in frames_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _normalize_video_frames(frames_np: Any) -> Any:
    """Ensure frames are shaped (F, H, W[, C]) with channels in the last dim."""
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required to normalize LTX2 frames") from exc

    arr = frames_np
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # Drop batch dimension if present.
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4:
        if arr.shape[-1] in {1, 3, 4}:
            return arr
        if arr.shape[1] in {1, 3, 4}:
            return arr.transpose(0, 2, 3, 1)
        if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            return arr.transpose(1, 2, 3, 0)
    if arr.ndim == 3:
        return arr

    raise RuntimeError(f"Unexpected LTX2 frame shape: {arr.shape}")


def _load_init_image(source: Any) -> Optional[Any]:
    """Load an init image from URL, base64 data URL, raw base64, or local path."""
    if source is None:
        return None
    if Image is None:
        raise RuntimeError("PIL is required for init_image support")

    try:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, str):
            text = source.strip()
            if not text:
                return None
            if text.startswith("data:image"):
                header, b64_data = text.split(",", 1)
                img_bytes = base64.b64decode(b64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if text.startswith("http://") or text.startswith("https://"):
                if requests is None:
                    raise RuntimeError("requests is required to fetch init_image URLs")
                resp = requests.get(text, timeout=20)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            # Assume local file path
            return Image.open(text).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to load init_image: {exc}") from exc
    return None


def run_ltx2(
    job: Dict[str, Any],
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    outputs_dir: Optional[Path] = None,
    read_gpu_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    utilization_hint: int = 0,
    model_id: str = "maxin-cn/Latte-1",
) -> Tuple[Dict[str, Any], int, Optional[Path]]:
    """Run an LTX2 (Latte-1) job. Returns metrics, utilization, and video path."""
    log_fn = log_fn or _default_logger
    outputs_dir = outputs_dir or Path.cwd() / "outputs"
    read_gpu_stats = read_gpu_stats or (lambda: {})

    job_id = str(job.get("task_id") or job.get("job_id") or "ltx2")
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        return ({"status": "failed", "error": "prompt is required"}, utilization_hint, None)

    seed_raw = job.get("seed")
    try:
        seed = int(seed_raw)
    except Exception:
        return ({"status": "failed", "error": "seed is required"}, utilization_hint, None)

    steps = _clamp_int("steps", job.get("steps", 20), 20, 1, 50, log_fn)
    guidance = _clamp_float("guidance", job.get("guidance", 7.0), 7.0, 0.0, 12.0, log_fn)
    width = _clamp_int("width", job.get("width", 512), 512, 256, 768, log_fn)
    height = _clamp_int("height", job.get("height", 512), 512, 256, 768, log_fn)
    frames = _clamp_int("frames", job.get("frames", 16), 16, 1, 16, log_fn)
    fps = _clamp_int("fps", job.get("fps", 8), 8, 1, 12, log_fn)
    negative_prompt = str(job.get("negative_prompt") or "").strip()
    timeout = int(job.get("timeout", 0) or LTX2_JOB_TIMEOUT)
    init_image_raw = (
        job.get("init_image")
        or job.get("init_image_url")
        or job.get("init_image_b64")
        or job.get("init_image_path")
        or job.get("image")
        or job.get("image_b64")
    )
    init_image = None
    if init_image_raw:
        init_image = _load_init_image(init_image_raw)
    model_ref = os.environ.get("LTX2_MODEL_PATH") or os.environ.get("LTX2_MODEL_ID") or model_id

    output_path = outputs_dir / f"video_{job_id}.mp4"

    log_fn(
        f"[{job_id}] LTX2 start: steps={steps}, frames={frames}, "
        f"fps={fps}, {width}x{height}, guidance={guidance}, "
        f"timeout={timeout}s"
    )

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

    # --- step-level progress callback ---
    _timed_out = threading.Event()

    def _step_callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
        elapsed = time.time() - started
        log_fn(
            f"[{job_id}] step {step + 1}/{steps} "
            f"({elapsed:.1f}s elapsed)"
        )
        if _timed_out.is_set():
            raise _JobTimeout(
                f"Generation exceeded {timeout}s timeout at step {step + 1}"
            )
        return kwargs

    # --- timeout watchdog thread ---
    def _watchdog() -> None:
        if timeout <= 0:
            return
        deadline = started + timeout
        while time.time() < deadline:
            if _timed_out.wait(timeout=1.0):
                return  # generation finished early
        _timed_out.set()

    watchdog_thread: Optional[threading.Thread] = None
    if timeout > 0:
        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

    try:
        if torch is None:
            raise RuntimeError("torch is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        video_frames = generate_video_frames(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            frames=frames,
            seed=seed,
            model_id=model_ref,
            init_image=init_image,
            callback_on_step_end=_step_callback,
        )
        frames_np = None
        if torch is not None and isinstance(video_frames, torch.Tensor):
            frames_np = (video_frames.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        elif isinstance(video_frames, (list, tuple)):
            if not video_frames:
                raise RuntimeError("ltx2 returned empty frame list")
            first = video_frames[0]
            if torch is not None and isinstance(first, torch.Tensor):
                stack = torch.stack(list(video_frames), dim=0)
                frames_np = (stack.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
            else:
                try:
                    import numpy as np  # type: ignore
                except Exception as exc:
                    raise RuntimeError("numpy is required to convert LTX2 frames") from exc
                frames_np = np.stack([np.array(frame) for frame in video_frames])
                if frames_np.dtype != "uint8":
                    frames_np = (frames_np * 255).clip(0, 255).astype("uint8")
        else:
            try:
                import numpy as np  # type: ignore
            except Exception as exc:
                raise RuntimeError("numpy is required to convert LTX2 frames") from exc
            if isinstance(video_frames, np.ndarray):
                frames_np = video_frames
                if frames_np.dtype != "uint8":
                    frames_np = (frames_np * 255).clip(0, 255).astype("uint8")
        if frames_np is None:
            raise RuntimeError("Unsupported LTX2 frame format")
        frames_np = _normalize_video_frames(frames_np)
        _save_video_frames(frames_np, output_path, fps, log_fn)
    except _JobTimeout as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] LTX2 TIMEOUT: {error_msg}")
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except RuntimeError as exc:
        status = "failed"
        error_msg = str(exc)
        if "out of memory" in error_msg.lower() and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log_fn(f"[{job_id}] LTX2 generation failed: {error_msg}")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] LTX2 generation failed: {error_msg}")
    finally:
        # Signal watchdog to stop regardless of outcome
        _timed_out.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)

    duration = time.time() - started
    log_fn(
        f"[{job_id}] LTX2 finished: status={status}, "
        f"duration={duration:.1f}s"
    )
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)

    metrics: Dict[str, Any] = {
        "status": status,
        "model_name": str(job.get("model_name") or "ltx2"),
        "reward_weight": float(job.get("reward_weight", 1.0)),
        "task_type": "video_gen",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "seed": seed,
        "timeout": timeout,
        "timed_out": "timeout" in error_msg.lower() if error_msg else False,
        "output_path": str(output_path) if status == "success" else None,
    }
    if status == "failed":
        metrics["error"] = error_msg or "ltx2 generation error"
    return metrics, int(util), output_path if status == "success" else None


def video_to_b64(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

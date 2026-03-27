"""LTX-Video 2.3 job runner.

This is the main entry-point called by ``client.py`` for VIDEO_GEN jobs
that target the ``ltx_video`` pipeline family.  It orchestrates:

  1. Parameter validation and clamping
  2. Base generation (via ``generator.generate_frames``)
  3. Optional upscaler stages (spatial / temporal)
  4. Frame normalization → MP4 encoding
  5. Metrics collection

The legacy ``engines/ltx2/ltx2_runner.py`` (Latte-1) remains untouched
for backward compatibility.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image
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
    import requests as _requests  # type: ignore
except Exception:  # pragma: no cover
    _requests = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from .config import LTXVideoConfig, load_config
from .generator import generate_frames
from .upscaler import apply_spatial_upscale, apply_temporal_upscale

LogFn = Callable[[str], None]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT = int(os.environ.get("HAVNAI_LTX_VIDEO_JOB_TIMEOUT", "600"))


class _JobTimeout(Exception):
    pass


def _default_logger(message: str) -> None:
    print(message)


# ---------------------------------------------------------------------------
# Helpers (shared with ltx2_runner — kept local to avoid cross-engine deps)
# ---------------------------------------------------------------------------

def _clamp_int(name: str, value: Any, default: int, lo: int, hi: int, log_fn: LogFn) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    clamped = max(lo, min(v, hi))
    if v != clamped:
        log_fn(f"Adjusted {name} from {v} to {clamped}")
    return clamped


def _clamp_float(name: str, value: Any, default: float, lo: float, hi: float, log_fn: LogFn) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    clamped = max(lo, min(v, hi))
    if v != clamped:
        log_fn(f"Adjusted {name} from {v} to {clamped}")
    return clamped


def _load_init_image(source: Any) -> Optional[Any]:
    """Load an init image from URL, base64 data-URL, raw base64, or path."""
    if source is None:
        return None
    if Image is None:
        raise RuntimeError("PIL is required for init_image support")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return None
        if text.startswith("data:image"):
            _, b64 = text.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if text.startswith(("http://", "https://")):
            if _requests is None:
                raise RuntimeError("requests is required to fetch init_image URLs")
            resp = _requests.get(text, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(text).convert("RGB")
    return None


def _normalize_frames(raw: Any) -> Any:
    """Ensure frames are (F, H, W, C) uint8 numpy array."""
    if np is None:
        raise RuntimeError("numpy is required")

    arr = raw
    if torch is not None and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu()
        if arr.is_floating_point():
            arr = (arr * 255).clamp(0, 255).byte()
        arr = arr.numpy()
    elif isinstance(arr, (list, tuple)):
        if not arr:
            raise RuntimeError("Empty frame list")
        first = arr[0]
        if torch is not None and isinstance(first, torch.Tensor):
            stack = torch.stack(list(arr), dim=0)
            if stack.is_floating_point():
                stack = (stack * 255).clamp(0, 255).byte()
            arr = stack.cpu().numpy()
        else:
            arr = np.stack([np.array(f) for f in arr])

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)

    # Fix channel ordering
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            return arr
        if arr.shape[1] in (1, 3, 4):
            return arr.transpose(0, 2, 3, 1)
    if arr.ndim == 3:
        return arr
    raise RuntimeError(f"Unexpected frame shape: {arr.shape}")


def _save_video(frames_np: Any, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(str(path), frames_np, fps=fps)
        return
    if cv2 is None:
        raise RuntimeError("imageio or opencv-python-headless is required to write mp4")
    h, w = frames_np[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    try:
        for frame in frames_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_ltx_video(
    job: Dict[str, Any],
    *,
    log_fn: Optional[LogFn] = None,
    outputs_dir: Optional[Path] = None,
    read_gpu_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    utilization_hint: int = 0,
    config: Optional[LTXVideoConfig] = None,
) -> Tuple[Dict[str, Any], int, Optional[Path]]:
    """Execute an LTX-Video 2.3 job.

    Returns ``(metrics_dict, gpu_utilization, output_path_or_None)``.
    """
    log_fn = log_fn or _default_logger
    outputs_dir = outputs_dir or Path.cwd() / "outputs"
    read_gpu_stats = read_gpu_stats or (lambda: {})
    config = config or load_config()

    job_id = str(job.get("task_id") or job.get("job_id") or "ltx_video")
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        return _fail(job, "prompt is required", utilization_hint)

    seed_raw = job.get("seed")
    try:
        seed = int(seed_raw)
    except Exception:
        return _fail(job, "seed is required", utilization_hint)

    # Pipeline mode selection
    pipeline_mode = str(job.get("pipeline_mode") or config.defaults.get("pipeline_mode", "two_stage"))
    try:
        mode_cfg = config.mode_config(pipeline_mode)
    except Exception as exc:
        return _fail(job, str(exc), utilization_hint)

    # Validate assets
    missing = config.validate_mode_assets(pipeline_mode)
    if missing:
        return _fail(
            job,
            f"Pipeline mode {pipeline_mode!r} missing assets: {', '.join(missing)}",
            utilization_hint,
        )

    # Resolve defaults — payload overrides > mode defaults
    mode_defaults = config.mode_defaults(pipeline_mode)
    limits = config.mode_limits(pipeline_mode)

    steps = _clamp_int("steps", job.get("steps", mode_defaults["steps"]),
                        mode_defaults["steps"], 1, 150, log_fn)
    guidance = _clamp_float("guidance", job.get("guidance", mode_defaults["guidance"]),
                            mode_defaults["guidance"], 0.0, 20.0, log_fn)
    width = _clamp_int("width", job.get("width", mode_defaults["width"]),
                        mode_defaults["width"], 256, limits["max_width"], log_fn)
    height = _clamp_int("height", job.get("height", mode_defaults["height"]),
                         mode_defaults["height"], 256, limits["max_height"], log_fn)
    frames = _clamp_int("frames", job.get("frames", mode_defaults["frames"]),
                         mode_defaults["frames"], 1, limits["max_frames"], log_fn)
    fps = _clamp_int("fps", job.get("fps", mode_defaults["fps"]),
                      mode_defaults["fps"], 1, 60, log_fn)
    negative_prompt = str(job.get("negative_prompt") or "").strip()

    # Checkpoint variant
    checkpoint_variant = str(
        job.get("checkpoint_variant")
        or config.defaults.get("checkpoint_variant", "dev")
    )
    # Distilled modes force the distilled checkpoint
    required_ckpts = mode_cfg.get("requires_checkpoint", [])
    if required_ckpts and checkpoint_variant not in required_ckpts:
        checkpoint_variant = required_ckpts[0]
        log_fn(f"[{job_id}] Overriding checkpoint to {checkpoint_variant!r} for mode {pipeline_mode!r}")

    # Upscaler selection
    upscaler = str(job.get("upscaler") or mode_cfg.get("default_upscaler") or "").strip()
    enable_temporal_upscale = bool(job.get("temporal_upscale", False))
    if pipeline_mode == "two_stage_hq":
        enable_temporal_upscale = True

    # Init image
    init_image_raw = (
        job.get("init_image") or job.get("init_image_url")
        or job.get("init_image_b64") or job.get("image") or None
    )
    init_image = None
    if init_image_raw:
        if not mode_cfg.get("supports_image_input", False):
            log_fn(f"[{job_id}] Warning: mode {pipeline_mode!r} does not support image input; ignoring init_image")
        else:
            init_image = _load_init_image(init_image_raw)

    # Timeout
    timeout_raw = job.get("timeout", 0)
    try:
        timeout = int(timeout_raw or 0)
    except Exception:
        timeout = 0
    if timeout <= 0:
        timeout = _DEFAULT_TIMEOUT

    output_path = outputs_dir / f"video_{job_id}.mp4"

    log_fn(
        f"[{job_id}] LTX-Video start: mode={pipeline_mode}, "
        f"ckpt={checkpoint_variant}, steps={steps}, frames={frames}, "
        f"fps={fps}, {width}x{height}, guidance={guidance}, "
        f"upscaler={upscaler or 'none'}, temporal_up={enable_temporal_upscale}, "
        f"timeout={timeout}s"
    )

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

    # Timeout machinery
    _timed_out = threading.Event()
    _progress_started: Dict[str, float] = {}

    def _step_cb(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
        now = time.time()
        if "start" not in _progress_started:
            _progress_started["start"] = now
            log_fn(f"[{job_id}] First denoise step after {now - started:.1f}s warmup")
        elapsed = now - _progress_started["start"]
        log_fn(f"[{job_id}] step {step + 1}/{steps} ({elapsed:.1f}s elapsed)")
        if _timed_out.is_set():
            raise _JobTimeout(f"Exceeded {timeout}s timeout at step {step + 1}")
        return kwargs

    def _watchdog() -> None:
        if timeout <= 0:
            return
        while not _timed_out.wait(timeout=1.0):
            if "start" in _progress_started and time.time() >= _progress_started["start"] + timeout:
                _timed_out.set()
                return

    wd_thread: Optional[threading.Thread] = None
    if timeout > 0:
        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        wd_thread.start()

    try:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # --- Stage 1: base generation ---
        video_frames = generate_frames(
            config,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            frames=frames,
            seed=seed,
            checkpoint_variant=checkpoint_variant,
            init_image=init_image,
            callback_on_step_end=_step_cb,
        )

        gen_time = time.time() - started
        log_fn(f"[{job_id}] Base generation complete in {gen_time:.1f}s")

        # --- Stage 2: optional upscalers ---
        if upscaler and upscaler.startswith("spatial_upscaler"):
            scale = "x2" if "x2" in upscaler else "x1_5"
            up_start = time.time()
            video_frames = apply_spatial_upscale(video_frames, config, scale=scale)
            log_fn(f"[{job_id}] Spatial upscale ({scale}) done in {time.time() - up_start:.1f}s")

        if enable_temporal_upscale:
            up_start = time.time()
            video_frames = apply_temporal_upscale(video_frames, config)
            log_fn(f"[{job_id}] Temporal upscale done in {time.time() - up_start:.1f}s")

        # --- Normalize and save ---
        frames_np = _normalize_frames(video_frames)
        _save_video(frames_np, output_path, fps)

    except _JobTimeout as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] LTX-Video TIMEOUT: {error_msg}")
        _cuda_cleanup()
    except RuntimeError as exc:
        status = "failed"
        error_msg = str(exc)
        if "out of memory" in error_msg.lower():
            _cuda_cleanup()
        log_fn(f"[{job_id}] LTX-Video failed: {error_msg}")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] LTX-Video failed: {error_msg}")
    finally:
        _timed_out.set()
        if wd_thread:
            wd_thread.join(timeout=2.0)

    duration = time.time() - started
    log_fn(f"[{job_id}] LTX-Video finished: status={status}, duration={duration:.1f}s")

    end_stats = read_gpu_stats()
    util = max(
        start_stats.get("utilization", 0),
        end_stats.get("utilization", 0),
        utilization_hint,
    )

    metrics: Dict[str, Any] = {
        "status": status,
        "model_name": str(job.get("model_name") or "ltx_video"),
        "model_family": "ltx_video",
        "model_version": config.version,
        "reward_weight": float(job.get("reward_weight", 1.0)),
        "task_type": "video_gen",
        "pipeline_mode": pipeline_mode,
        "checkpoint_variant": checkpoint_variant,
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
        "upscaler": upscaler or None,
        "temporal_upscale": enable_temporal_upscale,
        "timed_out": "timeout" in error_msg.lower() if error_msg else False,
        "output_path": str(output_path) if status == "success" else None,
    }
    if status == "failed":
        metrics["error"] = error_msg or "ltx_video generation error"

    return metrics, int(util), output_path if status == "success" else None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fail(job: Dict[str, Any], error: str, util: int) -> Tuple[Dict[str, Any], int, None]:
    return (
        {
            "status": "failed",
            "task_type": "video_gen",
            "model_name": str(job.get("model_name") or "ltx_video"),
            "model_family": "ltx_video",
            "reward_weight": float(job.get("reward_weight", 1.0)),
            "error": error,
        },
        util,
        None,
    )


def _cuda_cleanup() -> None:
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def video_to_b64(path: Path) -> Optional[str]:
    """Encode an MP4 file to base64."""
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

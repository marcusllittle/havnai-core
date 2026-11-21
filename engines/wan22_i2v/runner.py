"""Wan 2.2 I2V job runner.

The client calls ``run_wan22_i2v`` the same way it calls ``run_ltx_video``:
validate the job, generate frames, write an MP4, and return metrics plus the
output path. Encoding matches the LTX 0.9.8 runner (imageio, then OpenCV).
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

from .config import Wan22I2VConfig, load_config, resolve_settings
from .generator import generate_frames

LogFn = Callable[[str], None]


class _JobTimeout(Exception):
    pass


def _default_logger(message: str) -> None:
    print(message)


def _load_init_image(source: Any) -> Optional[Any]:
    if source is None:
        return None
    if Image is None:
        raise RuntimeError("PIL is required to read the Wan 2.2 source image")
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
                raise RuntimeError("requests is required to fetch the source image")
            response = _requests.get(text, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        return Image.open(text).convert("RGB")
    return None


def _normalize_frames(raw: Any) -> Any:
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
            raise RuntimeError("Wan 2.2 returned no frames")
        first = arr[0]
        if torch is not None and isinstance(first, torch.Tensor):
            stack = torch.stack(list(arr), dim=0)
            if stack.is_floating_point():
                stack = (stack * 255).clamp(0, 255).byte()
            arr = stack.cpu().numpy()
        else:
            arr = np.stack([np.array(frame) for frame in arr])
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.dtype != np.uint8:
        scale = 255 if getattr(arr, "max", lambda: 1)() <= 1.0 else 1
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            return arr
        if arr.shape[1] in (1, 3, 4):
            return arr.transpose(0, 2, 3, 1)
    if arr.ndim == 3:
        return arr
    raise RuntimeError(f"Unexpected Wan frame shape: {getattr(arr, 'shape', None)}")


def _save_video(frames_np: Any, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(str(path), frames_np, fps=fps)
        return
    if cv2 is None:
        raise RuntimeError("imageio or opencv-python-headless is required to write mp4")
    height, width = frames_np[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    try:
        for frame in frames_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _fail(job: Dict[str, Any], error: str, util: int) -> Tuple[Dict[str, Any], int, None]:
    return (
        {
            "status": "failed",
            "task_type": "video_gen",
            "model_name": str(job.get("model_name") or "wan22-i2v-a14b"),
            "model_family": "wan22_i2v",
            "reward_weight": float(job.get("reward_weight", 1.0)),
            "error": error,
        },
        util,
        None,
    )


def _cuda_cleanup() -> None:
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _backend() -> str:
    name = os.environ.get("HAI_WAN22_BACKEND", "comfy").strip().lower() or "comfy"
    return name if name in {"comfy", "diffusers"} else "comfy"


def run_wan22_i2v(
    job: Dict[str, Any],
    *,
    log_fn: Optional[LogFn] = None,
    outputs_dir: Optional[Path] = None,
    read_gpu_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    utilization_hint: int = 0,
    config: Optional[Wan22I2VConfig] = None,
    torch_module: Any = None,
    diffusers_module: Any = None,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], int, Optional[Path]]:
    """Execute one Wan 2.2 image-to-video job.

    The default backend is ComfyUI, which is the measured 3060 recipe.
    ``HAI_WAN22_BACKEND=diffusers`` keeps the in-process loader.
    """
    if _backend() != "diffusers":
        from .comfy import run_comfy_i2v

        return run_comfy_i2v(
            job,
            log_fn=log_fn or _default_logger,
            outputs_dir=outputs_dir or Path.cwd() / "outputs",
            read_gpu_stats=read_gpu_stats or (lambda: {}),
            utilization_hint=utilization_hint,
            config=config,
            progress_fn=progress_fn,
        )
    return _run_diffusers_i2v(
        job,
        log_fn=log_fn,
        outputs_dir=outputs_dir,
        read_gpu_stats=read_gpu_stats,
        utilization_hint=utilization_hint,
        config=config,
        torch_module=torch_module,
        diffusers_module=diffusers_module,
        progress_fn=progress_fn,
    )


def _run_diffusers_i2v(
    job: Dict[str, Any],
    *,
    log_fn: Optional[LogFn] = None,
    outputs_dir: Optional[Path] = None,
    read_gpu_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    utilization_hint: int = 0,
    config: Optional[Wan22I2VConfig] = None,
    torch_module: Any = None,
    diffusers_module: Any = None,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], int, Optional[Path]]:
    log_fn = log_fn or _default_logger
    outputs_dir = outputs_dir or Path.cwd() / "outputs"
    read_gpu_stats = read_gpu_stats or (lambda: {})
    config = config or load_config()
    job_id = str(job.get("task_id") or job.get("job_id") or "wan22_i2v")

    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        return _fail(job, "prompt is required", utilization_hint)
    try:
        seed = int(job.get("seed"))
    except (TypeError, ValueError):
        return _fail(job, "seed is required", utilization_hint)
    try:
        settings = resolve_settings(job, config)
    except ValueError as exc:
        return _fail(job, str(exc), utilization_hint)

    init_raw = (
        job.get("init_image")
        or job.get("init_image_url")
        or job.get("init_image_b64")
        or job.get("image")
        or None
    )
    if not init_raw:
        return _fail(job, "Wan 2.2 image-to-video requires a source image", utilization_hint)
    try:
        init_image = _load_init_image(init_raw)
    except Exception as exc:
        return _fail(job, f"Failed to load source image: {exc}", utilization_hint)
    if init_image is None:
        return _fail(job, "Wan 2.2 image-to-video requires a source image", utilization_hint)

    try:
        timeout = int(job.get("timeout") or config.defaults.get("timeout_seconds") or 1800)
    except (TypeError, ValueError):
        timeout = 1800
    timeout = max(60, min(timeout, 7200))
    output_path = outputs_dir / f"video_{job_id}.mp4"
    log_fn(
        f"[{job_id}] Wan 2.2 I2V start: {settings['width']}x{settings['height']} "
        f"frames={settings['frames']} steps={settings['steps']} "
        f"guidance={settings['guidance']}/{settings['guidance_2']} "
        f"shift={settings['flow_shift']} boundary={settings['boundary_ratio']:.3f} "
        f"offload={settings['offload']} preset={settings['motion_preset'] or '-'}"
    )
    if settings["requested_strength"] != 1.0:
        log_fn(f"[{job_id}] {settings['strength_note']} Requested strength={settings['requested_strength']}.")
    if settings["requested_size"] != (settings["width"], settings["height"]):
        log_fn(
            f"[{job_id}] Clamped {settings['requested_size'][0]}x{settings['requested_size'][1]} "
            f"to {settings['width']}x{settings['height']} for the 12 GB profile. "
            "Set HAI_WAN22_ALLOW_720P=1 or allow_720p on the job to keep a larger size."
        )

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""
    timed_out = threading.Event()
    cancel_event = job.get("_cancel_event")
    progress_started: Dict[str, float] = {}

    def _step_cb(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled_by_user")
        now = time.time()
        if "start" not in progress_started:
            progress_started["start"] = now
        if timed_out.is_set() and step + 1 < int(settings["steps"]):
            raise _JobTimeout(f"Exceeded {timeout}s timeout at step {step + 1}")
        if progress_fn is not None:
            done = step + 1
            progress_fn(min(90.0, 15.0 + (done / max(int(settings["steps"]), 1)) * 70.0), "sampling")
        return kwargs

    def _watchdog() -> None:
        while not timed_out.wait(timeout=1.0):
            if "start" in progress_started and time.time() >= progress_started["start"] + timeout:
                timed_out.set()
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()
    try:
        frames = generate_frames(
            config,
            prompt=settings["prompt"],
            negative_prompt=settings["negative_prompt"],
            image=init_image,
            settings=settings,
            seed=seed,
            callback_on_step_end=_step_cb,
            torch_module=torch_module,
            diffusers_module=diffusers_module,
        )
        _save_video(_normalize_frames(frames), output_path, int(settings["fps"]))
    except _JobTimeout as exc:
        status = "failed"
        error_msg = str(exc)
        _cuda_cleanup()
        log_fn(f"[{job_id}] Wan 2.2 timeout: {error_msg}")
    except RuntimeError as exc:
        status = "cancelled" if str(exc) == "cancelled_by_user" else "failed"
        error_msg = str(exc)
        if "out of memory" in error_msg.lower():
            _cuda_cleanup()
        log_fn(f"[{job_id}] Wan 2.2 failed: {error_msg}")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] Wan 2.2 failed: {error_msg}")
    finally:
        timed_out.set()
        watchdog.join(timeout=2.0)

    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    metrics: Dict[str, Any] = {
        "status": status,
        "model_name": str(job.get("model_name") or "wan22-i2v-a14b"),
        "model_family": "wan22_i2v",
        "model_version": config.version,
        "reward_weight": float(job.get("reward_weight", 1.0)),
        "task_type": "video_gen",
        "pipeline_mode": "i2v_4step",
        "checkpoint_variant": "q4_k_m",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
        "steps": settings["steps"],
        "guidance": settings["guidance"],
        "guidance_2": settings["guidance_2"],
        "width": settings["width"],
        "height": settings["height"],
        "frames": settings["frames"],
        "fps": settings["fps"],
        "flow_shift": settings["flow_shift"],
        "boundary_ratio": settings["boundary_ratio"],
        "high_noise_steps": settings["high_noise_steps"],
        "lora_scale_high": settings["lora_scale_high"],
        "lora_scale_low": settings["lora_scale_low"],
        "offload": settings["offload"],
        "strength": settings["strength"],
        "requested_strength": settings["requested_strength"],
        "motion_preset": settings["motion_preset"],
        "seed": seed,
        "timeout": timeout,
        "timed_out": "timeout" in error_msg.lower() if error_msg else False,
        "output_path": str(output_path) if status == "success" else None,
    }
    if status != "success":
        metrics["error"] = error_msg or "wan22 i2v generation error"
    return metrics, int(util), output_path if status == "success" else None


def video_to_b64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

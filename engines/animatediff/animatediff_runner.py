"""AnimateDiff entrypoint for video generation."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from .animatediff_generator import generate_animatediff_frames

LogFn = Callable[[str], None]


def _default_logger(message: str) -> None:
    print(message)


def _clamp_int(name: str, value: Any, default: int, min_value: int, max_value: int, log_fn: LogFn) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    clamped = max(min_value, min(parsed, max_value))
    if parsed != clamped:
        log_fn(f"Adjusted {name} from {parsed} to {clamped}")
    return clamped


def _clamp_float(name: str, value: Any, default: float, min_value: float, max_value: float, log_fn: LogFn) -> float:
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
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    log_fn(f"Wrote video to {output_path}")


def _normalize_video_frames(frames_np: Any) -> Any:
    """Ensure frames are shaped (F, H, W[, C]) with channels in the last dim."""
    if np is None:
        raise RuntimeError("numpy is required to normalize AnimateDiff frames")
    arr = frames_np
    if arr.ndim == 5:  # (B, F, C, H, W) or (B, F, H, W, C)
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):  # (F, C, H, W)
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):  # (F, H, W, C)
        return arr
    raise RuntimeError(f"Unexpected AnimateDiff frame shape: {arr.shape}")


def _load_init_image(source: Any) -> Optional[Any]:
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
            return Image.open(text).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to load init_image: {exc}") from exc
    return None


def run_animatediff(
    job: Dict[str, Any],
    model_id: str,
    outputs_dir: Optional[Path] = None,
    log_fn: Optional[LogFn] = None,
    read_gpu_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    utilization_hint: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Path]]:
    """Run an AnimateDiff job. Returns metrics, utilization, and video path."""
    log_fn = log_fn or _default_logger
    outputs_dir = outputs_dir or Path.cwd() / "outputs"
    read_gpu_stats = read_gpu_stats or (lambda: {})
    utilization_hint = utilization_hint or {}

    job_id = str(job.get("task_id") or job.get("job_id") or "animatediff")
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        return ({"status": "failed", "error": "prompt is required"}, utilization_hint, None)

    seed_raw = job.get("seed")
    try:
        seed = int(seed_raw)
    except Exception:
        return ({"status": "failed", "error": "seed is required"}, utilization_hint, None)

    steps = _clamp_int("steps", job.get("steps", 25), 25, 1, 50, log_fn)
    guidance = _clamp_float("guidance", job.get("guidance", 6.0), 6.0, 0.0, 12.0, log_fn)
    width = _clamp_int("width", job.get("width", 512), 512, 256, 768, log_fn)
    height = _clamp_int("height", job.get("height", 512), 512, 256, 768, log_fn)
    frames = _clamp_int("frames", job.get("frames", 16), 16, 1, 32, log_fn)
    fps = _clamp_int("fps", job.get("fps", 8), 8, 1, 24, log_fn)
    negative_prompt = str(job.get("negative_prompt") or "").strip()
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

    adapter_id = (
        os.environ.get("ANIMATEDIFF_MOTION_ADAPTER")
        or job.get("motion_adapter")
        or "guoyww/animatediff-motion-adapter-v1-5-2"
    )
    strength = job.get("strength")
    try:
        strength = float(strength) if strength is not None else None
    except (TypeError, ValueError):
        strength = None

    output_path = outputs_dir / f"animatediff_{job_id}.mp4"

    log_fn(
        f"[{job_id}] AnimateDiff start: steps={steps}, frames={frames}, "
        f"fps={fps}, {width}x{height}, guidance={guidance}"
    )

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""
    try:
        if torch is None:
            raise RuntimeError("torch is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        try:
            video_frames = generate_animatediff_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                frames=frames,
                seed=seed,
            model_id=model_id,
            adapter_id=adapter_id,
            init_image=init_image,
            strength=strength,
        )
        except RuntimeError as exc:
            if init_image is not None and "init_image" in str(exc) and "support" in str(exc).lower():
                log_fn(f"[{job_id}] AnimateDiff pipeline does not support init_image; retrying without init image.")
                video_frames = generate_animatediff_frames(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance=guidance,
                    width=width,
                    height=height,
                    frames=frames,
                    seed=seed,
                    model_id=model_id,
                    adapter_id=adapter_id,
                    init_image=None,
                    strength=None,
                )
            else:
                raise
        frames_np = None
        if torch is not None and isinstance(video_frames, torch.Tensor):
            frames_np = (video_frames.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        elif isinstance(video_frames, (list, tuple)):
            if not video_frames:
                raise RuntimeError("animatediff returned empty frame list")
            first = video_frames[0]
            if torch is not None and isinstance(first, torch.Tensor):
                stack = torch.stack(list(video_frames), dim=0)
                frames_np = (stack.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
            else:
                if np is None:
                    raise RuntimeError("numpy is required to convert AnimateDiff frames")
                frames_np = np.stack([np.array(frame) for frame in video_frames])
                if frames_np.dtype != "uint8":
                    frames_np = (frames_np * 255).clip(0, 255).astype("uint8")
        else:
            if np is None:
                raise RuntimeError("numpy is required to convert AnimateDiff frames")
            if isinstance(video_frames, np.ndarray):
                frames_np = video_frames
                if frames_np.dtype != "uint8":
                    frames_np = (frames_np * 255).clip(0, 255).astype("uint8")
        if frames_np is None:
            raise RuntimeError("Unsupported AnimateDiff frame format")
        frames_np = _normalize_video_frames(frames_np)
        _save_video_frames(frames_np, output_path, fps, log_fn)
    except RuntimeError as exc:
        status = "failed"
        error_msg = str(exc)
        if "out of memory" in error_msg.lower() and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log_fn(f"[{job_id}] AnimateDiff generation failed: {error_msg}")
    except Exception as exc:  # pragma: no cover - safety net
        status = "failed"
        error_msg = str(exc)
        log_fn(f"[{job_id}] AnimateDiff generation failed: {error_msg}")

    duration = time.time() - started
    total_ms = int(duration * 1000)
    log_fn(
        f"[{job_id}] AnimateDiff finished: status={status}, "
        f"duration={duration:.1f}s"
    )
    end_stats = read_gpu_stats()
    metrics: Dict[str, Any] = {
        "status": status,
        "model_name": str(job.get("model_name") or "animatediff"),
        "task_type": "animatediff",
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
        "elapsed_ms": total_ms,
        "init_image": bool(init_image_raw),
        "output_path": str(output_path) if status == "success" else None,
    }
    if status != "success":
        metrics["error"] = error_msg or "animatediff generation error"
        return metrics, utilization_hint, None

    if isinstance(utilization_hint, dict):
        utilization = utilization_hint.copy()
    else:
        try:
            utilization_val = float(utilization_hint)
        except (TypeError, ValueError):
            utilization_val = 0.0
        utilization = {"utilization": utilization_val}
    utilization.update({"gpu_start": start_stats, "gpu_end": end_stats})
    return metrics, utilization, output_path


def video_to_b64(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")

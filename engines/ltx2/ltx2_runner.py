"""LTX2 entrypoint for Latte-1 video generation."""
from __future__ import annotations

import base64
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

from .ltx2_generator import generate_video_frames

LogFn = Callable[[str], None]


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

    steps = _clamp_int("steps", job.get("steps", 25), 25, 1, 50, log_fn)
    guidance = _clamp_float("guidance", job.get("guidance", 6.0), 6.0, 0.0, 12.0, log_fn)
    width = _clamp_int("width", job.get("width", 512), 512, 256, 512, log_fn)
    height = _clamp_int("height", job.get("height", 512), 512, 256, 512, log_fn)
    frames = _clamp_int("frames", job.get("frames", 48), 48, 1, 48, log_fn)
    fps = _clamp_int("fps", job.get("fps", 8), 8, 1, 8, log_fn)
    negative_prompt = str(job.get("negative_prompt") or "").strip()
    model_ref = os.environ.get("LTX2_MODEL_PATH") or os.environ.get("LTX2_MODEL_ID") or model_id

    output_path = outputs_dir / f"video_{job_id}.mp4"

    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    error_msg = ""

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
        )
        frames_np = (video_frames.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        _save_video_frames(frames_np, output_path, fps, log_fn)
    except RuntimeError as exc:
        status = "failed"
        error_msg = str(exc)
        if "out of memory" in error_msg.lower() and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log_fn(f"LTX2 generation failed: {error_msg}")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        log_fn(f"LTX2 generation failed: {error_msg}")

    duration = time.time() - started
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

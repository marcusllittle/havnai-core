"""AnimateDiff pipeline helpers."""
from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import Any, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from diffusers import (  # type: ignore
        AnimateDiffPipeline,
        MotionAdapter,
        StableDiffusionPipeline,
        DDIMScheduler,
    )
except Exception:  # pragma: no cover
    AnimateDiffPipeline = None  # type: ignore
    MotionAdapter = None  # type: ignore
    StableDiffusionPipeline = None  # type: ignore
    DDIMScheduler = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

_PIPE_LOCK = threading.Lock()
_PIPE: Optional[Any] = None
_PIPE_ID: Optional[str] = None
_ADAPTER_ID: Optional[str] = None


def load_animatediff_pipeline(
    model_id: str,
    adapter_id: str,
    device: str = "cuda",
) -> Any:
    """Load and cache AnimateDiff pipeline with a motion adapter."""
    if (
        AnimateDiffPipeline is None
        or MotionAdapter is None
        or StableDiffusionPipeline is None
        or torch is None
    ):
        raise RuntimeError("AnimateDiff dependencies are unavailable")

    global _PIPE, _PIPE_ID, _ADAPTER_ID
    with _PIPE_LOCK:
        if _PIPE is None or _PIPE_ID != model_id or _ADAPTER_ID != adapter_id:
            model_path = Path(model_id).expanduser()
            base_kwargs = {
                "torch_dtype": torch.float16,
                "safety_checker": None,
                "requires_safety_checker": False,
                "feature_extractor": None,
            }
            if model_path.exists() and model_path.is_file() and hasattr(StableDiffusionPipeline, "from_single_file"):
                base = StableDiffusionPipeline.from_single_file(
                    str(model_path), **base_kwargs
                )
            else:
                base = StableDiffusionPipeline.from_pretrained(
                    model_id, **base_kwargs
                )
            adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pipe(base, motion_adapter=adapter)
            if DDIMScheduler is not None:
                try:
                    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                except Exception:
                    pass
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            _PIPE = pipe.to(device)
            _PIPE_ID = model_id
            _ADAPTER_ID = adapter_id
        return _PIPE


def generate_animatediff_frames(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    frames: int,
    seed: int,
    model_id: str,
    adapter_id: str,
    init_image: Optional[Any] = None,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for AnimateDiff generation")

    pipe = load_animatediff_pipeline(model_id, adapter_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "num_frames": frames,
    }

    supports_image = False
    try:
        sig = inspect.signature(pipe.__call__)
        supports_image = "image" in sig.parameters
        if "num_frames" not in sig.parameters and "video_length" in sig.parameters:
            call_kwargs["video_length"] = frames
            call_kwargs.pop("num_frames", None)
        if "output_type" in sig.parameters:
            call_kwargs["output_type"] = "pt"
    except Exception:
        pass

    if init_image is not None:
        if not supports_image:
            raise RuntimeError("AnimateDiff pipeline does not support init_image")
        prepared_image = init_image
        if Image is not None and isinstance(init_image, Image.Image):
            prepared_image = init_image.convert("RGB").resize(
                (width, height), resample=Image.LANCZOS
            )
        call_kwargs["image"] = prepared_image

    result = pipe(**call_kwargs)
    return result.frames[0]

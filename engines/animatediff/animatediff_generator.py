"""AnimateDiff pipeline helpers."""
from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from diffusers import (  # type: ignore
        AnimateDiffPipeline,
        AnimateDiffVideoToVideoPipeline,
        MotionAdapter,
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        DDIMScheduler,
    )
except Exception:  # pragma: no cover
    AnimateDiffPipeline = None  # type: ignore
    AnimateDiffVideoToVideoPipeline = None  # type: ignore
    MotionAdapter = None  # type: ignore
    StableDiffusionPipeline = None  # type: ignore
    StableDiffusionImg2ImgPipeline = None  # type: ignore
    DDIMScheduler = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

_PIPE_LOCK = threading.Lock()
_PIPE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_LOGGER = logging.getLogger(__name__)
_LAST_PIPELINE_CACHE_HIT = False
_LAST_PIPELINE_LOAD_MS = 0


def _read_animatediff_pipeline_cache_size() -> int:
    raw = os.environ.get("HAI_ANIMATEDIFF_PIPELINE_CACHE_SIZE") or "1"
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return 1


_ANIMATEDIFF_PIPELINE_CACHE_SIZE = _read_animatediff_pipeline_cache_size()


def _set_last_pipeline_stats(cache_hit: bool, load_ms: int) -> None:
    global _LAST_PIPELINE_CACHE_HIT, _LAST_PIPELINE_LOAD_MS
    _LAST_PIPELINE_CACHE_HIT = bool(cache_hit)
    _LAST_PIPELINE_LOAD_MS = max(0, int(load_ms))


def get_last_pipeline_stats() -> dict:
    return {
        "pipeline_cache_hit": bool(_LAST_PIPELINE_CACHE_HIT),
        "pipeline_load_ms": int(_LAST_PIPELINE_LOAD_MS),
    }


def _release_pipeline(pipe: Any) -> None:
    if pipe is None:
        return
    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    except Exception:
        pass
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _animatediff_cache_key(model_id: str, adapter_id: str, mode: str) -> str:
    return f"{model_id}|{adapter_id}|{mode}"


def unload_pipeline() -> None:
    """Release the cached AnimateDiff pipeline and free GPU memory."""
    with _PIPE_LOCK:
        to_release = list(_PIPE_CACHE.values())
        _PIPE_CACHE.clear()
    for pipe in to_release:
        _release_pipeline(pipe)
    if to_release:
        _LOGGER.info("AnimateDiff pipeline unloaded")


def load_animatediff_pipeline(
    model_id: str,
    adapter_id: str,
    use_init_image: bool = False,
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

    mode = "i2v" if use_init_image else "t2v"
    cache_key = _animatediff_cache_key(model_id, adapter_id, mode)

    if _ANIMATEDIFF_PIPELINE_CACHE_SIZE > 0:
        with _PIPE_LOCK:
            cached = _PIPE_CACHE.pop(cache_key, None)
            if cached is not None:
                _PIPE_CACHE[cache_key] = cached
                _set_last_pipeline_stats(True, 0)
                return cached

    build_started = time.time()
    # Temporarily disable HF_HUB_OFFLINE for model loading so downloads
    # can succeed when the model isn't cached yet.
    _offline_was = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        model_path = Path(model_id).expanduser()
        base_kwargs = {
            "torch_dtype": torch.float16,
            "safety_checker": None,
            "requires_safety_checker": False,
            "feature_extractor": None,
        }
        if use_init_image:
            if AnimateDiffVideoToVideoPipeline is None or StableDiffusionImg2ImgPipeline is None:
                raise RuntimeError("AnimateDiff init_image requires AnimateDiffVideoToVideoPipeline")
            if model_path.exists() and model_path.is_file() and hasattr(StableDiffusionImg2ImgPipeline, "from_single_file"):
                base = StableDiffusionImg2ImgPipeline.from_single_file(
                    str(model_path), **base_kwargs
                )
            else:
                base = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id, **base_kwargs
                )
        else:
            if model_path.exists() and model_path.is_file() and hasattr(StableDiffusionPipeline, "from_single_file"):
                base = StableDiffusionPipeline.from_single_file(
                    str(model_path), **base_kwargs
                )
            else:
                base = StableDiffusionPipeline.from_pretrained(
                    model_id, **base_kwargs
                )
        adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch.float16)
        if use_init_image and AnimateDiffVideoToVideoPipeline is not None:
            pipe = AnimateDiffVideoToVideoPipeline.from_pipe(base, motion_adapter=adapter)
        else:
            pipe = AnimateDiffPipeline.from_pipe(base, motion_adapter=adapter)
    finally:
        if _offline_was is not None:
            os.environ["HF_HUB_OFFLINE"] = _offline_was
    if DDIMScheduler is not None:
        try:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass
    offload_enabled = False
    # Enable CPU offload for GPUs with < 14GB VRAM (covers RTX 3060 12GB).
    # AnimateDiff + motion adapter need ~10-12GB; without offload, 12GB cards OOM.
    force_offload = os.environ.get("ANIMATEDIFF_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
    disable_offload = os.environ.get("ANIMATEDIFF_NO_OFFLOAD", "").lower() in ("1", "true", "yes")

    if not disable_offload and (
        force_offload
        or (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3)
    ):
        try:
            pipe.enable_model_cpu_offload()
            offload_enabled = True
            _LOGGER.info("CPU offload enabled (low VRAM or forced)")
        except Exception:
            pass
    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        else:
            pipe.enable_vae_slicing()
    except Exception:
        pass
    if not offload_enabled:
        pipe = pipe.to(device)
    load_ms = int((time.time() - build_started) * 1000)

    if _ANIMATEDIFF_PIPELINE_CACHE_SIZE <= 0:
        _set_last_pipeline_stats(False, load_ms)
        return pipe

    evicted: list[Any] = []
    with _PIPE_LOCK:
        _PIPE_CACHE[cache_key] = pipe
        while len(_PIPE_CACHE) > _ANIMATEDIFF_PIPELINE_CACHE_SIZE:
            _, old_pipe = _PIPE_CACHE.popitem(last=False)
            evicted.append(old_pipe)
    for old_pipe in evicted:
        _release_pipeline(old_pipe)
    _set_last_pipeline_stats(False, load_ms)
    return pipe


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
    strength: Optional[float] = None,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for AnimateDiff generation")

    transient_pipeline = _ANIMATEDIFF_PIPELINE_CACHE_SIZE <= 0
    pipe = load_animatediff_pipeline(model_id, adapter_id, use_init_image=init_image is not None)
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
    if init_image is not None:
        call_kwargs["strength"] = 0.55 if strength is None else strength

    supports_image = False
    image_param = None
    sig_params = None
    try:
        sig = inspect.signature(pipe.__call__)
        sig_params = set(sig.parameters.keys())
        if "image" in sig.parameters:
            supports_image = True
            image_param = "image"
        elif "init_image" in sig.parameters:
            supports_image = True
            image_param = "init_image"
        elif "video" in sig.parameters:
            supports_image = True
            image_param = "video"
        if "num_frames" not in sig.parameters and "video_length" in sig.parameters:
            call_kwargs["video_length"] = frames
            call_kwargs.pop("num_frames", None)
        if "output_type" in sig.parameters:
            call_kwargs["output_type"] = "pt"
    except Exception:
        pass
    if sig_params is not None:
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig_params}

    if init_image is not None:
        if not supports_image:
            raise RuntimeError("AnimateDiff pipeline does not support init_image")
        prepared_image = init_image
        if Image is not None and isinstance(init_image, Image.Image):
            prepared_image = init_image.convert("RGB").resize(
                (width, height), resample=Image.LANCZOS
            )
        if image_param == "video":
            call_kwargs[image_param] = [prepared_image] * max(1, frames)
        elif image_param:
            call_kwargs[image_param] = prepared_image
        else:
            _LOGGER.warning("AnimateDiff init_image provided but no supported image param found.")

    try:
        result = pipe(**call_kwargs)
        return result.frames[0]
    finally:
        if transient_pipeline:
            _release_pipeline(pipe)

"""LTX-Video 2.3 pipeline loading and frame generation.

Wraps the real Lightricks LTXPipeline / LTXImageToVideoPipeline from
``diffusers`` (>= 0.32.0).  The old Latte-1 pipeline in ``engines/ltx2``
is left intact for backward compatibility.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# Import the real LTX-Video pipeline classes from diffusers.
# These ship with diffusers >= 0.32.0 and are distinct from LattePipeline.
_LTXPipeline: Any = None
_LTXImageToVideoPipeline: Any = None
try:
    from diffusers import LTXPipeline as _LTXPipeline  # type: ignore
except ImportError:
    pass
try:
    from diffusers import LTXImageToVideoPipeline as _LTXImageToVideoPipeline  # type: ignore
except ImportError:
    pass

from .config import LTXVideoConfig, load_config

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline cache — keyed by (checkpoint_path, pipeline_class_name)
# ---------------------------------------------------------------------------
_PIPE_LOCK = threading.Lock()
_PIPES: Dict[str, Any] = {}  # cache_key → pipeline instance


def _cache_key(ckpt_path: str, cls_name: str) -> str:
    return f"{ckpt_path}::{cls_name}"


def unload_pipelines() -> None:
    """Release all cached LTX-Video pipelines and free GPU memory."""
    with _PIPE_LOCK:
        _PIPES.clear()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _LOGGER.info("All LTX-Video pipelines unloaded")


def _resolve_dtype(dtype_str: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for LTX-Video")
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def _load_pipeline(
    pipeline_cls: Any,
    checkpoint_path: Path,
    text_encoder_id: Optional[str],
    dtype_str: str = "bfloat16",
    device: str = "cuda",
    cpu_offload_threshold_gb: float = 16.0,
    vae_slicing: bool = True,
) -> Any:
    """Load (or retrieve cached) an LTX-Video pipeline instance."""
    if pipeline_cls is None:
        raise RuntimeError(
            "LTX-Video pipeline class not available. "
            "Install diffusers >= 0.32.0: pip install -U diffusers"
        )
    if torch is None:
        raise RuntimeError("torch is required for LTX-Video generation")

    key = _cache_key(str(checkpoint_path), pipeline_cls.__name__)
    with _PIPE_LOCK:
        if key in _PIPES:
            _LOGGER.debug("LTX-Video pipeline cache HIT: %s", key)
            return _PIPES[key]
        _LOGGER.info("LTX-Video pipeline cache MISS — loading fresh: %s", key)

        dtype = _resolve_dtype(dtype_str)
        _LOGGER.info("Loading LTX-Video pipeline: %s from %s", pipeline_cls.__name__, checkpoint_path)

        # Allow HF downloads during load
        offline_was = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}

            # Supply text encoder if the pipeline init accepts it
            if text_encoder_id:
                try:
                    from transformers import AutoTokenizer, AutoModel  # type: ignore
                    sig = inspect.signature(pipeline_cls.from_pretrained)
                    if "text_encoder" in sig.parameters:
                        load_kwargs["text_encoder"] = AutoModel.from_pretrained(
                            text_encoder_id, torch_dtype=dtype
                        )
                    if "tokenizer" in sig.parameters:
                        load_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(text_encoder_id)
                except Exception as te_exc:
                    _LOGGER.warning("Could not pre-load Gemma text encoder: %s", te_exc)

            if checkpoint_path.is_file() and hasattr(pipeline_cls, "from_single_file"):
                pipe = pipeline_cls.from_single_file(str(checkpoint_path), **load_kwargs)
            else:
                pipe = pipeline_cls.from_pretrained(str(checkpoint_path), **load_kwargs)

            pipe = pipe.to(device)
        finally:
            if offline_was is not None:
                os.environ["HF_HUB_OFFLINE"] = offline_was

        # Memory management
        force_offload = os.environ.get("LTX_VIDEO_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
        disable_offload = os.environ.get("LTX_VIDEO_NO_OFFLOAD", "").lower() in ("1", "true", "yes")
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        _LOGGER.info("LTX-Video pipeline loaded on %s (VRAM=%.1fGB)", device, vram_gb)

        if not disable_offload and (force_offload or vram_gb < cpu_offload_threshold_gb):
            try:
                pipe.enable_model_cpu_offload()
                _LOGGER.info("LTX-Video CPU offload enabled (VRAM=%.1fGB, threshold=%.1fGB)", vram_gb, cpu_offload_threshold_gb)
            except Exception as exc:
                _LOGGER.warning("LTX-Video CPU offload failed (non-fatal): %s", exc)
        else:
            _LOGGER.info("LTX-Video running full GPU mode (VRAM=%.1fGB)", vram_gb)

        if vae_slicing:
            try:
                pipe.enable_vae_slicing()
                _LOGGER.debug("VAE slicing enabled")
            except Exception as exc:
                _LOGGER.warning("VAE slicing failed (non-fatal): %s", exc)

        _PIPES[key] = pipe
        _LOGGER.info("LTX-Video pipeline ready: %s", pipeline_cls.__name__)
        return pipe


# ---------------------------------------------------------------------------
# Public generation API
# ---------------------------------------------------------------------------

def generate_frames(
    config: LTXVideoConfig,
    *,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 50,
    guidance: float = 3.0,
    width: int = 768,
    height: int = 512,
    frames: int = 97,
    seed: int = 0,
    checkpoint_variant: str = "dev",
    init_image: Optional[Any] = None,
    callback_on_step_end: Optional[Callable] = None,
    device: str = "cuda",
) -> Any:
    """Run a single-pass LTX-Video generation and return raw frames.

    This handles both text-to-video and image-to-video depending on
    whether *init_image* is provided.
    """
    if torch is None:
        raise RuntimeError("torch is required for LTX-Video")

    ckpt_path = config.checkpoint_path(checkpoint_variant)
    text_encoder_id = config.asset_repo_id("text_encoder")
    defaults = config.defaults

    # Select pipeline class based on input type
    if init_image is not None and _LTXImageToVideoPipeline is not None:
        pipeline_cls = _LTXImageToVideoPipeline
    else:
        pipeline_cls = _LTXPipeline

    pipe = _load_pipeline(
        pipeline_cls,
        ckpt_path,
        text_encoder_id=text_encoder_id,
        dtype_str=defaults.get("dtype", "bfloat16"),
        device=device,
        cpu_offload_threshold_gb=defaults.get("cpu_offload_threshold_gb", 16.0),
        vae_slicing=defaults.get("vae_slicing", True),
    )

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Build call kwargs — introspect the pipeline's __call__ signature to
    # handle API differences across diffusers versions.
    call_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
    }

    try:
        sig = inspect.signature(pipe.__call__)
        param_names = set(sig.parameters.keys())

        # Frame count parameter naming varies
        if "num_frames" in param_names:
            call_kwargs["num_frames"] = frames
        elif "video_length" in param_names:
            call_kwargs["video_length"] = frames

        if "output_type" in param_names:
            call_kwargs["output_type"] = "pt"

        # Image input for i2v
        if init_image is not None and "image" in param_names:
            prepared = init_image
            if Image is not None and isinstance(init_image, Image.Image):
                prepared = init_image.convert("RGB").resize((width, height), resample=Image.LANCZOS)
            call_kwargs["image"] = prepared

        # Step callback
        if callback_on_step_end is not None:
            if "callback_on_step_end" in param_names:
                call_kwargs["callback_on_step_end"] = callback_on_step_end
            elif "callback" in param_names:
                def _compat_cb(step: int, timestep: Any, latents: Any) -> None:
                    callback_on_step_end(pipe, step, timestep, {})
                call_kwargs["callback"] = _compat_cb

    except Exception:
        # Best-effort fallback
        call_kwargs["num_frames"] = frames

    result = pipe(**call_kwargs)

    # Normalize output — diffusers may return .frames, .videos, or direct tensor
    if hasattr(result, "frames"):
        out = result.frames
        if isinstance(out, (list, tuple)) and len(out) > 0:
            return out[0]
        return out
    if hasattr(result, "videos"):
        return result.videos
    return result

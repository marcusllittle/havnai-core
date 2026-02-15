"""Latte-1 (LTX2) pipeline helpers."""
from __future__ import annotations

import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from diffusers import LattePipeline, AutoencoderKLTemporalDecoder  # type: ignore
except Exception:  # pragma: no cover
    LattePipeline = None  # type: ignore
    AutoencoderKLTemporalDecoder = None  # type: ignore

_PIPE_LOCK = threading.Lock()
_PIPE: Optional[Any] = None
_PIPE_ID: Optional[str] = None
_LOGGER = logging.getLogger(__name__)


def unload_pipeline() -> None:
    """Release the cached LTX2 pipeline and free GPU memory."""
    global _PIPE, _PIPE_ID
    with _PIPE_LOCK:
        if _PIPE is not None:
            del _PIPE
            _PIPE = None
            _PIPE_ID = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            _LOGGER.info("LTX2 pipeline unloaded")


def load_latte_pipeline(model_id: str, device: str = "cuda") -> Any:
    """Load and cache LattePipeline + temporal VAE."""
    if LattePipeline is None or AutoencoderKLTemporalDecoder is None or torch is None:
        raise RuntimeError("LattePipeline dependencies are unavailable")

    global _PIPE, _PIPE_ID
    with _PIPE_LOCK:
        if _PIPE is None or _PIPE_ID != model_id:
            # Temporarily disable HF_HUB_OFFLINE for model loading so downloads
            # can succeed when the model isn't cached yet.
            _offline_was = os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                model_path = Path(model_id).expanduser()
                if model_path.exists() and model_path.is_file() and hasattr(LattePipeline, "from_single_file"):
                    pipe = LattePipeline.from_single_file(str(model_path), torch_dtype=torch.float16).to(device)
                else:
                    pipe = LattePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            finally:
                if _offline_was is not None:
                    os.environ["HF_HUB_OFFLINE"] = _offline_was
            vae = AutoencoderKLTemporalDecoder.from_pretrained(
                model_id,
                subfolder="vae_temporal_decoder",
                torch_dtype=torch.float16,
            ).to(device)
            pipe.vae = vae
            # Only enable CPU offload if VRAM is low (< 10GB) or forced via env var
            force_offload = os.environ.get("LTX2_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
            disable_offload = os.environ.get("LTX2_NO_OFFLOAD", "").lower() in ("1", "true", "yes")

            if not disable_offload and (force_offload or (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3)):
                try:
                    pipe.enable_model_cpu_offload()
                    _LOGGER.info("CPU offload enabled (low VRAM or forced)")
                    print("🐌 LTX2 CPU offload ENABLED (this will be slow)")
                except Exception:
                    pass
            else:
                print("⚡ LTX2 CPU offload DISABLED (full GPU mode)")
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            _PIPE = pipe
            _PIPE_ID = model_id
        return _PIPE


def generate_video_frames(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    frames: int,
    seed: int,
    model_id: str,
    init_image: Optional[Any] = None,
    callback_on_step_end: Optional[Any] = None,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for LTX2 generation")

    pipe = load_latte_pipeline(model_id)
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
    }
    # Diffusers LattePipeline changed "num_frames" -> "video_length" in newer releases.
    supports_image = True
    try:
        sig = inspect.signature(pipe.__call__)
        if "num_frames" in sig.parameters:
            call_kwargs["num_frames"] = frames
        elif "video_length" in sig.parameters:
            call_kwargs["video_length"] = frames
        if "output_type" in sig.parameters:
            call_kwargs["output_type"] = "pt"
        supports_image = "image" in sig.parameters
    except Exception:
        # Best-effort fallback if signature introspection fails.
        call_kwargs["num_frames"] = frames

    if init_image is not None:
        if not supports_image:
            raise RuntimeError("LTX2 pipeline does not support init_image")
        prepared_image = init_image
        if Image is not None and isinstance(init_image, Image.Image):
            prepared_image = init_image.convert("RGB").resize((width, height), resample=Image.LANCZOS)
        call_kwargs["image"] = prepared_image

    # Attach step-progress callback if the pipeline supports it.
    if callback_on_step_end is not None:
        try:
            sig = inspect.signature(pipe.__call__)
            if "callback_on_step_end" in sig.parameters:
                call_kwargs["callback_on_step_end"] = callback_on_step_end
            elif "callback" in sig.parameters:
                # Older diffusers API uses positional (step, timestep, latents)
                def _compat_cb(step: int, timestep: Any, latents: Any) -> None:
                    callback_on_step_end(pipe, step, timestep, {})
                call_kwargs["callback"] = _compat_cb
        except Exception:
            pass  # Skip callback if signature introspection fails

    result = pipe(**call_kwargs)
    return result.frames[0]

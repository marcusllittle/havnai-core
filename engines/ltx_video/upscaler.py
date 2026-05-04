"""LTX-Video 2.3 upscaler stages (spatial and temporal).

These are applied as post-processing after the base generation pass
in two-stage and two-stage-hq pipeline modes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from .config import LTXVideoConfig

_LOGGER = logging.getLogger(__name__)


def _load_upscaler_model(checkpoint_path: Path, device: str = "cuda") -> Any:
    """Load an upscaler checkpoint.

    LTX-Video upscaler checkpoints are standalone safetensors models.
    The exact loading mechanism depends on how Lightricks packages them.
    This implementation supports both a direct state_dict approach and
    a diffusers-compatible from_single_file path.
    """
    if torch is None:
        raise RuntimeError("torch is required for upscaling")

    # Try diffusers pipeline loader first (if Lightricks ships a pipeline wrapper)
    try:
        from diffusers import DiffusionPipeline  # type: ignore
        if hasattr(DiffusionPipeline, "from_single_file"):
            model = DiffusionPipeline.from_single_file(
                str(checkpoint_path), torch_dtype=torch.bfloat16
            )
            return model.to(device)
    except Exception:
        pass

    # Fallback: load raw state dict for manual model construction.
    # This is a placeholder for when the upscaler model class is known.
    try:
        from safetensors.torch import load_file  # type: ignore
        state_dict = load_file(str(checkpoint_path))
        _LOGGER.info("Loaded upscaler state_dict from %s (%d keys)", checkpoint_path, len(state_dict))
        return state_dict
    except Exception as exc:
        raise RuntimeError(f"Failed to load upscaler from {checkpoint_path}: {exc}") from exc


def apply_spatial_upscale(
    frames: Any,
    config: LTXVideoConfig,
    scale: str = "x2",
    device: str = "cuda",
) -> Any:
    """Apply spatial upscaling to generated frames.

    Args:
        frames: Video frames as tensor (F,C,H,W) or (F,H,W,C) or numpy array.
        config: LTX-Video family config for asset resolution.
        scale: ``"x2"`` or ``"x1_5"`` — selects the upscaler asset.
        device: Target device.

    Returns:
        Upscaled frames in the same format as input.
    """
    asset_key = f"spatial_upscaler_{scale}"
    ckpt_path = config.asset_path(asset_key)
    if ckpt_path is None or not ckpt_path.exists():
        _LOGGER.warning("Spatial upscaler %s not found at %s — skipping", scale, ckpt_path)
        return frames

    _LOGGER.info("Applying spatial upscale (%s) from %s", scale, ckpt_path)
    model_or_sd = _load_upscaler_model(ckpt_path, device=device)

    # If the loader returned a pipeline with a __call__, use it directly
    if callable(getattr(model_or_sd, "__call__", None)):
        result = model_or_sd(frames)
        if hasattr(result, "frames"):
            return result.frames[0] if isinstance(result.frames, (list, tuple)) else result.frames
        return result

    # Otherwise, this is a state_dict — apply a basic interpolation fallback.
    # When the official upscaler model class is available, replace this block.
    _LOGGER.warning(
        "Spatial upscaler loaded as raw state_dict; "
        "using torch interpolation fallback until model class is integrated"
    )
    return _torch_interpolate_fallback(frames, scale, device)


def apply_temporal_upscale(
    frames: Any,
    config: LTXVideoConfig,
    device: str = "cuda",
) -> Any:
    """Apply temporal upscaling (frame interpolation) to generated frames.

    Doubles the frame count by interpolating between consecutive frames.
    """
    asset_key = "temporal_upscaler_x2"
    ckpt_path = config.asset_path(asset_key)
    if ckpt_path is None or not ckpt_path.exists():
        _LOGGER.warning("Temporal upscaler not found at %s — skipping", ckpt_path)
        return frames

    _LOGGER.info("Applying temporal upscale (x2) from %s", ckpt_path)
    model_or_sd = _load_upscaler_model(ckpt_path, device=device)

    if callable(getattr(model_or_sd, "__call__", None)):
        result = model_or_sd(frames)
        if hasattr(result, "frames"):
            return result.frames[0] if isinstance(result.frames, (list, tuple)) else result.frames
        return result

    _LOGGER.warning(
        "Temporal upscaler loaded as raw state_dict; "
        "using linear interpolation fallback until model class is integrated"
    )
    return _temporal_interpolate_fallback(frames, device)


# ---------------------------------------------------------------------------
# Fallbacks — used when the real upscaler model class is not yet integrated.
# These produce correct-shaped output so the rest of the pipeline works.
# ---------------------------------------------------------------------------

def _torch_interpolate_fallback(frames: Any, scale: str, device: str) -> Any:
    """Bilinear upscale as a placeholder for the real spatial upscaler."""
    if torch is None:
        return frames

    scale_factor = 2.0 if scale == "x2" else 1.5
    tensor = frames
    if not isinstance(tensor, torch.Tensor):
        if np is not None:
            tensor = torch.from_numpy(np.array(tensor))
        else:
            return frames

    original_device = tensor.device
    tensor = tensor.to(device)

    # Expect (F, C, H, W) or (F, H, W, C)
    needs_permute = tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4)
    if needs_permute:
        tensor = tensor.permute(0, 3, 1, 2)  # → (F, C, H, W)

    tensor = tensor.float()
    upscaled = torch.nn.functional.interpolate(
        tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )

    if needs_permute:
        upscaled = upscaled.permute(0, 2, 3, 1)

    return upscaled.to(original_device)


def _temporal_interpolate_fallback(frames: Any, device: str) -> Any:
    """Simple frame-averaging interpolation as a placeholder."""
    if torch is None or np is None:
        return frames

    tensor = frames
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(np.array(tensor)).to(device)

    if tensor.shape[0] < 2:
        return tensor

    interpolated = []
    for i in range(tensor.shape[0] - 1):
        interpolated.append(tensor[i])
        mid = (tensor[i].float() + tensor[i + 1].float()) / 2.0
        interpolated.append(mid.to(tensor.dtype))
    interpolated.append(tensor[-1])

    return torch.stack(interpolated, dim=0)

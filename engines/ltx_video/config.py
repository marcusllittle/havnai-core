"""LTX-Video 2.3 configuration loader and asset resolver.

Loads ``ltx_video_config.json`` and provides helpers to:
  - resolve checkpoint / asset paths on disk
  - validate that a requested pipeline mode has its prerequisites
  - report which capabilities a node can actually serve
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = os.environ.get(
    "HAVNAI_LTX_VIDEO_CONFIG",
    str(Path(__file__).resolve().parent.parent.parent / "server" / "manifests" / "ltx_video_config.json"),
)
_DEFAULT_BASE_DIR = os.environ.get(
    "HAVNAI_LTX_VIDEO_DIR",
    "/mnt/d/havnai-storage/models/ltx_video",
)


class LTXVideoAssetError(Exception):
    """Raised when a required asset cannot be located."""


# ---------------------------------------------------------------------------
# Family config container
# ---------------------------------------------------------------------------

class LTXVideoConfig:
    """Parsed and validated LTX-Video family configuration."""

    def __init__(self, raw: Dict[str, Any], base_dir_override: Optional[str] = None) -> None:
        self.raw = raw
        self.version: str = raw.get("version", "2.3")
        self.base_dir = Path(base_dir_override or raw.get("base_dir", _DEFAULT_BASE_DIR))
        self.checkpoints: Dict[str, Dict[str, Any]] = raw.get("checkpoints", {})
        self.assets: Dict[str, Dict[str, Any]] = raw.get("assets", {})
        self.pipeline_modes: Dict[str, Dict[str, Any]] = raw.get("pipeline_modes", {})
        self.capabilities: List[str] = raw.get("capabilities", [])
        self.defaults: Dict[str, Any] = raw.get("defaults", {})

    # -- path resolution ---------------------------------------------------

    def checkpoint_path(self, variant: str) -> Path:
        """Return absolute path to a checkpoint variant's file."""
        entry = self.checkpoints.get(variant)
        if entry is None:
            _LOGGER.error("Unknown checkpoint variant: %r (available: %s)", variant, list(self.checkpoints.keys()))
            raise LTXVideoAssetError(f"Unknown checkpoint variant: {variant!r}")
        filename = entry["filename"]
        # Allow env-var override per-variant: HAVNAI_LTX_VIDEO_CKPT_DEV=/some/path
        env_key = f"HAVNAI_LTX_VIDEO_CKPT_{variant.upper()}"
        override = os.environ.get(env_key, "").strip()
        if override:
            _LOGGER.info("Checkpoint %r overridden via %s → %s", variant, env_key, override)
            return Path(override)
        resolved = self.base_dir / filename
        _LOGGER.debug("Checkpoint %r resolved to %s (exists=%s)", variant, resolved, resolved.exists())
        return resolved

    def asset_path(self, asset_key: str) -> Optional[Path]:
        """Return absolute path to an auxiliary asset, or *None* if HuggingFace-hosted."""
        entry = self.assets.get(asset_key)
        if entry is None:
            raise LTXVideoAssetError(f"Unknown asset key: {asset_key!r}")
        if entry.get("type") == "huggingface":
            return None  # handled via HF hub, not local file
        filename = entry["filename"]
        env_key = f"HAVNAI_LTX_VIDEO_ASSET_{asset_key.upper()}"
        override = os.environ.get(env_key, "").strip()
        if override:
            return Path(override)
        return self.base_dir / filename

    def asset_repo_id(self, asset_key: str) -> Optional[str]:
        """Return HuggingFace repo ID for a hub-hosted asset, or *None*."""
        entry = self.assets.get(asset_key)
        if entry and entry.get("type") == "huggingface":
            return entry.get("repo_id")
        return None

    # -- pipeline mode helpers ---------------------------------------------

    def mode_config(self, mode: str) -> Dict[str, Any]:
        """Return the pipeline mode definition, or raise."""
        cfg = self.pipeline_modes.get(mode)
        if cfg is None:
            raise LTXVideoAssetError(
                f"Unknown pipeline mode: {mode!r}. "
                f"Available: {', '.join(sorted(self.pipeline_modes))}"
            )
        return cfg

    def validate_mode_assets(self, mode: str) -> List[str]:
        """Check that all required assets for *mode* exist on disk.

        Returns a list of missing asset descriptions (empty = all good).
        """
        mode_cfg = self.mode_config(mode)
        missing: List[str] = []

        for variant in mode_cfg.get("requires_checkpoint", []):
            p = self.checkpoint_path(variant)
            if not p.exists():
                missing.append(f"checkpoint:{variant} ({p})")
                _LOGGER.warning("Mode %r: checkpoint %r NOT FOUND at %s", mode, variant, p)
            else:
                _LOGGER.debug("Mode %r: checkpoint %r OK at %s", mode, variant, p)

        for asset_key in mode_cfg.get("requires_assets", []):
            p = self.asset_path(asset_key)
            if p is not None and not p.exists():
                missing.append(f"asset:{asset_key} ({p})")
                _LOGGER.warning("Mode %r: asset %r NOT FOUND at %s", mode, asset_key, p)
            # HF-hosted assets validated at load time, not here

        if missing:
            _LOGGER.warning("Mode %r has %d missing assets: %s", mode, len(missing), missing)
        else:
            _LOGGER.info("Mode %r: all required assets present", mode)
        return missing

    def available_modes(self) -> List[str]:
        """Return pipeline modes whose required assets are all present."""
        available = []
        for mode in self.pipeline_modes:
            if not self.validate_mode_assets(mode):
                available.append(mode)
        return available

    def mode_supports(self, mode: str, capability: str) -> bool:
        """Check if a pipeline mode supports a given capability flag."""
        mode_cfg = self.mode_config(mode)
        flag_map = {
            "image_to_video": mode_cfg.get("supports_image_input", False),
            "audio_to_video": mode_cfg.get("supports_audio_input", False),
            "keyframe_interpolation": mode_cfg.get("supports_keyframes", False),
            "retake": mode_cfg.get("supports_retake", False),
        }
        return flag_map.get(capability, False)

    def mode_defaults(self, mode: str) -> Dict[str, Any]:
        """Return default generation parameters for a pipeline mode."""
        mode_cfg = self.mode_config(mode)
        return {
            "steps": mode_cfg.get("default_steps", 50),
            "guidance": mode_cfg.get("default_guidance", 3.0),
            "width": mode_cfg.get("default_width", 768),
            "height": mode_cfg.get("default_height", 512),
            "frames": mode_cfg.get("default_frames", 97),
            "fps": mode_cfg.get("default_fps", 24),
        }

    def mode_limits(self, mode: str) -> Dict[str, int]:
        """Return max parameter bounds for a pipeline mode."""
        mode_cfg = self.mode_config(mode)
        return {
            "max_frames": mode_cfg.get("max_frames", 257),
            "max_width": mode_cfg.get("max_width", 1280),
            "max_height": mode_cfg.get("max_height", 1280),
        }

    def advertised_capabilities(self) -> Set[str]:
        """Return the set of capabilities a node can advertise based on available assets."""
        caps: Set[str] = set()
        available = self.available_modes()
        if not available:
            return caps
        caps.add("text_to_video")
        for mode in available:
            mode_cfg = self.pipeline_modes[mode]
            if mode_cfg.get("supports_image_input"):
                caps.add("image_to_video")
            if mode_cfg.get("supports_audio_input"):
                caps.add("audio_to_video")
            if mode_cfg.get("supports_keyframes"):
                caps.add("keyframe_interpolation")
            if mode_cfg.get("supports_retake"):
                caps.add("retake")
            # Upscaler capabilities
            for asset_key in mode_cfg.get("optional_assets", []) + mode_cfg.get("requires_assets", []):
                if "spatial_upscaler" in asset_key:
                    p = self.asset_path(asset_key)
                    if p is None or p.exists():
                        caps.add("spatial_upscale")
                if "temporal_upscaler" in asset_key:
                    p = self.asset_path(asset_key)
                    if p is None or p.exists():
                        caps.add("temporal_upscale")
            if "distilled" in str(mode_cfg.get("requires_checkpoint", [])):
                caps.add("distilled_inference")
        return caps


# ---------------------------------------------------------------------------
# Module-level loader
# ---------------------------------------------------------------------------

_CACHED_CONFIG: Optional[LTXVideoConfig] = None


def load_config(
    config_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    force_reload: bool = False,
) -> LTXVideoConfig:
    """Load and cache the LTX-Video family config."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload:
        return _CACHED_CONFIG

    path = Path(config_path or _DEFAULT_CONFIG_PATH)
    if not path.exists():
        _LOGGER.warning("LTX-Video config not found at %s — using empty defaults", path)
        _CACHED_CONFIG = LTXVideoConfig({}, base_dir_override=base_dir)
        return _CACHED_CONFIG

    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        _LOGGER.error("Failed to parse LTX-Video config at %s: %s", path, exc)
        _CACHED_CONFIG = LTXVideoConfig({}, base_dir_override=base_dir)
        return _CACHED_CONFIG

    _CACHED_CONFIG = LTXVideoConfig(raw, base_dir_override=base_dir)
    _LOGGER.info(
        "Loaded LTX-Video %s config: %d checkpoints, %d assets, %d pipeline modes",
        _CACHED_CONFIG.version,
        len(_CACHED_CONFIG.checkpoints),
        len(_CACHED_CONFIG.assets),
        len(_CACHED_CONFIG.pipeline_modes),
    )
    return _CACHED_CONFIG

"""Wan 2.2 I2V configuration, asset paths, and per-job overrides.

The JSON file is the source of the recommended 12 GB defaults. Environment
variables override the file. Values on the job override both. Nothing here
imports torch or diffusers, so the coordinator can read preset names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_LOGGER_NAME = "engines.wan22_i2v.config"

PIPELINE_NAMES = {"wan22-i2v", "wan22_i2v"}
FAMILY_NAMES = {"wan22_i2v", "wan22-i2v"}
MODEL_NAME = "wan22-i2v-a14b"
DEFAULT_ROOT = "/mnt/d/models/wan22"

# Job key, then accepted aliases. The first alias is the canonical name.
_JOB_KEYS: Dict[str, Sequence[str]] = {
    "steps": ("steps",),
    "guidance": ("guidance", "guidance_scale"),
    "guidance_2": ("guidance_2", "guidance_scale_2"),
    "width": ("width",),
    "height": ("height",),
    "frames": ("frames", "num_frames"),
    "fps": ("fps",),
    "flow_shift": ("flow_shift", "shift"),
    "boundary_ratio": ("boundary_ratio",),
    "high_noise_steps": ("high_noise_steps",),
    "lora_scale_high": ("lora_scale_high", "lora_strength_high"),
    "lora_scale_low": ("lora_scale_low", "lora_strength_low"),
    "offload": ("offload",),
    "scheduler": ("scheduler",),
}

_ENV_KEYS = {
    "steps": "HAI_WAN22_STEPS",
    "guidance": "HAI_WAN22_GUIDANCE",
    "guidance_2": "HAI_WAN22_GUIDANCE_2",
    "width": "HAI_WAN22_WIDTH",
    "height": "HAI_WAN22_HEIGHT",
    "frames": "HAI_WAN22_FRAMES",
    "fps": "HAI_WAN22_FPS",
    "flow_shift": "HAI_WAN22_SHIFT",
    "boundary_ratio": "HAI_WAN22_BOUNDARY",
    "high_noise_steps": "HAI_WAN22_HIGH_NOISE_STEPS",
    "lora_scale_high": "HAI_WAN22_LORA_HIGH",
    "lora_scale_low": "HAI_WAN22_LORA_LOW",
    "offload": "HAI_WAN22_OFFLOAD",
    "scheduler": "HAI_WAN22_SCHEDULER",
}

_INT_KEYS = {"steps", "width", "height", "frames", "fps", "high_noise_steps"}
_FLOAT_KEYS = {
    "guidance",
    "guidance_2",
    "flow_shift",
    "boundary_ratio",
    "lora_scale_high",
    "lora_scale_low",
}
_REQUIRED_FILES = (
    "transformer_high",
    "transformer_low",
    "vae",
    "lora_high",
    "lora_low",
)

_CACHED: Optional["Wan22I2VConfig"] = None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """The worker advertises this model only when the flag is on."""
    return _env_on("HAI_WAN22_I2V")


def is_wan22_pipeline(pipeline: str) -> bool:
    return str(pipeline or "").strip().lower() in PIPELINE_NAMES


def is_wan22_entry(entry: Any) -> bool:
    if isinstance(entry, Mapping):
        pipeline = entry.get("pipeline")
        family = entry.get("model_family")
    else:
        pipeline = getattr(entry, "pipeline", "")
        family = getattr(entry, "model_family", "")
    return is_wan22_pipeline(str(pipeline or "")) or str(family or "").strip().lower() in FAMILY_NAMES


def default_config_path() -> Path:
    """Prefer the bundled client config, then a copy next to this module."""
    override = os.environ.get("HAI_WAN22_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    bundled = repo_root / "client" / "configs" / "wan22_i2v.json"
    if bundled.is_file():
        return bundled
    return Path(__file__).resolve().parent / "wan22_i2v.json"


class Wan22I2VConfig:
    """Parsed Wan 2.2 I2V config. Paths are resolved against the weights root."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = dict(raw)
        self.family = str(raw.get("family") or "wan22_i2v")
        self.version = str(raw.get("version") or "2.2-i2v-a14b-q4")
        self.license = str(raw.get("license") or "apache-2.0")
        self.diffusers_repo = str(raw.get("diffusers_repo") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        self.files = dict(raw.get("files") or {})
        self.defaults = dict(raw.get("defaults") or {})
        self.negative_prompt = str(raw.get("negative_prompt") or "")
        presets = raw.get("motion_presets") or {}
        self.motion_presets = dict(presets) if isinstance(presets, Mapping) else {}
        fallback = raw.get("text_encoder_fallback") or {}
        self.text_encoder_fallback = dict(fallback) if isinstance(fallback, Mapping) else {}

    def weights_root(self) -> Path:
        override = os.environ.get("HAI_WAN22_ROOT", "").strip()
        if override:
            return Path(override).expanduser()
        return Path(str(self.raw.get("weights_root") or DEFAULT_ROOT)).expanduser()

    def asset_path(self, key: str) -> Path:
        relative = str(self.files.get(key) or "").strip()
        if not relative:
            raise KeyError(f"Wan 2.2 config has no file for {key!r}")
        return self.weights_root() / relative

    def missing_assets(self, *, require_local_text_encoder: bool = False) -> List[str]:
        """Relative paths that are not on disk.

        The fp8 text encoder is optional. ``auto`` mode falls back to the
        diffusers umT5 when the Comfy file cannot be loaded, so a missing
        local encoder does not block admission unless the operator forced it.
        """
        missing: List[str] = []
        keys = list(_REQUIRED_FILES)
        mode = os.environ.get("HAI_WAN22_TEXT_ENCODER", "auto").strip().lower() or "auto"
        if require_local_text_encoder or mode == "local":
            keys.append("text_encoder")
        for key in keys:
            path = self.asset_path(key)
            if not path.is_file():
                missing.append(str(self.files.get(key) or key))
        return missing

    def preset(self, name: str) -> Optional[Dict[str, str]]:
        key = str(name or "").strip()
        if not key:
            return None
        entry = self.motion_presets.get(key)
        if not isinstance(entry, Mapping):
            return None
        return {
            "id": key,
            "label": str(entry.get("label") or key),
            "prompt": str(entry.get("prompt") or "").strip(),
        }

    def preset_names(self) -> List[str]:
        return sorted(self.motion_presets)


def load_config(config_path: Optional[str] = None, *, force: bool = False) -> Wan22I2VConfig:
    global _CACHED
    if _CACHED is not None and not force and config_path is None:
        return _CACHED
    path = Path(config_path).expanduser() if config_path else default_config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Wan 2.2 config must be a JSON object: {path}")
    loaded = Wan22I2VConfig(raw)
    if config_path is None:
        _CACHED = loaded
    return loaded


def reset_config_cache() -> None:
    global _CACHED
    _CACHED = None


def boundary_ratio_for_split(
    timesteps: Sequence[float],
    high_noise_steps: int,
    num_train_timesteps: float = 1000.0,
) -> float:
    """Place the expert switch between step ``high_noise_steps`` and the next.

    Diffusers compares ``timestep >= boundary_ratio * num_train_timesteps``.
    The midpoint between the last high-noise timestep and the first low-noise
    timestep is the 2/2 split LightX2V uses for a 4-step list of
    ``[1000, 750, 500, 250]``: ratio 0.625, threshold 625.
    """
    values = [float(item) for item in timesteps]
    if not values or num_train_timesteps <= 0:
        return 0.0
    steps = int(high_noise_steps)
    if steps <= 0:
        return 0.0
    if steps >= len(values):
        return 1.0
    midpoint = (values[steps - 1] + values[steps]) / 2.0
    return midpoint / float(num_train_timesteps)


def sigmas_from_timesteps(timesteps: Sequence[float]) -> List[float]:
    """Flow-match sigmas. The trailing 0 is the fully denoised end."""
    sigmas = [float(item) / 1000.0 for item in timesteps]
    if not sigmas or sigmas[-1] != 0.0:
        sigmas.append(0.0)
    return sigmas


def _job_value(job: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in job and job.get(name) not in (None, ""):
            return job.get(name)
    return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != int(number):
        return None
    return int(number)


def _parse_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _layered(job: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> Any:
    """Job, then env, then the config file."""
    raw = _job_value(job, _JOB_KEYS.get(key, (key,)))
    source = "job" if raw not in (None, "") else ""
    if raw in (None, ""):
        env_name = _ENV_KEYS.get(key)
        if env_name:
            env_raw = os.environ.get(env_name, "").strip()
            if env_raw:
                raw = env_raw
                source = "env"
    if raw in (None, ""):
        raw = defaults.get(key)
        source = "config"
    if key in _INT_KEYS:
        parsed = _parse_int(raw)
        if parsed is None:
            parsed = _parse_int(defaults.get(key))
            source = "config"
        return parsed, source
    if key in _FLOAT_KEYS:
        parsed_f = _parse_float(raw)
        if parsed_f is None:
            parsed_f = _parse_float(defaults.get(key))
            source = "config"
        return parsed_f, source
    text = str(raw or "").strip()
    if not text:
        text = str(defaults.get(key) or "").strip()
        source = "config"
    return text, source


def _snap_frames(frames: int, limit: int) -> int:
    """Wan latent time wants 4n+1 frames. Snap to the nearest count that fits."""
    frames = max(5, min(int(frames), max(5, int(limit))))
    down = ((frames - 1) // 4) * 4 + 1
    up = down + 4
    if up <= limit and abs(up - frames) < abs(frames - down):
        snapped = up
    else:
        snapped = down
    if snapped > limit:
        snapped = ((limit - 1) // 4) * 4 + 1
    return max(5, snapped)


def _fit_frame(width: int, height: int, max_edge: int, max_pixels: int) -> tuple:
    """Keep a 480p-class frame inside the 12 GB budget, portrait or landscape."""
    width = max(64, int(width))
    height = max(64, int(height))
    edge = max(width, height)
    if edge > max_edge and edge > 0:
        scale = max_edge / float(edge)
        width = max(64, int(width * scale))
        height = max(64, int(height * scale))
    if width * height > max_pixels and width * height > 0:
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(64, int(width * scale))
        height = max(64, int(height * scale))
    width = max(64, width - (width % 16))
    height = max(64, height - (height % 16))
    while width * height > max_pixels and (width > 64 or height > 64):
        if width >= height and width > 64:
            width -= 16
        elif height > 64:
            height -= 16
        else:
            break
    return width, height


def allow_720p(job: Optional[Mapping[str, Any]] = None) -> bool:
    if _env_on("HAI_WAN22_ALLOW_720P"):
        return True
    if not job:
        return False
    return str(job.get("allow_720p") or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_settings(job: Optional[Mapping[str, Any]] = None, config: Optional[Wan22I2VConfig] = None) -> Dict[str, Any]:
    """Merge config, env, and job into the values the pipeline will run.

    ``strength`` is recorded but not applied as an img2img denoise. Wan I2V
    conditions on the source frame for the whole clip, which is full strength.
    """
    config = config or load_config()
    job = dict(job or {})
    defaults = config.defaults
    sources: Dict[str, str] = {}
    resolved: Dict[str, Any] = {}
    for key in _JOB_KEYS:
        value, source = _layered(job, defaults, key)
        resolved[key] = value
        sources[key] = source

    safe_w = int(defaults.get("safe_width") or 480)
    safe_h = int(defaults.get("safe_height") or 832)
    safe_pixels = safe_w * safe_h
    requested_size = (int(resolved["width"]), int(resolved["height"]))
    if allow_720p(job):
        max_edge = max(int(defaults.get("max_width") or 832), int(defaults.get("max_height") or 1280))
        max_pixels = int(defaults.get("max_width") or 720) * int(defaults.get("max_height") or 1280)
        over_safe = requested_size[0] * requested_size[1] > safe_pixels
        frame_limit = int(defaults.get("max_frames_720p") or 49) if over_safe else int(defaults.get("max_frames") or 81)
    else:
        max_edge = max(safe_w, safe_h)
        max_pixels = safe_pixels
        frame_limit = int(defaults.get("safe_frames") or defaults.get("max_frames") or 81)
    resolved["width"], resolved["height"] = _fit_frame(
        requested_size[0], requested_size[1], max_edge, max_pixels
    )
    resolved["frames"] = _snap_frames(int(resolved["frames"]), frame_limit)
    resolved["steps"] = max(1, min(int(resolved["steps"]), 40))
    resolved["fps"] = max(8, min(int(resolved["fps"]), 30))
    resolved["guidance"] = max(0.0, min(float(resolved["guidance"]), 7.0))
    resolved["guidance_2"] = max(0.0, min(float(resolved["guidance_2"]), 7.0))
    resolved["flow_shift"] = max(0.0, min(float(resolved["flow_shift"]), 16.0))
    resolved["lora_scale_high"] = max(0.0, min(float(resolved["lora_scale_high"]), 2.0))
    resolved["lora_scale_low"] = max(0.0, min(float(resolved["lora_scale_low"]), 2.0))
    resolved["high_noise_steps"] = max(0, min(int(resolved["high_noise_steps"]), int(resolved["steps"])))
    timesteps = list(defaults.get("denoising_timesteps") or [1000, 750, 500, 250])
    # An explicit boundary on the job or in the env wins. Otherwise the split
    # follows high_noise_steps so a 4-step list stays 2 high + 2 low.
    if sources.get("boundary_ratio") == "config":
        resolved["boundary_ratio"] = boundary_ratio_for_split(
            timesteps, int(resolved["high_noise_steps"])
        )
        sources["boundary_ratio"] = "split"
    resolved["denoising_timesteps"] = timesteps
    resolved["scheduler"] = str(resolved["scheduler"] or "flow_match_euler")
    offload = str(resolved["offload"] or "model").strip().lower()
    if offload not in {"model", "sequential", "group"}:
        offload = "model"
    resolved["offload"] = offload

    requested_strength = _parse_float(job.get("strength"))
    if requested_strength is None:
        requested_strength = _parse_float(defaults.get("strength"))
        strength_source = "config"
    else:
        strength_source = "job"
    if requested_strength is None:
        requested_strength = 1.0
    resolved["requested_strength"] = max(0.0, min(float(requested_strength), 1.0))
    resolved["strength"] = 1.0
    sources["strength"] = "i2v_full"
    sources["requested_strength"] = strength_source
    resolved["strength_note"] = (
        "Wan I2V keeps the source frame as a condition for the whole clip. "
        "Denoise strength stays at 1. Partial img2img strength is not applied."
    )

    preset_name = str(job.get("motion_preset") or job.get("motionPreset") or "").strip()
    preset = config.preset(preset_name) if preset_name else None
    if preset_name and preset is None:
        known = ", ".join(config.preset_names())
        raise ValueError(f"unknown motion preset {preset_name!r}. Known presets: {known}")
    prompt = str(job.get("prompt") or "").strip()
    if preset and preset["prompt"] and preset["prompt"].lower() not in prompt.lower():
        prompt = f"{prompt}, {preset['prompt']}" if prompt else preset["prompt"]
    negative = str(job.get("negative_prompt") or "").strip()
    if not negative:
        negative = config.negative_prompt
        sources["negative_prompt"] = "config"
    else:
        sources["negative_prompt"] = "job"
    resolved["prompt"] = prompt
    resolved["negative_prompt"] = negative
    resolved["motion_preset"] = preset["id"] if preset else ""
    resolved["requested_size"] = requested_size
    resolved["dtype"] = str(defaults.get("dtype") or "bfloat16")
    resolved["vae_tiling"] = bool(defaults.get("vae_tiling", True))
    resolved["vae_slicing"] = bool(defaults.get("vae_slicing", True))
    resolved["cpu_offload_threshold_gb"] = float(defaults.get("cpu_offload_threshold_gb") or 16)
    resolved["sources"] = sources
    return resolved


def admission(flag_on: bool, missing: Sequence[str]) -> Dict[str, Any]:
    """Whether the node should advertise the model. Used by the client."""
    if not flag_on:
        return {
            "advertise": False,
            "files_present": False,
            "error": "HAI_WAN22_I2V is off",
            "capabilities": ["image_to_video"],
        }
    if missing:
        return {
            "advertise": False,
            "files_present": False,
            "error": "missing Wan 2.2 files: " + ", ".join(missing),
            "capabilities": ["image_to_video"],
        }
    return {
        "advertise": True,
        "files_present": True,
        "error": "",
        "capabilities": ["image_to_video"],
    }

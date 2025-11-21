"""Local asset registry for WAN 2.2 GGUF video generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Preferred on-disk layout — we attempt to pick the first existing root.
DEFAULT_BASE_CANDIDATES = [
    os.getenv("WAN_W2V_ROOT"),
    "D:/havnai-storage/w2v/models",
    "D:/havnai-storage/models",
    "/mnt/d/havnai-storage/w2v/models",
    "/mnt/d/havnai-storage/models",
]

GGUF_FILENAME = "Rapid-WAN-2.2-I2V-Q4_K_M.gguf"
DEFAULT_LORA_FILES = [
    "WAN-2.2-I2V-Handjob-HIGH-v1.safetensors",
    "DR34ML4Y_I2V_14B_HIGH.safetensors",
    "DR34ML4Y_I2V_14B_LOW.safetensors",
    "56Low-noise-Cumshot-Aesthetics.safetensors",
    "23High-noise-Cumshot-Aesthetics.safetensors",
    "wan22-ultimatedeepthroat-i2v-102epoc-high-k3nk.safetensors",
]
DEFAULT_MOTION_FILES = {
    "high": "high_noise_model.safetensors",
    "low": "low_noise_model.safetensors",
}

SD15_CHECKPOINTS = [
    "perfectdeliberate_v5SD15.safetensors",
    "realisticVisionV60B1_v51HyperVAE.safetensors",
]


@dataclass(frozen=True)
class EnginePaths:
    """Resolved filesystem layout for the WAN 2.2 GGUF stack."""

    root: Path
    gguf_path: Path
    loras_dir: Path
    motion_dir: Path
    creator_dir: Path
    output_dir: Path
    frames_dir: Path
    videos_dir: Path


def _first_existing(paths: Iterable[Optional[str]]) -> Optional[Path]:
    for candidate in paths:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def discover_base_path() -> Path:
    """Pick the best-fit root path based on the requested layout."""

    match = _first_existing(DEFAULT_BASE_CANDIDATES)
    if match:
        return match
    # Fall back to the primary Windows-style location, even if not present yet.
    return Path(DEFAULT_BASE_CANDIDATES[1] or "D:/havnai-storage/models").expanduser()


class ModelRegistry:
    """Lightweight inventory of WAN GGUF, LoRAs, motion modules, and SD1.5 bases."""

    def __init__(self, base_path: Optional[Path] = None) -> None:
        root = base_path or discover_base_path()
        gguf_path = (root / "gguf" / GGUF_FILENAME).expanduser()
        loras_dir = (root / "loras").expanduser()
        motion_dir = (root / "motion").expanduser()
        creator_dir = (root / "creator").expanduser()
        output_dir = (root / "output").expanduser()
        frames_dir = output_dir / "frames"
        videos_dir = output_dir / "videos"

        self.paths = EnginePaths(
            root=root,
            gguf_path=gguf_path,
            loras_dir=loras_dir,
            motion_dir=motion_dir,
            creator_dir=creator_dir,
            output_dir=output_dir,
            frames_dir=frames_dir,
            videos_dir=videos_dir,
        )
        self.using_fallback = False
        try:
            for directory in (loras_dir, motion_dir, creator_dir, frames_dir, videos_dir):
                directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # If the primary drive is not writable in the current environment, fall back locally.
            repo_root = Path(__file__).resolve().parents[3]
            fallback_root = Path(os.getenv("WAN_W2V_FALLBACK_ROOT", repo_root / "w2v-cache")).expanduser()
            fallback_output = fallback_root / "output"
            self.paths = EnginePaths(
                root=fallback_root,
                gguf_path=fallback_root / "gguf" / GGUF_FILENAME,
                loras_dir=fallback_root / "loras",
                motion_dir=fallback_root / "motion",
                creator_dir=fallback_root / "creator",
                output_dir=fallback_output,
                frames_dir=fallback_output / "frames",
                videos_dir=fallback_output / "videos",
            )
            for directory in (
                self.paths.loras_dir,
                self.paths.motion_dir,
                self.paths.creator_dir,
                self.paths.frames_dir,
                self.paths.videos_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            self.using_fallback = True

    # ------------------------------------------------------------------ #
    # LoRAs
    # ------------------------------------------------------------------ #

    def list_loras(self) -> Dict[str, Path]:
        available: Dict[str, Path] = {}
        for file in DEFAULT_LORA_FILES:
            path = self.paths.loras_dir / file
            if path.exists():
                key = path.stem.lower()
                available[key] = path
        # Include any ad-hoc LoRAs dropped into the folder
        for extra in self.paths.loras_dir.glob("*.safetensors"):
            key = extra.stem.lower()
            available.setdefault(key, extra)
        return available

    def resolve_loras(self, requested: Iterable[str]) -> List[Path]:
        lookup = self.list_loras()
        resolved: List[Path] = []
        for name in requested:
            key = str(name).strip().lower()
            if not key:
                continue
            path = lookup.get(key)
            if path:
                resolved.append(path)
        return resolved

    # ------------------------------------------------------------------ #
    # Motion modules
    # ------------------------------------------------------------------ #

    def motion_modules(self) -> Dict[str, Path]:
        modules: Dict[str, Path] = {}
        for key, filename in DEFAULT_MOTION_FILES.items():
            path = self.paths.motion_dir / filename
            if path.exists():
                modules[key] = path
        return modules

    def resolve_motion(self, motion_type: str) -> Optional[Path]:
        key = (motion_type or "").strip().lower() or "high"
        return self.motion_modules().get(key)

    # ------------------------------------------------------------------ #
    # Base checkpoints (SD 1.5)
    # ------------------------------------------------------------------ #

    def sd15_bases(self) -> Dict[str, Path]:
        bases: Dict[str, Path] = {}
        for filename in SD15_CHECKPOINTS:
            path = self.paths.creator_dir / filename
            if path.exists():
                bases[path.stem.lower()] = path
        return bases

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def required_assets_available(self) -> Tuple[bool, List[str]]:
        missing: List[str] = []
        if not self.paths.gguf_path.exists():
            missing.append(str(self.paths.gguf_path))
        if not self.motion_modules():
            missing.append(str(self.paths.motion_dir / DEFAULT_MOTION_FILES["high"]))
            missing.append(str(self.paths.motion_dir / DEFAULT_MOTION_FILES["low"]))
        if not self.list_loras():
            missing.append(str(self.paths.loras_dir))
        return (len(missing) == 0, missing)

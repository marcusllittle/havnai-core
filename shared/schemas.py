"""Shared schema definitions used by coordinator and nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class JobSpec:
    """Represents a work item distributed by the coordinator."""

    wallet: str
    model: str
    prompt: str
    # Optional structured fields for richer job types such as AnimateDiff video generation.
    # Coordinators and nodes may choose
    # to use these directly or pass them through inside ``settings``.
    negative_prompt: str = ""
    frames: int = 0
    fps: int = 0
    motion: str = ""
    base_model: str = ""
    width: int = 0
    height: int = 0
    seed: int | None = None
    lora_strength: float | None = None
    init_image: str | None = None
    scheduler: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "IMAGE_GEN"


@dataclass
class ModelEntry:
    """Describes a model artifact available to the network."""

    name: str
    path: str
    pipeline: str
    type: str
    tags: List[str] = field(default_factory=list)
    reward_weight: float = 0.0
    task_type: str = "IMAGE_GEN"
    vae_path: str = ""
    controlnet_path: str = ""
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    sampler: str | None = None
    negative_prompt_default: str = ""


@dataclass
class Manifest:
    """Container listing all model entries the coordinator exposes."""

    models: List[ModelEntry] = field(default_factory=list)

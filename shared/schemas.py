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
    weight: float = 0.0
    task_type: str = "IMAGE_GEN"


@dataclass
class Manifest:
    """Container listing all model entries the coordinator exposes."""

    models: List[ModelEntry] = field(default_factory=list)

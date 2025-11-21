"""WAN motion module resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .model_registry import ModelRegistry


class MotionLoader:
    """Selects high/low noise motion modules used by the WAN pipeline."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def resolve(self, motion_type: str) -> Optional[Path]:
        return self.registry.resolve_motion(motion_type)

    def available(self) -> dict:
        return {key: str(path) for key, path in self.registry.motion_modules().items()}

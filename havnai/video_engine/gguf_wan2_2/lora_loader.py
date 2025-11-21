"""LoRA resolution and (optional) attachment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

try:  # Optional dependency; not required for scaffolding
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .model_registry import EnginePaths, ModelRegistry


class LoRALoader:
    """Discovers LoRAs and provides a minimal hook to attach them to a pipeline."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.paths: EnginePaths = registry.paths

    def available(self) -> dict:
        return self.registry.list_loras()

    def resolve(self, requested: Iterable[str]) -> List[Path]:
        return self.registry.resolve_loras(requested)

    def attach(self, pipeline: object, loras: Iterable[Path]) -> None:
        """
        Attach LoRAs to a pipeline if the runtime supports it.
        This is intentionally light-touch: most WAN runtimes expose their own
        APIs. For now we just surface the intended files on the pipeline object.
        """

        resolved = list(loras)
        if not resolved:
            return
        if hasattr(pipeline, "loras"):
            pipeline.loras = resolved  # type: ignore[attr-defined]
        # If a torch-compatible pipeline is provided, users can extend this hook.
        if torch is None or not hasattr(pipeline, "load_lora_weights"):
            return
        try:
            for lora_path in resolved:
                pipeline.load_lora_weights(str(lora_path))  # type: ignore[call-arg]
        except Exception:
            # Non-fatal; keep the pipeline usable even if loading fails.
            pass

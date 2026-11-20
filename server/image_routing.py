"""Pure routing rules for Z-Image versus the SDXL image path.

Kept separate from the Flask app so tests can check catalog and node matching
without starting the coordinator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def is_zimage_pipeline(pipeline: str) -> bool:
    return str(pipeline or "").strip().lower().startswith("zimage")


def positive_suffix_override(pipeline: str) -> Optional[str]:
    """Empty suffix for Turbo. None means the caller keeps the SDXL/SD1.5 suffix."""
    if is_zimage_pipeline(pipeline):
        return ""
    return None


def negative_prompt_override(pipeline: str) -> Optional[str]:
    """Turbo ignores negatives. None means the caller keeps the SD negative."""
    if is_zimage_pipeline(pipeline):
        return ""
    return None


def clamp_guidance(pipeline: str, guidance: float) -> float:
    """SDXL stays at 1.0 or above. Turbo is allowed to stay at 0."""
    value = float(guidance)
    if is_zimage_pipeline(pipeline):
        return max(0.0, min(15.0, value))
    return max(1.0, min(15.0, value))


def node_advertises_model(node: Dict[str, Any], model_name: str, pipeline: str) -> bool:
    """The model and pipeline lists the worker sends on heartbeat."""
    models = {str(item).lower() for item in node.get("models") or [] if isinstance(item, str)}
    pipelines = {str(item).lower() for item in node.get("pipelines") or [] if isinstance(item, str)}
    if models and str(model_name or "").lower() not in models:
        return False
    if pipelines and str(pipeline or "").lower() not in pipelines:
        return False
    return True

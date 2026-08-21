"""Manifest-backed video workflow presets."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_WORKFLOW_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_NUMERIC_SETTINGS = {
    "steps",
    "guidance",
    "width",
    "height",
    "frames",
    "fps",
    "strength",
    "lora_strength",
}
_STRING_SETTINGS = {"pipeline_mode", "checkpoint_variant"}


class VideoWorkflowError(ValueError):
    """Raised when a requested workflow is not advertised by the model."""

    code = "unknown_video_workflow"


class VideoWorkflowRequirementError(VideoWorkflowError):
    """Raised when a workflow-specific input requirement is not met."""

    code = "workflow_init_image_required"


def public_video_workflows(model_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the validated, frontend-safe workflows from a model manifest entry."""
    workflows: List[Dict[str, Any]] = []
    raw_workflows = model_cfg.get("video_workflows")
    if not isinstance(raw_workflows, list):
        return workflows

    for raw in raw_workflows:
        if not isinstance(raw, dict):
            continue
        workflow_id = str(raw.get("id") or "").strip().lower()
        label = str(raw.get("label") or "").strip()
        if not _WORKFLOW_ID_RE.fullmatch(workflow_id) or not label:
            continue

        raw_settings = raw.get("settings")
        settings: Dict[str, Any] = {}
        if isinstance(raw_settings, dict):
            for key in _NUMERIC_SETTINGS:
                value = raw_settings.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    settings[key] = value
            for key in _STRING_SETTINGS:
                value = raw_settings.get(key)
                if isinstance(value, str) and value.strip():
                    settings[key] = value.strip()

        workflow: Dict[str, Any] = {
            "id": workflow_id,
            "label": label,
            "settings": settings,
        }
        description = str(raw.get("description") or "").strip()
        if description:
            workflow["description"] = description
        if raw.get("default") is True:
            workflow["default"] = True
        if raw.get("requires_init_image") is True:
            workflow["requires_init_image"] = True
        workflows.append(workflow)
    return workflows


def apply_video_workflow(
    model_cfg: Dict[str, Any], payload: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Apply preset settings while retaining explicit request overrides."""
    workflow_id = str(payload.get("workflow_id") or "").strip().lower()
    if not workflow_id:
        return dict(payload), None

    workflow = next(
        (item for item in public_video_workflows(model_cfg) if item["id"] == workflow_id),
        None,
    )
    if workflow is None:
        raise VideoWorkflowError(f"Unknown video workflow '{workflow_id}' for this model")

    init_image_fields = ("init_image", "init_image_url", "init_image_b64")
    has_init_image = any(
        isinstance(payload.get(key), str) and bool(payload[key].strip())
        for key in init_image_fields
    )
    if workflow.get("requires_init_image") is True and not has_init_image:
        raise VideoWorkflowRequirementError(
            f"Video workflow '{workflow_id}' requires an init image"
        )

    merged = dict(workflow.get("settings") or {})
    merged.update(payload)
    merged["workflow_id"] = workflow_id
    return merged, workflow


__all__ = [
    "VideoWorkflowError",
    "VideoWorkflowRequirementError",
    "apply_video_workflow",
    "public_video_workflows",
]

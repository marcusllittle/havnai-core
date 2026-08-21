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
_PROMPT_ENHANCER_MODES = {"T", "TI", "T1", "TI1"}


class VideoWorkflowError(ValueError):
    """Raised when a requested workflow is not advertised by the model."""

    code = "unknown_video_workflow"


class VideoWorkflowRequirementError(VideoWorkflowError):
    """Raised when a workflow-specific input requirement is not met."""

    code = "workflow_init_image_required"


class VideoWorkflowSettingError(VideoWorkflowError):
    """Raised when a workflow setting is not supported by the runtime contract."""

    code = "invalid_video_workflow_setting"


def normalize_prompt_enhancer(value: Any) -> str:
    """Return a WanGP LTX prompt-enhancer mode or reject an unsafe value."""
    mode = str(value or "").strip().upper()
    if not mode:
        return ""
    if mode not in _PROMPT_ENHANCER_MODES:
        raise VideoWorkflowSettingError(
            f"Unsupported LTX prompt enhancer mode '{mode}'"
        )
    return mode


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
            try:
                prompt_enhancer = normalize_prompt_enhancer(
                    raw_settings.get("prompt_enhancer")
                )
            except VideoWorkflowSettingError:
                prompt_enhancer = ""
            if prompt_enhancer:
                settings["prompt_enhancer"] = prompt_enhancer

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
    if "prompt_enhancer" in merged:
        merged["prompt_enhancer"] = normalize_prompt_enhancer(
            merged.get("prompt_enhancer")
        )
    merged["workflow_id"] = workflow_id
    return merged, workflow


__all__ = [
    "VideoWorkflowError",
    "VideoWorkflowRequirementError",
    "VideoWorkflowSettingError",
    "apply_video_workflow",
    "normalize_prompt_enhancer",
    "public_video_workflows",
]

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

import video_workflows


MODEL = {
    "video_workflows": [
        {
            "id": "faithful_i2v",
            "label": "Maximum fidelity",
            "default": True,
            "requires_init_image": True,
            "settings": {
                "steps": 8,
                "guidance": 1.0,
                "frames": 97,
                "fps": 24,
                "strength": 0.95,
                "lora_strength": 0.35,
                "prompt_enhancer": "ti",
                "prompt": "must not be exposed or applied",
            },
        }
    ]
}


def test_public_workflows_only_expose_supported_settings() -> None:
    workflows = video_workflows.public_video_workflows(MODEL)

    assert workflows == [
        {
            "id": "faithful_i2v",
            "label": "Maximum fidelity",
            "default": True,
            "requires_init_image": True,
            "settings": {
                "steps": 8,
                "guidance": 1.0,
                "frames": 97,
                "fps": 24,
                "strength": 0.95,
                "lora_strength": 0.35,
                "prompt_enhancer": "TI",
            },
        }
    ]


def test_explicit_request_values_override_workflow_defaults() -> None:
    payload, selected = video_workflows.apply_video_workflow(
        MODEL,
        {
            "workflow_id": "faithful_i2v",
            "frames": 121,
            "prompt": "user prompt",
            "init_image": "data:image/png;base64,source",
        },
    )

    assert selected and selected["id"] == "faithful_i2v"
    assert payload["frames"] == 121
    assert payload["strength"] == 0.95
    assert payload["lora_strength"] == 0.35
    assert payload["prompt_enhancer"] == "TI"
    assert payload["prompt"] == "user prompt"


def test_unknown_workflow_is_rejected() -> None:
    with pytest.raises(video_workflows.VideoWorkflowError):
        video_workflows.apply_video_workflow(MODEL, {"workflow_id": "missing"})


def test_invalid_prompt_enhancer_override_is_rejected() -> None:
    with pytest.raises(video_workflows.VideoWorkflowSettingError) as error:
        video_workflows.apply_video_workflow(
            MODEL,
            {
                "workflow_id": "faithful_i2v",
                "init_image": "source-image",
                "prompt_enhancer": "shell command",
            },
        )

    assert error.value.code == "invalid_video_workflow_setting"


def test_workflow_requiring_init_image_rejects_missing_source() -> None:
    with pytest.raises(video_workflows.VideoWorkflowRequirementError) as error:
        video_workflows.apply_video_workflow(
            MODEL,
            {"workflow_id": "faithful_i2v", "prompt": "user prompt"},
        )

    assert error.value.code == "workflow_init_image_required"


@pytest.mark.parametrize("value", ["", "   "])
def test_workflow_requiring_init_image_rejects_blank_source(value: str) -> None:
    with pytest.raises(video_workflows.VideoWorkflowRequirementError):
        video_workflows.apply_video_workflow(
            MODEL,
            {"workflow_id": "faithful_i2v", "init_image": value},
        )


@pytest.mark.parametrize("field", ["init_image", "init_image_url", "init_image_b64"])
def test_workflow_accepts_every_init_image_alias(field: str) -> None:
    payload, selected = video_workflows.apply_video_workflow(
        MODEL,
        {"workflow_id": "faithful_i2v", field: "source-image"},
    )

    assert selected and selected["id"] == "faithful_i2v"
    assert payload[field] == "source-image"


def test_ltx23_manifest_advertises_expected_workflows() -> None:
    manifest = json.loads((ROOT / "server" / "manifests" / "registry.json").read_text())
    model = next(item for item in manifest["models"] if item["name"] == "ltx23_wangp_distilled")
    workflows = video_workflows.public_video_workflows(model)

    assert [item["id"] for item in workflows] == [
        "faithful_i2v",
        "faithful_portrait_i2v",
        "balanced_i2v",
        "portrait_i2v",
        "dynamic_i2v",
    ]
    assert workflows[0]["default"] is True
    assert workflows[0]["label"] == "Maximum fidelity"
    assert workflows[0]["settings"]["strength"] == 0.98
    assert workflows[0]["settings"]["lora_strength"] == 0.35
    assert workflows[0]["settings"]["prompt_enhancer"] == "TI"
    portrait_fidelity = workflows[1]
    assert portrait_fidelity["settings"]["width"] == 704
    assert portrait_fidelity["settings"]["height"] == 1280
    assert portrait_fidelity["settings"]["strength"] == 0.98
    assert portrait_fidelity["settings"]["lora_strength"] == 0.35
    assert portrait_fidelity["settings"]["prompt_enhancer"] == "TI"

    payload, selected = video_workflows.apply_video_workflow(
        model,
        {
            "workflow_id": "faithful_portrait_i2v",
            "init_image": "source-image",
        },
    )
    assert selected and selected["id"] == "faithful_portrait_i2v"
    assert payload["width"] == 704
    assert payload["height"] == 1280
    assert payload["strength"] == 0.98
    assert payload["lora_strength"] == 0.35
    assert payload["prompt_enhancer"] == "TI"

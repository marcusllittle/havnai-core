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
            "label": "Source fidelity",
            "default": True,
            "requires_init_image": True,
            "settings": {
                "steps": 8,
                "guidance": 1.0,
                "frames": 97,
                "fps": 24,
                "strength": 0.95,
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
            "label": "Source fidelity",
            "default": True,
            "requires_init_image": True,
            "settings": {
                "steps": 8,
                "guidance": 1.0,
                "frames": 97,
                "fps": 24,
                "strength": 0.95,
            },
        }
    ]


def test_explicit_request_values_override_workflow_defaults() -> None:
    payload, selected = video_workflows.apply_video_workflow(
        MODEL,
        {"workflow_id": "faithful_i2v", "frames": 121, "prompt": "user prompt"},
    )

    assert selected and selected["id"] == "faithful_i2v"
    assert payload["frames"] == 121
    assert payload["strength"] == 0.95
    assert payload["prompt"] == "user prompt"


def test_unknown_workflow_is_rejected() -> None:
    with pytest.raises(video_workflows.VideoWorkflowError):
        video_workflows.apply_video_workflow(MODEL, {"workflow_id": "missing"})


def test_ltx23_manifest_advertises_expected_workflows() -> None:
    manifest = json.loads((ROOT / "server" / "manifests" / "registry.json").read_text())
    model = next(item for item in manifest["models"] if item["name"] == "ltx23_wangp_distilled")
    workflows = video_workflows.public_video_workflows(model)

    assert [item["id"] for item in workflows] == [
        "faithful_i2v",
        "balanced_i2v",
        "portrait_i2v",
        "dynamic_i2v",
    ]
    assert workflows[0]["default"] is True
    assert workflows[0]["settings"]["strength"] == 0.95

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engines.wangp.runner import (
    DEFAULT_LTX23_LORA_FILENAME,
    _resolve_ltx23_lora,
    _resolve_prompt_enhancer,
)
from engines.wangp.worker import (
    CONTINUATION_INSTRUCTION,
    SOURCE_PRESERVATION_INSTRUCTION,
    _apply_lora_settings,
    _prepare_prompt_enhancer_config,
    _prompt_with_source_preservation,
)


def test_resolves_installed_default_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lora_path = tmp_path / "loras" / "ltx2" / DEFAULT_LTX23_LORA_FILENAME
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()
    monkeypatch.delenv("HAVNAI_LTX23_LORA", raising=False)
    monkeypatch.delenv("HAVNAI_LTX23_LORA_STRENGTH", raising=False)
    monkeypatch.delenv("HAVNAI_LTX23_LORA_ENABLED", raising=False)

    assert _resolve_ltx23_lora(tmp_path) == (DEFAULT_LTX23_LORA_FILENAME, 0.8)


def test_missing_or_disabled_lora_is_not_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HAVNAI_LTX23_LORA", raising=False)
    assert _resolve_ltx23_lora(tmp_path) is None

    lora_path = tmp_path / "loras" / "ltx2" / DEFAULT_LTX23_LORA_FILENAME
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()
    monkeypatch.setenv("HAVNAI_LTX23_LORA_ENABLED", "false")
    assert _resolve_ltx23_lora(tmp_path) is None


def test_workflow_strength_overrides_environment_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lora_path = tmp_path / "loras" / "ltx2" / DEFAULT_LTX23_LORA_FILENAME
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()
    monkeypatch.setenv("HAVNAI_LTX23_LORA_STRENGTH", "0.9")

    assert _resolve_ltx23_lora(tmp_path, 0.35) == (
        DEFAULT_LTX23_LORA_FILENAME,
        0.35,
    )


def test_rejects_lora_path_outside_wangp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAVNAI_LTX23_LORA", "../outside.safetensors")

    with pytest.raises(ValueError, match="must be a filename"):
        _resolve_ltx23_lora(tmp_path)


def test_prompt_enhancer_requires_supported_mode_and_source() -> None:
    assert _resolve_prompt_enhancer("ti", has_source=True) == "TI"
    assert _resolve_prompt_enhancer(None, has_source=False) == ""
    with pytest.raises(ValueError, match="requires an init image"):
        _resolve_prompt_enhancer("TI", has_source=False)
    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_prompt_enhancer("TIM", has_source=True)


def test_prompt_enhancer_uses_isolated_automatic_config(tmp_path: Path) -> None:
    root = tmp_path / "Wan2GP"
    root.mkdir()
    source_config = {
        "enhancer_enabled": 3,
        "enhancer_mode": 1,
        "profile": 4,
        "prompt_enhancer_randomize_seed": True,
    }
    (root / "wgp_config.json").write_text(json.dumps(source_config))

    mode, config_path = _prepare_prompt_enhancer_config(root, tmp_path / "job", "ti")

    assert mode == "TI"
    assert config_path is not None
    generated = json.loads(config_path.read_text())
    assert generated["enhancer_mode"] == 0
    assert generated["enhancer_enabled"] == 3
    assert generated["prompt_enhancer_randomize_seed"] is False
    assert json.loads((root / "wgp_config.json").read_text()) == source_config


def test_prompt_enhancer_rejects_disabled_runtime(tmp_path: Path) -> None:
    root = tmp_path / "Wan2GP"
    root.mkdir()
    (root / "wgp_config.json").write_text(json.dumps({"enhancer_enabled": 0}))

    with pytest.raises(RuntimeError, match="not enabled"):
        _prepare_prompt_enhancer_config(root, tmp_path / "job", "TI")


def test_image_aware_prompt_enhancer_receives_source_preservation_instruction() -> None:
    prepared = _prompt_with_source_preservation("She turns slowly.", "TI", True)

    assert prepared.startswith("She turns slowly.\n@ ")
    assert prepared.endswith(SOURCE_PRESERVATION_INSTRUCTION)
    assert "one continuous, chronological shot" in prepared
    assert "visible features spatially and anatomically consistent" in prepared


def test_continuation_prompt_advances_without_restarting_action() -> None:
    prepared = _prompt_with_source_preservation(
        "She continues turning.", "TI", True, continuation=True
    )

    assert SOURCE_PRESERVATION_INSTRUCTION in prepared
    assert prepared.endswith(CONTINUATION_INSTRUCTION)
    assert "do not restart or repeat" in prepared
    assert "across the clip boundary" in prepared


@pytest.mark.parametrize(
    ("mode", "has_source"),
    [("T", True), ("TI", False), ("", True)],
)
def test_source_preservation_instruction_requires_image_aware_enhancement(
    mode: str, has_source: bool
) -> None:
    assert (
        _prompt_with_source_preservation("Original prompt", mode, has_source)
        == "Original prompt"
    )


def test_worker_applies_lora_filename_and_multiplier() -> None:
    settings = {"activated_loras": None, "loras_multipliers": None}

    _apply_lora_settings(
        settings,
        {
            "activated_loras": [DEFAULT_LTX23_LORA_FILENAME],
            "loras_multipliers": "0.6",
        },
    )

    assert settings["activated_loras"] == [DEFAULT_LTX23_LORA_FILENAME]
    assert settings["loras_multipliers"] == "0.6"

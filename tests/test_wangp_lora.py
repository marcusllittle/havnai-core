from __future__ import annotations

from pathlib import Path

import pytest

from engines.wangp.runner import DEFAULT_LTX23_LORA_FILENAME, _resolve_ltx23_lora
from engines.wangp.worker import _apply_lora_settings


def test_resolves_installed_default_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lora_path = tmp_path / "loras" / "ltx2" / DEFAULT_LTX23_LORA_FILENAME
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()
    monkeypatch.delenv("HAVNAI_LTX23_LORA", raising=False)
    monkeypatch.delenv("HAVNAI_LTX23_LORA_STRENGTH", raising=False)
    monkeypatch.delenv("HAVNAI_LTX23_LORA_ENABLED", raising=False)

    assert _resolve_ltx23_lora(tmp_path) == (DEFAULT_LTX23_LORA_FILENAME, 0.6)


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

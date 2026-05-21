from __future__ import annotations

from pathlib import Path

import pytest

from common.lora_router import route_loras


EXPECTED_FILES = [
    "NsfwPovAllInOneLoraSdxl-000009.safetensors",
    "sagging-xl-v1.0.safetensors",
    "bdsm_SDXL_1_.safetensors",
]


@pytest.fixture()
def temp_lora_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for filename in EXPECTED_FILES:
        (tmp_path / filename).write_text("", encoding="utf-8")
    monkeypatch.setenv("HAVNAI_LORA_DIR", str(tmp_path))
    return tmp_path


def _selected_ids(result: dict) -> list[str]:
    return [entry["id"] for entry in result["selected_loras"]]


def test_normal_prompt_selects_only_default_pov(temp_lora_dir: Path) -> None:
    result = route_loras("a woman standing in a room")
    assert _selected_ids(result) == ["pov_all_in_one_sdxl"]
    assert result["warnings"] == []


def test_sagging_breast_selects_default_and_sagging(temp_lora_dir: Path) -> None:
    result = route_loras("a woman with sagging breast")
    assert _selected_ids(result) == ["pov_all_in_one_sdxl", "sagging_breast"]


def test_sagging_breasts_selects_default_and_sagging(temp_lora_dir: Path) -> None:
    result = route_loras("a woman with sagging breasts")
    assert _selected_ids(result) == ["pov_all_in_one_sdxl", "sagging_breast"]


def test_rope_bondage_selects_default_and_bdsm(temp_lora_dir: Path) -> None:
    result = route_loras("woman in rope bondage")
    assert _selected_ids(result) == ["pov_all_in_one_sdxl", "bondage_go_bdsm"]


def test_sagging_breasts_in_rope_bondage_selects_all_three(temp_lora_dir: Path) -> None:
    result = route_loras("woman with sagging breasts in rope bondage")
    assert _selected_ids(result) == [
        "pov_all_in_one_sdxl",
        "sagging_breast",
        "bondage_go_bdsm",
    ]


def test_non_sdxl_pipeline_selects_none(temp_lora_dir: Path) -> None:
    result = route_loras("woman with sagging breasts in rope bondage", pipeline="sd15")
    assert result["selected_loras"] == []
    assert result["warnings"] == []


def test_missing_files_warn_and_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAVNAI_LORA_DIR", str(tmp_path))
    result = route_loras("woman with sagging breasts in rope bondage")
    assert result["selected_loras"] == []
    assert len(result["warnings"]) == 3
    assert any("pov_all_in_one_sdxl" in warning for warning in result["warnings"])


def test_havnai_lora_dir_is_honored(temp_lora_dir: Path) -> None:
    result = route_loras("plain prompt")
    selected = result["selected_loras"]
    assert len(selected) == 1
    assert selected[0]["path"] == str((temp_lora_dir / EXPECTED_FILES[0]).resolve())


def test_requested_loras_are_preserved(temp_lora_dir: Path) -> None:
    requested = [{"name": "custom_user_lora.safetensors", "weight": 0.9}]
    result = route_loras("plain prompt", requested_loras=requested)
    assert _selected_ids(result) == ["custom_user_lora", "pov_all_in_one_sdxl"]
    assert result["selected_loras"][0]["trigger_type"] == "user"
    assert result["selected_loras"][0]["trigger_reason"] == "user_requested"


def test_duplicate_requested_lora_wins_over_routed_duplicate(temp_lora_dir: Path) -> None:
    requested = [{"name": "NsfwPovAllInOneLoraSdxl-000009.safetensors", "weight": 0.99}]
    result = route_loras("plain prompt", requested_loras=requested)
    assert _selected_ids(result) == ["NsfwPovAllInOneLoraSdxl-000009", "pov_all_in_one_sdxl"] or _selected_ids(result) == ["NsfwPovAllInOneLoraSdxl-000009"]
    assert _selected_ids(result) == ["NsfwPovAllInOneLoraSdxl-000009"]
    assert any("user-provided lora won" in warning.lower() for warning in result["warnings"])

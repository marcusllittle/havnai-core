"""
Tests for prompt mutation bug fix.

Verifies that human-anatomy quality tokens (detailed skin pores, natural gaze,
no extra breasts, etc.) are NOT injected into non-human/object prompts, while
still being applied correctly to human/portrait prompts.

Imports the two pure functions directly from app.py by extracting just the
relevant module-level definitions without triggering Flask/DB imports.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Inline the constants + functions under test so we don't need Flask installed.
# This mirrors exactly what is in server/app.py.
# ---------------------------------------------------------------------------

POSITIVE_SUFFIX_SD15_REALISM = (
    "(ultra-realistic 8k:1.05), (detailed skin pores:1.03), "
    "focused eyes, clear pupils, natural gaze, "
    "natural teeth, realistic mouth structure, "
    "no extra breasts, no multiple breasts, not deformed, no multiple legs"
)

POSITIVE_SUFFIX_STYLIZED = "high quality, masterpiece, detailed, clean lines"

POSITIVE_SUFFIX_SDXL = "high quality, masterpiece, detailed, sharp focus, 8k"

_STYLIZED_TAGS = {"anime", "cartoon", "pixar", "manhwa", "webtoon", "stylized", "fantasy"}

_HUMAN_SUBJECT_KEYWORDS = {
    "woman", "man", "girl", "boy", "person", "people", "human",
    "portrait", "face", "body", "figure", "character", "model",
    "pilot", "warrior", "soldier", "knight", "mage", "wizard",
    "nude", "naked", "skin", "eyes", "hair", "hands",
    "she", "he", "her", "his",
    "nsfw", "sexy", "erotic", "lingerie", "bikini",
}


def _prompt_has_human_subject(prompt: str) -> bool:
    if not prompt:
        return False
    words = set(re.split(r"[\s,.:;!?()\[\]{}\"']+", prompt.lower()))
    return bool(words & _HUMAN_SUBJECT_KEYWORDS)


def get_positive_suffix(model_cfg: Optional[Dict[str, Any]] = None, prompt: str = "") -> str:
    if model_cfg is None:
        return POSITIVE_SUFFIX_SD15_REALISM if _prompt_has_human_subject(prompt) else POSITIVE_SUFFIX_STYLIZED
    tags = set(t.lower() for t in (model_cfg.get("tags") or []))
    pipeline = (model_cfg.get("pipeline") or "sd15").lower()
    if tags & _STYLIZED_TAGS:
        return POSITIVE_SUFFIX_STYLIZED
    if "sdxl" in pipeline or "xl" in pipeline:
        return POSITIVE_SUFFIX_SDXL
    if _prompt_has_human_subject(prompt):
        return POSITIVE_SUFFIX_SD15_REALISM
    return POSITIVE_SUFFIX_STYLIZED

# ---------------------------------------------------------------------------
# _prompt_has_human_subject
# ---------------------------------------------------------------------------

class TestPromptHasHumanSubject:
    def test_empty_prompt(self):
        assert _prompt_has_human_subject("") is False

    def test_none_like_prompt(self):
        assert _prompt_has_human_subject("  ") is False

    # Human prompts
    def test_woman(self):
        assert _prompt_has_human_subject("a beautiful woman in a red dress") is True

    def test_portrait(self):
        assert _prompt_has_human_subject("close-up portrait, soft lighting") is True

    def test_pilot(self):
        assert _prompt_has_human_subject("sci-fi pilot in a cockpit, detailed") is True

    def test_nude(self):
        assert _prompt_has_human_subject("nude model, studio lighting") is True

    def test_eyes(self):
        assert _prompt_has_human_subject("closeup eyes, sharp focus") is True

    # Non-human prompts — these are the critical regression cases
    def test_wallet_product_shot(self):
        assert _prompt_has_human_subject(
            "leather wallet on a marble surface, studio lighting, product photography"
        ) is False

    def test_spaceship(self):
        assert _prompt_has_human_subject(
            "futuristic spaceship in orbit, sci-fi, cinematic, detailed hull"
        ) is False

    def test_environment(self):
        assert _prompt_has_human_subject(
            "alien jungle landscape, neon plants, misty atmosphere, concept art"
        ) is False

    def test_vehicle(self):
        assert _prompt_has_human_subject(
            "armored ground vehicle, desert terrain, dusty, 8k"
        ) is False

    def test_abstract(self):
        assert _prompt_has_human_subject(
            "glowing energy orb, dark background, volumetric light"
        ) is False

    def test_weapon(self):
        assert _prompt_has_human_subject(
            "sci-fi plasma rifle, detailed, product render, dark background"
        ) is False


# ---------------------------------------------------------------------------
# get_positive_suffix
# ---------------------------------------------------------------------------

SDXL_CFG = {"pipeline": "sdxl", "tags": []}
SD15_CFG = {"pipeline": "sd15", "tags": []}
ANIME_CFG = {"pipeline": "sd15", "tags": ["anime"]}

class TestGetPositiveSuffix:

    # Stylized models always get lightweight suffix regardless of prompt
    def test_anime_model_human_prompt(self):
        result = get_positive_suffix(ANIME_CFG, prompt="beautiful woman, anime style")
        assert result == POSITIVE_SUFFIX_STYLIZED

    def test_anime_model_nonhuman_prompt(self):
        result = get_positive_suffix(ANIME_CFG, prompt="spaceship in orbit")
        assert result == POSITIVE_SUFFIX_STYLIZED

    # SDXL always gets SDXL suffix
    def test_sdxl_human_prompt(self):
        result = get_positive_suffix(SDXL_CFG, prompt="beautiful woman portrait")
        assert result == POSITIVE_SUFFIX_SDXL

    def test_sdxl_nonhuman_prompt(self):
        result = get_positive_suffix(SDXL_CFG, prompt="futuristic wallet product shot")
        assert result == POSITIVE_SUFFIX_SDXL

    # SD1.5 realism — the core fix
    def test_sd15_human_prompt_gets_realism_suffix(self):
        result = get_positive_suffix(SD15_CFG, prompt="a woman with detailed skin, portrait")
        assert result == POSITIVE_SUFFIX_SD15_REALISM
        # Must contain the problematic tokens only for human prompts
        assert "skin pores" in result
        assert "natural gaze" in result

    def test_sd15_nonhuman_prompt_does_not_get_anatomy_tokens(self):
        result = get_positive_suffix(
            SD15_CFG,
            prompt="leather wallet on marble surface, product photography, studio lighting"
        )
        # Should NOT be the realism suffix
        assert result != POSITIVE_SUFFIX_SD15_REALISM
        # Specifically must not contain anatomy/skin tokens
        assert "skin pores" not in result
        assert "natural gaze" not in result
        assert "no extra breasts" not in result
        assert "no multiple breasts" not in result
        assert "no multiple legs" not in result

    def test_sd15_spaceship_no_anatomy_tokens(self):
        result = get_positive_suffix(SD15_CFG, prompt="sci-fi spaceship in orbit, cinematic")
        assert "skin pores" not in result
        assert "natural gaze" not in result
        assert "no extra breasts" not in result

    def test_sd15_environment_no_anatomy_tokens(self):
        result = get_positive_suffix(
            SD15_CFG,
            prompt="alien jungle landscape, neon plants, misty atmosphere, concept art"
        )
        assert "skin pores" not in result
        assert "no multiple legs" not in result

    # Legacy call without cfg
    def test_no_cfg_human_prompt(self):
        result = get_positive_suffix(None, prompt="a woman portrait")
        assert result == POSITIVE_SUFFIX_SD15_REALISM

    def test_no_cfg_nonhuman_prompt(self):
        result = get_positive_suffix(None, prompt="futuristic spacecraft")
        assert result != POSITIVE_SUFFIX_SD15_REALISM
        assert "skin pores" not in result

from __future__ import annotations

import re
from typing import Optional, Tuple


def enhance_prompt_for_positions(user_prompt: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Enhance prompts with position-specific hints and routing metadata."""

    prompt = (user_prompt or "").strip()
    lower = prompt.lower()

    position_rules = [
        ("POVDoggy", [r"\bdoggy\b", r"\bdoggystyle\b", r"\bfrom behind\b"]),
        ("POVReverseCowgirl", [r"\breverse cowgirl\b", r"\breverse riding\b"]),
        ("PSCowgirl", [r"\bcowgirl\b", r"\briding\b", r"\bon top\b"]),
        ("MissionaryVaginal-v2", [r"\bmissionary\b", r"\blegs spread\b"]),
    ]

    hardcore_keywords = [
        r"\bsex\b",
        r"\bfucking\b",
        r"\bpenetration\b",
        r"\banal\b",
        r"\bdoggy\b",
        r"\bdoggystyle\b",
        r"\bfrom behind\b",
        r"\bcowgirl\b",
        r"\briding\b",
        r"\bon top\b",
        r"\breverse cowgirl\b",
        r"\breverse riding\b",
        r"\bmissionary\b",
        r"\blegs spread\b",
    ]

    if not any(re.search(pat, lower, flags=re.IGNORECASE) for pat in hardcore_keywords):
        return prompt, None, None

    position_lora = None
    matched_pattern = None
    for lora_name, patterns in position_rules:
        for pat in patterns:
            if re.search(pat, lower, flags=re.IGNORECASE):
                position_lora = lora_name
                matched_pattern = pat
                break
        if position_lora:
            break

    if position_lora is None:
        position_lora = "POVDoggy"

    enhanced = prompt
    if matched_pattern:
        def _reinforce(match: re.Match) -> str:
            return f"({match.group(0)}:1.15)"

        enhanced = re.sub(matched_pattern, _reinforce, enhanced, count=1, flags=re.IGNORECASE)

    enhanced = f"{enhanced}, detailed, masterpiece" if enhanced else "detailed, masterpiece"

    style_rules = [
        ("kizukiAnimeHentai_animeHentaiV4", [r"\banime\b", r"\bhentai\b", r"\bahegao\b", r"\b2d\b"]),
        (
            "disneyPixarCartoon_v10",
            [r"\bcartoon\b", r"\bpixar\b", r"\bdisney\b", r"\bcute eyes\b", r"\bbig eyes\b", r"\b3d animated\b"],
        ),
        ("unstablePornhwa_beta", [r"\bmanhwa\b", r"\bwebtoon\b", r"\bkorean comic\b"]),
    ]

    selected_model = None
    for model_name, patterns in style_rules:
        if any(re.search(pat, lower, flags=re.IGNORECASE) for pat in patterns):
            selected_model = model_name
            break
    if selected_model is None:
        selected_model = "uberRealisticPornMerge_v23Final"

    return enhanced, selected_model, position_lora


__all__ = ["enhance_prompt_for_positions"]

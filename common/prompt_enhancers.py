from __future__ import annotations

import re
from typing import Optional, Tuple


POSITION_LORA_WEIGHTS = {
    "POVDoggy": 0.65,
    "POVReverseCowgirl": 0.65,
    "PSCowgirl": 0.65,
    "MissionaryVaginal-v2": 0.60,
}

EXTRA_POSITION_NEGATIVE = (
    ", merged bodies, conjoined twins, extra torso, duplicate torso, extra limbs, "
    "extra arms, extra legs, deformed anatomy, body fusion, siamese, malformed chest, "
    "missing penis, censored, bar censor, mosaic censor"
)

POSITION_REINFORCEMENTS = {
    "POVDoggy": "(doggystyle:1.05)",
    "POVReverseCowgirl": "(reverse cowgirl position:1.05)",
    "PSCowgirl": "(cowgirl position:1.05)",
    "MissionaryVaginal-v2": "(missionary position:1.05)",
}

POSITION_PENETRATION_SUFFIX = {
    "POVDoggy": "(deep penetration:0.9)",
    "POVReverseCowgirl": "(vaginal penetration:0.9), (penis inside:0.9)",
    "PSCowgirl": "(vaginal penetration:0.9), (penis inside:0.9)",
    "MissionaryVaginal-v2": "(vaginal penetration:0.9), (penis inside:0.9)",
}


def resolve_position_lora_weight(lora_name: str) -> float:
    return float(POSITION_LORA_WEIGHTS.get(lora_name, 0.65))


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
    reinforcement = POSITION_REINFORCEMENTS.get(position_lora, "(position:1.05)")
    if matched_pattern:
        def _reinforce(match: re.Match) -> str:
            return reinforcement

        enhanced = re.sub(matched_pattern, _reinforce, enhanced, count=1, flags=re.IGNORECASE)
    else:
        enhanced = f"{enhanced}, {reinforcement}" if enhanced else reinforcement

    penetration = POSITION_PENETRATION_SUFFIX.get(position_lora)
    if penetration:
        enhanced = f"{enhanced}, {penetration}"

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


__all__ = [
    "EXTRA_POSITION_NEGATIVE",
    "POSITION_LORA_WEIGHTS",
    "enhance_prompt_for_positions",
    "resolve_position_lora_weight",
]

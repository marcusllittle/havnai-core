from __future__ import annotations

import re
from typing import Optional, Tuple


POSITION_LORA_WEIGHTS = {
    "POVDoggy": 0.60,
    "POVReverseCowgirl": 0.65,
    "PSCowgirl": 0.65,
    "MissionaryVaginal-v2": 0.60,
}

BASE_POSITION_NEGATIVE = (
    "merged bodies, conjoined twins, extra torso, duplicate torso, extra limbs, "
    "extra arms, extra legs, deformed anatomy, body fusion, siamese, malformed chest, "
    "missing penis, censored, bar censor, mosaic censor"
)

SHARPNESS_NEGATIVE = "blurry, lowres, soft focus, pixelated, artifacts, low quality"

ANTI_OVERLAY_NEGATIVE = (
    "duplicate ass, layered buttocks, overlapping anatomy, double pussy, ghosting, "
    "transparent overlay, cloned body parts"
)

ANTI_ORAL_NEGATIVE = (
    "facesitting, sitting on face, cunnilingus, oral sex, anilingus, rimjob, blowjob"
)

ANTI_MULTIPLE_GIRLS_NEGATIVE = (
    "multiple girls, two women, duplicate face, cloned face, multiple heads, "
    "extra face, second woman, lesbian, yuri, girl on girl"
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

DOGGY_POV_SUFFIX = (
    "(pov from behind male viewer:1.1), (ass focus:1.1), "
    "(deep vaginal penetration:1.05), (penis inside:1.0), "
    "onegirl, solo female, faceless male"
)

DOGGY_ANTI_ORAL_NEGATIVE = ANTI_ORAL_NEGATIVE

POSITION_QUALITY_SUFFIX = (
    "ultra detailed skin, sharp focus, photorealistic, masterpiece, best quality, "
    "highres, 8k, intricate details, cinematic lighting"
)

HARDCORE_KEYWORDS = [
    r"\bsex\b",
    r"\bfucking\b",
    r"\bpenetration\b",
    r"\banal\b",
    r"doggy",
    r"doggystyle",
    r"doggy style",
    r"from behind",
    r"bent over",
    r"\bcowgirl\b",
    r"\briding\b",
    r"\bon top\b",
    r"\breverse cowgirl\b",
    r"\breverse riding\b",
    r"\bmissionary\b",
    r"\blegs spread\b",
]


def resolve_position_lora_weight(lora_name: str) -> float:
    return float(POSITION_LORA_WEIGHTS.get(lora_name, 0.65))

def has_hardcore_keywords(user_prompt: str) -> bool:
    prompt = (user_prompt or "").strip()
    if not prompt:
        return False
    return any(re.search(pat, prompt, flags=re.IGNORECASE) for pat in HARDCORE_KEYWORDS)


def enhance_prompt_for_positions(user_prompt: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Enhance prompts with position-specific hints and routing metadata."""

    prompt = (user_prompt or "").strip()
    lower = prompt.lower()

    position_rules = [
        ("POVDoggy", [r"doggy", r"doggystyle", r"doggy style", r"from behind", r"bent over"]),
        ("POVReverseCowgirl", [r"\breverse cowgirl\b", r"\breverse riding\b"]),
        ("PSCowgirl", [r"\bcowgirl\b", r"\briding\b", r"\bon top\b"]),
        ("MissionaryVaginal-v2", [r"\bmissionary\b", r"\blegs spread\b"]),
    ]

    if not any(re.search(pat, lower, flags=re.IGNORECASE) for pat in HARDCORE_KEYWORDS):
        return prompt, None, None, None

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
    if position_lora == "POVDoggy":
        enhanced = f"{enhanced}, {DOGGY_POV_SUFFIX}"

    enhanced = (
        f"{enhanced}, detailed, masterpiece, {POSITION_QUALITY_SUFFIX}"
        if enhanced
        else f"detailed, masterpiece, {POSITION_QUALITY_SUFFIX}"
    )

    style_rules = [
        ("kizukiAnimeHentai_animeHentaiV4", [r"\banime\b", r"\bhentai\b", r"\bahegao\b", r"\b2d\b"]),
        (
            "disneyPixarCartoon_v10",
            [r"\bcartoon\b", r"\bpixar\b", r"\bdisney\b", r"\bcute eyes\b", r"\bbig eyes\b", r"\b3d animated\b"],
        ),
        ("unstablePornhwa_beta", [r"\bmanhwa\b", r"\bwebtoon\b", r"\bkorean comic\b"]),
    ]

    selected_model = None
    if not position_lora:
        for model_name, patterns in style_rules:
            if any(re.search(pat, lower, flags=re.IGNORECASE) for pat in patterns):
                selected_model = model_name
                break
        if selected_model is None:
            selected_model = "uberRealisticPornMerge_v23Final"
    else:
        selected_model = "lazymixRealAmateur_v40"

    negative = None
    if position_lora:
        negatives = [
            BASE_POSITION_NEGATIVE,
            ANTI_MULTIPLE_GIRLS_NEGATIVE,
            SHARPNESS_NEGATIVE,
            ANTI_OVERLAY_NEGATIVE,
        ]
        if position_lora == "POVDoggy":
            negatives.append(DOGGY_ANTI_ORAL_NEGATIVE)
        negative = ", ".join([item for item in negatives if item])

    return enhanced, selected_model, position_lora, negative


__all__ = [
    "ANTI_MULTIPLE_GIRLS_NEGATIVE",
    "ANTI_ORAL_NEGATIVE",
    "ANTI_OVERLAY_NEGATIVE",
    "BASE_POSITION_NEGATIVE",
    "HARDCORE_KEYWORDS",
    "POSITION_LORA_WEIGHTS",
    "enhance_prompt_for_positions",
    "has_hardcore_keywords",
    "resolve_position_lora_weight",
]

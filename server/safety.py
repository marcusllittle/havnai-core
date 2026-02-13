"""Content safety filtering for HavnAI prompts."""

from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Phrase lists for content filtering
# ---------------------------------------------------------------------------

MINOR_PHRASES = [
    "child",
    "children",
    "kid",
    "kids",
    "toddler",
    "infant",
    "newborn",
    "preteen",
    "pre-teen",
    "underage",
    "minor",
    "teen",
    "teenage",
    "teenager",
    "schoolgirl",
    "schoolboy",
    "loli",
    "lolita",
    "shota",
    "lolicon",
    "high school",
    "middle school",
    "junior high",
    "elementary school",
    "barely legal",
]

AMBIGUOUS_YOUTH_PHRASES = [
    "girl",
    "boy",
    "young",
    "youthful",
]

SEXUAL_PHRASES = [
    "sex",
    "sexual",
    "sexy",
    "nude",
    "naked",
    "porn",
    "pornographic",
    "erotic",
    "nsfw",
    "topless",
    "bottomless",
    "breasts",
    "boobs",
    "ass",
    "butt",
    "genitals",
    "penis",
    "vagina",
    "vaginal",
    "anal",
    "oral",
    "blowjob",
    "handjob",
    "fellatio",
    "cum",
    "cumshot",
    "sperm",
    "masturbate",
    "masturbation",
    "orgasm",
    "intercourse",
]


def _compile_phrase_patterns(phrases: List[str]) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    for phrase in phrases:
        text = str(phrase).strip()
        if not text:
            continue
        escaped = re.escape(text).replace(r"\ ", r"\s+")
        patterns.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))
    return patterns


MINOR_PATTERNS = _compile_phrase_patterns(MINOR_PHRASES)
AMBIGUOUS_YOUTH_PATTERNS = _compile_phrase_patterns(AMBIGUOUS_YOUTH_PHRASES)
SEXUAL_PATTERNS = _compile_phrase_patterns(SEXUAL_PHRASES)

AGE_PATTERNS = [
    re.compile(r"\b(?:age|aged)\s*(?:of\s*)?(?:1[0-7]|[0-9])\b", re.IGNORECASE),
    re.compile(r"\b(?:1[0-7]|[0-9])\s*(?:yo|y/o|yrs?|years?\s*old|year[- ]?old)\b", re.IGNORECASE),
    re.compile(r"\b(?:under|below)\s*18\b", re.IGNORECASE),
]


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    if not text:
        return False
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def check_safety(prompt: str, negative_prompt: str = "") -> Optional[str]:
    """Check if prompt contains unsafe content.

    Returns a reason string if blocked, or None if safe.
    """
    combined = " ".join([segment for segment in (prompt, negative_prompt) if segment])
    if _matches_any(combined, MINOR_PATTERNS) or _matches_any(combined, AGE_PATTERNS):
        return "minor_content_detected"
    if _matches_any(combined, SEXUAL_PATTERNS) and _matches_any(combined, AMBIGUOUS_YOUTH_PATTERNS):
        return "ambiguous_age_in_sexual_context"
    return None

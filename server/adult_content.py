"""Adult-content public isolation helpers.

These helpers are intentionally conservative and dependency-light. They mark
obvious adult/NSFW signals so public publication and listing paths can fail
closed until a richer classifier exists.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Optional


ADULT_TERMS = {
    "adult",
    "explicit",
    "erotic",
    "hentai",
    "naked",
    "nude",
    "nudity",
    "nsfw",
    "porn",
    "pornographic",
    "sexual",
    "topless",
}


_WORD_PATTERNS = [
    re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    for term in sorted(ADULT_TERMS)
]


def classify(*values: Any) -> Optional[str]:
    """Return a stable reason when values contain adult/NSFW signals."""
    for value in _flatten(values):
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in ADULT_TERMS:
            return "adult_content_signal"
        if any(pattern.search(text) for pattern in _WORD_PATTERNS):
            return "adult_content_signal"
    return None


def from_job(job: Mapping[str, Any], *extra_values: Any) -> Optional[str]:
    """Classify a job plus route-specific public metadata."""
    values = [
        _get(job, "model"),
        _get(job, "task_type"),
        _get(job, "prompt"),
        _get(job, "negative_prompt"),
        _get(job, "resolved_spec"),
        *extra_values,
    ]
    return classify(*values)


def _get(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row.get(key)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            return row[key]
        except (KeyError, IndexError):
            return None


def _flatten(values: Iterable[Any]) -> Iterable[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            yield from _flatten(value.values())
            continue
        if isinstance(value, (list, tuple, set)):
            yield from _flatten(value)
            continue
        if isinstance(value, str):
            parsed = _try_json(value)
            if parsed is not None:
                yield from _flatten([parsed])
                continue
        yield value


def _try_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

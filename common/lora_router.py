from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(__file__).with_name("lora_routes.json")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE = "sdxl"
DEFAULT_TASK_TYPE = "image"
LORA_DIR_ENV = "HAVNAI_LORA_DIR"
DEFAULT_LORA_DIR = "models/loras"


def load_lora_routes(config_path: Path | None = None) -> Dict[str, Any]:
    path = config_path or CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_lora_dir(config: Dict[str, Any] | None = None) -> Path:
    config = config or load_lora_routes()
    env_name = str(config.get("lora_dir_env") or LORA_DIR_ENV)
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value).expanduser()

    default_dir = str(config.get("default_lora_dir") or DEFAULT_LORA_DIR)
    return (REPO_ROOT / default_dir).resolve()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_filename(value: Any) -> str:
    return Path(str(value or "").strip()).name.lower()


def _normalize_stem(value: Any) -> str:
    filename = _normalize_filename(value)
    return Path(filename).stem.lower() if filename else ""


def _normalize_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.replace("\\", "/").lower()


def _route_matches_prompt(route: Dict[str, Any], prompt: str) -> tuple[bool, Optional[str]]:
    trigger_type = _normalize_text(route.get("trigger_type"))
    prompt_lc = _normalize_text(prompt)

    if trigger_type == "always":
        return True, "default route for matching pipeline"

    if not prompt_lc:
        return False, None

    if trigger_type == "exact_phrase":
        for phrase in route.get("phrases") or []:
            phrase_lc = _normalize_text(phrase)
            if phrase_lc and phrase_lc in prompt_lc:
                return True, f"matched exact phrase: {phrase}"
        return False, None

    if trigger_type == "keyword":
        for keyword in route.get("keywords") or []:
            keyword_lc = _normalize_text(keyword)
            if keyword_lc and keyword_lc in prompt_lc:
                return True, f"matched keyword: {keyword}"
        return False, None

    return False, None


def _normalize_requested_lora(item: Any, resolved_dir: Path) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw = dict(item)
    else:
        name = str(item or "").strip()
        if not name:
            return None
        raw = {"name": name}

    name = str(raw.get("name") or raw.get("filename") or raw.get("id") or "").strip()
    filename = str(raw.get("filename") or Path(name).name or "").strip()
    path = str(raw.get("path") or "").strip()
    if not path and filename:
        path = str((resolved_dir / filename).resolve())

    lora_id = str(raw.get("id") or Path(filename or name).stem or name).strip()
    weight = raw.get("weight")

    normalized = {
        **raw,
        "id": lora_id,
        "filename": filename,
        "path": path,
        "weight": weight,
        "trigger_type": str(raw.get("trigger_type") or "user"),
        "trigger_reason": str(raw.get("trigger_reason") or "user_requested"),
    }
    return normalized


def _candidate_keys(entry: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for value in (
        entry.get("id"),
        entry.get("filename"),
        entry.get("path"),
        entry.get("name"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        keys.add(text.lower())
        keys.add(_normalize_filename(text))
        keys.add(_normalize_stem(text))
        keys.add(_normalize_path(text))
    return {key for key in keys if key}


def route_loras(
    prompt: str,
    pipeline: str = DEFAULT_PIPELINE,
    task_type: str = DEFAULT_TASK_TYPE,
    requested_loras: list | None = None,
) -> Dict[str, Any]:
    config = load_lora_routes()
    warnings: List[str] = []

    resolved_dir = resolve_lora_dir(config)
    selected_loras: List[Dict[str, Any]] = []
    winner_keys: set[str] = set()

    for item in requested_loras or []:
        normalized = _normalize_requested_lora(item, resolved_dir)
        if not normalized:
            continue
        selected_loras.append(normalized)
        winner_keys.update(_candidate_keys(normalized))

    if not bool(config.get("enabled", True)):
        return {"selected_loras": selected_loras, "warnings": warnings}

    if _normalize_text(task_type) != "image":
        return {"selected_loras": selected_loras, "warnings": warnings}

    pipeline_lc = _normalize_text(pipeline)
    for route in config.get("routes") or []:
        if not bool(route.get("enabled", True)):
            continue

        route_pipeline = _normalize_text(route.get("pipeline") or DEFAULT_PIPELINE)
        if route_pipeline and pipeline_lc != route_pipeline:
            continue

        matched, trigger_reason = _route_matches_prompt(route, prompt)
        if not matched:
            continue

        filename = str(route.get("filename") or "").strip()
        if not filename:
            warnings.append(f"Route '{route.get('id', '?')}' is missing filename and was skipped")
            continue

        path = (resolved_dir / filename).resolve()
        if not path.exists():
            warnings.append(f"Configured LoRA file missing for route '{route.get('id', '?')}': {filename}")
            continue

        routed = {
            "id": str(route.get("id") or Path(filename).stem),
            "filename": filename,
            "path": str(path),
            "weight": route.get("weight"),
            "trigger_type": str(route.get("trigger_type") or ""),
            "trigger_reason": trigger_reason or "matched route",
            "name": Path(filename).stem,
        }

        duplicate_keys = _candidate_keys(routed)
        if winner_keys.intersection(duplicate_keys):
            warnings.append(
                f"User-provided LoRA won over routed duplicate for route '{routed['id']}'"
            )
            continue

        selected_loras.append(routed)
        winner_keys.update(duplicate_keys)

    return {"selected_loras": selected_loras, "warnings": warnings}

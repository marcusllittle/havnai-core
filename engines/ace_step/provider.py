"""HTTP provider for the supported ACE-Step 1.5 asynchronous API."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


class AceStepError(RuntimeError):
    """Raised when the ACE-Step service rejects or fails a generation."""


class AceStepCancelled(AceStepError):
    """Raised when HavnAI cancels an in-flight generation."""


@dataclass(frozen=True)
class AceStepResult:
    path: Path
    content_type: str
    metadata: Dict[str, Any]


class AceStepProvider:
    """Small adapter around ACE-Step's release/query/audio endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        request_timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("ACESTEP_API_URL") or "http://localhost:8001"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("ACESTEP_API_KEY", "")
        self.session = session or requests.Session()
        self.request_timeout = max(1.0, float(request_timeout))
        self.poll_interval = max(0.1, float(poll_interval))

    @property
    def headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _unwrap(response: requests.Response) -> Any:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise AceStepError("ACE-Step returned an invalid response")
        if int(payload.get("code", 200)) != 200 or payload.get("error"):
            raise AceStepError(str(payload.get("error") or "ACE-Step request failed"))
        return payload.get("data")

    def probe(self) -> Dict[str, Any]:
        health = self._unwrap(
            self.session.get(self._url("/health"), headers=self.headers, timeout=5)
        )
        try:
            models = self._unwrap(
                self.session.get(self._url("/v1/model_inventory"), headers=self.headers, timeout=5)
            )
        except (requests.RequestException, AceStepError, ValueError):
            models = self._unwrap(
                self.session.get(self._url("/v1/models"), headers=self.headers, timeout=5)
            )
        model_rows = models.get("models", []) if isinstance(models, dict) else models
        names = [
            str(item.get("name") or item.get("id"))
            for item in model_rows or []
            if isinstance(item, dict) and (item.get("name") or item.get("id"))
        ]
        default_model = models.get("default_model") if isinstance(models, dict) else None
        if not default_model and isinstance(health, dict):
            default_model = health.get("loaded_model")
        if default_model and str(default_model) not in names:
            names.append(str(default_model))
        return {
            "service": (health or {}).get("service") if isinstance(health, dict) else "ACE-Step API",
            "version": (health or {}).get("version") if isinstance(health, dict) else None,
            "models": names,
            "default_model": default_model,
        }

    @staticmethod
    def _metadata_value(value: Any, fallback: Any = None) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "n/a", "none", "null"}:
            return fallback
        return value if value is not None else fallback

    @staticmethod
    def _generation_payload(settings: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(settings.get("prompt") or "").strip()
        style = str(settings.get("style") or "").strip()
        instrumental = bool(settings.get("instrumental"))
        if style:
            prompt = f"{prompt}. Style: {style}"
        if instrumental:
            prompt = f"{prompt}. Instrumental only, no vocals."
        payload: Dict[str, Any] = {
            "task_type": "text2music",
            "prompt": prompt,
            "lyrics": "" if instrumental else str(settings.get("lyrics") or ""),
            "audio_duration": float(settings["duration"]),
            "audio_format": str(settings.get("audio_format") or "mp3"),
            "batch_size": 1,
            "thinking": bool(settings.get("thinking", False)),
            "model": str(settings.get("engine_model") or "acestep-v15-turbo"),
        }
        for source, target in (("bpm", "bpm"), ("key", "key_scale")):
            if settings.get(source) not in (None, ""):
                payload[target] = settings[source]
        if settings.get("seed") not in (None, ""):
            payload["seed"] = int(settings["seed"])
            payload["use_random_seed"] = False
        return payload

    def generate(
        self,
        settings: Dict[str, Any],
        output_dir: Path,
        *,
        progress: Optional[ProgressCallback] = None,
        cancelled: Optional[CancelCallback] = None,
        timeout: float = 900.0,
    ) -> AceStepResult:
        notify = progress or (lambda _value, _stage: None)
        is_cancelled = cancelled or (lambda: False)
        notify(5, "preparing")
        data = self._unwrap(
            self.session.post(
                self._url("/release_task"),
                json=self._generation_payload(settings),
                headers=self.headers,
                timeout=self.request_timeout,
            )
        )
        task_id = str((data or {}).get("task_id") or "") if isinstance(data, dict) else ""
        if not task_id:
            raise AceStepError("ACE-Step did not return a task id")

        started = time.monotonic()
        result_item: Optional[Dict[str, Any]] = None
        while result_item is None:
            if is_cancelled():
                raise AceStepCancelled("cancelled_by_user")
            elapsed = time.monotonic() - started
            if elapsed > max(1.0, float(timeout)):
                raise AceStepError("ACE-Step generation timed out")
            query = self._unwrap(
                self.session.post(
                    self._url("/query_result"),
                    json={"task_id_list": [task_id]},
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
            )
            row = query[0] if isinstance(query, list) and query else {}
            status = int(row.get("status", 0)) if isinstance(row, dict) else 0
            if status == 2:
                raise AceStepError(str(row.get("error") or row.get("result") or "ACE-Step generation failed"))
            if status == 1:
                raw_result = row.get("result")
                try:
                    parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise AceStepError("ACE-Step returned invalid result metadata") from exc
                if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
                    raise AceStepError("ACE-Step returned no audio result")
                result_item = parsed[0]
                break
            notify(min(88.0, 12.0 + elapsed / max(float(timeout), 1.0) * 76.0), "creating")
            time.sleep(self.poll_interval)

        audio_ref = str(result_item.get("file") or "").strip()
        if not audio_ref:
            raise AceStepError("ACE-Step result is missing audio")
        notify(90, "finishing")
        audio_url = urljoin(f"{self.base_url}/", audio_ref)
        response = self.session.get(
            audio_url,
            headers=self.headers,
            stream=True,
            timeout=max(self.request_timeout, 120.0),
        )
        response.raise_for_status()
        content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
        suffix = Path(urlparse(audio_url).path).suffix.lower()
        if not suffix:
            suffix = mimetypes.guess_extension(content_type) or ".mp3"
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"music{suffix}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if is_cancelled():
                        raise AceStepCancelled("cancelled_by_user")
                    if chunk:
                        handle.write(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        metas = result_item.get("metas") if isinstance(result_item.get("metas"), dict) else {}
        metadata = {
            "duration": self._metadata_value(metas.get("duration"), settings.get("duration")),
            "bpm": self._metadata_value(metas.get("bpm"), settings.get("bpm")),
            "key": self._metadata_value(metas.get("keyscale"), settings.get("key")),
            "seed": result_item.get("seed_value", settings.get("seed")),
            "model": result_item.get("dit_model", settings.get("engine_model")),
            "prompt": result_item.get("prompt", settings.get("prompt")),
            "style": settings.get("style"),
            "instrumental": bool(settings.get("instrumental")),
            "lyrics": result_item.get("lyrics", settings.get("lyrics")),
            "ace_step_task_id": task_id,
            "created_at": result_item.get("create_time"),
        }
        return AceStepResult(
            path=destination,
            content_type=content_type or mimetypes.guess_type(destination.name)[0] or "audio/mpeg",
            metadata={key: value for key, value in metadata.items() if value not in (None, "")},
        )


def runtime_probe() -> tuple[bool, Dict[str, Any]]:
    try:
        return True, AceStepProvider().probe()
    except Exception as exc:
        return False, {"error": str(exc)}

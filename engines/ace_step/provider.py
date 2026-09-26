"""HTTP provider for the supported ACE-Step 1.5 asynchronous API.

Covers the full task surface exposed by ``acestep-api``:

``text2music``  caption + lyrics -> song
``cover``       restyle an existing track, keeping its structure
``repaint``     regenerate one time range of an existing track
``extract``     isolate a single named stem from a mix        (base model only)
``lego``        generate one named stem over existing audio   (base model only)
``complete``    arrange missing tracks into a partial mix     (base model only)

A single call may return several takes (``batch_size``), so :meth:`AceStepProvider.generate`
returns a list of :class:`AceStepResult` ordered as the service returned them.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

#: Every task the upstream engine understands (``acestep/constants.py``).
TASK_TYPES: Tuple[str, ...] = (
    "text2music",
    "repaint",
    "cover",
    "cover-nofsq",
    "extract",
    "lego",
    "complete",
)

#: Turbo checkpoints only implement a subset; base/sft implement everything.
TURBO_TASK_TYPES: Tuple[str, ...] = ("text2music", "repaint", "cover", "cover-nofsq")

#: Named stems usable with ``extract`` / ``lego`` / ``complete``.
TRACK_NAMES: Tuple[str, ...] = (
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
)

#: Tasks that require a source/context audio file to be supplied.
TASKS_REQUIRING_SOURCE: Tuple[str, ...] = (
    "repaint",
    "cover",
    "cover-nofsq",
    "extract",
    "lego",
    "complete",
)

#: Tasks that operate on a single named stem.
TASKS_REQUIRING_TRACK: Tuple[str, ...] = ("extract", "lego")

AUDIO_FORMATS: Tuple[str, ...] = ("mp3", "flac", "wav", "wav32", "opus", "aac")

REPAINT_MODES: Tuple[str, ...] = ("conservative", "balanced", "aggressive")

MAX_BATCH_SIZE = 8

_STATUS_PENDING = 0
_STATUS_SUCCESS = 1
_STATUS_FAILED = 2


class AceStepError(RuntimeError):
    """Raised when the ACE-Step service rejects or fails a generation."""


class AceStepCancelled(AceStepError):
    """Raised when HavnAI cancels an in-flight generation."""


class AceStepCapabilityError(AceStepError):
    """Raised when the requested task is not supported by the selected checkpoint."""


class AceStepModelMismatch(AceStepError):
    """Raised when the service generated with a different checkpoint than requested.

    ACE-Step does not fail a job whose ``model`` is unknown or not loaded into a
    slot — it logs, falls back to its primary checkpoint and returns HTTP 200.
    For HavnAI that is worse than an error: the audio is attributed, rewarded and
    published against a model that never ran. The only signal is ``dit_model`` on
    the result, so we check it on every take.
    """


@dataclass(frozen=True)
class AceStepResult:
    path: Path
    content_type: str
    metadata: Dict[str, Any]
    index: int = 0
    is_primary: bool = True


@dataclass(frozen=True)
class AceStepModel:
    """One DiT checkpoint the service has on disk."""

    name: str
    is_default: bool = False
    is_loaded: bool = False
    supported_task_types: Tuple[str, ...] = field(default_factory=tuple)

    def supports(self, task_type: str) -> bool:
        if not self.supported_task_types:
            # Inventory did not tell us; assume the conservative turbo subset.
            return task_type in TURBO_TASK_TYPES
        return task_type in self.supported_task_types


def _coerce_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class AceStepProvider:
    """Adapter around ACE-Step's release/query/audio endpoints."""

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
        self.on_demand = str(
            os.getenv("HAVNAI_ACESTEP_ON_DEMAND", "")
        ).strip().lower() in {"1", "true", "yes"}

    # ------------------------------------------------------------------ helpers

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

    @staticmethod
    def _metadata_value(value: Any, fallback: Any = None) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "n/a", "none", "null"}:
            return fallback
        return value if value is not None else fallback

    # ------------------------------------------------------------------- probing

    def inventory(self) -> List[AceStepModel]:
        """Return the checkpoints the service can serve, with their task types."""
        try:
            data = self._unwrap(
                self.session.get(
                    self._url("/v1/model_inventory"), headers=self.headers, timeout=5
                )
            )
        except (requests.RequestException, AceStepError, ValueError):
            data = self._unwrap(
                self.session.get(self._url("/v1/models"), headers=self.headers, timeout=5)
            )
        rows = data.get("models", []) if isinstance(data, dict) else data
        default_model = data.get("default_model") if isinstance(data, dict) else None
        models: List[AceStepModel] = []
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("id") or "").strip()
            if not name:
                continue
            declared = item.get("supported_task_types")
            supported = tuple(
                str(value) for value in declared if str(value) in TASK_TYPES
            ) if isinstance(declared, (list, tuple)) else ()
            models.append(
                AceStepModel(
                    name=name,
                    is_default=bool(item.get("is_default")) or name == default_model,
                    is_loaded=bool(item.get("is_loaded")),
                    supported_task_types=supported,
                )
            )
        return models

    def probe(self) -> Dict[str, Any]:
        health = self._unwrap(
            self.session.get(self._url("/health"), headers=self.headers, timeout=5)
        )
        health = health if isinstance(health, dict) else {}
        models = self.inventory()
        names = [model.name for model in models]
        default_model = next((model.name for model in models if model.is_default), None)
        if not default_model:
            default_model = health.get("loaded_model")
        if default_model and str(default_model) not in names:
            names.append(str(default_model))

        # The inventory is disk-driven: it lists every acestep-* directory under
        # the checkpoints dir, loaded or not. Only a LOADED checkpoint is actually
        # servable — asking for an unloaded one silently returns primary-model
        # audio at HTTP 200 (see _verify_engine_model), so the two lists are kept
        # apart and callers gate on `loaded_models`.
        loaded = [model.name for model in models if model.is_loaded]
        health_loaded_model = str(health.get("loaded_model") or "").strip()
        if (
            health.get("models_initialized") is not False
            and health_loaded_model
            and health_loaded_model not in loaded
        ):
            loaded.append(health_loaded_model)

        # When the service runs with ACESTEP_ON_DEMAND_MODEL_LOAD=true it swaps its
        # primary slot to whatever `model` asks for, so an on-disk checkpoint IS
        # servable even though it is not resident yet. The operator opts in with
        # HAVNAI_ACESTEP_ON_DEMAND=1 on the node to match their service config; the
        # dit_model check on every result stays as the backstop either way.
        if self.on_demand:
            loaded = list(dict.fromkeys(loaded + names))

        capabilities: set[str] = set()
        for model in models:
            if model.name not in loaded:
                continue
            capabilities.update(model.supported_task_types or TURBO_TASK_TYPES)


        return {
            "service": health.get("service") or "ACE-Step API",
            "version": health.get("version"),
            "models": names,
            "loaded_models": loaded,
            "default_model": default_model,
            "loaded_model": health.get("loaded_model"),
            "loaded_lm_model": health.get("loaded_lm_model"),
            "llm_initialized": bool(health.get("llm_initialized")),
            "on_demand": self.on_demand,
            "task_types": sorted(capabilities),
            "model_task_types": {
                model.name: list(model.supported_task_types)
                for model in models
                if model.name in loaded
            },
        }

    # ------------------------------------------------------------------ payloads

    @staticmethod
    def _compose_caption(settings: Mapping[str, Any]) -> str:
        prompt = str(settings.get("prompt") or "").strip()
        style = str(settings.get("style") or "").strip()
        if style:
            prompt = f"{prompt}. Style: {style}" if prompt else f"Style: {style}"
        if bool(settings.get("instrumental")):
            prompt = f"{prompt}. Instrumental only, no vocals." if prompt else "Instrumental only, no vocals."
        return prompt

    @classmethod
    def _generation_payload(cls, settings: Mapping[str, Any]) -> Dict[str, Any]:
        task_type = str(settings.get("task_type") or "text2music").strip().lower()
        if task_type not in TASK_TYPES:
            raise AceStepCapabilityError(f"unsupported_task_type:{task_type}")

        instrumental = bool(settings.get("instrumental"))
        audio_format = str(settings.get("audio_format") or "mp3").lower()
        if audio_format not in AUDIO_FORMATS:
            audio_format = "mp3"

        batch_size = _coerce_int(settings.get("batch_size"), 1) or 1
        batch_size = max(1, min(MAX_BATCH_SIZE, batch_size))

        payload: Dict[str, Any] = {
            "task_type": task_type,
            "prompt": cls._compose_caption(settings),
            "lyrics": "" if instrumental else str(settings.get("lyrics") or ""),
            "audio_duration": float(settings["duration"]),
            "audio_format": audio_format,
            "batch_size": batch_size,
            "thinking": bool(settings.get("thinking", False)),
            "model": str(settings.get("engine_model") or "acestep-v15-turbo"),
        }

        # --- musical metadata -------------------------------------------------
        for source, target in (
            ("bpm", "bpm"),
            ("key", "key_scale"),
            ("time_signature", "time_signature"),
            ("vocal_language", "vocal_language"),
        ):
            if settings.get(source) not in (None, ""):
                payload[target] = settings[source]

        # --- sampler ----------------------------------------------------------
        steps = _coerce_int(settings.get("inference_steps"))
        if steps:
            payload["inference_steps"] = max(1, min(200, steps))
        guidance = _coerce_float(settings.get("guidance_scale"))
        if guidance is not None:
            payload["guidance_scale"] = guidance
        shift = _coerce_float(settings.get("shift"))
        if shift is not None:
            payload["shift"] = shift
        infer_method = str(settings.get("infer_method") or "").strip().lower()
        if infer_method in {"ode", "sde"}:
            payload["infer_method"] = infer_method

        # --- seeds ------------------------------------------------------------
        seed = settings.get("seed")
        if seed not in (None, ""):
            payload["seed"] = seed if isinstance(seed, str) else int(seed)
            payload["use_random_seed"] = False
        else:
            payload["use_random_seed"] = True

        # --- edit / conditioning ---------------------------------------------
        if task_type in {"cover", "cover-nofsq"}:
            strength = _coerce_float(settings.get("audio_cover_strength"))
            if strength is not None:
                payload["audio_cover_strength"] = max(0.0, min(1.0, strength))
            noise = _coerce_float(settings.get("cover_noise_strength"))
            if noise is not None:
                payload["cover_noise_strength"] = max(0.0, min(1.0, noise))

        if task_type == "repaint":
            start = _coerce_float(settings.get("repainting_start"), 0.0)
            end = _coerce_float(settings.get("repainting_end"), -1.0)
            payload["repainting_start"] = start
            payload["repainting_end"] = end
            payload["chunk_mask_mode"] = "explicit"
            mode = str(settings.get("repaint_mode") or "balanced").lower()
            payload["repaint_mode"] = mode if mode in REPAINT_MODES else "balanced"
            if payload["repaint_mode"] == "balanced":
                strength = _coerce_float(settings.get("repaint_strength"), 0.5)
                payload["repaint_strength"] = max(0.0, min(1.0, strength or 0.5))
            crossfade = _coerce_float(settings.get("repaint_wav_crossfade_sec"))
            if crossfade is not None:
                payload["repaint_wav_crossfade_sec"] = max(0.0, crossfade)

        if task_type in TASKS_REQUIRING_TRACK:
            track = str(settings.get("track_name") or "").strip().lower()
            if track not in TRACK_NAMES:
                raise AceStepCapabilityError(f"invalid_track_name:{track or 'missing'}")
            payload["track_name"] = track

        if task_type == "complete":
            classes = settings.get("track_classes") or []
            tracks = [
                str(value).strip().lower()
                for value in classes
                if str(value).strip().lower() in TRACK_NAMES
            ]
            if not tracks:
                raise AceStepCapabilityError("invalid_track_classes")
            payload["track_classes"] = tracks

        if settings.get("global_caption"):
            payload["global_caption"] = str(settings["global_caption"])
        if settings.get("instruction"):
            payload["instruction"] = str(settings["instruction"])
        if settings.get("audio_code_string"):
            payload["audio_code_string"] = str(settings["audio_code_string"])

        return payload

    @staticmethod
    def _multipart_files(settings: Mapping[str, Any]) -> Dict[str, Tuple[str, Any, str]]:
        """Open src/reference audio for multipart upload, when local paths are given."""
        files: Dict[str, Tuple[str, Any, str]] = {}
        for key, field_name in (("src_audio_path", "src_audio"), ("reference_audio_path", "ref_audio")):
            raw = settings.get(key)
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_file():
                raise AceStepError(f"missing_audio_input:{field_name}")
            content_type = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
            files[field_name] = (path.name, path.open("rb"), content_type)
        return files

    # --------------------------------------------------------------- validation

    def validate(self, settings: Mapping[str, Any], *, models: Optional[Sequence[AceStepModel]] = None) -> None:
        """Raise if the requested task cannot run on the selected checkpoint."""
        task_type = str(settings.get("task_type") or "text2music").strip().lower()
        if task_type not in TASK_TYPES:
            raise AceStepCapabilityError(f"unsupported_task_type:{task_type}")
        if task_type in TASKS_REQUIRING_SOURCE and not settings.get("src_audio_path"):
            raise AceStepCapabilityError(f"source_audio_required:{task_type}")
        engine_model = str(settings.get("engine_model") or "").strip()
        if not engine_model:
            return
        inventory = list(models) if models is not None else self.inventory()
        entry = next((item for item in inventory if item.name == engine_model), None)
        if entry and not entry.supports(task_type):
            raise AceStepCapabilityError(
                f"task_unsupported_by_model:{task_type}:{engine_model}"
            )
        # Present in the inventory only means "downloaded". A checkpoint that is
        # not loaded into a slot is not selectable: the service would fall back to
        # its primary model and return that audio as if we had asked for it.
        if entry and not entry.is_loaded and not entry.is_default and not self.on_demand:
            raise AceStepCapabilityError(f"model_not_loaded:{engine_model}")

    # --------------------------------------------------------------- generation

    def generate(
        self,
        settings: Dict[str, Any],
        output_dir: Path,
        *,
        progress: Optional[ProgressCallback] = None,
        cancelled: Optional[CancelCallback] = None,
        timeout: float = 900.0,
    ) -> List[AceStepResult]:
        notify = progress or (lambda _value, _stage: None)
        is_cancelled = cancelled or (lambda: False)
        notify(5, "preparing")

        payload = self._generation_payload(settings)
        files = self._multipart_files(settings)
        try:
            if files:
                response = self.session.post(
                    self._url("/release_task"),
                    data={key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in payload.items()},
                    files=files,
                    headers=self.headers,
                    timeout=max(self.request_timeout, 120.0),
                )
            else:
                response = self.session.post(
                    self._url("/release_task"),
                    json=payload,
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
            data = self._unwrap(response)
        finally:
            for _name, handle, _ctype in files.values():
                try:
                    handle.close()
                except Exception:  # pragma: no cover - best effort
                    pass

        task_id = str((data or {}).get("task_id") or "") if isinstance(data, dict) else ""
        if not task_id:
            raise AceStepError("ACE-Step did not return a task id")

        items = self._await_results(task_id, notify, is_cancelled, timeout)
        self._verify_engine_model(items, settings)
        notify(90, "finishing")

        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[AceStepResult] = []
        for index, item in enumerate(items):
            results.append(
                self._download_result(
                    item,
                    output_dir,
                    settings,
                    task_id,
                    index,
                    is_cancelled,
                )
            )
        if not results:
            raise AceStepError("ACE-Step returned no audio result")
        return results

    @staticmethod
    def _verify_engine_model(items: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> None:
        """Reject a result the service produced with a different checkpoint."""
        requested = str(settings.get("engine_model") or "").strip()
        if not requested:
            return
        for item in items:
            served = str(item.get("dit_model") or "").strip()
            # Without dit_model, the requested checkpoint cannot be verified.
            if not served:
                raise AceStepModelMismatch(f"model_unverified:requested={requested}")
            if served != requested:
                raise AceStepModelMismatch(
                    f"model_fallback:requested={requested}:served={served}"
                )

    def _await_results(
        self,
        task_id: str,
        notify: ProgressCallback,
        is_cancelled: CancelCallback,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        started = time.monotonic()
        while True:
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
            status = _coerce_int(row.get("status"), _STATUS_PENDING) if isinstance(row, dict) else _STATUS_PENDING
            if status == _STATUS_FAILED:
                raise AceStepError(
                    str(row.get("error") or row.get("result") or "ACE-Step generation failed")
                )
            if status == _STATUS_SUCCESS:
                raw_result = row.get("result")
                try:
                    parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise AceStepError("ACE-Step returned invalid result metadata") from exc
                if not isinstance(parsed, list) or not parsed:
                    raise AceStepError("ACE-Step returned no audio result")
                return [item for item in parsed if isinstance(item, dict)]
            notify(min(88.0, 12.0 + elapsed / max(float(timeout), 1.0) * 76.0), "creating")
            time.sleep(self.poll_interval)

    def _download_result(
        self,
        result_item: Mapping[str, Any],
        output_dir: Path,
        settings: Mapping[str, Any],
        task_id: str,
        index: int,
        is_cancelled: CancelCallback,
    ) -> AceStepResult:
        audio_ref = str(result_item.get("file") or "").strip()
        if not audio_ref:
            raise AceStepError("ACE-Step result is missing audio")
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

        stem = "music" if index == 0 else f"music-{index + 1}"
        destination = output_dir / f"{stem}{suffix}"
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
            "time_signature": self._metadata_value(metas.get("timesignature"), settings.get("time_signature")),
            "genres": self._metadata_value(metas.get("genres")),
            "seed": result_item.get("seed_value", settings.get("seed")),
            "model": result_item.get("dit_model", settings.get("engine_model")),
            "lm_model": result_item.get("lm_model"),
            "prompt": result_item.get("prompt", settings.get("prompt")),
            "style": settings.get("style"),
            "instrumental": bool(settings.get("instrumental")),
            "lyrics": result_item.get("lyrics", settings.get("lyrics")),
            "task_type": settings.get("task_type") or "text2music",
            "track_name": settings.get("track_name"),
            "ace_step_task_id": task_id,
            "variation": index + 1,
            "created_at": result_item.get("create_time"),
        }
        return AceStepResult(
            path=destination,
            content_type=content_type or mimetypes.guess_type(destination.name)[0] or "audio/mpeg",
            metadata={key: value for key, value in metadata.items() if value not in (None, "")},
            index=index,
            is_primary=index == 0,
        )


def runtime_probe() -> tuple[bool, Dict[str, Any]]:
    try:
        return True, AceStepProvider().probe()
    except Exception as exc:
        return False, {"error": str(exc)}

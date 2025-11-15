from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

try:  # pragma: no cover - shared module lives in sibling repo
    from shared.schemas import Manifest, ModelEntry
except Exception:  # pragma: no cover

    @dataclass
    class ModelEntry:  # type: ignore
        name: str
        path: str
        pipeline: str
        type: str
        tags: List[str] = field(default_factory=list)
        reward_weight: float = 0.0
        task_type: str = "IMAGE_GEN"

    @dataclass
    class Manifest:  # type: ignore
        models: List[ModelEntry] = field(default_factory=list)


class ManifestError(RuntimeError):
    """Raised when manifest retrieval or validation failed."""


class ModelRegistry:
    """Manifest-driven registry fetched from the coordinator."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        endpoint: str = "/models/list",
    ) -> None:
        self.base_url = (base_url or os.environ.get("COORDINATOR_URL") or os.environ.get("SERVER_URL") or "http://127.0.0.1:5001").rstrip("/")
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self._lock = threading.Lock()
        self._manifest: Optional[Manifest] = None
        self._models: Dict[str, ModelEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> Manifest:
        """Download and cache the current manifest from the coordinator."""

        url = f"{self.base_url}{self.endpoint}"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        manifest = self._parse_manifest(payload)
        with self._lock:
            self._manifest = manifest
            self._models = {entry.name.lower(): entry for entry in manifest.models}
        return manifest

    def get(self, name: str) -> ModelEntry:
        """Return the manifest entry by name (case-insensitive)."""

        key = (name or "").lower()
        with self._lock:
            if not self._models:
                raise ManifestError("Model registry has not been refreshed yet")
            entry = self._models.get(key)
        if not entry:
            raise KeyError(f"Model '{name}' not found in manifest")
        return entry

    def list_types(self) -> List[str]:
        """Return the unique pipeline types available on this node."""

        with self._lock:
            values = {entry.pipeline for entry in self._models.values() if entry.pipeline}
        return sorted(values)

    def list_entries(self) -> List[ModelEntry]:
        """Return all manifest entries currently cached."""

        with self._lock:
            return list(self._models.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_manifest(payload: Dict[str, object]) -> Manifest:
        try:
            models_data = payload.get("models", [])  # type: ignore[assignment]
            if not isinstance(models_data, list):
                raise TypeError("models field must be a list")
            allowed = ModelEntry.__dataclass_fields__.keys()  # type: ignore[attr-defined]
            normalized = []
            for model in models_data:
                if not isinstance(model, dict):
                    continue
                filtered = {key: value for key, value in model.items() if key in allowed}
                normalized.append(ModelEntry(**filtered))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            raise ManifestError(f"Invalid manifest structure: {exc}") from exc
        return Manifest(models=normalized)


REGISTRY = ModelRegistry()


__all__ = [
    "ModelRegistry",
    "ModelEntry",
    "Manifest",
    "ManifestError",
    "REGISTRY",
]

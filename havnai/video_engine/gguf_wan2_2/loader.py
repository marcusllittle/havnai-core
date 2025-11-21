"""WAN 2.2 GGUF loader."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

try:  # Optional; only used when llama-cpp is installed alongside CUDA
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore

from .model_registry import EnginePaths


class WanGGUFLoader:
    """Loads the Rapid WAN 2.2 GGUF checkpoint and exposes a tiny inference hook."""

    def __init__(self, paths: EnginePaths) -> None:
        self.paths = paths
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self.backend: str = ""
        self.mock_mode: bool = False
        self.load_error: Optional[str] = None

    def ensure_loaded(self) -> None:
        """Attempt to load the GGUF model; fall back to mock mode if unavailable."""

        if self._model or self.mock_mode:
            return
        if self.load_error:
            return
        if not self.paths.gguf_path.exists():
            self.load_error = f"GGUF model missing at {self.paths.gguf_path}"
            self.mock_mode = True
            return

        with self._lock:
            if self._model or self.mock_mode:
                return
            if Llama is None:
                self.load_error = "llama-cpp-python not installed; running in mock mode"
                self.mock_mode = True
                return
            try:
                # Light-weight init; actual inference remains upstream specific.
                self._model = Llama(
                    model_path=str(self.paths.gguf_path),
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    logits_all=False,
                    embedding=False,
                )
                self.backend = "llama_cpp"
            except Exception as exc:  # pragma: no cover - runtime specific
                self.load_error = f"Failed to load GGUF: {exc}"
                self.mock_mode = True

    def can_generate(self) -> bool:
        """True when a real backend is available."""

        return self._model is not None and not self.mock_mode

    def generate_tokens(self, prompt: str, max_tokens: int = 64) -> str:
        """Tiny helper used by mock pipelines; not a full video decode."""

        if not self._model:
            raise RuntimeError(self.load_error or "WAN GGUF model not loaded")
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            temperature=0.8,
        )
        return "".join(chunk["choices"][0]["text"] for chunk in output)

    def describe(self) -> dict:
        return {
            "backend": self.backend or ("mock" if self.mock_mode else ""),
            "path": str(self.paths.gguf_path),
            "load_error": self.load_error,
        }

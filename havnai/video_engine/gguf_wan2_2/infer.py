"""Top-level orchestration for WAN 2.2 GGUF video generation."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import WanGGUFLoader
from .lora_loader import LoRALoader
from .model_registry import ModelRegistry
from .motion_loader import MotionLoader
from .pipeline_i2v import ImageToVideoPipeline
from .pipeline_t2v import TextToVideoPipeline
from .video_writer import VideoWriter


@dataclass
class VideoJobRequest:
    prompt: str
    negative_prompt: str = ""
    motion_type: str = "high"
    lora_list: List[str] = field(default_factory=list)
    init_image_b64: Optional[str] = None
    duration: float = 4.0
    fps: int = 24
    width: int = 720
    height: int = 512
    job_id: Optional[str] = None


@dataclass
class VideoJobResult:
    job_id: str
    status: str
    video_path: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VideoEngine:
    """Unified interface for T2V + I2V WAN flows."""

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.registry = ModelRegistry(base_path=base_path)
        self.loader = WanGGUFLoader(self.registry.paths)
        self.lora_loader = LoRALoader(self.registry)
        self.motion_loader = MotionLoader(self.registry)
        self.writer = VideoWriter(self.registry.paths)
        self.t2v = TextToVideoPipeline(self.registry.paths, self.loader, self.lora_loader, self.motion_loader, self.writer)
        self.i2v = ImageToVideoPipeline(self.registry.paths, self.loader, self.lora_loader, self.motion_loader, self.writer)
        self.inputs_dir = self.registry.paths.output_dir / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: VideoJobRequest) -> VideoJobResult:
        job_id = request.job_id or f"wan-gguf-{uuid.uuid4().hex[:8]}"
        prompt = (request.prompt or "").strip()
        if not prompt:
            return VideoJobResult(job_id=job_id, status="failed", error="Prompt is required")

        fps = max(1, min(int(request.fps or 24), 60))
        duration = max(0.5, float(request.duration or 4.0))
        width = max(64, int(request.width or 720))
        height = max(64, int(request.height or 512))

        init_image_path: Optional[Path] = None
        if request.init_image_b64:
            try:
                init_image_path = self._save_init_image(job_id, request.init_image_b64)
                if init_image_path.exists() and (width == 720 and height == 512):
                    try:
                        from PIL import Image  # type: ignore

                        with Image.open(init_image_path) as img:
                            width, height = img.size
                    except Exception:
                        pass
            except Exception as exc:
                return VideoJobResult(job_id=job_id, status="failed", error=f"Init image decode failed: {exc}")

        try:
            if init_image_path:
                payload = self.i2v.generate(
                    job_id=job_id,
                    prompt=prompt,
                    negative_prompt=request.negative_prompt,
                    motion_type=request.motion_type,
                    lora_list=request.lora_list,
                    duration=duration,
                    fps=fps,
                    init_image=init_image_path,
                    width=width,
                    height=height,
                )
            else:
                payload = self.t2v.generate(
                    job_id=job_id,
                    prompt=prompt,
                    negative_prompt=request.negative_prompt,
                    motion_type=request.motion_type,
                    lora_list=request.lora_list,
                    duration=duration,
                    fps=fps,
                    width=width,
                    height=height,
                )
        except Exception as exc:  # pragma: no cover - runtime path
            return VideoJobResult(job_id=job_id, status="failed", error=str(exc))

        return VideoJobResult(
            job_id=job_id,
            status="success",
            video_path=payload.get("video_path"),
            frame_paths=payload.get("frame_paths", []),
            metadata=payload,
        )

    def _save_init_image(self, job_id: str, image_b64: str) -> Path:
        cleaned = image_b64
        if "," in cleaned and cleaned.startswith("data:"):
            cleaned = cleaned.split(",", 1)[1]
        img_bytes = base64.b64decode(cleaned)
        path = self.inputs_dir / f"{job_id}_init.png"
        with path.open("wb") as fh:
            fh.write(img_bytes)
        return path

    def describe(self) -> dict:
        ready, missing = self.registry.required_assets_available()
        return {
            "paths": {
                "root": str(self.registry.paths.root),
                "gguf": str(self.registry.paths.gguf_path),
                "loras": str(self.registry.paths.loras_dir),
                "motion": str(self.registry.paths.motion_dir),
                "creator": str(self.registry.paths.creator_dir),
                "output": str(self.registry.paths.output_dir),
            },
            "loras": list(self.lora_loader.available().keys()),
            "motion_modules": self.motion_loader.available(),
            "sd15_bases": list(self.registry.sd15_bases().keys()),
            "gguf": self.loader.describe(),
            "ready": ready,
            "missing": missing,
            "using_fallback": getattr(self.registry, "using_fallback", False),
        }

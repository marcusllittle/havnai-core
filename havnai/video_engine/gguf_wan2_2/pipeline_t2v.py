"""Text-to-video pipeline, ComfyUI-style but pure Python."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .loader import WanGGUFLoader
from .lora_loader import LoRALoader
from .model_registry import EnginePaths
from .motion_loader import MotionLoader
from .video_writer import VideoWriter


class BaseWanPipeline:
    """Shared helpers for WAN pipelines (T2V / I2V)."""

    def __init__(
        self,
        paths: EnginePaths,
        loader: WanGGUFLoader,
        lora_loader: LoRALoader,
        motion_loader: MotionLoader,
        writer: VideoWriter,
    ) -> None:
        self.paths = paths
        self.loader = loader
        self.lora_loader = lora_loader
        self.motion_loader = motion_loader
        self.writer = writer

    @staticmethod
    def _coerce_frames(duration: float, fps: int) -> int:
        frames = max(1, int(duration * fps))
        # WAN prefers 4n+1 frame counts
        if frames % 4 != 1:
            frames = (frames // 4) * 4 + 1
        return frames

    def _mock_frames(
        self,
        num_frames: int,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        init_image: Optional[Path] = None,
    ) -> List[object]:
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception:  # pragma: no cover
            # Pillow unavailable; fall back to minimal PNG bytes to keep pipeline alive.
            import base64

            placeholder = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AArgB9z2slZkAAAAASUVORK5CYII="
            )
            return [placeholder for _ in range(num_frames)]

        base_image: Optional[Image.Image] = None  # type: ignore[attr-defined]
        if init_image and init_image.exists():
            try:
                base_image = Image.open(init_image).convert("RGB")
            except Exception:
                base_image = None

        frames: List[object] = []
        for idx in range(num_frames):
            canvas = (
                base_image.copy()
                if base_image
                else Image.new("RGB", (width, height), color=(12 + idx % 10, 12 + idx * 2 % 32, 18 + idx % 48))
            )
            draw = ImageDraw.Draw(canvas)
            text = f"WAN2.2 GGUF {'I2V' if init_image else 'T2V'}\n{prompt[:60]}"
            if negative_prompt:
                text += f"\n- {negative_prompt[:40]}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None  # type: ignore
            draw.text((24, 24 + idx % 12), text, fill=(235, 235, 235), font=font, align="left")
            frames.append(canvas)
        return frames

    def _apply_loras(self, pipeline: object, lora_list: Iterable[Path]) -> None:
        if not lora_list:
            return
        self.lora_loader.attach(pipeline, lora_list)


class TextToVideoPipeline(BaseWanPipeline):
    """Composition-friendly T2V pipeline."""

    def generate(
        self,
        job_id: str,
        prompt: str,
        negative_prompt: str,
        motion_type: str,
        lora_list: Sequence[str],
        duration: float,
        fps: int,
        width: int,
        height: int,
    ) -> dict:
        self.loader.ensure_loaded()

        loras = self.lora_loader.resolve(lora_list)
        motion_path = self.motion_loader.resolve(motion_type)
        num_frames = self._coerce_frames(duration, fps)

        # Placeholder/mock frames keep the pipeline runnable until the GGUF runtime is wired.
        frames = self._mock_frames(num_frames, prompt, negative_prompt, width, height, init_image=None)

        # Persist frames + encode video
        frame_paths = self.writer.write_frames(job_id, frames)
        video_path = self.writer.encode_video(job_id, fps)

        return {
            "job_id": job_id,
            "num_frames": len(frame_paths),
            "fps": fps,
            "video_path": str(video_path),
            "frame_paths": [str(f) for f in frame_paths],
            "loras": [str(path) for path in loras],
            "motion_module": str(motion_path) if motion_path else None,
            "backend": self.loader.backend or ("mock" if self.loader.mock_mode else ""),
            "width": width,
            "height": height,
        }

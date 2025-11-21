"""Image-to-video pipeline built atop WAN 2.2 GGUF."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .lora_loader import LoRALoader
from .loader import WanGGUFLoader
from .model_registry import EnginePaths
from .motion_loader import MotionLoader
from .pipeline_t2v import BaseWanPipeline
from .video_writer import VideoWriter


class ImageToVideoPipeline(BaseWanPipeline):
    """Supports picture → video chaining (I2V)."""

    def __init__(
        self,
        paths: EnginePaths,
        loader: WanGGUFLoader,
        lora_loader: LoRALoader,
        motion_loader: MotionLoader,
        writer: VideoWriter,
    ) -> None:
        super().__init__(paths, loader, lora_loader, motion_loader, writer)

    def generate(
        self,
        job_id: str,
        prompt: str,
        negative_prompt: str,
        motion_type: str,
        lora_list: Sequence[str],
        duration: float,
        fps: int,
        init_image: Path,
        width: int,
        height: int,
    ) -> dict:
        self.loader.ensure_loaded()

        loras = self.lora_loader.resolve(lora_list)
        motion_path = self.motion_loader.resolve(motion_type)
        num_frames = self._coerce_frames(duration, fps)

        frames = self._mock_frames(num_frames, prompt, negative_prompt, width, height, init_image=init_image)

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

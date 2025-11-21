"""Frame + video persistence helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

try:  # Optional; used when ffmpeg is absent
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None  # type: ignore

from .model_registry import EnginePaths


class VideoWriter:
    """Writes frame sequences and encodes them into MP4."""

    def __init__(self, paths: EnginePaths) -> None:
        self.paths = paths
        self.paths.frames_dir.mkdir(parents=True, exist_ok=True)
        self.paths.videos_dir.mkdir(parents=True, exist_ok=True)

    def write_frames(self, job_id: str, frames: Iterable[object]) -> List[Path]:
        job_frames_dir = self.paths.frames_dir / job_id
        job_frames_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for idx, frame in enumerate(frames):
            frame_path = job_frames_dir / f"frame_{idx:05d}.png"
            self._save_single_frame(frame, frame_path)
            saved.append(frame_path)
        return saved

    def _save_single_frame(self, frame: object, target: Path) -> None:
        if isinstance(frame, Path):
            shutil.copy2(frame, target)
            return
        if isinstance(frame, (bytes, bytearray, memoryview)):
            target.write_bytes(bytes(frame))
            return
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to serialize frames") from exc

        if isinstance(frame, Image.Image):  # type: ignore[attr-defined]
            frame.save(target)
            return
        if hasattr(frame, "save"):
            frame.save(target)  # type: ignore[call-arg]
            return
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def encode_video(self, job_id: str, fps: int) -> Path:
        job_frames_dir = self.paths.frames_dir / job_id
        video_path = self.paths.videos_dir / f"{job_id}.mp4"

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(job_frames_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            subprocess.run(cmd, check=True)
            return video_path

        if iio is None:
            video_path.write_text("ffmpeg/imageio unavailable – placeholder video file.")
            return video_path

        images: List[object] = []
        for frame_path in sorted(job_frames_dir.glob("frame_*.png")):
            images.append(iio.imread(frame_path))
        if not images:
            raise RuntimeError("No frames were written; unable to encode video")
        iio.imwrite(video_path, images, fps=fps)
        return video_path

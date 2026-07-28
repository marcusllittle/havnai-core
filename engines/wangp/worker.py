"""WanGP-venv worker for one HavnAI LTX-2.3 generation request."""

from __future__ import annotations

import json
import shutil
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


ACTIVE_JOB: Any = None
CANCEL_REQUESTED = False


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _handle_signal(_signum: int, _frame: Any) -> None:
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if ACTIVE_JOB is not None:
        try:
            ACTIVE_JOB.cancel()
        except Exception:
            pass


def main() -> int:
    global ACTIVE_JOB
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker.py REQUEST_JSON")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root = Path(request["wangp_root"])
    status_path = Path(request["status_path"])
    result_path = Path(request["result_path"])
    output_path = Path(request["output_path"])
    output_dir = Path(request["output_dir"])

    sys.path.insert(0, str(root))
    from shared.api import init

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _atomic_json(status_path, {"phase": "loading_video_model", "progress": 0})

    session = init(
        root=root,
        output_dir=output_dir,
        console_output=False,
        console_isatty=False,
    )
    settings = session.get_default_settings(str(request["model_type"]))
    settings.update(
        {
            "model_type": str(request["model_type"]),
            "prompt": str(request["prompt"]),
            "negative_prompt": str(request.get("negative_prompt") or ""),
            "resolution": str(request["resolution"]),
            "num_inference_steps": int(request["steps"]),
            "guidance_scale": float(request["guidance"]),
            "video_length": int(request["frames"]),
            "duration_seconds": float(request["duration_seconds"]),
            "force_fps": int(request["fps"]),
            "prompt_enhancer": "",
            "seed": int(request["seed"]),
        }
    )
    source_image = str(request.get("source_image") or "").strip()
    if source_image:
        settings.update(
            {
                "image_prompt_type": "S",
                "image_start": [source_image],
                "input_video_strength": float(request.get("source_strength") or 1.0),
            }
        )

    ACTIVE_JOB = session.submit_task(settings)
    last_update: Optional[tuple[Any, ...]] = None
    for event in ACTIVE_JOB.events.iter(timeout=0.5):
        if CANCEL_REQUESTED:
            ACTIVE_JOB.cancel()
        if event.kind != "progress":
            continue
        update = event.data
        key = (update.phase, update.progress, update.current_step, update.total_steps)
        if key == last_update:
            continue
        last_update = key
        _atomic_json(
            status_path,
            {
                "phase": str(update.phase or "generating"),
                "status": str(update.status or ""),
                "progress": int(update.progress or 0),
                "current_step": update.current_step,
                "total_steps": update.total_steps,
            },
        )

    result = ACTIVE_JOB.result()
    generated = [Path(value) for value in result.generated_files if Path(value).is_file()]
    videos = [path for path in generated if path.suffix.lower() == ".mp4"]
    if result.success and videos:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(videos[-1], output_path)
    errors = [getattr(error, "message", str(error)) for error in result.errors]
    success = bool(result.success and output_path.is_file() and output_path.stat().st_size > 0)
    if result.success and not videos:
        errors.append("WanGP completed without an MP4 output")
    _atomic_json(
        result_path,
        {
            "success": success,
            "cancelled": bool(result.cancelled or CANCEL_REQUESTED),
            "generated_files": [str(path) for path in generated],
            "output_path": str(output_path) if success else None,
            "errors": errors,
        },
    )
    return 0 if success else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        try:
            request_data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
            _atomic_json(
                Path(request_data["result_path"]),
                {
                    "success": False,
                    "errors": [str(exc), traceback.format_exc(limit=8)],
                },
            )
        except Exception:
            pass
        raise

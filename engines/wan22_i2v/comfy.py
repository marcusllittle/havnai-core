"""Wan 2.2 image-to-video through the worker's ComfyUI.

This is the path that matches the measured RTX 3060 recipe: GGUF Q4_K_M
experts, the LightX2V 4-step LoRA, euler, shift 5, and 2 high-noise steps
followed by 2 low-noise steps. The graph is ComfyUI's API format. A local
ComfyUI (``HAI_COMFY_URL``, default ``http://127.0.0.1:8188``) must already
be running, with ComfyUI-GGUF installed and the weight folders pointed at
``/mnt/d/models/wan22``.

``HAI_WAN22_WORKFLOW`` may name an API-format JSON file. String values that
are exactly ``{{prompt}}``, ``{{negative}}``, ``{{image}}``, ``{{width}}``,
``{{height}}``, ``{{frames}}``, ``{{steps}}``, ``{{seed}}``, ``{{shift}}``,
or ``{{fps}}`` are replaced before the graph is submitted.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import Wan22I2VConfig, load_config


PLACEHOLDERS = (
    "prompt",
    "negative",
    "image",
    "width",
    "height",
    "frames",
    "steps",
    "seed",
    "shift",
    "fps",
)


def comfy_url() -> str:
    return os.environ.get("HAI_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def _basename(config: Wan22I2VConfig, key: str) -> str:
    return Path(str(config.files.get(key) or "")).name


def build_workflow(
    settings: Dict[str, Any],
    *,
    config: Wan22I2VConfig,
    image_name: str,
    seed: int,
) -> Dict[str, Any]:
    """The measured 4-step graph. Node ids are stable so tests can read them."""
    high = _basename(config, "transformer_high")
    low = _basename(config, "transformer_low")
    steps = int(settings["steps"])
    split = int(settings["high_noise_steps"])
    shift = float(settings["flow_shift"])
    guidance = float(settings["guidance"])
    frames = int(settings["frames"])
    width = int(settings["width"])
    height = int(settings["height"])
    fps = float(settings["fps"])
    sampler = str(settings.get("scheduler") or "euler")
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": high}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": low}},
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": _basename(config, "lora_high"),
                "strength_model": float(settings["lora_scale_high"]),
            },
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": _basename(config, "lora_low"),
                "strength_model": float(settings["lora_scale_low"]),
            },
        },
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": shift}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": shift}},
        "7": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": _basename(config, "text_encoder"), "type": "wan"},
        },
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": _basename(config, "vae")}},
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["7", 0], "text": settings["prompt"]},
        },
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["7", 0], "text": settings["negative_prompt"]},
        },
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["9", 0],
                "negative": ["10", 0],
                "vae": ["8", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
                "start_image": ["11", 0],
            },
        },
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["5", 0],
                "add_noise": "enable",
                "noise_seed": int(seed),
                "steps": steps,
                "cfg": guidance,
                "sampler_name": sampler,
                "scheduler": "simple",
                "positive": ["12", 0],
                "negative": ["12", 1],
                "latent_image": ["12", 2],
                "start_at_step": 0,
                "end_at_step": split,
                "return_with_leftover_noise": "enable",
            },
        },
        "14": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["6", 0],
                "add_noise": "disable",
                "noise_seed": int(seed),
                "steps": steps,
                "cfg": guidance,
                "sampler_name": sampler,
                "scheduler": "simple",
                "positive": ["12", 0],
                "negative": ["12", 1],
                "latent_image": ["13", 0],
                "start_at_step": split,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": fps}},
        "17": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["16", 0],
                "filename_prefix": "havnai/wan22",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def workflow_from_file(path: Path, mapping: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Wan workflow must be a JSON object: {path}")
    return _fill(raw, {f"{{{{{name}}}}}": mapping[name] for name in PLACEHOLDERS if name in mapping})


def _fill(value: Any, mapping: Dict[str, Any]) -> Any:
    if isinstance(value, str) and value in mapping:
        return mapping[value]
    if isinstance(value, dict):
        return {key: _fill(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, mapping) for item in value]
    return value


def _video_file(history: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    outputs = history.get("outputs") if isinstance(history, dict) else None
    if not isinstance(outputs, dict):
        return None
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for key in ("videos", "gifs", "images"):
            files = node.get(key)
            if not isinstance(files, list):
                continue
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("filename") or "")
                if name.lower().endswith((".mp4", ".webm", ".mkv")):
                    return item
    return None


class ComfyError(RuntimeError):
    pass


def run_graph(
    workflow: Dict[str, Any],
    image_bytes: bytes,
    image_name: str,
    *,
    base_url: str,
    session: Any,
    timeout: float,
    progress_fn: Optional[Callable[[float, str], None]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Upload, queue, poll, download, then ask ComfyUI to free the card."""
    if progress_fn:
        progress_fn(5, "uploading_source")
    upload = session.post(
        f"{base_url}/upload/image",
        files={"image": (image_name, image_bytes, "image/png")},
        data={"overwrite": "true", "type": "input"},
        timeout=60,
    )
    upload.raise_for_status()
    body = upload.json()
    if isinstance(body, dict) and body.get("name"):
        workflow = _rename_image(workflow, image_name, str(body["name"]))
    client_id = uuid.uuid4().hex
    queued = session.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    queued.raise_for_status()
    queued_body = queued.json()
    if isinstance(queued_body, dict) and queued_body.get("error"):
        raise ComfyError(f"ComfyUI rejected the graph: {queued_body['error']}")
    prompt_id = str((queued_body or {}).get("prompt_id") or "")
    if not prompt_id:
        raise ComfyError("ComfyUI did not return a prompt id")

    started = clock()
    history: Dict[str, Any] = {}
    while clock() - started < timeout:
        if progress_fn:
            elapsed = clock() - started
            progress_fn(min(90.0, 12.0 + (elapsed / max(timeout, 1)) * 70.0), "sampling")
        response = session.get(f"{base_url}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        entry = payload.get(prompt_id) if isinstance(payload, dict) else None
        if isinstance(entry, dict) and entry.get("outputs"):
            history = entry
            break
        status = (entry or {}).get("status") if isinstance(entry, dict) else None
        if isinstance(status, dict) and status.get("status_str") == "error":
            raise ComfyError("ComfyUI reported an error for this Wan job")
        sleeper(2.0)
    else:
        raise ComfyError(f"ComfyUI Wan job exceeded {int(timeout)}s")

    video = _video_file(history)
    if not video:
        raise ComfyError("ComfyUI finished without an mp4")
    if progress_fn:
        progress_fn(92, "downloading_video")
    params = {
        "filename": video.get("filename"),
        "subfolder": video.get("subfolder") or "",
        "type": video.get("type") or "output",
    }
    downloaded = session.get(f"{base_url}/view", params=params, timeout=120)
    downloaded.raise_for_status()
    data = getattr(downloaded, "content", b"")
    if not data:
        raise ComfyError("ComfyUI returned an empty video")
    try:
        session.post(
            f"{base_url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
        )
    except Exception:
        pass
    return data


def _rename_image(workflow: Dict[str, Any], previous: str, uploaded: str) -> Dict[str, Any]:
    if previous == uploaded:
        return workflow
    cloned = json.loads(json.dumps(workflow))
    for node in cloned.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict) and inputs.get("image") == previous:
            inputs["image"] = uploaded
    return cloned


def image_to_png(image: Any) -> bytes:
    import io

    from PIL import Image

    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as opened:
            frame = opened.convert("RGB")
    else:
        frame = image.convert("RGB")
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    return buffer.getvalue()


def resolve_workflow(
    settings: Dict[str, Any],
    config: Wan22I2VConfig,
    image_name: str,
    seed: int,
) -> Dict[str, Any]:
    override = os.environ.get("HAI_WAN22_WORKFLOW", "").strip()
    if override:
        mapping = {
            "prompt": settings["prompt"],
            "negative": settings["negative_prompt"],
            "image": image_name,
            "width": int(settings["width"]),
            "height": int(settings["height"]),
            "frames": int(settings["frames"]),
            "steps": int(settings["steps"]),
            "seed": int(seed),
            "shift": float(settings["flow_shift"]),
            "fps": float(settings["fps"]),
        }
        return workflow_from_file(Path(override).expanduser(), mapping)
    return build_workflow(settings, config=config, image_name=image_name, seed=seed)


def run_comfy_i2v(
    job: Dict[str, Any],
    *,
    log_fn: Callable[[str], None],
    outputs_dir: Path,
    read_gpu_stats: Callable[[], Dict[str, Any]],
    utilization_hint: int,
    config: Optional[Wan22I2VConfig] = None,
    progress_fn: Optional[Callable[[float, str], None]] = None,
    session: Any = None,
) -> tuple:
    """Run one job and return ``(metrics, utilization, path)``."""
    from .runner import _fail, _load_init_image

    config = config or load_config()
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        return _fail(job, "prompt is required", utilization_hint)
    try:
        seed = int(job.get("seed"))
    except (TypeError, ValueError):
        return _fail(job, "seed is required", utilization_hint)
    init_raw = job.get("init_image") or job.get("init_image_url") or job.get("image")
    if not init_raw:
        return _fail(job, "Wan 2.2 image-to-video requires a source image", utilization_hint)
    try:
        image = _load_init_image(init_raw)
    except Exception as exc:
        return _fail(job, f"Failed to load source image: {exc}", utilization_hint)
    if image is None:
        return _fail(job, "Wan 2.2 image-to-video requires a source image", utilization_hint)
    if getattr(image, "size", None):
        job = dict(job)
        job.setdefault("source_width", image.size[0])
        job.setdefault("source_height", image.size[1])
    try:
        from .config import resolve_settings

        settings = resolve_settings(job, config)
    except ValueError as exc:
        return _fail(job, str(exc), utilization_hint)

    task_id = str(job.get("task_id") or job.get("job_id") or "wan22")
    safe_id = "".join(ch for ch in task_id if ch.isalnum() or ch in {"-", "_"})[:80] or "wan22"
    image_name = f"havnai_{safe_id}.png"
    try:
        timeout = int(job.get("timeout") or config.defaults.get("timeout_seconds") or 3600)
    except (TypeError, ValueError):
        timeout = 3600
    timeout = max(60, min(timeout, 7200))
    log_fn(
        f"[{safe_id}] Wan 2.2 ComfyUI: {settings['width']}x{settings['height']} "
        f"frames={settings['frames']} fps={settings['fps']} steps={settings['steps']} "
        f"shift={settings['flow_shift']} sampler={settings['scheduler']} "
        f"high_noise_steps={settings['high_noise_steps']}"
    )
    if session is None:
        import requests

        session = requests.Session()
    try:
        workflow = resolve_workflow(settings, config, image_name, seed)
        video = run_graph(
            workflow,
            image_to_png(image),
            image_name,
            base_url=comfy_url(),
            session=session,
            timeout=timeout,
            progress_fn=progress_fn,
        )
    except Exception as exc:
        return _fail(job, f"ComfyUI Wan 2.2 failed: {exc}", utilization_hint)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"video_{safe_id}.mp4"
    output_path.write_bytes(video)
    stats = read_gpu_stats() or {}
    try:
        utilization = int(float(stats.get("utilization", utilization_hint)))
    except (TypeError, ValueError):
        utilization = utilization_hint
    metrics = {
        "status": "success",
        "task_type": "video_gen",
        "model_name": str(job.get("model_name") or "wan22-i2v-a14b"),
        "model_family": "wan22_i2v",
        "model_version": config.version,
        "backend": "comfy",
        "sampler": settings["scheduler"],
        "scheduler": "simple",
        "flow_shift": settings["flow_shift"],
        "high_noise_steps": settings["high_noise_steps"],
        "steps": settings["steps"],
        "guidance": settings["guidance"],
        "frames": settings["frames"],
        "fps": settings["fps"],
        "width": settings["width"],
        "height": settings["height"],
        "seed": seed,
        "motion_preset": settings["motion_preset"],
        "strength": 1.0,
        "requested_strength": settings["requested_strength"],
        "resolved_prompt": settings["prompt"],
        "resolved_negative_prompt": settings["negative_prompt"],
        "chained": False,
    }
    log_fn(f"[{safe_id}] Wan 2.2 ComfyUI wrote {output_path.name}")
    return metrics, utilization, output_path

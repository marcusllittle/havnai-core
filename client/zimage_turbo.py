"""Q5_K_M Z-Image-Turbo for the HavnAI image worker.

The transformer stays on the GPU. Each job moves the Qwen3 text encoder to
the GPU for one encode, then back to the CPU before denoising. Guidance is
always 0 and the negative prompt is not used. The worker only advertises
this model when ``HAI_ZIMAGE_TURBO`` is on, so SDXL nodes stay as they are.

The old ``zimage`` branch loaded base ``Tongyi-MAI/Z-Image`` in bf16 with no
offload. This module does not do that.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILENAME = "z-image-turbo-Q5_K_M.gguf"
PIPELINE_NAMES = {"zimage-turbo", "zimage_turbo", "z-image-turbo"}
DEFAULT_STEPS = 9
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MODEL_NAME = "zimage-turbo"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def resolution_cap() -> Dict[str, int]:
    """Z-Image limits. These names are not the SDXL caps on the 3060 unit."""
    max_width = _positive_int("HAI_ZIMAGE_MAX_WIDTH", DEFAULT_WIDTH)
    max_height = _positive_int("HAI_ZIMAGE_MAX_HEIGHT", DEFAULT_HEIGHT)
    max_pixels = _positive_int("HAI_ZIMAGE_MAX_PIXELS", max_width * max_height)
    return {
        "max_width": max_width,
        "max_height": max_height,
        "max_pixels": max_pixels,
    }


def max_steps() -> int:
    return _positive_int("HAI_ZIMAGE_MAX_STEPS", DEFAULT_STEPS)


def cap_resolution(width: int, height: int) -> tuple:
    """Fit a request inside the Z-Image cap. Does not read HAI_SDXL_*."""
    limits = resolution_cap()
    width = max(16, int(width))
    height = max(16, int(height))
    width = min(width, limits["max_width"])
    height = min(height, limits["max_height"])
    budget = limits["max_pixels"]
    if width * height > budget:
        scale = (budget / float(width * height)) ** 0.5
        width = max(16, int(width * scale))
        height = max(16, int(height * scale))
    width = max(16, width - (width % 16))
    height = max(16, height - (height % 16))
    while width * height > budget and (width > 16 or height > 16):
        if width >= height and width > 16:
            width -= 16
        elif height > 16:
            height -= 16
        else:
            break
    return width, height


def enabled() -> bool:
    raw = os.environ.get("HAI_ZIMAGE_TURBO", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_zimage_pipeline(pipeline: str) -> bool:
    return str(pipeline or "").strip().lower() in PIPELINE_NAMES


def is_zimage_entry(entry: Any) -> bool:
    return is_zimage_pipeline(getattr(entry, "pipeline", ""))


def should_release_on_switch(keep: str) -> bool:
    """True when a model switch should drop the resident Z-Image pipeline."""
    return str(keep or "") != "zimage"


def transformer_load_kwargs(snapshot: str, dtype: Any, quant_config: Any) -> Dict[str, Any]:
    """Arguments that make from_single_file read transformer/config.json."""
    return {
        "quantization_config": quant_config,
        "torch_dtype": dtype,
        "local_files_only": True,
        "config": snapshot,
        "subfolder": "transformer",
    }


class Runtime:
    """One resident pipeline for the worker process."""

    def __init__(self) -> None:
        self.pipe: Any = None

    def unload(self, torch_module: Any = None) -> None:
        pipe = self.pipe
        self.pipe = None
        if pipe is None:
            return
        remove_hooks = getattr(pipe, "remove_all_hooks", None)
        if callable(remove_hooks):
            try:
                remove_hooks()
            except Exception:
                pass
        for name in ("text_encoder", "transformer", "vae"):
            module = getattr(pipe, name, None)
            if module is None or not hasattr(module, "to"):
                continue
            try:
                module.to("cpu")
            except Exception:
                pass
        del pipe
        gc.collect()
        torch_mod = torch_module if torch_module is not None else _optional_torch()
        cuda = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
        empty = getattr(cuda, "empty_cache", None)
        if callable(empty):
            try:
                if getattr(cuda, "is_available", lambda: False)():
                    empty()
            except Exception:
                pass

    def load(
        self,
        *,
        torch_module: Any = None,
        diffusers_module: Any = None,
        hub_module: Any = None,
        cache_dir: Optional[str] = None,
        gpu_index: int = 0,
    ) -> Any:
        if self.pipe is not None:
            return self.pipe
        torch_mod = torch_module or _require_torch()
        diffusers_mod = diffusers_module or _require_diffusers()
        hub = hub_module or _require_hub()
        dtype = torch_mod.bfloat16
        snapshot = hub.snapshot_download(
            repo_id=ZIMAGE_REPO,
            cache_dir=cache_dir,
            ignore_patterns=["transformer/*.safetensors", "transformer/*.bin"],
        )
        gguf_path = hub.hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILENAME,
            cache_dir=cache_dir,
        )
        quant = diffusers_mod.GGUFQuantizationConfig(compute_dtype=dtype)
        kwargs = transformer_load_kwargs(snapshot, dtype, quant)
        transformer = _call_with_dtype(
            lambda **load_kwargs: diffusers_mod.ZImageTransformer2DModel.from_single_file(gguf_path, **load_kwargs),
            dtype,
            {key: value for key, value in kwargs.items() if key != "torch_dtype"},
        )
        pipe = _call_with_dtype(
            lambda **load_kwargs: diffusers_mod.ZImagePipeline.from_pretrained(snapshot, **load_kwargs),
            dtype,
            {
                "transformer": transformer,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            },
        )
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if torch_mod.cuda.is_available():
            device = torch_mod.device("cuda", gpu_index)
            pipe.transformer.to(device)
            pipe.vae.to(device)
        self.pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        gpu_index: int = 0,
        torch_module: Any = None,
        diffusers_module: Any = None,
        hub_module: Any = None,
        cache_dir: Optional[str] = None,
    ) -> Any:
        torch_mod = torch_module or _require_torch()
        width, height = cap_resolution(width, height)
        steps = min(max(1, int(steps)), max_steps())
        pipe = self.load(
            torch_module=torch_mod,
            diffusers_module=diffusers_module,
            hub_module=hub_module,
            cache_dir=cache_dir,
            gpu_index=gpu_index,
        )
        if not torch_mod.cuda.is_available():
            raise RuntimeError("Z-Image-Turbo needs a CUDA GPU")
        device = torch_mod.device("cuda", gpu_index)
        pipe.text_encoder.to(device)
        embeds, _negative = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=False,
        )
        pipe.text_encoder.to("cpu")
        gc.collect()
        torch_mod.cuda.empty_cache()
        pipe.transformer.to(device)
        pipe.vae.to(device)
        prompt_embeds = [_move_embed(item, device) for item in embeds]
        generator = torch_mod.Generator(device=device).manual_seed(int(seed))
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            height=int(height),
            width=int(width),
            generator=generator,
        )
        return result.images[0]


RUNTIME = Runtime()


def unload() -> None:
    RUNTIME.unload()


def load(**kwargs: Any) -> Any:
    return RUNTIME.load(**kwargs)


def generate_job(prompt: str, **kwargs: Any) -> Any:
    return RUNTIME.generate(prompt, **kwargs)


def _move_embed(item: Any, device: Any) -> Any:
    move = getattr(item, "to", None)
    if callable(move):
        return move(device)
    return item


def _call_with_dtype(fn: Any, dtype: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return fn(torch_dtype=dtype, **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        return fn(dtype=dtype, **kwargs)


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError("Z-Image-Turbo needs torch")
    return torch


def _require_diffusers() -> Any:
    try:
        import diffusers
    except ImportError as exc:
        raise RuntimeError(
            "Z-Image-Turbo needs diffusers>=0.43.0 and gguf. "
            "Install them before setting HAI_ZIMAGE_TURBO=1."
        ) from exc
    return diffusers


def _require_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError("Z-Image-Turbo needs huggingface_hub") from exc
    return huggingface_hub

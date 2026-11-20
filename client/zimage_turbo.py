"""Q5_K_M Z-Image-Turbo for the HavnAI image worker.

The GGUF transformer and the VAE stay on the GPU across jobs. The Qwen3 text
encoder is about 8 GB in bf16 and about twice that in fp32, so it must not sit
on the GPU at the same time. The default ``HAI_ZIMAGE_ENCODER_MODE=cpu`` loads
that encoder in fp32 and encodes on the CPU. ``swap`` moves the transformer
and VAE off, encodes on the GPU, then moves them back. Guidance is always 0
and the negative prompt is not used. The worker only advertises this model
when ``HAI_ZIMAGE_TURBO`` is on. ``HAI_ZIMAGE_PRELOAD=1`` loads it at startup,
and only when the turbo flag is also on.

The old ``zimage`` branch loaded base ``Tongyi-MAI/Z-Image`` in bf16 with no
offload. This module does not do that.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILENAME = "z-image-turbo-Q5_K_M.gguf"
PIPELINE_NAMES = {"zimage-turbo", "zimage_turbo", "z-image-turbo"}
DEFAULT_STEPS = 9
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MODEL_NAME = "zimage-turbo"
WARMUP_PROMPT = "a"
LOAD_STAGE_NAMES = (
    "gguf_transformer",
    "text_encoder",
    "fp32_cast",
    "vae",
    "move_to_cuda",
)


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


def encoder_mode() -> str:
    """``cpu`` (default) or ``swap``. Anything else is a configuration error."""
    raw = os.environ.get("HAI_ZIMAGE_ENCODER_MODE", "cpu").strip().lower()
    if raw in {"", "cpu"}:
        return "cpu"
    if raw == "swap":
        return "swap"
    raise ValueError("HAI_ZIMAGE_ENCODER_MODE must be 'cpu' or 'swap'")


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _env_on("HAI_ZIMAGE_TURBO")


def preload_enabled() -> bool:
    """Startup preload is opt-in and does nothing unless the model itself is on.

    ``HAI_PRELOAD_IMAGE_ON_STARTUP`` remains the SDXL switch. This flag is
    separate so the default worker still warms SDXL. The worker honors it
    only when ``HAI_ZIMAGE_TURBO`` is also on, and the size-1 image cache
    cannot hold SDXL beside this pipeline.
    """
    return enabled() and _env_on("HAI_ZIMAGE_PRELOAD")


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
        self.last_stats: Dict[str, Any] = {}
        self.load_seconds: Optional[Dict[str, float]] = None
        self.first_encode_seconds: Optional[float] = None

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
        self.load_seconds = None
        self.first_encode_seconds = None
        self.last_stats = {}
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
        fp32 = getattr(torch_mod, "float32", None)
        started = time.perf_counter()
        stages = {name: 0.0 for name in LOAD_STAGE_NAMES}
        snapshot_started = time.perf_counter()
        snapshot = hub.snapshot_download(
            repo_id=ZIMAGE_REPO,
            cache_dir=cache_dir,
            ignore_patterns=["transformer/*.safetensors", "transformer/*.bin"],
        )
        stages["snapshot"] = round(time.perf_counter() - snapshot_started, 3)
        with _stage(stages, "gguf_transformer"):
            gguf_path = hub.hf_hub_download(
                repo_id=GGUF_REPO,
                filename=GGUF_FILENAME,
                cache_dir=cache_dir,
            )
            quant = diffusers_mod.GGUFQuantizationConfig(compute_dtype=dtype)
            kwargs = transformer_load_kwargs(snapshot, dtype, quant)
            transformer = _call_with_dtype(
                lambda **load_kwargs: diffusers_mod.ZImageTransformer2DModel.from_single_file(
                    gguf_path, **load_kwargs
                ),
                dtype,
                {key: value for key, value in kwargs.items() if key != "torch_dtype"},
            )
        with _stage(stages, "text_encoder"):
            # Read the encoder as fp32. A second cast after a bf16 load walks the
            # weights again; fp32_cast below stays near zero when this dtype sticks.
            text_encoder = _load_text_encoder(diffusers_mod, snapshot, fp32)
        with _stage(stages, "fp32_cast"):
            _ensure_encoder_fp32_cpu(text_encoder, fp32)
        with _stage(stages, "vae"):
            vae = _load_vae(diffusers_mod, snapshot, dtype)
        assemble_started = time.perf_counter()
        pipe = _call_with_dtype(
            lambda **load_kwargs: diffusers_mod.ZImagePipeline.from_pretrained(snapshot, **load_kwargs),
            dtype,
            {
                "transformer": transformer,
                "text_encoder": text_encoder,
                "vae": vae,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            },
        )
        if getattr(pipe, "text_encoder", None) is not text_encoder:
            pipe.text_encoder = text_encoder
        if getattr(pipe, "vae", None) is not vae:
            pipe.vae = vae
        if hasattr(transformer, "to"):
            pipe.transformer = transformer
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        stages["assemble"] = round(time.perf_counter() - assemble_started, 3)
        with _stage(stages, "move_to_cuda"):
            if torch_mod.cuda.is_available():
                device = torch_mod.device("cuda", gpu_index)
                _move_module(pipe.transformer, device)
                _move_module(pipe.vae, device)
        self.pipe = pipe
        self._warmup_cpu_encoder(pipe)
        stages["total"] = round(time.perf_counter() - started, 3)
        self.load_seconds = stages
        self._remember_load_stats()
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
        mode = encoder_mode()
        _reset_peak(torch_mod)
        encode_started = time.perf_counter()
        if mode == "cpu":
            # fp32 on the host. This CPU has no bf16 hardware, and bf16 here is an emulator.
            # Load already builds the encoder in fp32. Skip the cast when it is still there
            # so a warm job does not walk the weights again.
            fp32 = getattr(torch_mod, "float32", None)
            if not _encoder_is_fp32_cpu(pipe.text_encoder, fp32):
                _move_module(pipe.text_encoder, "cpu", fp32)
            _refuse_encoder_overlap(pipe)
            embeds = self._encode_timed(pipe, prompt, "cpu")
        else:
            _warn_swap_needs_free_vram()
            _move_module(pipe.transformer, "cpu")
            _move_module(pipe.vae, "cpu")
            gc.collect()
            _empty_cache(torch_mod)
            _move_module(pipe.text_encoder, device, getattr(torch_mod, "bfloat16", None))
            _refuse_encoder_overlap(pipe)
            embeds = self._encode_timed(pipe, prompt, device)
            _move_module(pipe.text_encoder, "cpu")
            gc.collect()
            _empty_cache(torch_mod)
            _refuse_encoder_overlap(pipe)
            _move_module(pipe.transformer, device)
            _move_module(pipe.vae, device)
        encode_seconds = time.perf_counter() - encode_started
        prompt_embeds = [
            _move_embed(item, device, getattr(torch_mod, "bfloat16", None)) for item in embeds
        ]
        generator = torch_mod.Generator(device=device).manual_seed(int(seed))
        denoise_started = time.perf_counter()
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            height=int(height),
            width=int(width),
            generator=generator,
        )
        denoise_seconds = time.perf_counter() - denoise_started
        peak_mib, warning = _vram_guard(torch_mod, mode)
        self.last_stats = {
            "encoder_mode": mode,
            "encode_seconds": round(encode_seconds, 3),
            "denoise_seconds": round(denoise_seconds, 3),
            "peak_allocated_mib": peak_mib,
            "vram_warning": warning,
        }
        self._remember_load_stats()
        return result.images[0]

    def _encode_timed(self, pipe: Any, prompt: str, device: Any) -> List[Any]:
        started = time.perf_counter()
        embeds = encode_prompt_unpadded(pipe, prompt, device)
        if self.first_encode_seconds is None:
            self.first_encode_seconds = round(time.perf_counter() - started, 3)
        return embeds

    def _warmup_cpu_encoder(self, pipe: Any) -> None:
        """One short CPU forward so the first real prompt is not the first forward.

        Swap mode is left cold here. A GPU warmup would pull the encoder onto
        the device while the transformer is resident.
        """
        if self.first_encode_seconds is not None:
            return
        try:
            mode = encoder_mode()
        except ValueError:
            return
        if mode != "cpu":
            return
        try:
            self._encode_timed(pipe, WARMUP_PROMPT, "cpu")
        except Exception:
            LOGGER.warning("Z-Image warmup encode failed", exc_info=True)

    def _remember_load_stats(self) -> None:
        if self.load_seconds is not None:
            self.last_stats["load_seconds"] = dict(self.load_seconds)
        if self.first_encode_seconds is not None:
            self.last_stats["first_encode_seconds"] = self.first_encode_seconds


RUNTIME = Runtime()


def unload() -> None:
    RUNTIME.unload()


def load(**kwargs: Any) -> Any:
    return RUNTIME.load(**kwargs)


def generate_job(prompt: str, **kwargs: Any) -> Any:
    return RUNTIME.generate(prompt, **kwargs)


def encode_prompt_unpadded(pipe: Any, prompt: str, device: Any) -> List[Any]:
    """Match ZImagePipeline._encode_prompt without padding the prompt out to 512.

    Diffusers 0.43 applies the Qwen chat template, runs the text encoder, and
    keeps hidden_states[-2]. It then drops padding with the attention mask.
    Qwen3 is causal and right-padded, so the real tokens match the unpadded
    forward. Padding to 512 is what made a 43-token prompt take 42 s in bf16.
    """
    tokenizer = pipe.tokenizer
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    text_inputs = tokenizer(
        templated,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()
    hidden_states = pipe.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]
    return [hidden_states[index][attention_mask[index]] for index in range(len(hidden_states))]


_SWAP_WARNED = False


def _warn_swap_needs_free_vram() -> None:
    global _SWAP_WARNED
    if _SWAP_WARNED:
        return
    _SWAP_WARNED = True
    message = (
        "Z-Image encoder mode swap needs about 9 GB free VRAM. "
        "It spills into shared memory when ACE-Step or another model is resident."
    )
    LOGGER.warning(message)
    print(message, file=sys.stderr, flush=True)


def _module_device(module: Any) -> str:
    if module is None:
        return "cpu"
    direct = getattr(module, "device", None)
    if isinstance(direct, str):
        return direct
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            return str(next(parameters()).device)
        except (StopIteration, Exception):
            pass
    if direct is not None and not callable(direct):
        return str(direct)
    return "cpu"


def _on_cuda(module: Any) -> bool:
    return _module_device(module).startswith("cuda")


def _refuse_encoder_overlap(pipe: Any) -> None:
    if _on_cuda(getattr(pipe, "text_encoder", None)) and _on_cuda(getattr(pipe, "transformer", None)):
        raise RuntimeError("refusing to run while the text encoder and transformer are both on CUDA")


class _stage:
    """Record one load stage. The timer survives a failure so the partial is kept."""

    def __init__(self, stages: Dict[str, float], name: str) -> None:
        self.stages = stages
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "_stage":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.stages[self.name] = round(time.perf_counter() - self.started, 3)
        return False


def _encoder_is_fp32_cpu(module: Any, fp32: Any) -> bool:
    if module is None or fp32 is None:
        return False
    if _on_cuda(module):
        return False
    return getattr(module, "dtype", None) == fp32


def _ensure_encoder_fp32_cpu(module: Any, fp32: Any) -> None:
    """Cast only when the loader did not already return fp32 on CPU.

    Loading with ``torch_dtype=float32`` does the conversion while the weights
    are read. A second ``.to(float32)`` walks every parameter again, which is
    the cast this skips when the dtype is already right.
    """
    if _encoder_is_fp32_cpu(module, fp32):
        return
    _move_module(module, "cpu", fp32)


def _load_text_encoder(diffusers_mod: Any, snapshot: str, dtype: Any) -> Any:
    loader = getattr(diffusers_mod, "load_zimage_text_encoder", None)
    kwargs = {
        "subfolder": "text_encoder",
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if callable(loader):
        return _call_with_dtype(lambda **load_kwargs: loader(snapshot, **load_kwargs), dtype, kwargs)
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("Z-Image-Turbo needs transformers with Qwen3") from exc
    return _call_with_dtype(
        lambda **load_kwargs: AutoModel.from_pretrained(snapshot, **load_kwargs),
        dtype,
        kwargs,
    )


def _load_vae(diffusers_mod: Any, snapshot: str, dtype: Any) -> Any:
    loader = getattr(diffusers_mod, "load_zimage_vae", None)
    kwargs = {"subfolder": "vae", "local_files_only": True}
    if callable(loader):
        return _call_with_dtype(lambda **load_kwargs: loader(snapshot, **load_kwargs), dtype, kwargs)
    vae_cls = getattr(diffusers_mod, "AutoencoderKL", None)
    if vae_cls is None:
        raise RuntimeError("Z-Image-Turbo needs diffusers.AutoencoderKL")
    return _call_with_dtype(
        lambda **load_kwargs: vae_cls.from_pretrained(snapshot, **load_kwargs),
        dtype,
        kwargs,
    )


def _move_module(module: Any, device: Any, dtype: Any = None) -> None:
    if module is None or not hasattr(module, "to"):
        return
    if dtype is not None:
        try:
            module.to(device=device, dtype=dtype)
            return
        except TypeError:
            pass
    module.to(device)


def _empty_cache(torch_mod: Any) -> None:
    cuda = getattr(torch_mod, "cuda", None)
    empty = getattr(cuda, "empty_cache", None)
    if callable(empty):
        try:
            empty()
        except Exception:
            pass


def _reset_peak(torch_mod: Any) -> None:
    cuda = getattr(torch_mod, "cuda", None)
    reset = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass


def _vram_guard(torch_mod: Any, mode: str) -> tuple:
    """Warn when this job's peak allocation exceeds 95% of total VRAM."""
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return None, False
    max_allocated = getattr(cuda, "max_memory_allocated", None)
    props_of = getattr(cuda, "get_device_properties", None)
    if not callable(max_allocated) or not callable(props_of):
        return None, False
    try:
        peak = int(max_allocated())
        total = int(props_of(0).total_memory)
    except Exception:
        return None, False
    peak_mib = round(peak / (1024.0 * 1024.0), 1)
    if total <= 0 or peak <= 0.95 * total:
        return peak_mib, False
    total_mib = total / (1024.0 * 1024.0)
    message = (
        f"Z-Image encoder mode {mode} peak allocated {peak_mib:.0f} MiB, "
        f"above 95% of {total_mib:.0f} MiB VRAM"
    )
    LOGGER.warning(message)
    print(message, file=sys.stderr, flush=True)
    return peak_mib, True


def _move_embed(item: Any, device: Any, dtype: Any = None) -> Any:
    move = getattr(item, "to", None)
    if not callable(move):
        return item
    if dtype is not None:
        try:
            return move(device=device, dtype=dtype)
        except TypeError:
            try:
                return move(device, dtype)
            except TypeError:
                return move(device)
    try:
        return move(device)
    except TypeError:
        return move(device=device)


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

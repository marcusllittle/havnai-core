"""Wan 2.2 I2V pipeline loading and a single generation call.

Mirrors ``engines/ltx_video/generator.py``: one cached diffusers pipeline,
dtype from the config, and CPU offload when VRAM is under the threshold.
The differences are the two GGUF experts, the LightX2V LoRA on each expert,
and the Wan 2.1 VAE. LoRAs are attached, never fused. Fusing into a GGUF
expert is not supported by diffusers and would unpack the quantized weights.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from .config import Wan22I2VConfig, load_config, sigmas_from_timesteps

_LOGGER = logging.getLogger(__name__)

_PIPE_LOCK = threading.Lock()
_PIPES: Dict[str, Any] = {}


def _cache_key(root: str, repo: str, settings: Dict[str, Any]) -> str:
    return (
        f"{root}::{repo}::wan22-i2v"
        f"::{settings.get('lora_scale_high')}::{settings.get('lora_scale_low')}"
        f"::{settings.get('offload')}"
    )


def unload_pipelines() -> None:
    """Drop the cached pipeline and release CUDA memory."""
    with _PIPE_LOCK:
        _PIPES.clear()
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    _LOGGER.info("Wan 2.2 I2V pipeline unloaded")


def _resolve_dtype(torch_module: Any, dtype_str: str) -> Any:
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return mapping.get(str(dtype_str or "bfloat16"), torch_module.bfloat16)


def _diffusers(module: Any = None) -> Any:
    if module is not None:
        return module
    try:
        import diffusers  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Wan 2.2 I2V needs diffusers>=0.35 with WanImageToVideoPipeline, "
            "WanTransformer3DModel, AutoencoderKLWan, and GGUFQuantizationConfig, "
            "plus the gguf package. Install them before setting HAI_WAN22_I2V=1."
        ) from exc
    return diffusers


def _call_with_dtype(fn: Callable[..., Any], dtype: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return fn(torch_dtype=dtype, **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        return fn(dtype=dtype, **kwargs)


def _load_gguf_expert(
    diffusers_mod: Any,
    path: Path,
    *,
    repo: str,
    subfolder: str,
    dtype: Any,
) -> Any:
    quant_cls = getattr(diffusers_mod, "GGUFQuantizationConfig", None)
    model_cls = getattr(diffusers_mod, "WanTransformer3DModel", None)
    if quant_cls is None or model_cls is None or not hasattr(model_cls, "from_single_file"):
        raise RuntimeError(
            "diffusers is missing WanTransformer3DModel.from_single_file or "
            "GGUFQuantizationConfig. Wan 2.2 GGUF needs diffusers>=0.35."
        )
    quant = quant_cls(compute_dtype=dtype)
    return _call_with_dtype(
        lambda **load_kwargs: model_cls.from_single_file(str(path), **load_kwargs),
        dtype,
        {
            "quantization_config": quant,
            "config": repo,
            "subfolder": subfolder,
        },
    )


def _load_vae(diffusers_mod: Any, path: Path, repo: str, dtype: Any) -> Any:
    vae_cls = getattr(diffusers_mod, "AutoencoderKLWan", None)
    if vae_cls is None:
        raise RuntimeError("diffusers is missing AutoencoderKLWan")
    if path.is_file() and hasattr(vae_cls, "from_single_file"):
        try:
            return _call_with_dtype(
                lambda **load_kwargs: vae_cls.from_single_file(str(path), **load_kwargs),
                dtype,
                {"config": repo, "subfolder": "vae"},
            )
        except Exception as exc:
            _LOGGER.warning("Wan 2.1 VAE single-file load failed (%s); trying the diffusers repo", exc)
    return _call_with_dtype(
        lambda **load_kwargs: vae_cls.from_pretrained(repo, **load_kwargs),
        dtype,
        {"subfolder": "vae"},
    )


def _load_text_encoder(diffusers_mod: Any, config: Wan22I2VConfig, dtype: Any) -> tuple:
    """Return ``(encoder, tokenizer, source)``.

    ``source`` is ``local-fp8`` or ``diffusers-umt5``. The Comfy fp8 file is
    not a diffusers checkpoint. When ``from_single_file`` rejects it, the
    umT5 in the Wan 2.2 diffusers repo is the fallback.
    """
    encoder_cls = getattr(diffusers_mod, "UMT5EncoderModel", None)
    tokenizer_cls = getattr(diffusers_mod, "T5TokenizerFast", None)
    if tokenizer_cls is None:
        try:
            from transformers import T5TokenizerFast as tokenizer_cls  # type: ignore
        except ImportError:
            tokenizer_cls = None
    if encoder_cls is None:
        try:
            from transformers import UMT5EncoderModel as encoder_cls  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Wan 2.2 I2V needs UMT5EncoderModel") from exc

    fallback = config.text_encoder_fallback or {}
    repo = str(fallback.get("repo_id") or config.diffusers_repo)
    encoder_sub = str(fallback.get("subfolder") or "text_encoder")
    tokenizer_sub = str(fallback.get("tokenizer_subfolder") or "tokenizer")
    mode = os.environ.get("HAI_WAN22_TEXT_ENCODER", "auto").strip().lower() or "auto"
    local = config.asset_path("text_encoder")

    def _tokenizer() -> Any:
        if tokenizer_cls is None:
            raise RuntimeError("Wan 2.2 I2V needs T5TokenizerFast")
        return tokenizer_cls.from_pretrained(repo, subfolder=tokenizer_sub)

    if mode != "diffusers" and local.is_file() and hasattr(encoder_cls, "from_single_file"):
        try:
            encoder = _call_with_dtype(
                lambda **load_kwargs: encoder_cls.from_single_file(str(local), **load_kwargs),
                dtype,
                {"config": repo, "subfolder": encoder_sub},
            )
            return encoder, _tokenizer(), "local-fp8"
        except Exception as exc:
            if mode == "local":
                raise RuntimeError(
                    f"HAI_WAN22_TEXT_ENCODER=local but the fp8 umT5 at {local} "
                    f"did not load: {exc}. The Comfy scaled-fp8 file is not a "
                    "diffusers checkpoint. Leave the flag on auto to download "
                    f"the umT5 in {repo}."
                ) from exc
            _LOGGER.warning(
                "Comfy fp8 umT5 did not load (%s). Falling back to %s/%s. "
                "That download is the diffusers umT5, not the fp8 file.",
                exc,
                repo,
                encoder_sub,
            )
    encoder = _call_with_dtype(
        lambda **load_kwargs: encoder_cls.from_pretrained(repo, **load_kwargs),
        dtype,
        {"subfolder": encoder_sub},
    )
    return encoder, _tokenizer(), "diffusers-umt5"


def _attach_lora(module: Any, path: Path, adapter_name: str, scale: float) -> str:
    """Attach one LightX2V LoRA to one expert. Never fuse it into the GGUF."""
    if hasattr(module, "load_lora_adapter"):
        try:
            module.load_lora_adapter(str(path), adapter_name=adapter_name, prefix=None)
        except TypeError:
            module.load_lora_adapter(str(path), adapter_name=adapter_name)
        if hasattr(module, "set_adapters"):
            try:
                module.set_adapters([adapter_name], weights=[float(scale)])
            except TypeError:
                module.set_adapters([adapter_name], adapter_weights=[float(scale)])
        return "load_lora_adapter"
    if hasattr(module, "load_lora_weights"):
        module.load_lora_weights(str(path), adapter_name=adapter_name)
        if hasattr(module, "set_adapters"):
            try:
                module.set_adapters([adapter_name], adapter_weights=[float(scale)])
            except TypeError:
                module.set_adapters([adapter_name])
        return "load_lora_weights"
    raise RuntimeError(
        f"{type(module).__name__} cannot load a LoRA. "
        "diffusers WanTransformer3DModel.load_lora_adapter is required. "
        "Fusing a LoRA into a GGUF expert is intentionally not attempted."
    )


class _DistillScheduler:
    """Feed LightX2V's step list when the pipeline calls set_timesteps(n)."""

    def __init__(self, base: Any, sigmas: list) -> None:
        self._base = base
        self._sigmas = list(sigmas)

    def set_timesteps(self, num_inference_steps: Any = None, device: Any = None, sigmas: Any = None, **kwargs: Any) -> Any:
        chosen = self._sigmas if sigmas is None else sigmas
        try:
            return self._base.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=chosen,
                **kwargs,
            )
        except TypeError:
            return self._base.set_timesteps(num_inference_steps, device=device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def install_distill_schedule(pipe: Any, timesteps: list) -> Any:
    sigmas = sigmas_from_timesteps(timesteps)
    pipe.scheduler = _DistillScheduler(pipe.scheduler, sigmas)
    return pipe.scheduler


def _apply_offload(pipe: Any, settings: Dict[str, Any], torch_module: Any) -> str:
    """Match the LTX rule: offload below the VRAM threshold, else ``.to(cuda)``."""
    disable = os.environ.get("HAI_WAN22_NO_OFFLOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    force = os.environ.get("HAI_WAN22_CPU_OFFLOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    vram_gb = 0.0
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        try:
            vram_gb = cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 0.0
    threshold = float(settings.get("cpu_offload_threshold_gb") or 16)
    mode = str(settings.get("offload") or "model")
    use_offload = not disable and (force or vram_gb <= 0 or vram_gb < threshold)
    if not use_offload:
        device = "cuda" if cuda is not None and cuda.is_available() else "cpu"
        pipe.to(device)
        return "cuda"
    if mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return "sequential"
    if mode == "group" and hasattr(pipe, "enable_group_offload"):
        pipe.enable_group_offload()
        return "group"
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return "model"
    pipe.to("cpu")
    return "cpu"


def _enable_vae_memory(pipe: Any, settings: Dict[str, Any]) -> None:
    if settings.get("vae_slicing") and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception as exc:
            _LOGGER.warning("Wan VAE slicing failed: %s", exc)
    vae = getattr(pipe, "vae", None)
    if settings.get("vae_tiling") and vae is not None and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception as exc:
            _LOGGER.warning("Wan VAE tiling failed: %s", exc)


def _load_pipeline(
    config: Wan22I2VConfig,
    settings: Dict[str, Any],
    *,
    torch_module: Any = None,
    diffusers_module: Any = None,
) -> Any:
    torch_mod = torch_module or torch
    if torch_mod is None:
        raise RuntimeError("torch is required for Wan 2.2 I2V")
    diffusers_mod = _diffusers(diffusers_module)
    pipeline_cls = getattr(diffusers_mod, "WanImageToVideoPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError(
            "diffusers.WanImageToVideoPipeline is missing. Install diffusers>=0.35."
        )
    root = str(config.weights_root())
    key = _cache_key(root, config.diffusers_repo, settings)
    with _PIPE_LOCK:
        cached = _PIPES.get(key)
        if cached is not None:
            _LOGGER.debug("Wan 2.2 pipeline cache HIT: %s", key)
            return cached

        dtype = _resolve_dtype(torch_mod, settings.get("dtype") or "bfloat16")
        high = _load_gguf_expert(
            diffusers_mod,
            config.asset_path("transformer_high"),
            repo=config.diffusers_repo,
            subfolder="transformer",
            dtype=dtype,
        )
        low = _load_gguf_expert(
            diffusers_mod,
            config.asset_path("transformer_low"),
            repo=config.diffusers_repo,
            subfolder="transformer_2",
            dtype=dtype,
        )
        _attach_lora(high, config.asset_path("lora_high"), "lightx2v_high", float(settings["lora_scale_high"]))
        _attach_lora(low, config.asset_path("lora_low"), "lightx2v_low", float(settings["lora_scale_low"]))
        vae = _load_vae(diffusers_mod, config.asset_path("vae"), config.diffusers_repo, dtype)
        text_encoder, tokenizer, encoder_source = _load_text_encoder(diffusers_mod, config, dtype)
        pipe = _call_with_dtype(
            lambda **load_kwargs: pipeline_cls.from_pretrained(config.diffusers_repo, **load_kwargs),
            dtype,
            {
                "transformer": high,
                "transformer_2": low,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
            },
        )
        pipe._wan22_text_encoder_source = encoder_source
        if hasattr(pipe, "register_to_config"):
            pipe.register_to_config(boundary_ratio=float(settings["boundary_ratio"]))
        install_distill_schedule(pipe, list(settings.get("denoising_timesteps") or []))
        offload = _apply_offload(pipe, settings, torch_mod)
        pipe._wan22_offload = offload
        _enable_vae_memory(pipe, settings)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        _PIPES[key] = pipe
        _LOGGER.info(
            "Wan 2.2 pipeline ready: offload=%s encoder=%s boundary=%.3f",
            offload,
            encoder_source,
            float(settings["boundary_ratio"]),
        )
        return pipe


def generate_frames(
    config: Wan22I2VConfig,
    *,
    prompt: str,
    negative_prompt: str = "",
    image: Any = None,
    settings: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    callback_on_step_end: Optional[Callable] = None,
    torch_module: Any = None,
    diffusers_module: Any = None,
) -> Any:
    """Run one image-to-video pass and return the pipeline frame batch."""
    torch_mod = torch_module or torch
    if torch_mod is None:
        raise RuntimeError("torch is required for Wan 2.2 I2V")
    if image is None:
        raise RuntimeError("Wan 2.2 I2V requires a source image")
    settings = dict(settings or {})
    pipe = _load_pipeline(
        config,
        settings,
        torch_module=torch_mod,
        diffusers_module=diffusers_module,
    )
    if hasattr(pipe, "register_to_config"):
        pipe.register_to_config(boundary_ratio=float(settings.get("boundary_ratio") or 0.625))

    width = int(settings.get("width") or 480)
    height = int(settings.get("height") or 832)
    prepared = image
    if Image is not None and isinstance(image, Image.Image):
        prepared = image.convert("RGB").resize((width, height), resample=Image.LANCZOS)

    device = "cpu"
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        device = "cuda"
    generator = torch_mod.Generator(device=device).manual_seed(int(seed))
    call_kwargs: Dict[str, Any] = {
        "image": prepared,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": int(settings.get("frames") or 81),
        "num_inference_steps": int(settings.get("steps") or 4),
        "guidance_scale": float(settings.get("guidance") or 1.0),
        "guidance_scale_2": float(settings.get("guidance_2") or settings.get("guidance") or 1.0),
        "generator": generator,
    }
    if callback_on_step_end is not None:
        call_kwargs["callback_on_step_end"] = callback_on_step_end
    try:
        result = pipe(**call_kwargs)
    except TypeError:
        call_kwargs.pop("guidance_scale_2", None)
        result = pipe(**call_kwargs)
    if hasattr(result, "frames"):
        frames = result.frames
        if isinstance(frames, (list, tuple)) and frames:
            return frames[0]
        return frames
    return result

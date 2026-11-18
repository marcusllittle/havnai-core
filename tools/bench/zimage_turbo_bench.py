#!/usr/bin/env python3
"""Standalone VRAM and speed bench for Z-Image-Turbo against juggernautXL.

This does not talk to the HavnAI worker, registry loader, or router. Run it
on the RTX 3060 with the node stopped. `--dry-run` prints the plan, and
`--smoke` exercises timing, OOM handling, reports, and the contact sheet
without a GPU or the real weights.

Low-VRAM route, checked against the diffusers 0.43.0 source:

- `ZImagePipeline` cannot load a GGUF file itself. The supported path is
  `ZImageTransformer2DModel.from_single_file(..., quantization_config=
  GGUFQuantizationConfig(...))`, then pass that module into
  `ZImagePipeline.from_pretrained`.
- The unquantized bf16 transformer file is 12.3 GB. The Qwen3 text encoder
  shards in Tongyi-MAI/Z-Image-Turbo are about 8.1 GB. Those two cannot both
  stay resident on a 12 GB card.
- `ZImagePipeline.model_cpu_offload_seq` is `text_encoder->transformer->vae`.
  `enable_model_cpu_offload()` follows that order, so the text encoder is on
  the GPU for prompt encoding and is moved off before the transformer runs.
  The first denoising step records each module's device so a real run can
  confirm that.

Default Turbo schedule is `num_inference_steps=9`. The model card says that
schedule performs 8 DiT forwards. The script counts the forwards it actually
sees. SDXL uses the juggernautXL checkpoint path, 30 steps, guidance 6.0, the
registry negative prompt, and the same sampler swap as `client.py`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "server" / "manifests" / "registry.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
SDXL_MODEL_NAME = "juggernautXL_ragnarokBy"

# Hub file sizes, not measured VRAM. The diffusers transformer shards are
# stored as fp32 (about 24.7 GB on disk) and land near 12.3 GB once cast to
# bf16, which matches the published BF16 GGUF.
HUB_FILE_GB = {
    "text_encoder": 8.1,
    "vae": 0.17,
    "transformer_bf16": 12.3,
    "q8_0": 7.2,
    "q5_k_m": 4.6,
}

# client.py treats a card at or under this as the 3060 low-VRAM profile.
WORKER_LOW_VRAM_GB = 12.5

ROUTE = {
    "choice": "gguf_transformer_plus_model_cpu_offload",
    "diffusers": "0.43.0",
    "why": (
        "Diffusers 0.43.0 can load the Z-Image transformer from GGUF, but only "
        "through ZImageTransformer2DModel.from_single_file and "
        "GGUFQuantizationConfig. The pipeline class itself cannot open a GGUF. "
        "The bf16 transformer is about 12.3 GB and the Qwen3 text encoder is "
        "about 8.1 GB, so leaving both on the GPU exceeds a 12 GB card before "
        "activations. enable_model_cpu_offload() follows "
        "text_encoder->transformer->vae, which is what takes the text encoder "
        "off the GPU after prompt encoding. Q8_0 (about 7.2 GB) and Q5_K_M "
        "(about 4.6 GB) are the default transformer files. An unquantized "
        "bf16 run is optional and is not expected to fit."
    ),
}

FALLBACK_SDXL = {
    "name": SDXL_MODEL_NAME,
    "path": "/mnt/d/havnai-storage/models/creator/juggernautXL_ragnarokBy.safetensors",
    "sampler": "dpm++_sde_karras",
    "steps": 30,
    "guidance": 6.0,
    "width": 832,
    "height": 1216,
    "negative_prompt_default": (
        "extra fingers, extra hands, extra arms, twisted wrists, deformed hands, "
        "dislocated joints, malformed feet, backward feet, duplicate face, "
        "background ghosts, extra people, bad anatomy, low quality, blurry"
    ),
}

# Portraits and backdrops follow the locked Astra Valkyries templates in
# server/astra_gen.py. The sign prompt is there to stress on-image text.
# havnai_default is the fallback string in client.py run_image_generation.
PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "portrait_nova",
        "seed": 41001,
        "kind": "astra_portrait",
        "text": (
            "Nova, elite adult female starfighter pilot, seated in a futuristic "
            "spacecraft cockpit, long platinum-blonde hair in high glossy ponytail, "
            "piercing bright blue eyes, confident smile, athletic build, glossy "
            "metallic royal-blue and black pilot bodysuit, sleek reflective chest "
            "panel and shoulder plating, black gloves and futuristic wrist cuffs, "
            "premium command chair, glowing control panels, cinematic blue "
            "highlights, polished anime-inspired realism, sharp linework, "
            "ultra-clean render, centered full body portrait, premium game "
            "character splash art, high contrast, Astra Valkyries"
        ),
    },
    {
        "id": "portrait_rex",
        "seed": 41002,
        "kind": "astra_portrait",
        "text": (
            "Rex, elite adult female tactical ace pilot, standing confidently "
            "inside a dark futuristic spacecraft cockpit, short sleek chin-length "
            "bob with dark roots fading to vivid neon teal tips, luminous cyan "
            "highlights framing her face, cool cyan eyes with a confident smirk, "
            "athletic build, glossy teal-black pilot bodysuit with subtle gold "
            "zipper detail, cropped jacket open, reflective form-fitting material, "
            "powerful stance, cockpit illuminated by teal accent lights, glowing "
            "interface panels, moody cyber-sci-fi atmosphere, cinematic low angle, "
            "ultra-detailed anime realism, premium character concept art, strong "
            "contrast, full body portrait, Astra Valkyries"
        ),
    },
    {
        "id": "backdrop_nebula",
        "seed": 41003,
        "kind": "astra_backdrop",
        "text": (
            "wide establishing shot of an empty futuristic starfighter runway at "
            "night, no people, no pilots, deep space and vibrant blue-violet "
            "nebulae drifting past the runway lights, glossy metal deck, cinematic "
            "sci-fi environment, premium game key art, Astra Valkyries"
        ),
    },
    {
        "id": "backdrop_abyss",
        "seed": 41004,
        "kind": "astra_backdrop",
        "text": (
            "empty spacecraft cockpit interior with no pilot and no people, "
            "windows opening onto the cold darkness of the abyss, faint teal "
            "starlight and drifting ice crystals, instrument glow on empty seats, "
            "cinematic sci-fi environment plate, Astra Valkyries"
        ),
    },
    {
        "id": "sign_astra",
        "seed": 41005,
        "kind": "on_image_text",
        "text": (
            "a clean brushed-metal hangar wall inside a starfighter bay, one large "
            "sign painted in white block letters that reads ASTRA, cyan rim light, "
            "no other writing, no people, sharp focus, cinematic sci-fi"
        ),
    },
    {
        "id": "havnai_default",
        "seed": 41006,
        "kind": "havnai_generic",
        "text": "a high quality photo of a golden retriever on a beach at sunset",
    },
]

VARIANTS: Dict[str, Dict[str, Any]] = {
    "q8_0": {
        "kind": "zimage_gguf",
        "quant": "Q8_0",
        "repo": GGUF_REPO,
        "filename": "z-image-turbo-Q8_0.gguf",
        "hub_file_gb": HUB_FILE_GB["q8_0"],
    },
    "q5_k_m": {
        "kind": "zimage_gguf",
        "quant": "Q5_K_M",
        "repo": GGUF_REPO,
        "filename": "z-image-turbo-Q5_K_M.gguf",
        "hub_file_gb": HUB_FILE_GB["q5_k_m"],
    },
    "bf16": {
        "kind": "zimage_bf16",
        "quant": "BF16",
        "repo": ZIMAGE_REPO,
        "filename": "",
        "hub_file_gb": HUB_FILE_GB["transformer_bf16"],
        "note": "Optional. The bf16 transformer alone is about 12.3 GB, so a 12 GB card is expected to run out of memory once activations are included.",
    },
    "juggernautxl": {
        "kind": "sdxl",
        "quant": "",
        "repo": "",
        "filename": "",
        "hub_file_gb": None,
    },
}

DEFAULT_VARIANTS = ("q8_0", "q5_k_m", "juggernautxl")


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def log(message: str) -> None:
    print(message, flush=True)


def is_oom(exc: BaseException) -> bool:
    # Same signals client.py treats as a CUDA OOM, plus the exception class name.
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "outofmemory" in text
        or "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cublas_status_alloc_failed" in text
    )


def round_seconds(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 3)


def bytes_to_mib(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 1)


def wsl_to_windows(path: str) -> Optional[str]:
    """Map a WSL mount such as /mnt/d/foo to D:\\foo."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    return None


def load_sdxl_registry() -> Dict[str, Any]:
    """Read the juggernautXL entry. Fall back to the known main-branch values."""
    info = dict(FALLBACK_SDXL)
    info["registry_path"] = str(REGISTRY_PATH)
    info["registry_loaded"] = False
    info["registry_error"] = ""
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        info["registry_error"] = str(exc)
        return info
    except json.JSONDecodeError as exc:
        info["registry_error"] = f"registry JSON is invalid: {exc}"
        return info
    models = payload.get("models")
    if not isinstance(models, list):
        info["registry_error"] = "registry has no models list"
        return info
    for entry in models:
        if isinstance(entry, dict) and entry.get("name") == SDXL_MODEL_NAME:
            for key in ("path", "sampler", "steps", "guidance", "width", "height", "negative_prompt_default"):
                if key in entry and entry[key] not in (None, ""):
                    info[key] = entry[key]
            info["registry_loaded"] = True
            return info
    info["registry_error"] = f"{SDXL_MODEL_NAME} is not in the registry"
    return info


def resolve_sdxl_checkpoint(explicit: str, registry_entry: Dict[str, Any]) -> Dict[str, Any]:
    registry_path = str(registry_entry.get("path") or FALLBACK_SDXL["path"])
    windows_candidate = wsl_to_windows(registry_path)
    if explicit:
        path = Path(explicit).expanduser()
        return {
            "path": str(path),
            "exists": path.is_file(),
            "source": "cli",
            "registry_path": registry_path,
            "windows_candidate": windows_candidate,
        }
    raw = Path(registry_path).expanduser()
    if raw.is_file():
        chosen = str(raw)
        source = "registry"
        exists = True
    elif windows_candidate and Path(windows_candidate).is_file():
        chosen = windows_candidate
        source = "registry_windows_translation"
        exists = True
    elif os.name == "nt" and windows_candidate:
        chosen = windows_candidate
        source = "registry_windows_translation"
        exists = False
    else:
        chosen = registry_path
        source = "registry"
        exists = False
    return {
        "path": chosen,
        "exists": exists,
        "source": source,
        "registry_path": registry_path,
        "windows_candidate": windows_candidate,
    }


def selected_variants(names: str) -> List[Dict[str, Any]]:
    chosen = []
    unknown = []
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        spec = VARIANTS.get(name)
        if spec is None:
            unknown.append(name)
            continue
        item = dict(spec)
        item["id"] = name
        chosen.append(item)
    if unknown:
        known = ", ".join(VARIANTS)
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}. Known variants: {known}")
    if not chosen:
        raise SystemExit("Select at least one variant.")
    return chosen


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    registry_entry = load_sdxl_registry()
    checkpoint = resolve_sdxl_checkpoint(args.sdxl_checkpoint, registry_entry)
    prompts = PROMPTS[: args.limit_prompts] if args.limit_prompts else list(PROMPTS)
    return {
        "script": "tools/bench/zimage_turbo_bench.py",
        "route": ROUTE,
        "gpu_planning_assumption_gb": 12,
        "worker_low_vram_threshold_gb": WORKER_LOW_VRAM_GB,
        "worker_note": (
            "client.py caches one image pipeline and frees it with gc plus "
            "torch.cuda.empty_cache. The 12.5 GB cap changes face-swap and LTX2 "
            "settings. The SDXL text-to-image path does not apply that cap. "
            "run_image_generation also clamps guidance to at least 1.0, so the "
            "current worker cannot run Turbo's required guidance of 0."
        ),
        "resolution": {"width": args.width, "height": args.height},
        "warm_runs": args.warm_runs,
        "turbo": {
            "repo": ZIMAGE_REPO,
            "num_inference_steps": args.turbo_steps,
            "expected_dit_forwards_when_steps_are_9": 8,
            "guidance_scale": args.turbo_guidance,
            "dtype": args.dtype,
            "negative_prompt": "ignored when guidance_scale is 0",
        },
        "sdxl": {
            "model": registry_entry["name"],
            "registry_loaded": registry_entry["registry_loaded"],
            "registry_error": registry_entry["registry_error"],
            "checkpoint": checkpoint,
            "sampler_in_registry": registry_entry["sampler"],
            "sampler_applied_like_worker": (
                "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) "
                "when the registry sampler contains dpmpp after '+' is replaced "
                "with 'p'. The worker does not set algorithm_type to "
                "sde-dpmsolver++ and does not set use_karras_sigmas."
            ),
            "steps": int(registry_entry["steps"]),
            "guidance": float(registry_entry["guidance"]),
            "registry_native_resolution": {
                "width": int(registry_entry["width"]),
                "height": int(registry_entry["height"]),
            },
            "negative_prompt": registry_entry["negative_prompt_default"],
            "dtype": "float16",
            "attention_slicing": "max",
            "bench_resolution_overrides_registry": True,
            "negative_contains_text_suppression": "text" in str(
                registry_entry["negative_prompt_default"]
            ).lower(),
        },
        "variants": selected_variants(args.variants),
        "prompts": prompts,
        "output_dir": args.output,
        "hf_cache": args.hf_cache,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bench Z-Image-Turbo GGUF variants against the juggernautXL SDXL baseline."
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants. Known: q8_0, q5_k_m, bf16, juggernautxl. Default: %(default)s",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=10,
        help="Extra timed repeats of the first prompt after the first image. Default: %(default)s",
    )
    parser.add_argument(
        "--turbo-steps",
        type=int,
        default=9,
        help="Z-Image num_inference_steps. 9 is the model-card schedule for 8 DiT forwards.",
    )
    parser.add_argument(
        "--turbo-guidance",
        type=float,
        default=0.0,
        help="Turbo guidance. Keep at 0. A value above 0 turns classifier-free guidance on.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--sdxl-checkpoint",
        default="",
        help="Override the juggernautXL path. Default: the path in server/manifests/registry.json.",
    )
    parser.add_argument("--hf-cache", default="", help="Hugging Face cache directory. Default: the library default.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", default="", help="Output directory. Default: tools/bench/outputs/<utc timestamp>.")
    parser.add_argument("--limit-prompts", type=int, default=0, help="Use only the first N prompts. For short trials.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan as JSON and exit.")
    parser.add_argument(
        "--check-support",
        action="store_true",
        help="Import diffusers and report whether Z-Image GGUF loading is available.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke test with a tiny fake pipeline. Writes reports under a temp directory unless --output is set.",
    )
    args = parser.parse_args(argv)
    if args.width % 16 or args.height % 16:
        parser.error("width and height must be multiples of 16")
    if args.width < 64 or args.height < 64:
        parser.error("width and height must be at least 64")
    if args.warm_runs < 1:
        parser.error("--warm-runs must be at least 1")
    if args.turbo_steps < 1:
        parser.error("--turbo-steps must be at least 1")
    return args


def query_nvidia_smi(gpu_index: int) -> Optional[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    line = output.strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "gpu_name": parts[0],
            "memory_total_mib": int(float(parts[1])),
            "memory_used_mib": int(float(parts[2])),
            "utilization_gpu": int(float(parts[3])),
        }
    except ValueError:
        return None


def torch_gpu_report(gpu_index: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch
    except ImportError:
        report["torch"] = "not installed"
        return report
    report["torch"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if not report["cuda_available"]:
        return report
    try:
        props = torch.cuda.get_device_properties(gpu_index)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    total = int(props.total_memory)
    report.update(
        {
            "gpu_index": gpu_index,
            "gpu_name": props.name,
            "total_vram_bytes": total,
            "total_vram_mib": bytes_to_mib(total),
            "total_vram_gb": round(total / float(1024**3), 2),
            "worker_would_treat_as_low_vram": (total / float(1024**3)) <= WORKER_LOW_VRAM_GB,
            "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        }
    )
    return report


class PeakMonitor:
    """Sample process RSS, system RAM, and nvidia-smi while a variant runs."""

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.peak_rss_bytes = 0
        self.peak_system_used_bytes = 0
        self.peak_smi_used_mib: Optional[int] = None
        self.peak_smi_sample: Optional[Dict[str, Any]] = None
        self.psutil_available = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample(self) -> None:
        try:
            import psutil

            self.psutil_available = True
            process = psutil.Process()
            rss = int(process.memory_info().rss)
            system_used = int(psutil.virtual_memory().used)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_system_used_bytes = max(self.peak_system_used_bytes, system_used)
        except ImportError:
            self.psutil_available = False
        except Exception:
            pass
        sample = query_nvidia_smi(self.gpu_index)
        if sample is None:
            return
        used = int(sample["memory_used_mib"])
        if self.peak_smi_used_mib is None or used >= self.peak_smi_used_mib:
            self.peak_smi_used_mib = used
            self.peak_smi_sample = sample

    def start(self) -> None:
        self.sample()

        def loop() -> None:
            while not self._stop.wait(0.5):
                self.sample()

        self._thread = threading.Thread(target=loop, name="vram-sample", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.sample()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "peak_process_rss_mib": bytes_to_mib(self.peak_rss_bytes) if self.psutil_available else None,
            "peak_system_used_mib": bytes_to_mib(self.peak_system_used_bytes) if self.psutil_available else None,
            "peak_nvidia_smi_used_mib": self.peak_smi_used_mib,
            "peak_nvidia_smi_sample": self.peak_smi_sample,
        }


def cuda_memory() -> Dict[str, Optional[float]]:
    try:
        import torch
    except ImportError:
        return {"allocated_mib": None, "reserved_mib": None, "max_allocated_mib": None, "max_reserved_mib": None}
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "max_allocated_mib": None, "max_reserved_mib": None}
    return {
        "allocated_mib": bytes_to_mib(torch.cuda.memory_allocated()),
        "reserved_mib": bytes_to_mib(torch.cuda.memory_reserved()),
        "max_allocated_mib": bytes_to_mib(torch.cuda.max_memory_allocated()),
        "max_reserved_mib": bytes_to_mib(torch.cuda.max_memory_reserved()),
    }


def reset_cuda_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def module_device(module: Any) -> Optional[str]:
    if module is None:
        return None
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return str(next(parameters()).device)
    except (StopIteration, Exception):
        return None


def module_stored_bytes(module: Any) -> Optional[int]:
    if module is None:
        return None
    total = 0
    try:
        tensors = list(module.parameters()) + list(module.buffers())
    except Exception:
        return None
    for tensor in tensors:
        try:
            total += int(tensor.numel()) * int(tensor.element_size())
        except Exception:
            return None
    return total


def component_report(pipe: Any) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name in ("text_encoder", "text_encoder_2", "transformer", "vae", "unet"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        report[name] = {
            "device": module_device(module),
            "stored_bytes": module_stored_bytes(module),
        }
    memory = cuda_memory()
    report["cuda_allocated_mib"] = memory["allocated_mib"]
    report["cuda_reserved_mib"] = memory["reserved_mib"]
    text_device = (report.get("text_encoder") or {}).get("device") or ""
    transformer_device = (report.get("transformer") or {}).get("device") or ""
    report["text_encoder_off_gpu"] = text_device.startswith("cpu")
    report["transformer_on_gpu"] = transformer_device.startswith("cuda")
    return report


def write_reports(directory: Path, payload: Dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "results.md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: Dict[str, Any]) -> str:
    gpu = payload.get("gpu") or {}
    lines = [
        "# Z-Image-Turbo bench",
        "",
        payload.get("route", {}).get("why", ""),
        "",
        f"- GPU: {gpu.get('gpu_name') or (gpu.get('nvidia_smi') or {}).get('gpu_name') or 'not measured'}",
        f"- Total VRAM: {gpu.get('total_vram_gb', 'not measured')} GB"
        f" ({gpu.get('total_vram_mib', 'n/a')} MiB from torch,"
        f" nvidia-smi total {((gpu.get('nvidia_smi') or {}).get('memory_total_mib', 'n/a'))} MiB)",
        f"- Planning assumption: {payload.get('gpu_planning_assumption_gb', 12)} GB, exact card size comes from this run",
        f"- Measured on GPU: {payload.get('measured_on_gpu')}",
        "",
        "| variant | status | oom | cold load s | first image s | warm s/img | warm n | peak allocated MiB | peak nvidia-smi MiB | peak RSS MiB | residual allocated MiB | residual nvidia-smi MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("variants") or []:
        lines.append(
            "| {id} | {status} | {oom} | {cold} | {first} | {warm} | {warm_n} | {alloc} | {smi} | {rss} | {residual} | {residual_smi} |".format(
                id=_cell(row.get("id")),
                status=_cell(row.get("status")),
                oom="yes" if row.get("oom") else "no",
                cold=_num(row.get("cold_load_seconds")),
                first=_num(row.get("first_image_seconds")),
                warm=_num(row.get("warm_seconds_per_image")),
                warm_n=_cell(row.get("warm_runs_completed")),
                alloc=_num((row.get("peak_vram") or {}).get("max_allocated_mib")),
                smi=_num((row.get("peak_host") or {}).get("peak_nvidia_smi_used_mib")),
                rss=_num((row.get("peak_host") or {}).get("peak_process_rss_mib")),
                residual=_num((row.get("after_unload") or {}).get("allocated_mib")),
                residual_smi=_num((row.get("after_unload") or {}).get("nvidia_smi_used_mib")),
            )
        )
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("Per-image records, module devices, and the prompt text are in `results.json`.")
    lines.append("")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "/")


def _num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_contact_sheet(
    directory: Path,
    variants: List[Dict[str, Any]],
    prompts: List[Dict[str, Any]],
) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        eprint("Pillow is not installed, so the contact sheet was skipped.")
        return None
    thumb = 256
    label_w = 150
    label_h = 32
    columns = max(1, len(variants))
    rows = max(1, len(prompts))
    sheet = Image.new("RGB", (label_w + columns * thumb, label_h + rows * thumb), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col, variant in enumerate(variants):
        draw.text((label_w + col * thumb + 8, 8), str(variant.get("id")), fill=(230, 230, 230), font=font)
    for row, prompt in enumerate(prompts):
        draw.text((8, label_h + row * thumb + 8), str(prompt.get("id")), fill=(230, 230, 230), font=font)
        for col, variant in enumerate(variants):
            tile = Image.new("RGB", (thumb, thumb), (48, 48, 52))
            saved = ((variant.get("images") or {}).get(prompt["id"]) or {}).get("path")
            label = "missing"
            if saved and Path(saved).is_file():
                try:
                    with Image.open(saved) as source:
                        frame = source.convert("RGB")
                        frame.thumbnail((thumb, thumb))
                        left = (thumb - frame.width) // 2
                        top = (thumb - frame.height) // 2
                        tile.paste(frame, (left, top))
                    label = ""
                except OSError:
                    label = "unreadable"
            else:
                image_row = (variant.get("images") or {}).get(prompt["id"]) or {}
                if image_row.get("oom"):
                    label = "OOM"
                elif image_row.get("error"):
                    label = "error"
            if label:
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((12, thumb // 2), label, fill=(220, 180, 120), font=font)
            sheet.paste(tile, (label_w + col * thumb, label_h + row * thumb))
    destination = directory / "contact_sheet.png"
    sheet.save(destination)
    return destination


def run_variant(
    spec: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    settings: Dict[str, Any],
    image_dir: Path,
    backend: Any,
    gpu_index: int,
) -> Dict[str, Any]:
    """Load one variant, time it, then unload and record whatever VRAM is left."""
    record: Dict[str, Any] = {
        "id": spec["id"],
        "kind": spec.get("kind"),
        "status": "error",
        "oom": False,
        "error": "",
        "cold_load_seconds": None,
        "first_image_seconds": None,
        "warm_seconds_per_image": None,
        "warm_runs_requested": settings["warm_runs"],
        "warm_runs_completed": 0,
        "images": {},
        "warm_seconds": [],
        "prepare": {},
        "after_load": {},
        "denoise_step0": None,
        "peak_vram": {},
        "peak_host": {},
        "before_load": {},
        "after_unload": {},
    }
    monitor = PeakMonitor(gpu_index)
    pipe = None
    try:
        log(f"[{spec['id']}] preparing weights")
        started = time.perf_counter()
        record["prepare"] = backend.prepare(spec, settings)
        record["prepare"]["seconds"] = round_seconds(time.perf_counter() - started)
        reset_cuda_peak()
        record["before_load"] = {**cuda_memory(), "nvidia_smi": query_nvidia_smi(gpu_index)}
        monitor.start()
        log(f"[{spec['id']}] loading")
        started = time.perf_counter()
        pipe = backend.load(spec, settings)
        record["cold_load_seconds"] = round_seconds(time.perf_counter() - started)
        record["after_load"] = component_report(pipe)
        record["scheduler"] = backend.scheduler_name(pipe)

        sheet_times: List[Dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            image_record = generate_timed(backend, pipe, spec, prompt, settings, image_dir, save=True)
            record["images"][prompt["id"]] = image_record
            sheet_times.append(image_record)
            if index == 0:
                record["first_image_seconds"] = image_record.get("seconds")
                record["denoise_step0"] = backend.last_probe
                record["denoising_iterations"] = backend.last_iterations
            if image_record.get("oom"):
                record["oom"] = True

        warm_times: List[float] = []
        warm_prompt = prompts[0]
        for warm_index in range(settings["warm_runs"]):
            image_record = generate_timed(
                backend,
                pipe,
                spec,
                warm_prompt,
                settings,
                image_dir,
                save=False,
                label=f"warm_{warm_index:02d}",
            )
            if image_record.get("oom"):
                record["oom"] = True
                record["warm_seconds"].append(image_record)
                continue
            if image_record.get("seconds") is not None and not image_record.get("error"):
                warm_times.append(float(image_record["seconds"]))
            record["warm_seconds"].append({"seconds": image_record.get("seconds"), "error": image_record.get("error", "")})
        record["warm_runs_completed"] = len(warm_times)
        if warm_times:
            record["warm_seconds_per_image"] = round_seconds(sum(warm_times) / len(warm_times))
        if record["oom"] and any(item.get("path") for item in sheet_times):
            record["status"] = "partial_oom"
        elif record["oom"]:
            record["status"] = "oom"
        elif any(item.get("error") for item in sheet_times):
            record["status"] = "error"
        else:
            record["status"] = "ok"
    except Exception as exc:
        record["oom"] = is_oom(exc)
        record["status"] = "oom" if record["oom"] else "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=20)
        log(f"[{spec['id']}] {record['status']}: {record['error']}")
    finally:
        monitor.stop()
        record["peak_host"] = monitor.as_dict()
        record["peak_vram"] = cuda_memory()
        try:
            backend.release(pipe)
        except Exception as exc:
            record["release_error"] = f"{type(exc).__name__}: {exc}"
        pipe = None
        release_cuda_cache()
        record["after_unload"] = {**cuda_memory(), "nvidia_smi_used_mib": None}
        smi = query_nvidia_smi(gpu_index)
        if smi is not None:
            record["after_unload"]["nvidia_smi_used_mib"] = smi["memory_used_mib"]
            record["after_unload"]["nvidia_smi"] = smi
        log(
            f"[{spec['id']}] status={record['status']} cold={record['cold_load_seconds']} "
            f"first={record['first_image_seconds']} warm={record['warm_seconds_per_image']} "
            f"residual_allocated_mib={record['after_unload'].get('allocated_mib')}"
        )
    return record


def generate_timed(
    backend: Any,
    pipe: Any,
    spec: Dict[str, Any],
    prompt: Dict[str, Any],
    settings: Dict[str, Any],
    image_dir: Path,
    save: bool,
    label: str = "",
) -> Dict[str, Any]:
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "prompt_id": prompt["id"],
        "seed": prompt["seed"],
        "label": label or prompt["id"],
        "seconds": None,
        "oom": False,
        "error": "",
        "path": "",
    }
    try:
        image = backend.generate(pipe, spec, prompt, settings)
        result["seconds"] = round_seconds(time.perf_counter() - started)
        if save:
            destination_dir = image_dir / spec["id"]
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{prompt['id']}.png"
            image.save(destination)
            result["path"] = str(destination)
            result["width"], result["height"] = image.size
    except Exception as exc:
        result["seconds"] = round_seconds(time.perf_counter() - started)
        result["oom"] = is_oom(exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
        log(f"[{spec['id']}] {result['label']} failed: {result['error']}")
        release_cuda_cache()
    return result


class FakeBackend:
    """Tiny stand-in used by --smoke. One instance can be told to raise OOM."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.last_probe: Optional[Dict[str, Any]] = None
        self.last_iterations: Optional[int] = None

    def prepare(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"downloaded": False, "smoke": True}

    def load(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, str]:
        if spec.get("fail_load"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return {"id": spec["id"]}

    def scheduler_name(self, pipe: Any) -> str:
        return "fake"

    def generate(self, pipe: Any, spec: Dict[str, Any], prompt: Dict[str, Any], settings: Dict[str, Any]) -> Any:
        from PIL import Image

        if self.fail or spec.get("fail_generate"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        time.sleep(0.01)
        self.last_iterations = 1
        self.last_probe = {
            "measured": False,
            "text_encoder": {"device": "cpu"},
            "transformer": {"device": "cpu"},
            "text_encoder_off_gpu": True,
            "transformer_on_gpu": False,
            "note": "smoke backend, not a GPU measurement",
        }
        color = ((int(prompt["seed"]) * 40) % 200 + 20, 70, 140 if spec["id"] == "smoke_ok" else 40)
        return Image.new("RGB", (64, 64), color)

    def release(self, pipe: Any) -> None:
        return None


def torch_dtype(name: str) -> Any:
    import torch

    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def call_with_dtype(fn: Callable[..., Any], dtype: Any, kwargs: Dict[str, Any]) -> Any:
    """diffusers has used both torch_dtype and dtype. Try the older name first."""
    try:
        return fn(torch_dtype=dtype, **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        return fn(dtype=dtype, **kwargs)


class ZImageBackend:
    def __init__(self) -> None:
        self.last_probe: Optional[Dict[str, Any]] = None
        self.last_iterations: Optional[int] = None
        self._snapshot: Optional[str] = None
        self._gguf_path: Optional[str] = None

    def prepare(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        from huggingface_hub import hf_hub_download, snapshot_download

        cache = settings.get("hf_cache") or None
        ignore = ["transformer/*.safetensors", "transformer/*.bin"]
        if spec["kind"] == "zimage_bf16":
            ignore = []
        self._snapshot = snapshot_download(
            repo_id=ZIMAGE_REPO,
            cache_dir=cache,
            ignore_patterns=ignore or None,
        )
        info: Dict[str, Any] = {"snapshot": self._snapshot, "gguf": ""}
        if spec["kind"] == "zimage_gguf":
            self._gguf_path = hf_hub_download(
                repo_id=spec["repo"],
                filename=spec["filename"],
                cache_dir=cache,
            )
            info["gguf"] = self._gguf_path
        return info

    def load(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Any:
        import torch
        from diffusers import ZImagePipeline

        dtype = torch_dtype(settings["dtype"])
        kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if spec["kind"] == "zimage_gguf":
            from diffusers import GGUFQuantizationConfig
            from diffusers.models import ZImageTransformer2DModel

            if not self._gguf_path:
                raise RuntimeError("GGUF file was not prepared before load")
            transformer = call_with_dtype(
                lambda **load_kwargs: ZImageTransformer2DModel.from_single_file(self._gguf_path, **load_kwargs),
                dtype,
                {
                    "quantization_config": GGUFQuantizationConfig(compute_dtype=dtype),
                    "local_files_only": True,
                },
            )
            kwargs["transformer"] = transformer
        source = self._snapshot or ZIMAGE_REPO
        pipe = call_with_dtype(
            lambda **load_kwargs: ZImagePipeline.from_pretrained(source, **load_kwargs),
            dtype,
            kwargs,
        )
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError("This diffusers build has no enable_model_cpu_offload")
        # Parks every module on CPU and streams one of text_encoder, transformer, vae to the GPU.
        pipe.enable_model_cpu_offload(gpu_id=settings["gpu_index"])
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def scheduler_name(self, pipe: Any) -> str:
        scheduler = getattr(pipe, "scheduler", None)
        return type(scheduler).__name__ if scheduler is not None else ""

    def generate(self, pipe: Any, spec: Dict[str, Any], prompt: Dict[str, Any], settings: Dict[str, Any]) -> Any:
        import torch

        self.last_probe = None
        self.last_iterations = 0

        def on_step(pipeline: Any, step: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            self.last_iterations = (self.last_iterations or 0) + 1
            if step == 0 and self.last_probe is None:
                probe = component_report(pipeline)
                probe["measured"] = True
                probe["step"] = int(step)
                self.last_probe = probe
            return callback_kwargs

        device = "cuda" if torch.cuda.is_available() else "cpu"
        execution = getattr(pipe, "_execution_device", None)
        if execution is not None:
            device = str(execution)
        generator = torch.Generator(device=device).manual_seed(int(prompt["seed"]))
        call = dict(
            prompt=prompt["text"],
            height=settings["height"],
            width=settings["width"],
            num_inference_steps=settings["turbo_steps"],
            guidance_scale=settings["turbo_guidance"],
            generator=generator,
        )
        if settings["turbo_guidance"] > 0 and settings.get("negative_prompt"):
            call["negative_prompt"] = settings["negative_prompt"]
        try:
            result = pipe(callback_on_step_end=on_step, **call)
        except TypeError as exc:
            if "callback_on_step_end" not in str(exc):
                raise
            result = pipe(**call)
        return result.images[0]

    def release(self, pipe: Any) -> None:
        release_pipeline(pipe)


class SdxlBackend:
    """Match the live worker: fp16, attention slicing, one DPMSolver swap, then .to(cuda)."""

    def __init__(self) -> None:
        self.last_probe: Optional[Dict[str, Any]] = None
        self.last_iterations: Optional[int] = None
        self.scheduler_config: Dict[str, Any] = {}

    def prepare(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(settings["sdxl_checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(
                f"juggernautXL checkpoint not found at {path}. "
                "Pass --sdxl-checkpoint with the Windows path."
            )
        return {"checkpoint": str(path), "downloaded": False}

    def load(self, spec: Dict[str, Any], settings: Dict[str, Any]) -> Any:
        import torch
        from diffusers import StableDiffusionXLPipeline

        dtype = torch.float16
        path = settings["sdxl_checkpoint"]
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                path,
                torch_dtype=dtype,
                safety_checker=None,
            )
        except TypeError:
            pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=dtype)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        apply_worker_sampler(pipe, settings["sdxl_sampler"])
        self.scheduler_config = scheduler_facts(pipe)
        xformers_flag = os.environ.get("HAI_ENABLE_XFORMERS", "").strip().lower()
        if xformers_flag in {"1", "true", "yes", "on"} and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            self.scheduler_config["xformers"] = True
        else:
            self.scheduler_config["xformers"] = False
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        return pipe

    def scheduler_name(self, pipe: Any) -> str:
        scheduler = getattr(pipe, "scheduler", None)
        name = type(scheduler).__name__ if scheduler is not None else ""
        if self.scheduler_config:
            return f"{name} {json.dumps(self.scheduler_config, sort_keys=True)}"
        return name

    def generate(self, pipe: Any, spec: Dict[str, Any], prompt: Dict[str, Any], settings: Dict[str, Any]) -> Any:
        import torch

        self.last_iterations = settings["sdxl_steps"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(prompt["seed"]))
        # The worker clamps guidance into [1, 15] and steps into [5, 50].
        steps = max(5, min(50, int(settings["sdxl_steps"])))
        guidance = max(1.0, min(15.0, float(settings["sdxl_guidance"])))
        result = pipe(
            prompt["text"],
            negative_prompt=settings.get("negative_prompt") or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            height=settings["height"],
            width=settings["width"],
        )
        self.last_probe = component_report(pipe)
        self.last_probe["measured"] = torch.cuda.is_available()
        self.last_probe["note"] = "sampled after the SDXL call; the worker keeps the whole pipeline on one device"
        return result.images[0]

    def release(self, pipe: Any) -> None:
        release_pipeline(pipe)


def apply_worker_sampler(pipe: Any, sampler: str) -> None:
    """Copy of client.py _apply_image_sampler, kept local so this script stays standalone."""
    if not sampler or not hasattr(pipe, "scheduler"):
        return
    sampler_norm = sampler.replace("+", "p")
    if "dpmpp" not in sampler_norm:
        return
    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


def scheduler_facts(pipe: Any) -> Dict[str, Any]:
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    facts: Dict[str, Any] = {"class": type(pipe.scheduler).__name__}
    if config is None:
        return facts
    for key in ("algorithm_type", "use_karras_sigmas", "solver_order", "prediction_type"):
        if key in config:
            value = config[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                facts[key] = value
    return facts


def release_pipeline(pipe: Any) -> None:
    """Drop a pipeline the way the worker drops its single cached slot."""
    if pipe is None:
        return
    remove_hooks = getattr(pipe, "remove_all_hooks", None)
    if callable(remove_hooks):
        try:
            remove_hooks()
        except Exception:
            pass
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    components = getattr(pipe, "components", None)
    modules = list(components.values()) if isinstance(components, dict) else []
    if torch is not None:
        for module in modules:
            if isinstance(module, torch.nn.Module):
                try:
                    module.to("cpu")
                except Exception:
                    pass
        if not modules:
            try:
                pipe.to("cpu")
            except Exception:
                pass
    del pipe


def make_backend(spec: Dict[str, Any]) -> Any:
    if spec["kind"] in {"zimage_gguf", "zimage_bf16"}:
        return ZImageBackend()
    if spec["kind"] == "sdxl":
        return SdxlBackend()
    raise RuntimeError(f"No backend for {spec['kind']}")


def runtime_settings(args: argparse.Namespace, plan: Dict[str, Any]) -> Dict[str, Any]:
    sdxl = plan["sdxl"]
    return {
        "width": args.width,
        "height": args.height,
        "warm_runs": args.warm_runs,
        "turbo_steps": args.turbo_steps,
        "turbo_guidance": args.turbo_guidance,
        "dtype": args.dtype,
        "hf_cache": args.hf_cache,
        "gpu_index": args.gpu_index,
        "sdxl_checkpoint": sdxl["checkpoint"]["path"],
        "sdxl_sampler": sdxl["sampler_in_registry"],
        "sdxl_steps": sdxl["steps"],
        "sdxl_guidance": sdxl["guidance"],
        "negative_prompt": sdxl["negative_prompt"],
    }


def output_directory(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_ROOT / stamp


def run_bench(args: argparse.Namespace, plan: Dict[str, Any]) -> int:
    directory = output_directory(args)
    settings = runtime_settings(args, plan)
    prompts = plan["prompts"]
    gpu = {
        "nvidia_smi": query_nvidia_smi(args.gpu_index),
        **torch_gpu_report(args.gpu_index),
    }
    if gpu.get("nvidia_smi"):
        log(
            f"GPU {gpu['nvidia_smi']['gpu_name']} "
            f"total {gpu['nvidia_smi']['memory_total_mib']} MiB (nvidia-smi)"
        )
    elif gpu.get("gpu_name"):
        log(f"GPU {gpu['gpu_name']} total {gpu.get('total_vram_mib')} MiB (torch)")
    else:
        log("No CUDA GPU was visible. Real variants will fail until this runs on the 3060.")
    notes = [
        "Cold load is timed after weights are already on disk. Download time is in prepare.seconds.",
        "The first image is prompt portrait_nova (or the first selected prompt) and is not part of the warm mean.",
        "Warm seconds are extra repeats of that first prompt at the same seed.",
        "Residual VRAM is measured after the worker-style unload: hooks off, modules to CPU, gc, empty_cache.",
        "SDXL uses the juggernautXL registry negative prompt on every image. That negative does not ban text.",
        "Turbo guidance stays at 0, so a negative prompt is not applied. sign_astra is still the on-image text check.",
        plan["route"]["why"],
    ]
    payload: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_on_gpu": bool(gpu.get("cuda_available")),
        "gpu_planning_assumption_gb": 12,
        "gpu": gpu,
        "route": plan["route"],
        "plan": {
            "resolution": plan["resolution"],
            "turbo": plan["turbo"],
            "sdxl": plan["sdxl"],
            "prompts": prompts,
            "variants": [item["id"] for item in plan["variants"]],
        },
        "notes": notes,
        "variants": [],
    }
    write_reports(directory, payload)
    for spec in plan["variants"]:
        backend = make_backend(spec)
        record = run_variant(spec, prompts, settings, directory / "images", backend, args.gpu_index)
        payload["variants"].append(record)
        write_reports(directory, payload)
    sheet = build_contact_sheet(directory, payload["variants"], prompts)
    if sheet is not None:
        payload["contact_sheet"] = str(sheet)
        write_reports(directory, payload)
    log(f"Wrote {directory / 'results.json'}")
    log(f"Wrote {directory / 'results.md'}")
    return 0


def check_support() -> int:
    try:
        import diffusers
        import torch
        from diffusers import GGUFQuantizationConfig, ZImagePipeline
        from diffusers.models import ZImageTransformer2DModel
    except Exception as exc:
        eprint(f"import failed: {type(exc).__name__}: {exc}")
        eprint("Install tools/bench/requirements.txt on top of a CUDA torch build, then retry.")
        return 2
    sequence = getattr(ZImagePipeline, "model_cpu_offload_seq", None)
    has_loader = hasattr(ZImageTransformer2DModel, "from_single_file")
    log(f"diffusers {diffusers.__version__}")
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"ZImageTransformer2DModel.from_single_file={has_loader}")
    log(f"GGUFQuantizationConfig={GGUFQuantizationConfig}")
    log(f"ZImagePipeline.model_cpu_offload_seq={sequence}")
    if has_loader and sequence == "text_encoder->transformer->vae":
        log("support ok: GGUF transformer load plus text-encoder offload sequence")
        return 0
    eprint("support check failed: this diffusers build is missing the GGUF loader or the offload sequence")
    return 2


def run_smoke(args: argparse.Namespace) -> int:
    """Exercise the harness on CPU with a 64px fake pipeline, including one OOM."""
    import tempfile

    from PIL import Image

    directory = Path(args.output).expanduser() if args.output else Path(tempfile.mkdtemp(prefix="zimage-smoke-"))
    prompts = [
        {"id": "portrait_nova", "seed": 41001, "kind": "smoke", "text": "smoke portrait"},
        {"id": "sign_astra", "seed": 41005, "kind": "smoke", "text": "smoke sign"},
    ]
    settings = {
        "width": 64,
        "height": 64,
        "warm_runs": 2,
        "turbo_steps": 1,
        "turbo_guidance": 0.0,
        "dtype": "bf16",
        "hf_cache": "",
        "gpu_index": 0,
        "sdxl_checkpoint": "",
        "sdxl_sampler": "dpm++_sde_karras",
        "sdxl_steps": 30,
        "sdxl_guidance": 6.0,
        "negative_prompt": "",
    }
    specs = [
        {"id": "smoke_ok", "kind": "fake"},
        {"id": "smoke_oom", "kind": "fake", "fail_generate": True},
    ]
    payload: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_on_gpu": False,
        "gpu_planning_assumption_gb": 12,
        "gpu": {"nvidia_smi": query_nvidia_smi(0), "note": "smoke does not load a model"},
        "route": ROUTE,
        "notes": ["Smoke run. Timings and the contact sheet are from a fake pipeline."],
        "variants": [],
    }
    backend_ok = FakeBackend(fail=False)
    backend_oom = FakeBackend(fail=True)
    for spec, backend in zip(specs, (backend_ok, backend_oom)):
        payload["variants"].append(run_variant(spec, prompts, settings, directory / "images", backend, 0))
    sheet = build_contact_sheet(directory, payload["variants"], prompts)
    payload["contact_sheet"] = str(sheet) if sheet else ""
    write_reports(directory, payload)

    problems = []
    ok = payload["variants"][0]
    oom = payload["variants"][1]
    if ok["status"] != "ok":
        problems.append(f"smoke_ok status was {ok['status']}")
    if ok["warm_runs_completed"] != 2 or not ok["warm_seconds_per_image"]:
        problems.append("warm mean was not computed from 2 runs")
    if not ok["first_image_seconds"]:
        problems.append("first image was not timed")
    if not ok["images"]["portrait_nova"].get("path"):
        problems.append("portrait image was not saved")
    if oom["status"] != "oom" or not oom["oom"]:
        problems.append("OOM variant did not continue as an OOM record")
    if ok.get("after_unload") is None:
        problems.append("unload residual was not recorded")
    results_path = directory / "results.json"
    if not results_path.is_file() or not (directory / "results.md").is_file():
        problems.append("reports were not written")
    if sheet is None or not sheet.is_file():
        problems.append("contact sheet was not written")
    else:
        with Image.open(sheet) as opened:
            if opened.size[0] < 100 or opened.size[1] < 100:
                problems.append(f"contact sheet is unexpectedly small: {opened.size}")
    # The OOM variant must not prevent the successful variant from staying in the report.
    if [row["id"] for row in payload["variants"]] != ["smoke_ok", "smoke_oom"]:
        problems.append("a failed variant aborted the run")
    if problems:
        for problem in problems:
            eprint(f"smoke failed: {problem}")
        return 1
    log(f"smoke ok: {directory}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        plan = build_plan(args)
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
        checkpoint = plan["sdxl"]["checkpoint"]
        if not checkpoint["exists"]:
            eprint(f"juggernautXL checkpoint is not on this machine: {checkpoint['path']}")
        return 0
    if args.check_support:
        return check_support()
    if args.smoke:
        return run_smoke(args)
    plan = build_plan(args)
    return run_bench(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())

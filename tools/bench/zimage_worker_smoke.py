#!/usr/bin/env python3
"""Three Z-Image jobs through the worker runtime in one process.

havnai-node.service launches ``python -m client.client``. That process calls
``zimage_turbo.generate_job`` after evicting the SDXL slot. This script calls
that same function three times: a cold job, the same prompt again, then a
second prompt. Stop the node first so the 3060 is empty. ACE-Step should be
unloaded too; it was resident during the 456 s run.

    systemctl --user stop havnai-node.service
    HAI_ZIMAGE_TURBO=1 python tools/bench/zimage_worker_smoke.py \\
        --encoder-mode cpu \\
        --output /mnt/d/havnai-bench/zimage-worker.png

``--encoder-mode cpu`` (the default) keeps the text encoder in float32 on the
CPU and does not pad the prompt out to 512 tokens. Embeddings are cast to
bf16 when they move to the GPU. ``swap`` moves the transformer and VAE off
the GPU before the encoder is moved on, and needs about 9 GB free.
The SDXL caps are not read. Z-Image defaults to 1024x1024.

Requires diffusers>=0.43.0, gguf>=0.10.0, and a transformers build with
Qwen3Model, on top of the node's CUDA torch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))

import zimage_turbo  # noqa: E402


def _save(image: object, output: Path, label: str) -> Path:
    destination = output.with_name(f"{output.stem}-{label}{output.suffix or '.png'}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)  # type: ignore[attr-defined]
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Run three Z-Image jobs through the worker runtime.")
    parser.add_argument(
        "--prompt",
        default="a clean metal hangar wall in a starfighter bay, a sign that reads ASTRA, cyan rim light",
    )
    parser.add_argument(
        "--prompt-b",
        default="wide empty starfighter runway at night, blue nebula, no people",
    )
    parser.add_argument("--seed", type=int, default=41005)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--encoder-mode", choices=("cpu", "swap"), default="cpu")
    parser.add_argument("--output", default="zimage-worker.png")
    args = parser.parse_args()
    os.environ["HAI_ZIMAGE_TURBO"] = "1"
    os.environ["HAI_ZIMAGE_ENCODER_MODE"] = args.encoder_mode
    width, height = zimage_turbo.cap_resolution(args.width, args.height)
    steps = min(max(1, args.steps), zimage_turbo.max_steps())
    print(
        f"encoder_mode={args.encoder_mode} cap {width}x{height} steps={steps} guidance=0",
        flush=True,
    )
    output = Path(args.output).expanduser()
    jobs = (
        ("cold", args.prompt, args.seed),
        ("warm", args.prompt, args.seed),
        ("second", args.prompt_b, args.seed + 1),
    )
    for label, prompt, seed in jobs:
        started = time.perf_counter()
        image = zimage_turbo.generate_job(
            prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
        )
        elapsed = time.perf_counter() - started
        stats = zimage_turbo.RUNTIME.last_stats
        saved = _save(image, output, label)
        peak = stats.get("peak_allocated_mib")
        peak_text = "n/a" if peak is None else f"{peak:.1f}"
        print(
            f"{label} seconds={elapsed:.3f} "
            f"encode_seconds={stats.get('encode_seconds')} "
            f"denoise_seconds={stats.get('denoise_seconds')} "
            f"peak_allocated_mib={peak_text} "
            f"mode={stats.get('encoder_mode')} "
            f"file={saved}",
            flush=True,
        )
    zimage_turbo.unload()
    print("unloaded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One real Z-Image job through the worker runtime in client/zimage_turbo.py.

havnai-node.service launches ``python -m client.client``. That process calls
``zimage_turbo.generate_job`` after evicting the SDXL slot. This script calls
that same function. Stop the node first so the 3060 is empty.

    systemctl --user stop havnai-node.service
    HAI_ZIMAGE_TURBO=1 python tools/bench/zimage_worker_smoke.py \\
        --output /mnt/d/havnai-bench/zimage-worker.png

Requires diffusers>=0.43.0, gguf>=0.10.0, and a transformers build with
Qwen3Model, on top of the node's CUDA torch. The SDXL caps
(HAI_SDXL_CAP_WIDTH, HAI_SDXL_CAP_HEIGHT, HAI_SDXL_MAX_PIXELS,
HAI_SDXL_MAX_STEPS) are not read. Z-Image defaults to 1024x1024 via
HAI_ZIMAGE_MAX_WIDTH, HAI_ZIMAGE_MAX_HEIGHT, and HAI_ZIMAGE_MAX_PIXELS.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Z-Image job through the worker runtime.")
    parser.add_argument(
        "--prompt",
        default="a clean metal hangar wall in a starfighter bay, a sign that reads ASTRA, cyan rim light",
    )
    parser.add_argument("--seed", type=int, default=41005)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--output", default="zimage-worker.png")
    args = parser.parse_args()
    os.environ["HAI_ZIMAGE_TURBO"] = "1"
    width, height = zimage_turbo.cap_resolution(args.width, args.height)
    steps = min(max(1, args.steps), zimage_turbo.max_steps())
    print(
        f"cap {width}x{height} steps={steps} guidance=0 "
        f"limits={zimage_turbo.resolution_cap()}",
        flush=True,
    )
    started = time.perf_counter()
    image = zimage_turbo.generate_job(
        args.prompt,
        seed=args.seed,
        width=width,
        height=height,
        steps=steps,
    )
    elapsed = time.perf_counter() - started
    destination = Path(args.output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(f"wrote {destination} in {elapsed:.1f}s size={image.size}", flush=True)
    zimage_turbo.unload()
    print("unloaded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

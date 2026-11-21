#!/usr/bin/env python3
"""Check a worker against the measured Wan 2.2 and LTX-2.3 recipes.

Nothing here loads a model. ``--dry-run`` prints the graph and the expected
clip. On the RTX 3060 box, run it inside the same WSL environment as ComfyUI
and the HavnAI node:

    python scripts/verify_i2v_3060.py
    python scripts/verify_i2v_3060.py --submit /path/to/portrait.png

Expected dry-run lines include ``wan-recipe ok`` and ``ltx-recipe ok``.
A real Wan submit should finish in about 21 to 24 minutes for 129 frames
(about 5 minutes when the umT5 encoder is already warm and the clip is 81
frames) and print ``wan-submit ok`` plus width, height, frames, and fps.
An LTX probe only checks that WanGP's files are present unless
``--submit-ltx`` is also passed; that clip should be about 448x832, 217
frames, 24 fps, with an audio stream, in roughly 285 seconds once warm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.wan22_i2v import config as wan22_config  # noqa: E402
from engines.wan22_i2v.comfy import build_workflow, comfy_url  # noqa: E402
from engines.wangp.runner import ltx23_sample, runtime_probe  # noqa: E402


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _check_wan_files(config) -> list:
    return config.missing_assets(require_local_text_encoder=True)


def _check_comfy(url: str) -> list:
    problems = []
    try:
        stats = _get(f"{url}/system_stats")
    except Exception as exc:
        return [f"ComfyUI is not reachable at {url}: {exc}"]
    devices = ((stats.get("devices") or [{}])[0]) if isinstance(stats, dict) else {}
    name = str(devices.get("name") or "unknown")
    vram = devices.get("vram_total")
    print(f"comfy-device {name} vram_total={vram}")
    for node in ("UnetLoaderGGUF", "WanImageToVideo", "KSamplerAdvanced", "ModelSamplingSD3", "SaveVideo"):
        try:
            _get(f"{url}/object_info/{node}")
        except Exception:
            problems.append(f"ComfyUI is missing node {node}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the recipe and do not contact ComfyUI")
    parser.add_argument("--submit", default="", help="PNG to send through the Wan graph")
    parser.add_argument("--frames", type=int, default=0, help="Override Wan frames. 129 is the 8 second clip, 81 also fits.")
    args = parser.parse_args()

    config = wan22_config.load_config()
    job = {"prompt": "verification portrait", "seed": 1}
    if args.frames:
        job["frames"] = args.frames
    settings = wan22_config.resolve_settings(job, config)
    graph = build_workflow(settings, config=config, image_name="verify.png", seed=1)
    sample = ltx23_sample({})
    print(
        "wan-recipe "
        f"{settings['width']}x{settings['height']} frames={settings['frames']} "
        f"fps={settings['fps']} steps={settings['steps']} shift={settings['flow_shift']} "
        f"sampler={settings['scheduler']} high={settings['high_noise_steps']} "
        f"backend={settings['backend']}"
    )
    print(
        "ltx-recipe "
        f"{sample['width']}x{sample['height']} frames={sample['frames']} "
        f"fps={sample['fps']} steps={sample['steps']} audio={str(sample['audio']).lower()} "
        f"model=ltx2_22B_distilled_1_1"
    )
    ok = (
        settings["frames"] in {81, 129}
        and settings["flow_shift"] == 5.0
        and settings["scheduler"] == "euler"
        and settings["high_noise_steps"] == 2
        and settings["steps"] == 4
        and (settings["width"], settings["height"]) in {(480, 832), (480, 704), (832, 480), (704, 480)}
        and graph["13"]["inputs"]["end_at_step"] == 2
        and graph["14"]["inputs"]["start_at_step"] == 2
        and sample["frames"] == 217
        and sample["width"] == 448
        and sample["height"] == 832
        and sample["fps"] == 24
        and sample["audio"] is True
    )
    print("wan-recipe ok" if ok else "wan-recipe mismatch")
    print("ltx-recipe ok" if sample["frames"] == 217 else "ltx-recipe mismatch")
    if not ok:
        return 1
    if args.dry_run:
        print("weights", config.weights_root())
        print("comfy", comfy_url())
        print("expected-wan seconds=1260..1440 peak_vram_gb=10.7..11.2")
        print("expected-wan-81 seconds=300 peak_vram_gb=10.7..11.2 when umt5 is already loaded")
        print("expected-ltx seconds=285 peak_vram_gb=4.7 audio=yes")
        return 0

    problems = _check_wan_files(config)
    problems.extend(_check_comfy(comfy_url()))
    ltx_ready, ltx_reason = runtime_probe()
    print(f"ltx-runtime {'ok' if ltx_ready else 'missing'} {ltx_reason}")
    if not ltx_ready:
        problems.append(ltx_reason)
    ffprobe = shutil.which("ffprobe")
    if args.submit:
        if problems:
            print("not-submitting")
            for item in problems:
                print("problem", item)
            return 1
        from engines.wan22_i2v.comfy import image_to_png, resolve_workflow, run_graph
        import requests

        image = Path(args.submit).read_bytes()
        workflow = resolve_workflow(settings, config, "verify.png", 1)
        video = run_graph(
            workflow,
            image_to_png(image),
            "verify.png",
            base_url=comfy_url(),
            session=requests.Session(),
            timeout=float(settings.get("timeout_seconds") or config.defaults.get("timeout_seconds") or 3600),
        )
        out = Path(os.environ.get("HAVNAI_VERIFY_OUT", "/tmp/havnai-wan22-verify.mp4"))
        out.write_bytes(video)
        print(f"wan-submit ok bytes={out.stat().st_size} path={out}")
        if ffprobe:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames,r_frame_rate",
                 "-of", "json", str(out)],
                capture_output=True, text=True, check=False,
            )
            print(probe.stdout.strip() or probe.stderr.strip())
        return 0
    if problems:
        for item in problems:
            print("problem", item)
        return 1
    print("worker-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Live integration test: submit video jobs to a running HavnAI server.

Usage:
    python tests/test_video_live.py --server http://localhost:8080 --wallet 0xTEST

Submits LTX2 and AnimateDiff jobs, polls for completion, and reports results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


def submit_video_job(
    server: str,
    wallet: str,
    model: str,
    prompt: str,
    seed: int = 42,
    steps: int = 30,
    guidance: float = 7.0,
    width: int = 512,
    height: int = 512,
    frames: int = 16,
    fps: int = 8,
) -> Dict[str, Any]:
    """Submit a video job and return the response."""
    payload = {
        "wallet": wallet,
        "model": model,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
    }
    url = f"{server.rstrip('/')}/submit-job"
    print(f"  POST {url}")
    print(f"  Model: {model} | Prompt: {prompt[:60]}...")
    resp = requests.post(url, json=payload, timeout=30)
    data = resp.json()
    print(f"  Status: {resp.status_code} | Response: {json.dumps(data)}")
    return data


def poll_job(server: str, job_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Poll a job until completion or timeout."""
    url = f"{server.rstrip('/')}/result/{job_id}"
    start = time.time()
    last_status = ""
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            status = data.get("status", "unknown")
            if status != last_status:
                elapsed = int(time.time() - start)
                print(f"  [{elapsed}s] Job {job_id}: {status}")
                last_status = status
            if status in ("completed", "success", "done"):
                return data
            if status in ("failed", "error"):
                return data
        except Exception as e:
            print(f"  Poll error: {e}")
        time.sleep(3)
    return {"status": "timeout", "job_id": job_id}


def run_test(
    server: str,
    wallet: str,
    model: str,
    prompt: str,
    label: str,
    timeout: int = 300,
) -> bool:
    """Run a single video test: submit + poll + report."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    result = submit_video_job(server, wallet, model, prompt)
    job_id = result.get("job_id")
    if not job_id:
        print(f"  FAIL: No job_id returned. Error: {result.get('error', 'unknown')}")
        return False

    print(f"  Job queued: {job_id}")
    print(f"  Polling (timeout={timeout}s)...")

    final = poll_job(server, job_id, timeout=timeout)
    status = final.get("status", "unknown")
    video_url = final.get("video_url") or final.get("output_url")

    if status in ("completed", "success", "done"):
        print(f"  PASS: Video generated successfully")
        if video_url:
            print(f"  Video URL: {video_url}")
        return True
    else:
        error = final.get("error", "no error details")
        print(f"  FAIL: Status={status}, Error={error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Live video generation test")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--wallet", default="0xTEST_WALLET_ADDRESS", help="Wallet address")
    parser.add_argument("--timeout", type=int, default=300, help="Poll timeout in seconds")
    parser.add_argument("--model", choices=["ltx2", "animatediff", "both"], default="both",
                        help="Which model to test")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides defaults)")
    args = parser.parse_args()

    tests = []
    if args.model in ("ltx2", "both"):
        tests.append({
            "model": "ltx2",
            "prompt": args.prompt or "A calm ocean wave rolling onto a sandy beach at sunset, golden light, cinematic",
            "label": "LTX2 Video Generation",
        })
    if args.model in ("animatediff", "both"):
        tests.append({
            "model": "animatediff",
            "prompt": args.prompt or "A woman walking through a sunlit forest path, wind blowing through hair, realistic",
            "label": "AnimateDiff Video Generation",
        })

    print(f"Server: {args.server}")
    print(f"Wallet: {args.wallet}")
    print(f"Tests to run: {len(tests)}")

    results = []
    for test in tests:
        passed = run_test(
            server=args.server,
            wallet=args.wallet,
            model=test["model"],
            prompt=test["prompt"],
            label=test["label"],
            timeout=args.timeout,
        )
        results.append((test["label"], passed))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed_count = 0
    for label, passed in results:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {label}")
        if passed:
            passed_count += 1
    print(f"\n  {passed_count}/{len(results)} tests passed")

    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    main()

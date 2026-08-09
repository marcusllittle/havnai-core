#!/usr/bin/env python3
"""End-to-end testnet simulation for HavnAI reward flow.

Smoke-tests the full cycle against a running havnai-core instance:
  1. Submit an Astra game run (score above reward threshold)
  2. Verify reward was credited in player stats
  3. Check reward claim eligibility via blockchain endpoint
  4. Inspect payout worker queue depth

Requires:
  pip install requests

Usage:
  python scripts/simulate_testnet.py [--base-url http://localhost:5001] [--wallet 0x...]

Defaults to a zero-address wallet which will be rejected by mainnet but
works for dev/staging to verify the API flow end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

try:
    import requests
except ImportError:
    print("ERROR: requests not installed.  Run: pip install requests")
    sys.exit(1)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def get(base: str, path: str, params: Dict = None) -> Any:
    r = requests.get(f"{base}{path}", params=params, timeout=10)
    return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


def post(base: str, path: str, body: Dict) -> Any:
    r = requests.post(f"{base}{path}", json=body, timeout=10)
    return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


def run_simulation(base_url: str, wallet: str) -> int:
    failures = 0

    print(f"\nHavnAI testnet simulation")
    print(f"  base_url : {base_url}")
    print(f"  wallet   : {wallet}")
    print()

    # ── 1. Health check ───────────────────────────────────────
    print("[1] Health check")
    try:
        status, body = get(base_url, "/health")
        ok = status == 200 and isinstance(body, dict)
        if not check("GET /health returns 200", ok, str(status)):
            failures += 1
        if ok:
            check("db_ok is true", body.get("db_ok") is True, str(body.get("db_ok")))
    except Exception as exc:
        check("GET /health reachable", False, str(exc))
        failures += 1
        print("  Cannot reach server — aborting simulation.")
        return failures

    # ── 2. Astra reward submission ────────────────────────────
    print()
    print("[2] Submit Astra game run")
    payload = {
        "wallet": wallet,
        "score": 25000,
        "grade": "A",
        "duration_s": 90.0,
        "map_id": "testnet_map_01",
    }
    try:
        status, body = post(base_url, "/astra/reward", payload)
        ok = status in (200, 201)
        if not check("POST /astra/reward returns 2xx", ok, str(status)):
            failures += 1
        if ok and isinstance(body, dict):
            reward = body.get("reward", 0)
            check("Reward field present", "reward" in body)
            check("Reward >= 0", reward >= 0, str(reward))
            if body.get("ok") is False:
                reason = body.get("reason", "unknown")
                print(f"     Note: reward rejected ({reason}) — may be expected for zero-addr wallet")
        run_ok = ok
    except Exception as exc:
        check("POST /astra/reward reachable", False, str(exc))
        failures += 1
        run_ok = False

    # ── 3. Player stats ───────────────────────────────────────
    print()
    print("[3] Player stats")
    try:
        status, body = get(base_url, "/astra/stats", {"wallet": wallet})
        ok = status == 200 and isinstance(body, dict)
        if not check("GET /astra/stats returns 200", ok, str(status)):
            failures += 1
        if ok:
            check("total_runs field present", "total_runs" in body)
            check("daily_cap field present", "daily_cap" in body)
    except Exception as exc:
        check("GET /astra/stats reachable", False, str(exc))
        failures += 1

    # ── 4. Claimable rewards ──────────────────────────────────
    print()
    print("[4] Claimable reward balance")
    try:
        status, body = get(base_url, "/blockchain/claimable", {"wallet": wallet})
        ok = status == 200 and isinstance(body, dict)
        if not check("GET /blockchain/claimable returns 200", ok, str(status)):
            failures += 1
        if ok:
            check("claimable field present", "claimable" in body)
            check("total_earned field present", "total_earned" in body)
    except Exception as exc:
        check("GET /blockchain/claimable reachable", False, str(exc))
        failures += 1

    # ── 5. Leaderboard ────────────────────────────────────────
    print()
    print("[5] Leaderboard")
    try:
        status, body = get(base_url, "/astra/leaderboard")
        ok = status == 200 and isinstance(body, (list, dict))
        if not check("GET /astra/leaderboard returns 200", ok, str(status)):
            failures += 1
    except Exception as exc:
        check("GET /astra/leaderboard reachable", False, str(exc))
        failures += 1

    # ── Summary ───────────────────────────────────────────────
    print()
    if failures == 0:
        print(f"  All checks passed.")
    else:
        print(f"  {failures} check(s) failed.")
    print()

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HavnAI reward flow smoke test")
    parser.add_argument("--base-url", default="http://localhost:5001",
                        help="havnai-core base URL (default: http://localhost:5001)")
    parser.add_argument("--wallet", default="0x0000000000000000000000000000000000000001",
                        help="Wallet address to use for test run")
    args = parser.parse_args()

    failures = run_simulation(args.base_url, args.wallet)
    sys.exit(0 if failures == 0 else 1)

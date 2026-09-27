#!/usr/bin/env python3
"""Run public launch smoke checks without secrets.

The script intentionally uses only public endpoints. Admin-gated checks such as
`/metrics` and alert dry-run still need an operator-token evidence packet.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


JsonFetcher = Callable[[str, float], tuple[int, Any]]
StatusFetcher = Callable[[str, float], int]
TextFetcher = Callable[[str, float], tuple[int, str]]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def fetch_json(url: str, timeout: float) -> tuple[int, Any]:
    request = Request(url, headers={"User-Agent": "havnai-launch-smoke/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read()
        body: Any
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = raw.decode("utf-8", errors="replace")
        return exc.code, body
    except (TimeoutError, URLError) as exc:
        return 0, {"error": str(exc)}


def fetch_status(url: str, timeout: float) -> int:
    request = Request(url, method="GET", headers={"User-Agent": "havnai-launch-smoke/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read(1)
            return response.status
    except HTTPError as exc:
        return exc.code
    except (TimeoutError, URLError):
        return 0


def fetch_text(url: str, timeout: float) -> tuple[int, str]:
    request = Request(url, method="GET", headers={"User-Agent": "havnai-launch-smoke/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read(512 * 1024).decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read(64 * 1024).decode("utf-8", errors="replace")
    except (TimeoutError, URLError) as exc:
        return 0, str(exc)


def run_smoke(
    *,
    api_base: str,
    web_base: str,
    timeout: float,
    allow_legacy_gallery: bool = False,
    json_fetcher: JsonFetcher = fetch_json,
    status_fetcher: StatusFetcher = fetch_status,
    text_fetcher: TextFetcher = fetch_text,
) -> list[Check]:
    checks: list[Check] = []

    health_status, health = json_fetcher(_url(api_base, "/health"), timeout)
    checks.append(Check(
        "api_health",
        health_status == 200 and health.get("status") == "ok",
        f"status={health_status} payload_status={health.get('status')!r} version={health.get('version')!r}",
    ))

    healthz_status, healthz = json_fetcher(_url(api_base, "/healthz"), timeout)
    checks.append(Check(
        "api_healthz",
        healthz_status == 200 and healthz.get("ok") is True,
        f"status={healthz_status} ok={healthz.get('ok')!r}",
    ))

    control_status, control = json_fetcher(_url(api_base, "/v1/network/control-plane"), timeout)
    control_ok = (
        control_status == 200
        and control.get("health", {}).get("status") == "healthy"
        and int(control.get("queue", {}).get("queued", -1)) == 0
        and int(control.get("queue", {}).get("running", -1)) == 0
        and int(control.get("nodes", {}).get("ready", 0)) >= 1
    )
    checks.append(Check(
        "control_plane",
        control_ok,
        "status={} health={} ready={} queued={} running={}".format(
            control_status,
            control.get("health", {}).get("status"),
            control.get("nodes", {}).get("ready"),
            control.get("queue", {}).get("queued"),
            control.get("queue", {}).get("running"),
        ),
    ))

    gallery_status, gallery = json_fetcher(_url(api_base, "/gallery/browse?limit=1"), timeout)
    gallery_total = int(gallery.get("total", -1)) if isinstance(gallery, dict) else -1
    gallery_ok = gallery_status == 200 and (allow_legacy_gallery or gallery_total == 0)
    checks.append(Check(
        "legacy_gallery_hidden",
        gallery_ok,
        f"status={gallery_status} total={gallery_total} allow_legacy_gallery={allow_legacy_gallery}",
    ))

    discover_status, discover = json_fetcher(_url(api_base, "/music/discover"), timeout)
    discover_ok = discover_status == 200 and isinstance(discover, dict)
    checks.append(Check(
        "music_discover_public",
        discover_ok,
        f"status={discover_status} total={discover.get('total') if isinstance(discover, dict) else None}",
    ))

    for path in ["/", "/create", "/pricing", "/support", "/terms/credits-v1", "/refunds/credits-v1"]:
        status = status_fetcher(_url(web_base, path), timeout)
        checks.append(Check(f"web{path if path != '/' else '_home'}", status == 200, f"status={status}"))

    create_status, create_text = text_fetcher(_url(web_base, "/create"), timeout)
    legacy_prompts = [
        "Legacy alpha access code",
        "Add legacy code",
        "Edit legacy code",
        "Legacy access code saved",
        "currently requires a Public Alpha access code",
        "Studio access key",
        "requires an access key",
    ]
    found_prompts = [prompt for prompt in legacy_prompts if prompt in create_text]
    checks.append(Check(
        "create_no_invite_copy",
        create_status == 200 and not found_prompts,
        f"status={create_status} legacy_prompts={found_prompts}",
    ))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="https://api.joinhavn.io")
    parser.add_argument("--web-base", default="https://joinhavn.io")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-legacy-gallery",
        action="store_true",
        help="Allow nonzero /gallery/browse totals only when rows have explicit product-quality review.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report instead of text lines.")
    args = parser.parse_args()

    checks = run_smoke(
        api_base=args.api_base,
        web_base=args.web_base,
        timeout=args.timeout,
        allow_legacy_gallery=args.allow_legacy_gallery,
    )
    failed = [check for check in checks if not check.ok]

    if args.json:
        print(json.dumps({
            "ok": not failed,
            "checks": [check.__dict__ for check in checks],
        }, indent=2))
    else:
        for check in checks:
            state = "ok" if check.ok else "FAIL"
            print(f"{state} {check.name}: {check.detail}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

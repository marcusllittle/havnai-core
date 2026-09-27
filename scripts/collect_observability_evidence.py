#!/usr/bin/env python3
"""Collect redacted HAVN-43/HAVN-17 observability evidence.

The collector is safe to run without secrets: public checks are always captured
and admin-gated checks are reported as missing when no token is supplied. For
final launch evidence, run it from an operator shell with
``--admin-token`` or ``HAVNAI_ADMIN_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError


REQUIRED_METRIC_GROUPS = {
    "jobs": ["havnai_jobs_total", "havnai_jobs"],
    "workers_online": ["havnai_worker_online", "havnai_nodes_online"],
    "disk_free": ["havnai_output_disk_free_bytes"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def redact_token(value: str) -> str:
    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def fetch(path: str, *, base_url: str, timeout: float, token: str = "") -> dict[str, Any]:
    headers = {"User-Agent": "havnai-observability-evidence/1"}
    if token:
        headers["X-HavnAI-Token"] = token
    req = request.Request(base_url + path, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            content_type = response.headers.get("content-type", "")
            return {"status": response.status, "content_type": content_type, "body": body}
    except HTTPError as exc:
        return {
            "status": exc.code,
            "content_type": exc.headers.get("content-type", ""),
            "body": exc.read().decode("utf-8", "replace"),
        }
    except (TimeoutError, URLError) as exc:
        return {"status": 0, "content_type": "", "body": str(exc)}


def parse_json(result: dict[str, Any]) -> Any:
    try:
        return json.loads(str(result.get("body") or "{}"))
    except json.JSONDecodeError:
        return {}


def summarize_metrics(result: dict[str, Any]) -> dict[str, Any]:
    body = str(result.get("body") or "")
    matched = {
        group: next((name for name in names if name in body), "")
        for group, names in REQUIRED_METRIC_GROUPS.items()
    }
    return {
        "status": result.get("status"),
        "content_type": result.get("content_type"),
        "required_metric_groups": matched,
        "required_metrics_present": [name for name in matched.values() if name],
        "required_metrics_missing": [group for group, name in matched.items() if not name],
        "line_count": len([line for line in body.splitlines() if line and not line.startswith("#")]),
    }


def summarize_control_plane(result: dict[str, Any]) -> dict[str, Any]:
    payload = parse_json(result)
    return {
        "status": result.get("status"),
        "schema_version": payload.get("schema_version"),
        "health": (payload.get("health") or {}).get("status"),
        "ready_nodes": (payload.get("nodes") or {}).get("ready"),
        "queued": (payload.get("queue") or {}).get("queued"),
        "running": (payload.get("queue") or {}).get("running"),
        "alerts": len(payload.get("alerts") or []),
    }


def summarize_alert_dry_run(result: dict[str, Any]) -> dict[str, Any]:
    payload = parse_json(result)
    matches = payload.get("matches") or payload.get("alerts") or []
    rules = payload.get("rules") or []
    if isinstance(payload.get("matched_count"), int):
        match_count = payload["matched_count"]
    elif isinstance(matches, list) and matches:
        match_count = len(matches)
    elif isinstance(rules, list):
        match_count = sum(1 for rule in rules if isinstance(rule, dict) and rule.get("matched") is True)
    else:
        match_count = 0
    return {
        "status": result.get("status"),
        "schema_version": payload.get("schema_version"),
        "delivery_mode": (payload.get("delivery") or {}).get("mode"),
        "sent": (payload.get("delivery") or {}).get("sent"),
        "match_count": match_count,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    base_url = clean_base_url(args.base_url)
    token = args.admin_token.strip()
    health = fetch("/health", base_url=base_url, timeout=args.timeout)
    healthz = fetch("/healthz", base_url=base_url, timeout=args.timeout)
    control = fetch("/v1/network/control-plane", base_url=base_url, timeout=args.timeout)

    metrics = fetch("/metrics", base_url=base_url, timeout=args.timeout, token=token)
    inject = parse.urlencode({"inject": args.inject})
    dry_run = fetch(f"/v1/network/alerts/dry-run?{inject}", base_url=base_url, timeout=args.timeout, token=token)

    health_payload = parse_json(health)
    healthz_payload = parse_json(healthz)
    metrics_summary = summarize_metrics(metrics)
    dry_run_summary = summarize_alert_dry_run(dry_run)
    control_summary = summarize_control_plane(control)
    token_required_missing = not token or metrics.get("status") == 401 or dry_run.get("status") == 401
    public_ok = (
        health.get("status") == 200
        and health_payload.get("status") == "ok"
        and healthz.get("status") == 200
        and healthz_payload.get("ok") is True
        and control_summary.get("status") == 200
        and control_summary.get("health") == "healthy"
    )
    admin_ok = (
        metrics_summary.get("status") == 200
        and not metrics_summary.get("required_metrics_missing")
        and dry_run_summary.get("status") == 200
        and dry_run_summary.get("schema_version") == "network-alert-dry-run.v1"
    )
    return {
        "schema": "havn-43-observability-evidence.v1",
        "generated_at": utc_now(),
        "base_url": base_url,
        "admin_token_present": bool(token),
        "admin_token_hash": redact_token(token),
        "public_ok": public_ok,
        "admin_ok": admin_ok,
        "passed": public_ok and admin_ok,
        "missing": ["admin_token_metrics_and_alert_dry_run"] if token_required_missing else [],
        "health": {"status": health.get("status"), "payload_status": health_payload.get("status"),
                   "version": health_payload.get("version"), "nodes": health_payload.get("nodes"),
                   "queue_depth": health_payload.get("queue_depth")},
        "healthz": {"status": healthz.get("status"), "ok": healthz_payload.get("ok")},
        "control_plane": control_summary,
        "metrics": metrics_summary,
        "alert_dry_run": dry_run_summary,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--admin-token", default=os.environ.get("HAVNAI_ADMIN_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--inject", default="model_load_failures,gpu_vram_exhaustion")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = collect(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect redacted production observability evidence for HAVN-43.

The collector reads health, metrics, control-plane, and alert dry-run/send
surfaces without printing secrets. Alert delivery is considered blocked, not
failed, when the coordinator reports that no webhook is configured.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


JsonFetcher = Callable[[str, str, float, str | None, Any | None], tuple[int, Any]]
TextFetcher = Callable[[str, float, str | None], tuple[int, str]]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _token_header(admin_token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "havnai-observability-evidence/1"}
    if admin_token:
        headers["X-HavnAI-Token"] = admin_token
    return headers


def fetch_json(url: str, method: str, timeout: float, admin_token: str | None, payload: Any | None = None) -> tuple[int, Any]:
    data = None
    headers = _token_header(admin_token)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read()
        try:
            body: Any = json.loads(raw.decode("utf-8"))
        except Exception:
            body = raw.decode("utf-8", errors="replace")
        return exc.code, body
    except (TimeoutError, URLError) as exc:
        return 0, {"error": str(exc)}


def fetch_text(url: str, timeout: float, admin_token: str | None) -> tuple[int, str]:
    request = Request(url, headers=_token_header(admin_token))
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read(1024 * 1024).decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read(256 * 1024).decode("utf-8", errors="replace")
    except (TimeoutError, URLError) as exc:
        return 0, str(exc)


def _metric_present(metrics: str, name: str) -> bool:
    return re.search(rf"^{re.escape(name)}(?:\{{|\s)", metrics, flags=re.MULTILINE) is not None


def _metric_value(metrics: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}\s+([^\s]+)", metrics, flags=re.MULTILINE)
    return match.group(1) if match else None


def _load_alert_waiver(path: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not path:
        return None, []
    failures = []
    with open(path, "r", encoding="utf-8") as handle:
        waiver = json.load(handle)
    if not isinstance(waiver, dict):
        return None, ["alert waiver must be a JSON object"]
    for field in ("approved_by", "expires_at", "mitigation", "reason"):
        if not waiver.get(field):
            failures.append(f"alert waiver missing {field}")
    return waiver, failures


def _load_join_token_waiver(path: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not path:
        return None, []
    failures = []
    with open(path, "r", encoding="utf-8") as handle:
        waiver = json.load(handle)
    if not isinstance(waiver, dict):
        return None, ["join-token waiver must be a JSON object"]
    for field in ("approved_by", "expires_at", "mitigation", "reason"):
        if not waiver.get(field):
            failures.append(f"join-token waiver missing {field}")
    return waiver, failures


def collect_observability(
    *,
    api_base: str,
    admin_token: str | None,
    timeout: float,
    alert_waiver: str | None = None,
    join_token_present: bool | None = None,
    require_join_token: bool = True,
    join_token_waiver: str | None = None,
    json_fetcher: JsonFetcher = fetch_json,
    text_fetcher: TextFetcher = fetch_text,
) -> dict[str, Any]:
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    checks: list[Check] = []
    waiver, waiver_failures = _load_alert_waiver(alert_waiver)
    join_waiver, join_waiver_failures = _load_join_token_waiver(join_token_waiver)

    if join_token_present is None:
        join_token_present = bool(os.getenv("SERVER_JOIN_TOKEN", "").strip())
    checks.append(Check(
        "join_token_config",
        bool(join_token_present) or bool(join_waiver and not join_waiver_failures) or not require_join_token,
        "required={} configured={} waiver_provided={}".format(
            require_join_token,
            bool(join_token_present),
            bool(join_waiver),
        ),
    ))

    health_status, health = json_fetcher(_url(api_base, "/health"), "GET", timeout, None, None)
    checks.append(Check(
        "public_health",
        health_status == 200 and isinstance(health, dict) and health.get("status") == "ok",
        f"status={health_status} version={health.get('version') if isinstance(health, dict) else None}",
    ))

    control_status, control = json_fetcher(_url(api_base, "/v1/network/control-plane"), "GET", timeout, None, None)
    control_health = control.get("health", {}) if isinstance(control, dict) else {}
    control_queue = control.get("queue", {}) if isinstance(control, dict) else {}
    control_nodes = control.get("nodes", {}) if isinstance(control, dict) else {}
    checks.append(Check(
        "control_plane",
        control_status == 200 and control_health.get("status") in {"healthy", "ok"},
        "status={} health={} ready={} queued={} running={}".format(
            control_status,
            control_health.get("status"),
            control_nodes.get("ready"),
            control_queue.get("queued"),
            control_queue.get("running"),
        ),
    ))

    metrics_status, metrics_text = text_fetcher(_url(api_base, "/metrics"), timeout, admin_token)
    required_metrics = [
        "havnai_jobs",
        "havnai_nodes_online",
        "havnai_artifacts_total",
        "havnai_artifact_bytes",
        "havnai_output_disk_free_bytes",
    ]
    missing_metrics = [name for name in required_metrics if not _metric_present(metrics_text, name)]
    checks.append(Check(
        "metrics_scrape",
        metrics_status == 200 and not missing_metrics,
        f"status={metrics_status} missing={missing_metrics}",
    ))

    dry_status, dry_run = json_fetcher(
        _url(api_base, "/v1/network/alerts/dry-run?inject=model_load_failures,gpu_vram_exhaustion"),
        "GET",
        timeout,
        admin_token,
        None,
    )
    dry_delivery = dry_run.get("delivery", {}) if isinstance(dry_run, dict) else {}
    dry_rules = dry_run.get("rules", []) if isinstance(dry_run, dict) else []
    dry_matched = [
        rule.get("code") or rule.get("name") or "unknown"
        for rule in dry_rules
        if isinstance(rule, dict) and rule.get("matched")
    ]
    checks.append(Check(
        "alert_dry_run",
        dry_status == 200 and isinstance(dry_run, dict) and dry_run.get("schema_version") == "network-alert-dry-run.v1",
        f"status={dry_status} delivery_mode={dry_delivery.get('mode')} matched={dry_matched}",
    ))

    send_status, send = json_fetcher(
        _url(api_base, "/v1/network/alerts/send"),
        "POST",
        timeout,
        admin_token,
        {"inject": ["model_load_failures", "gpu_vram_exhaustion"]},
    )
    send_delivery = send.get("delivery", {}) if isinstance(send, dict) else {}
    delivery_reason = send_delivery.get("reason")
    delivery_blocked = delivery_reason == "alert_webhook_not_configured"
    checks.append(Check(
        "alert_delivery",
        send_status == 200 and (send_delivery.get("sent") is True or delivery_blocked),
        "status={} configured={} sent={} reason={}".format(
            send_status,
            send_delivery.get("configured"),
            send_delivery.get("sent"),
            delivery_reason,
        ),
    ))

    failures = [check for check in checks if not check.ok]
    blockers = []
    if delivery_blocked:
        if waiver and not waiver_failures:
            checks.append(Check(
                "alert_delivery_waiver",
                True,
                "approved_by={} expires_at={} mitigation_present=True".format(
                    waiver.get("approved_by"),
                    waiver.get("expires_at"),
                ),
            ))
        else:
            blockers.append("HAVNAI_ALERT_WEBHOOK is not configured; external alert delivery receipt still required or waived.")
    if waiver_failures:
        blockers.extend(waiver_failures)
    if not admin_token:
        blockers.append("HAVNAI_ADMIN_TOKEN was not provided; admin-gated evidence is expected to fail.")
    if require_join_token and not join_token_present:
        if join_waiver and not join_waiver_failures:
            checks.append(Check(
                "join_token_waiver",
                True,
                "approved_by={} expires_at={} mitigation_present=True".format(
                    join_waiver.get("approved_by"),
                    join_waiver.get("expires_at"),
                ),
            ))
        else:
            blockers.append("SERVER_JOIN_TOKEN is not configured; node join-token hardening proof is required or waived.")
    if join_waiver_failures:
        blockers.extend(join_waiver_failures)

    return {
        "schema_version": "havnai.observability-evidence.v1",
        "generated_at": generated_at,
        "api_base": api_base,
        "passed": not failures and not blockers,
        "checks": [check.__dict__ for check in checks],
        "summary": {
            "health_status": health.get("status") if isinstance(health, dict) else None,
            "health_version": health.get("version") if isinstance(health, dict) else None,
            "control_plane_status": control_health.get("status"),
            "nodes_ready": control_nodes.get("ready"),
            "queue_queued": control_queue.get("queued"),
            "queue_running": control_queue.get("running"),
            "metrics_status": metrics_status,
            "output_disk_free_bytes": _metric_value(metrics_text, "havnai_output_disk_free_bytes"),
            "alert_dry_run_matched": dry_matched,
            "alert_delivery": {
                "configured": send_delivery.get("configured"),
                "sent": send_delivery.get("sent"),
                "reason": delivery_reason,
                "destination_host_present": bool(send_delivery.get("destination_host")),
            },
            "alert_waiver": {
                "provided": bool(waiver),
                "approved_by": waiver.get("approved_by") if waiver else None,
                "expires_at": waiver.get("expires_at") if waiver else None,
                "mitigation_present": bool(waiver and waiver.get("mitigation")),
            },
            "join_token_config": {
                "required": require_join_token,
                "configured": bool(join_token_present),
                "waiver_provided": bool(join_waiver),
                "waiver_approved_by": join_waiver.get("approved_by") if join_waiver else None,
                "waiver_expires_at": join_waiver.get("expires_at") if join_waiver else None,
                "waiver_mitigation_present": bool(join_waiver and join_waiver.get("mitigation")),
            },
        },
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="https://api.joinhavn.io")
    parser.add_argument("--admin-token", default=os.getenv("HAVNAI_ADMIN_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--alert-waiver", help="Optional dated waiver JSON when HAVNAI_ALERT_WEBHOOK is not configured.")
    parser.add_argument("--join-token-waiver", help="Optional dated waiver JSON when SERVER_JOIN_TOKEN is not configured.")
    parser.add_argument("--no-require-join-token", action="store_true", help="Do not require SERVER_JOIN_TOKEN presence in the evidence packet.")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSON.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    report = collect_observability(
        api_base=args.api_base,
        admin_token=args.admin_token.strip() or None,
        timeout=args.timeout,
        alert_waiver=args.alert_waiver,
        require_join_token=not args.no_require_join_token,
        join_token_waiver=args.join_token_waiver,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

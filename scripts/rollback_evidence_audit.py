#!/usr/bin/env python3
"""Validate HAVN-46 rollback evidence or explicit waivers."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


REQUIRED_PACKET_FIELDS = [
    "environment",
    "operator",
    "utc_start",
    "utc_end",
    "source_version",
    "target_version",
    "health_checks",
    "outcome",
]


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _check_packet(path: str, surface: str) -> tuple[bool, list[str], dict[str, Any]]:
    data = _load_json(path)
    failures = [field for field in REQUIRED_PACKET_FIELDS if not data.get(field)]
    health = data.get("health_checks")
    if not isinstance(health, dict) or not health:
        failures.append("health_checks must be a non-empty object")
    elif any(value not in (True, "ok", "passed", 200) for value in health.values()):
        failures.append("one or more health_checks did not pass")
    if data.get("rollback_safe") is not True and not data.get("forward_fix_rationale"):
        failures.append("rollback_safe true or forward_fix_rationale is required")
    return not failures, failures, {"surface": surface, **data}


def _check_waiver(path: str, surface: str) -> tuple[bool, list[str], dict[str, Any]]:
    data = _load_json(path)
    failures = [
        field for field in ("approved_by", "expires_at", "mitigation", "reason")
        if not data.get(field)
    ]
    return not failures, failures, {"surface": surface, **data}


def audit_rollback_evidence(
    *,
    coordinator_packet: str | None,
    web_packet: str | None,
    node_packet: str | None,
    coordinator_waiver: str | None,
    web_waiver: str | None,
    node_waiver: str | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    for surface, packet, waiver in [
        ("coordinator", coordinator_packet, coordinator_waiver),
        ("web", web_packet, web_waiver),
        ("node", node_packet, node_waiver),
    ]:
        if packet:
            ok, failures, data = _check_packet(packet, surface)
            checks.append({"name": f"{surface}_rollback_packet", "ok": ok, "path": packet, "failures": failures})
        elif waiver:
            ok, failures, data = _check_waiver(waiver, surface)
            checks.append({"name": f"{surface}_rollback_waiver", "ok": ok, "path": waiver, "failures": failures})
        else:
            data = {"surface": surface}
            blockers.append(f"{surface} rollback packet or dated waiver is missing")
        data.pop("commands", None)
        data.pop("tokens", None)

    return {
        "schema_version": "havnai.rollback-evidence-audit.v1",
        "passed": all(check["ok"] for check in checks) and not blockers,
        "checks": checks,
        "summary": {
            "surfaces_checked": [check["name"].replace("_rollback_packet", "").replace("_rollback_waiver", "") for check in checks],
            "missing_surfaces": [blocker.split()[0] for blocker in blockers],
        },
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-packet")
    parser.add_argument("--web-packet")
    parser.add_argument("--node-packet")
    parser.add_argument("--coordinator-waiver")
    parser.add_argument("--web-waiver")
    parser.add_argument("--node-waiver")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_rollback_evidence(
        coordinator_packet=args.coordinator_packet,
        web_packet=args.web_packet,
        node_packet=args.node_packet,
        coordinator_waiver=args.coordinator_waiver,
        web_waiver=args.web_waiver,
        node_waiver=args.node_waiver,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"passed={report['passed']}")
        for check in report["checks"]:
            print(f"{'ok' if check['ok'] else 'FAIL'} {check['name']} {check['path']}")
        for blocker in report["blockers"]:
            print(f"BLOCKER {blocker}", file=sys.stderr)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

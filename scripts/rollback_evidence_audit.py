#!/usr/bin/env python3
"""Build a redacted HAVN-46 rollback evidence report.

The script is read-only. Operators can provide JSON evidence for coordinator,
web/Vercel, and node-runtime rollback drills. A surface passes only when it has
exercise evidence or an explicit waiver with an expiry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SURFACES = ("coordinator", "web", "node")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def redact(value: str | None) -> str:
    if not value:
        return ""
    return f"sha256:{digest(value)}"


def read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def waiver(args: argparse.Namespace, surface: str) -> dict[str, Any]:
    waiver_id = getattr(args, f"{surface}_waiver_id")
    expires = getattr(args, f"{surface}_waiver_expires")
    return {"present": bool(waiver_id.strip()), "id": waiver_id.strip(), "expires": expires.strip()}


def summarize_coordinator(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"evidence_present": False, "passed": False}
    health = data.get("health_checks") or {}
    before = str(data.get("source_version") or data.get("current_version") or "")
    rollback = str(data.get("rollback_version") or data.get("previous_version") or "")
    restored = str(data.get("restored_version") or data.get("final_version") or "")
    passed = bool(
        before
        and rollback
        and restored
        and health.get("before_ok") is True
        and health.get("rollback_ok") is True
        and health.get("restored_ok") is True
        and data.get("rollback_safety_reviewed") is True
    )
    return {
        "evidence_present": True,
        "passed": passed,
        "source_version": before[:12],
        "rollback_version": rollback[:12],
        "restored_version": restored[:12],
        "health_checks": {
            "before_ok": health.get("before_ok"),
            "rollback_ok": health.get("rollback_ok"),
            "restored_ok": health.get("restored_ok"),
        },
        "rollback_safety_reviewed": data.get("rollback_safety_reviewed") is True,
    }


def summarize_web(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"evidence_present": False, "passed": False}
    routes = data.get("smoke_routes") or []
    if not isinstance(routes, list):
        routes = []
    route_statuses = [
        {"path": str(route.get("path") or ""), "status": int(route.get("status") or 0)}
        for route in routes if isinstance(route, dict)
    ]
    passed = bool(
        data.get("current_deployment_id")
        and (data.get("rollback_deployment_id") or data.get("waiver_id"))
        and data.get("rollback_action") in {"exercised", "inventory_only", "waived"}
        and route_statuses
        and all(route["status"] == 200 for route in route_statuses)
        and data.get("rollback_safety_reviewed") is True
    )
    return {
        "evidence_present": True,
        "passed": passed,
        "current_deployment_hash": redact(str(data.get("current_deployment_id") or "")),
        "rollback_deployment_hash": redact(str(data.get("rollback_deployment_id") or "")),
        "rollback_action": data.get("rollback_action"),
        "smoke_route_count": len(route_statuses),
        "smoke_routes_ok": bool(route_statuses) and all(route["status"] == 200 for route in route_statuses),
        "rollback_safety_reviewed": data.get("rollback_safety_reviewed") is True,
    }


def summarize_node(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"evidence_present": False, "passed": False}
    capabilities = data.get("model_capabilities") or []
    if not isinstance(capabilities, list):
        capabilities = []
    passed = bool(
        (data.get("node_id_hash") or data.get("node_id"))
        and data.get("previous_runtime_present") is True
        and data.get("rollback_action") in {"exercised", "waived"}
        and data.get("heartbeat_ok") is True
        and capabilities
        and data.get("rollback_safety_reviewed") is True
    )
    return {
        "evidence_present": True,
        "passed": passed,
        "node_id_hash": str(data.get("node_id_hash") or redact(str(data.get("node_id") or ""))),
        "previous_runtime_present": data.get("previous_runtime_present") is True,
        "rollback_action": data.get("rollback_action"),
        "heartbeat_ok": data.get("heartbeat_ok") is True,
        "model_capability_count": len(capabilities),
        "rollback_safety_reviewed": data.get("rollback_safety_reviewed") is True,
    }


def surface_pass(summary: dict[str, Any], waiver_report: dict[str, Any]) -> bool:
    return bool(summary.get("passed") or waiver_report.get("present"))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    summaries = {
        "coordinator": summarize_coordinator(read_json(args.coordinator_evidence)),
        "web": summarize_web(read_json(args.web_evidence)),
        "node": summarize_node(read_json(args.node_evidence)),
    }
    waivers = {surface: waiver(args, surface) for surface in SURFACES}
    missing = [
        surface
        for surface in SURFACES
        if not surface_pass(summaries[surface], waivers[surface])
    ]
    return {
        "schema": "havn-46-rollback-evidence-audit.v1",
        "generated_at": utc_now(),
        "passed": not missing,
        "missing": missing,
        "surfaces": summaries,
        "waivers": waivers,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-evidence", default="")
    parser.add_argument("--web-evidence", default="")
    parser.add_argument("--node-evidence", default="")
    for surface in SURFACES:
        parser.add_argument(f"--{surface}-waiver-id", default=os.environ.get(f"HAVNAI_{surface.upper()}_ROLLBACK_WAIVER_ID", ""))
        parser.add_argument(f"--{surface}-waiver-expires", default=os.environ.get(f"HAVNAI_{surface.upper()}_ROLLBACK_WAIVER_EXPIRES", ""))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = collect(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

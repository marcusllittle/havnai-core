#!/usr/bin/env python3
"""Audit public production surfaces for HAVN-69 security evidence.

The audit is intentionally read-only. It checks anonymous public web/API
responses for obvious secret/internal-path leaks and verifies account-owned API
routes reject unauthenticated callers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


SECRET_PATTERNS = {
    "stripe_secret_key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{12,}"),
    "stripe_webhook_secret": re.compile(r"\bwhsec_[A-Za-z0-9]{12,}"),
    "clerk_secret_key": re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{12,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "internal_home_path": re.compile(r"/home/marcus/|/mnt/[a-z]/|C:\\\\Users\\\\", re.IGNORECASE),
    "sqlite_path": re.compile(r"\bledger\.db\b|\bhavnai\.db\b"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return cleaned


def fetch(url: str, timeout: float) -> dict[str, Any]:
    req = request.Request(url, headers={"User-Agent": "havnai-security-public-surface-audit/1"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            headers = {key.lower(): value for key, value in response.headers.items()}
            return {"status": response.status, "headers": headers, "body": body}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        headers = {key.lower(): value for key, value in exc.headers.items()}
        return {"status": exc.code, "headers": headers, "body": body}
    except (TimeoutError, URLError) as exc:
        return {"status": 0, "headers": {}, "body": str(exc), "error": str(exc)}


def scan_body(body: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(body)]


def audit_web(web_base: str, paths: list[str], timeout: float) -> dict[str, Any]:
    rows = []
    for path in paths:
        result = fetch(web_base + path, timeout)
        matches = scan_body(str(result.get("body") or ""))
        rows.append({
            "path": path,
            "status": result.get("status"),
            "bytes": len(str(result.get("body") or "").encode("utf-8")),
            "findings": matches,
        })
    return {
        "routes": rows,
        "passed": all(row["status"] == 200 and not row["findings"] for row in rows),
    }


def audit_protected_api(api_base: str, paths: list[str], timeout: float) -> dict[str, Any]:
    rows = []
    for path in paths:
        result = fetch(api_base + path, timeout)
        body = str(result.get("body") or "")
        headers = result.get("headers") or {}
        rows.append({
            "path": path,
            "status": result.get("status"),
            "cache_control": headers.get("cache-control", ""),
            "body_findings": scan_body(body),
            "body_preview": body[:120],
        })
    return {
        "routes": rows,
        "passed": all(row["status"] in {401, 403} and not row["body_findings"] for row in rows),
    }


def audit_public_api(api_base: str, paths: list[str], timeout: float) -> dict[str, Any]:
    rows = []
    for path in paths:
        result = fetch(api_base + path, timeout)
        body = str(result.get("body") or "")
        rows.append({
            "path": path,
            "status": result.get("status"),
            "body_findings": scan_body(body),
        })
    return {
        "routes": rows,
        "passed": all(row["status"] == 200 and not row["body_findings"] for row in rows),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-base", default="https://joinhavn.io")
    parser.add_argument("--api-base", default="https://api.joinhavn.io")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--web-paths", default="/,/create,/pricing,/support,/terms/credits-v1,/refunds/credits-v1")
    parser.add_argument("--public-api-paths", default="/health,/healthz,/models/list,/gallery/browse?limit=5,/music/discover?limit=5")
    parser.add_argument("--protected-api-paths", default="/v2/account,/v2/account/credits,/v2/astra/session,/v2/astra/stats")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    web_base = clean_base_url(args.web_base)
    api_base = clean_base_url(args.api_base)
    report = {
        "schema": "havn-69-public-security-surface-audit.v1",
        "generated_at": utc_now(),
        "web_base": web_base,
        "api_base": api_base,
        "web": audit_web(web_base, parse_csv(args.web_paths), args.timeout),
        "public_api": audit_public_api(api_base, parse_csv(args.public_api_paths), args.timeout),
        "protected_api": audit_protected_api(api_base, parse_csv(args.protected_api_paths), args.timeout),
    }
    report["passed"] = bool(report["web"]["passed"] and report["public_api"]["passed"] and report["protected_api"]["passed"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

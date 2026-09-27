#!/usr/bin/env python3
"""Collect redacted HAVN-14 public/private content-isolation evidence.

The collector is intentionally generic: operators provide the exact deployed
owner, cross-account, public media, and social-preview paths for the adult or
private test artifact under review. Tokens are never printed.
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


DENIED_STATUSES = {401, 403, 404}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def redact(value: str) -> str:
    if not value:
        return ""
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def build_url(base_url: str, path_or_url: str) -> str:
    value = path_or_url.strip()
    if value.startswith(("http://", "https://")):
        return value
    return parse.urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))


def fetch(url: str, *, timeout: float, token: str = "") -> dict[str, Any]:
    headers = {"User-Agent": "havnai-content-isolation-evidence/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(128 * 1024).decode("utf-8", "replace")
            return {"status": response.status, "headers": dict(response.headers), "body": body}
    except HTTPError as exc:
        return {
            "status": exc.code,
            "headers": dict(exc.headers),
            "body": exc.read(128 * 1024).decode("utf-8", "replace"),
        }
    except (TimeoutError, URLError) as exc:
        return {"status": 0, "headers": {}, "body": str(exc)}


def cache_is_no_store(headers: dict[str, Any]) -> bool:
    cache_control = str(headers.get("Cache-Control") or headers.get("cache-control") or "").lower()
    pragma = str(headers.get("Pragma") or headers.get("pragma") or "").lower()
    expires = str(headers.get("Expires") or headers.get("expires") or "")
    return "no-store" in cache_control or pragma == "no-cache" or expires == "0"


def body_contains_any(body: str, values: list[str]) -> list[str]:
    lowered = body.lower()
    return [value for value in values if value and value.lower() in lowered]


def summarize_denial(name: str, result: dict[str, Any], forbidden_terms: list[str]) -> dict[str, Any]:
    leaked_terms = body_contains_any(str(result.get("body") or ""), forbidden_terms)
    denied = result.get("status") in DENIED_STATUSES
    no_store = cache_is_no_store(result.get("headers") or {})
    return {
        "name": name,
        "status": result.get("status"),
        "denied": denied,
        "no_store": no_store,
        "leaked_terms": leaked_terms,
        "passed": denied and no_store and not leaked_terms,
    }


def summarize_allowed(name: str, result: dict[str, Any], forbidden_terms: list[str]) -> dict[str, Any]:
    leaked_terms = body_contains_any(str(result.get("body") or ""), forbidden_terms)
    return {
        "name": name,
        "status": result.get("status"),
        "allowed": result.get("status") == 200,
        "cache_control": result.get("headers", {}).get("Cache-Control") or result.get("headers", {}).get("cache-control"),
        "leaked_terms": leaked_terms,
        "passed": result.get("status") == 200 and not leaked_terms,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    base_url = clean_base_url(args.base_url)
    forbidden_terms = [term.strip() for term in args.forbidden_term if term.strip()]
    checks: list[dict[str, Any]] = []

    discover = fetch(build_url(base_url, "/music/discover"), timeout=args.timeout)
    discover_leaks = body_contains_any(str(discover.get("body") or ""), forbidden_terms)
    checks.append({
        "name": "music_discover_public",
        "status": discover.get("status"),
        "leaked_terms": discover_leaks,
        "passed": discover.get("status") == 200 and not discover_leaks,
    })

    for index, path in enumerate(args.expect_denied):
        result = fetch(build_url(base_url, path), timeout=args.timeout)
        checks.append(summarize_denial(f"public_denied_{index + 1}", result, forbidden_terms))

    if args.owner_url:
        owner = fetch(build_url(base_url, args.owner_url), timeout=args.timeout, token=args.owner_token)
        checks.append(summarize_allowed("owner_private_access", owner, forbidden_terms))
    else:
        checks.append({"name": "owner_private_access", "passed": False, "missing": "owner_url"})

    if args.cross_account_url:
        cross = fetch(build_url(base_url, args.cross_account_url), timeout=args.timeout, token=args.cross_account_token)
        checks.append(summarize_denial("cross_account_denial", cross, forbidden_terms))
    else:
        checks.append({"name": "cross_account_denial", "passed": False, "missing": "cross_account_url"})

    if args.social_preview_url:
        social = fetch(build_url(base_url, args.social_preview_url), timeout=args.timeout)
        checks.append(summarize_denial("social_preview_denial", social, forbidden_terms))
    else:
        checks.append({"name": "social_preview_denial", "passed": False, "missing": "social_preview_url"})

    failed = [check["name"] for check in checks if not check.get("passed")]
    return {
        "schema": "havn-14-content-isolation-evidence.v1",
        "generated_at": utc_now(),
        "base_url": base_url,
        "owner_token_present": bool(args.owner_token.strip()),
        "owner_token_hash": redact(args.owner_token.strip()),
        "cross_account_token_present": bool(args.cross_account_token.strip()),
        "cross_account_token_hash": redact(args.cross_account_token.strip()),
        "passed": not failed,
        "failed": failed,
        "checks": checks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--forbidden-term", action="append", default=[],
                        help="Term/id/title that must not appear in public or denial bodies.")
    parser.add_argument("--expect-denied", action="append", default=[],
                        help="Public URL/path that should return 401/403/404 for the private/adult artifact.")
    parser.add_argument("--owner-url", default="", help="Owner-only URL/path expected to return 200.")
    parser.add_argument("--owner-token", default=os.environ.get("HAVNAI_ISOLATION_OWNER_TOKEN", ""))
    parser.add_argument("--cross-account-url", default="", help="Same resource URL/path using another account token.")
    parser.add_argument("--cross-account-token", default=os.environ.get("HAVNAI_ISOLATION_OTHER_TOKEN", ""))
    parser.add_argument("--social-preview-url", default="", help="Public OG/social preview URL/path expected to deny.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = collect(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

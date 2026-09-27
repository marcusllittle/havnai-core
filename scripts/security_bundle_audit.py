#!/usr/bin/env python3
"""Scan public web pages and linked static assets for obvious secret leaks.

This is a read-only HAVN-69 helper. It fetches selected public pages, discovers
same-origin JavaScript/CSS/static links, and scans bounded response bodies for
secret keys, internal file paths, and SQLite/database path leaks.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import re
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Any
from urllib import parse, request
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


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        for key in ("src", "href"):
            value = attr.get(key)
            if value:
                self.links.add(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return cleaned


def fetch(url: str, timeout: float, max_bytes: int) -> dict[str, Any]:
    req = request.Request(url, headers={"User-Agent": "havnai-security-bundle-audit/1"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            headers = {key.lower(): value for key, value in response.headers.items()}
            return {
                "status": response.status,
                "headers": headers,
                "body": body[:max_bytes].decode("utf-8", "replace"),
                "truncated": len(body) > max_bytes,
            }
    except HTTPError as exc:
        body = exc.read(max_bytes + 1)
        headers = {key.lower(): value for key, value in exc.headers.items()}
        return {
            "status": exc.code,
            "headers": headers,
            "body": body[:max_bytes].decode("utf-8", "replace"),
            "truncated": len(body) > max_bytes,
        }
    except (TimeoutError, URLError) as exc:
        return {"status": 0, "headers": {}, "body": str(exc), "truncated": False, "error": str(exc)}


def scan_body(body: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(body)]


def should_follow(url: str, base: str) -> bool:
    parsed = parse.urlparse(url)
    base_parsed = parse.urlparse(base)
    if parsed.scheme and parsed.netloc and parsed.netloc != base_parsed.netloc:
        return False
    path = parsed.path
    return (
        path.startswith("/_next/")
        or path.startswith("/static/")
        or path.startswith("/assets/")
        or path.startswith("/astra/")
        or path.startswith("/create/")
        or path.endswith((".js", ".css", ".json", ".map", ".txt", ".png", ".webp"))
    )


def discover_links(page_url: str, body: str, base: str) -> set[str]:
    parser = LinkParser()
    parser.feed(body)
    links: set[str] = set()
    for value in parser.links:
        absolute = parse.urljoin(page_url, value)
        if should_follow(absolute, base):
            links.add(absolute)
    return links


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def audit(base: str, paths: list[str], timeout: float, max_assets: int, max_bytes: int) -> dict[str, Any]:
    visited: set[str] = set()
    queue: deque[tuple[str, str]] = deque((parse.urljoin(base + "/", path), "page") for path in paths)
    rows: list[dict[str, Any]] = []

    while queue and len(visited) < max_assets:
        url, kind = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        result = fetch(url, timeout, max_bytes)
        body = str(result.get("body") or "")
        findings = scan_body(body)
        rows.append({
            "url": url,
            "kind": kind,
            "status": result.get("status"),
            "bytes_scanned": len(body.encode("utf-8")),
            "truncated": bool(result.get("truncated")),
            "findings": findings,
        })
        if kind == "page" and result.get("status") == 200:
            for link in sorted(discover_links(url, body, base)):
                if link not in visited and len(visited) + len(queue) < max_assets:
                    queue.append((link, "asset"))

    return {
        "assets_scanned": len(rows),
        "routes": rows,
        "passed": all(row["status"] == 200 and not row["findings"] for row in rows),
        "findings": [row for row in rows if row["findings"] or row["status"] != 200],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-base", default="https://joinhavn.io")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--paths", default="/,/create,/pricing,/support,/music,/discover,/library")
    parser.add_argument("--max-assets", type=int, default=120)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    base = clean_base_url(args.web_base)
    report = {
        "schema": "havn-69-public-bundle-secret-audit.v1",
        "generated_at": utc_now(),
        "web_base": base,
        "max_assets": args.max_assets,
        "max_bytes_per_asset": args.max_bytes,
        "audit": audit(base, parse_csv(args.paths), args.timeout, args.max_assets, args.max_bytes),
    }
    report["passed"] = bool(report["audit"]["passed"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe anonymous public content-isolation surfaces.

This is intentionally read-only and unauthenticated. It checks for accidental
exposure of a known restricted publication ID or sensitive terms in public API,
web, media, and metadata surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any


DEFAULT_API = "https://api.joinhavn.io"
DEFAULT_WEB = "https://joinhavn.io"
DEFAULT_PATHS = ["/", "/music", "/create", "/pricing", "/support", "/terms/credits-v1", "/refunds/credits-v1"]


class MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: list[dict[str, str]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "meta":
            item = {key.lower(): value or "" for key, value in attrs}
            interesting = {"name", "property", "content"}
            self.meta.append({key: value for key, value in item.items() if key in interesting})

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def _request(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "havnai-public-isolation-probe/1.0"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(512_000)
            headers = dict(response.headers.items())
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(64_000)
        headers = dict(exc.headers.items())
        status = exc.code
    elapsed_ms = round((time.monotonic() - started) * 1000)
    text = body.decode("utf-8", errors="replace")
    return {
        "url": url,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "content_type": headers.get("Content-Type", ""),
        "cache_control": headers.get("Cache-Control", ""),
        "cf_cache_status": headers.get("CF-Cache-Status", ""),
        "x_vercel_cache": headers.get("X-Vercel-Cache", ""),
        "body_size_bytes": len(body),
        "text": text,
    }


def _contains_any(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def _json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _redact_response(result: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    leaked_terms = _contains_any(result["text"], terms)
    return {
        "url": result["url"],
        "status": result["status"],
        "content_type": result["content_type"],
        "cache_control": result["cache_control"],
        "cf_cache_status": result["cf_cache_status"],
        "x_vercel_cache": result["x_vercel_cache"],
        "body_size_bytes": result["body_size_bytes"],
        "elapsed_ms": result["elapsed_ms"],
        "leaked_terms": leaked_terms,
    }


def _web_meta_summary(text: str, terms: list[str]) -> dict[str, Any]:
    parser = MetaParser()
    parser.feed(text)
    meta_text = json.dumps({"title": parser.title, "meta": parser.meta}, sort_keys=True)
    return {
        "title_present": bool(parser.title.strip()),
        "meta_count": len(parser.meta),
        "leaked_terms_in_html": _contains_any(text, terms),
        "leaked_terms_in_meta": _contains_any(meta_text, terms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--web-base", default=DEFAULT_WEB)
    parser.add_argument("--blocked-publication-id", required=True)
    parser.add_argument("--sensitive-term", action="append", default=[])
    parser.add_argument("--web-path", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=12)
    args = parser.parse_args()

    terms = [args.blocked_publication_id, *args.sensitive_term]
    api_base = args.api_base.rstrip("/")
    web_base = args.web_base.rstrip("/")
    encoded_search = urllib.parse.quote(args.sensitive_term[0] if args.sensitive_term else args.blocked_publication_id)
    web_paths = args.web_path or DEFAULT_PATHS

    checks: list[dict[str, Any]] = []

    discover = _request(f"{api_base}/music/discover", args.timeout)
    discover_json = _json_load(discover["text"]) or {}
    discover_ids = [str(item.get("id", "")) for item in discover_json.get("publications", []) if isinstance(item, dict)]
    checks.append({
        "name": "music_discover_excludes_blocked_publication",
        "ok": discover["status"] == 200 and args.blocked_publication_id not in discover_ids and not _contains_any(discover["text"], terms),
        "detail": {
            **_redact_response(discover, terms),
            "publication_count": len(discover_ids),
            "blocked_publication_present": args.blocked_publication_id in discover_ids,
        },
    })

    search = _request(f"{api_base}/music/discover?search={encoded_search}", args.timeout)
    search_json = _json_load(search["text"]) or {}
    search_ids = [str(item.get("id", "")) for item in search_json.get("publications", []) if isinstance(item, dict)]
    checks.append({
        "name": "music_search_excludes_sensitive_query",
        "ok": search["status"] == 200 and args.blocked_publication_id not in search_ids and not _contains_any(search["text"], terms),
        "detail": {
            **_redact_response(search, terms),
            "publication_count": len(search_ids),
            "blocked_publication_present": args.blocked_publication_id in search_ids,
        },
    })

    denied_paths = [
        f"/music/publications/{args.blocked_publication_id}",
        f"/music/publications/{args.blocked_publication_id}/audio",
        f"/music/publications/{args.blocked_publication_id}/cover",
    ]
    for path in denied_paths:
        result = _request(f"{api_base}{path}", args.timeout)
        safe_status = result["status"] in {401, 403, 404, 405, 410}
        no_leaks = not _contains_any(result["text"], terms)
        private_cache = "private" in result["cache_control"].lower() or "no-store" in result["cache_control"].lower() or not result["cache_control"]
        checks.append({
            "name": f"api_denies_{path.strip('/').replace('/', '_')}",
            "ok": safe_status and no_leaks and private_cache,
            "detail": _redact_response(result, terms),
        })

    logs = _request(f"{api_base}/logs", args.timeout)
    checks.append({
        "name": "public_logs_denied_without_leakage",
        "ok": logs["status"] in {401, 403, 404} and not _contains_any(logs["text"], terms),
        "detail": _redact_response(logs, terms),
    })

    for path in web_paths:
        if not path.startswith("/"):
            path = f"/{path}"
        result = _request(f"{web_base}{path}", args.timeout)
        meta_summary = _web_meta_summary(result["text"], terms)
        checks.append({
            "name": f"web_{path.strip('/') or 'home'}_no_restricted_meta_or_body",
            "ok": result["status"] == 200 and not meta_summary["leaked_terms_in_html"] and not meta_summary["leaked_terms_in_meta"],
            "detail": {**_redact_response(result, terms), **meta_summary},
        })

    report = {
        "schema_version": "havnai.public-content-isolation-probe.v1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_base": api_base,
        "web_base": web_base,
        "blocked_publication_id": args.blocked_publication_id,
        "sensitive_terms_checked": args.sensitive_term,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Update only api.joinhavn.io and verify the authoritative A record."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request


def fetch_text(url: str, *, timeout: int = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace").strip()


def dig(*arguments: str) -> list[str]:
    result = subprocess.run(
        ["dig", "+short", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return [line.rstrip(".") for line in result.stdout.splitlines() if line.strip()]


def authoritative_ipv4(hostname: str, domain: str) -> set[str]:
    nameservers = dig("NS", domain)
    if not nameservers:
        raise RuntimeError(f"no authoritative nameserver found for {domain}")
    addresses: set[str] = set()
    for nameserver in nameservers:
        addresses.update(dig(f"@{nameserver}", "A", hostname))
    return addresses


def alert(message: str) -> None:
    webhook = os.getenv("HAVNAI_ALERT_WEBHOOK", "").strip()
    if not webhook:
        return
    body = json.dumps({"service": "havnai-ddns", "error": message}).encode("utf-8")
    request = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except Exception:
        pass


def main() -> int:
    domain = os.getenv("DYNADOT_DOMAIN", "joinhavn.io").strip().lower()
    subdomain = os.getenv("DYNADOT_SUBDOMAIN", "api").strip().lower()
    password = os.getenv("DYNADOT_DDNS_PASSWORD", "").strip()
    ttl = int(os.getenv("DYNADOT_TTL", "300"))
    if not password:
        raise RuntimeError("DYNADOT_DDNS_PASSWORD is required")
    if domain != "joinhavn.io" or subdomain != "api":
        raise RuntimeError("this updater is intentionally restricted to api.joinhavn.io")
    if ttl != 300:
        raise RuntimeError("DYNADOT_TTL must remain 300 seconds")

    hostname = f"{subdomain}.{domain}"
    public_ip = fetch_text("https://api.ipify.org")
    if not public_ip or public_ip.count(".") != 3:
        raise RuntimeError("public IPv4 discovery returned an invalid value")

    current = authoritative_ipv4(hostname, domain)
    if current == {public_ip}:
        print(f"{hostname} already resolves authoritatively to {public_ip}")
        return 0

    query = urllib.parse.urlencode(
        {
            "containRoot": "false",
            "domain": domain,
            "ip": public_ip,
            "pwd": password,
            "subDomain": subdomain,
            "ttl": str(ttl),
            "type": "A",
        }
    )
    response = fetch_text(f"https://www.dynadot.com/set_ddns?{query}")
    lowered = response.lower()
    if "success" not in lowered and "good" not in lowered:
        raise RuntimeError("Dynadot did not confirm the DDNS update")

    for _ in range(12):
        time.sleep(5)
        if public_ip in authoritative_ipv4(hostname, domain):
            print(f"updated and verified {hostname} -> {public_ip}")
            return 0
    raise RuntimeError(f"authoritative DNS did not converge to {public_ip}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = str(exc)
        alert(message)
        print(f"DDNS update failed: {message}", file=sys.stderr)
        raise SystemExit(1)

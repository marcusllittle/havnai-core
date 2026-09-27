from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import content_isolation_evidence as evidence  # type: ignore


def test_content_isolation_packet_passes_without_leaking_tokens(monkeypatch, capsys) -> None:
    def fake_fetch(url: str, *, timeout: float, token: str = "") -> dict[str, object]:
        if url.endswith("/music/discover"):
            return {"status": 200, "headers": {}, "body": json.dumps({"total": 0, "publications": []})}
        if url.endswith("/public/audio"):
            return {"status": 404, "headers": {"Cache-Control": "no-store"}, "body": "{}"}
        if url.endswith("/owner/artifact"):
            assert token == "owner-secret"
            return {"status": 200, "headers": {"Cache-Control": "private, no-store"}, "body": "private binary"}
        if url.endswith("/other/artifact"):
            assert token == "other-secret"
            return {"status": 404, "headers": {"Pragma": "no-cache"}, "body": "{}"}
        if url.endswith("/share/card"):
            return {"status": 404, "headers": {"Expires": "0"}, "body": "{}"}
        raise AssertionError(url)

    monkeypatch.setattr(evidence, "fetch", fake_fetch)
    exit_code = evidence.main([
        "--base-url", "https://api.example.test",
        "--forbidden-term", "adult-fixture-title",
        "--expect-denied", "/public/audio",
        "--owner-url", "/owner/artifact",
        "--owner-token", "owner-secret",
        "--cross-account-url", "/other/artifact",
        "--cross-account-token", "other-secret",
        "--social-preview-url", "/share/card",
    ])
    stdout = capsys.readouterr().out
    report = json.loads(stdout)

    assert exit_code == 0
    assert report["passed"] is True
    assert {check["name"] for check in report["checks"]} == {
        "music_discover_public",
        "public_denied_1",
        "owner_private_access",
        "cross_account_denial",
        "social_preview_denial",
    }
    assert "owner-secret" not in stdout
    assert "other-secret" not in stdout
    assert report["owner_token_hash"].startswith("sha256:")
    assert report["cross_account_token_hash"].startswith("sha256:")


def test_public_only_packet_keeps_private_and_social_gaps_open(monkeypatch) -> None:
    def fake_fetch(url: str, *, timeout: float, token: str = "") -> dict[str, object]:
        if url.endswith("/music/discover"):
            return {"status": 200, "headers": {}, "body": "{}"}
        return {"status": 404, "headers": {"Cache-Control": "no-store"}, "body": "{}"}

    monkeypatch.setattr(evidence, "fetch", fake_fetch)
    report = evidence.collect(evidence.parse_args([
        "--base-url", "https://api.example.test",
        "--expect-denied", "/public/audio",
    ]))

    assert report["passed"] is False
    assert report["failed"] == ["owner_private_access", "cross_account_denial", "social_preview_denial"]
    missing = {check["name"]: check.get("missing") for check in report["checks"] if check.get("missing")}
    assert missing == {
        "owner_private_access": "owner_url",
        "cross_account_denial": "cross_account_url",
        "social_preview_denial": "social_preview_url",
    }

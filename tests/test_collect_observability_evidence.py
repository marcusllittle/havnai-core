from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_observability_evidence as evidence  # type: ignore


def test_collect_reports_missing_admin_token_without_leaking_secret(monkeypatch) -> None:
    def fake_fetch(path: str, *, base_url: str, timeout: float, token: str = "") -> dict[str, object]:
        if path == "/health":
            return {"status": 200, "content_type": "application/json", "body": json.dumps({"status": "ok", "version": "test"})}
        if path == "/healthz":
            return {"status": 200, "content_type": "application/json", "body": json.dumps({"ok": True})}
        if path == "/v1/network/control-plane":
            return {
                "status": 200,
                "content_type": "application/json",
                "body": json.dumps({
                    "schema_version": "network-control-plane.v1",
                    "health": {"status": "healthy"},
                    "nodes": {"ready": 1},
                    "queue": {"queued": 0, "running": 0},
                    "alerts": [],
                }),
            }
        assert path.startswith(("/metrics", "/v1/network/alerts/dry-run"))
        assert token == ""
        return {"status": 401, "content_type": "application/json", "body": json.dumps({"error": "unauthorized"})}

    monkeypatch.setattr(evidence, "fetch", fake_fetch)

    args = evidence.parse_args(["--base-url", "https://api.example.test"])
    report = evidence.collect(args)

    assert report["public_ok"] is True
    assert report["admin_ok"] is False
    assert report["passed"] is False
    assert report["missing"] == ["admin_token_metrics_and_alert_dry_run"]
    assert report["admin_token_hash"] == ""


def test_collect_passes_token_backed_metrics_and_alerts_without_printing_token(monkeypatch, capsys) -> None:
    def fake_fetch(path: str, *, base_url: str, timeout: float, token: str = "") -> dict[str, object]:
        assert base_url == "https://api.example.test"
        if path == "/health":
            return {"status": 200, "content_type": "application/json", "body": json.dumps({"status": "ok", "version": "test"})}
        if path == "/healthz":
            return {"status": 200, "content_type": "application/json", "body": json.dumps({"ok": True})}
        if path == "/v1/network/control-plane":
            return {
                "status": 200,
                "content_type": "application/json",
                "body": json.dumps({
                    "schema_version": "network-control-plane.v1",
                    "health": {"status": "healthy"},
                    "nodes": {"ready": 1},
                    "queue": {"queued": 0, "running": 0},
                    "alerts": [],
                }),
            }
        assert token == "secret-token"
        if path == "/metrics":
            return {
                "status": 200,
                "content_type": "text/plain",
                "body": "\n".join([
                    "havnai_jobs_total 1",
                    "havnai_worker_online 1",
                    "havnai_output_disk_free_bytes 100000",
                ]),
            }
        assert path.startswith("/v1/network/alerts/dry-run?")
        return {
            "status": 200,
            "content_type": "application/json",
            "body": json.dumps({
                "schema_version": "network-alert-dry-run.v1",
                "delivery": {"mode": "dry_run", "sent": False},
                "matches": [{"name": "model_load_failures"}],
            }),
        }

    monkeypatch.setattr(evidence, "fetch", fake_fetch)

    exit_code = evidence.main([
        "--base-url",
        "https://api.example.test",
        "--admin-token",
        "secret-token",
    ])
    stdout = capsys.readouterr().out
    report = json.loads(stdout)

    assert exit_code == 0
    assert report["passed"] is True
    assert report["admin_ok"] is True
    assert report["admin_token_hash"].startswith("sha256:")
    assert "secret-token" not in stdout
    assert report["metrics"]["required_metrics_missing"] == []
    assert report["alert_dry_run"]["schema_version"] == "network-alert-dry-run.v1"


def test_live_metric_aliases_satisfy_required_observability_groups() -> None:
    summary = evidence.summarize_metrics({
        "status": 200,
        "content_type": "text/plain",
        "body": "\n".join([
            'havnai_jobs{state="cancelled"} 69',
            "havnai_nodes_online 1",
            "havnai_output_disk_free_bytes 100000",
        ]),
    })

    assert summary["required_metrics_missing"] == []
    assert summary["required_metric_groups"] == {
        "jobs": "havnai_jobs",
        "workers_online": "havnai_nodes_online",
        "disk_free": "havnai_output_disk_free_bytes",
    }

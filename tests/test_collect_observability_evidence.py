from scripts.collect_observability_evidence import collect_observability
import json


METRICS = "\n".join([
    "havnai_jobs{state=\"queued\"} 0",
    "havnai_nodes_online 1",
    "havnai_artifacts_total 12",
    "havnai_artifact_bytes 4096",
    "havnai_output_disk_free_bytes 123456",
])


def _json_fetcher(*, delivery=None):
    if delivery is None:
        delivery = {"mode": "webhook", "configured": False, "sent": False, "reason": "alert_webhook_not_configured"}

    def fetch(url, method, timeout, admin_token, payload=None):
        if url.endswith("/health"):
            return 200, {"status": "ok", "version": "2a61013"}
        if url.endswith("/v1/network/control-plane"):
            return 200, {
                "health": {"status": "healthy"},
                "nodes": {"ready": 1},
                "queue": {"queued": 0, "running": 0},
            }
        if "/v1/network/alerts/dry-run" in url:
            assert admin_token == "admin-token"
            return 200, {
                "schema_version": "network-alert-dry-run.v1",
                "delivery": {"mode": "dry_run", "sent": False},
                "rules": [
                    {"code": "model_load_failures", "matched": True},
                    {"code": "gpu_vram_exhaustion", "matched": True},
                ],
            }
        if url.endswith("/v1/network/alerts/send"):
            assert method == "POST"
            assert payload == {"inject": ["model_load_failures", "gpu_vram_exhaustion"]}
            return 200, {
                "schema_version": "network-alert-send.v1",
                "delivery": delivery,
                "rules": [{"name": "model_load_failures", "matched": True}],
            }
        raise AssertionError(f"unexpected url {url}")

    return fetch


def _text_fetcher(metrics=METRICS):
    def fetch(url, timeout, admin_token):
        assert url.endswith("/metrics")
        assert admin_token == "admin-token"
        return 200, metrics

    return fetch


def test_collect_observability_reports_webhook_gap_as_blocker():
    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        join_token_present=True,
        json_fetcher=_json_fetcher(),
        text_fetcher=_text_fetcher(),
    )

    assert report["schema_version"] == "havnai.observability-evidence.v1"
    assert report["passed"] is False
    assert report["summary"]["health_version"] == "2a61013"
    assert report["summary"]["output_disk_free_bytes"] == "123456"
    assert report["summary"]["alert_delivery"]["configured"] is False
    assert report["summary"]["alert_delivery"]["reason"] == "alert_webhook_not_configured"
    assert report["blockers"] == [
        "HAVNAI_ALERT_WEBHOOK is not configured; external alert delivery receipt still required or waived."
    ]
    assert all(check["ok"] for check in report["checks"])


def test_collect_observability_accepts_dated_alert_waiver(tmp_path):
    waiver = tmp_path / "alert-waiver.json"
    waiver.write_text(json.dumps({
        "approved_by": "Marcus Little",
        "expires_at": "2026-10-04T00:00:00Z",
        "mitigation": "Manual health/control-plane review every hour while webhook is unavailable.",
        "reason": "Webhook destination not available during launch hardening.",
    }), encoding="utf-8")

    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        alert_waiver=str(waiver),
        join_token_present=True,
        json_fetcher=_json_fetcher(),
        text_fetcher=_text_fetcher(),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is True
    assert report["blockers"] == []
    assert checks["alert_delivery_waiver"]["ok"] is True
    assert report["summary"]["alert_waiver"] == {
        "provided": True,
        "approved_by": "Marcus Little",
        "expires_at": "2026-10-04T00:00:00Z",
        "mitigation_present": True,
    }


def test_collect_observability_rejects_incomplete_alert_waiver(tmp_path):
    waiver = tmp_path / "alert-waiver.json"
    waiver.write_text(json.dumps({
        "approved_by": "Marcus Little",
        "reason": "Webhook destination not available during launch hardening.",
    }), encoding="utf-8")

    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        alert_waiver=str(waiver),
        join_token_present=True,
        json_fetcher=_json_fetcher(),
        text_fetcher=_text_fetcher(),
    )

    assert report["passed"] is False
    assert "alert waiver missing expires_at" in report["blockers"]
    assert "alert waiver missing mitigation" in report["blockers"]
    assert "HAVNAI_ALERT_WEBHOOK is not configured; external alert delivery receipt still required or waived." in report["blockers"]


def test_collect_observability_passes_when_alert_delivery_sends():
    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        join_token_present=True,
        json_fetcher=_json_fetcher(delivery={
            "mode": "webhook",
            "configured": True,
            "sent": True,
            "destination_host": "hooks.example",
            "status_code": 200,
        }),
        text_fetcher=_text_fetcher(),
    )

    assert report["passed"] is True
    assert report["blockers"] == []
    assert report["summary"]["alert_delivery"]["configured"] is True
    assert report["summary"]["alert_delivery"]["sent"] is True
    assert report["summary"]["alert_delivery"]["destination_host_present"] is True


def test_collect_observability_fails_missing_required_metric():
    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        join_token_present=True,
        json_fetcher=_json_fetcher(delivery={"mode": "webhook", "configured": True, "sent": True}),
        text_fetcher=_text_fetcher(metrics="havnai_jobs{state=\"queued\"} 0\n"),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is False
    assert checks["metrics_scrape"]["ok"] is False
    assert "havnai_nodes_online" in checks["metrics_scrape"]["detail"]


def test_collect_observability_blocks_missing_join_token_config():
    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        join_token_present=False,
        json_fetcher=_json_fetcher(delivery={
            "mode": "webhook",
            "configured": True,
            "sent": True,
        }),
        text_fetcher=_text_fetcher(),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is False
    assert checks["join_token_config"]["ok"] is False
    assert report["summary"]["join_token_config"] == {
        "required": True,
        "configured": False,
        "waiver_provided": False,
        "waiver_approved_by": None,
        "waiver_expires_at": None,
        "waiver_mitigation_present": False,
    }
    assert report["blockers"] == [
        "SERVER_JOIN_TOKEN is not configured; node join-token hardening proof is required or waived."
    ]


def test_collect_observability_accepts_dated_join_token_waiver(tmp_path):
    waiver = tmp_path / "join-token-waiver.json"
    waiver.write_text(json.dumps({
        "approved_by": "Marcus Little",
        "expires_at": "2026-10-04T00:00:00Z",
        "mitigation": "Restrict node enrollment to known hosts and review /register logs daily.",
        "reason": "Existing production node has not been rotated to a new join token yet.",
    }), encoding="utf-8")

    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
        join_token_present=False,
        join_token_waiver=str(waiver),
        json_fetcher=_json_fetcher(delivery={
            "mode": "webhook",
            "configured": True,
            "sent": True,
        }),
        text_fetcher=_text_fetcher(),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is True
    assert report["blockers"] == []
    assert checks["join_token_config"]["ok"] is True
    assert checks["join_token_waiver"]["ok"] is True
    assert report["summary"]["join_token_config"] == {
        "required": True,
        "configured": False,
        "waiver_provided": True,
        "waiver_approved_by": "Marcus Little",
        "waiver_expires_at": "2026-10-04T00:00:00Z",
        "waiver_mitigation_present": True,
    }

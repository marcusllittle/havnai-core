from scripts.collect_observability_evidence import collect_observability


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
                    {"name": "model_load_failures", "matched": True},
                    {"name": "gpu_vram_exhaustion", "matched": True},
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


def test_collect_observability_passes_when_alert_delivery_sends():
    report = collect_observability(
        api_base="https://api.example",
        admin_token="admin-token",
        timeout=1,
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
        json_fetcher=_json_fetcher(delivery={"mode": "webhook", "configured": True, "sent": True}),
        text_fetcher=_text_fetcher(metrics="havnai_jobs{state=\"queued\"} 0\n"),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is False
    assert checks["metrics_scrape"]["ok"] is False
    assert "havnai_nodes_online" in checks["metrics_scrape"]["detail"]

from scripts.launch_public_smoke import run_smoke


def _json_fixture(gallery_total=0):
    def fetch(url, timeout):
        if url.endswith("/health"):
            return 200, {"status": "ok", "version": "test"}
        if url.endswith("/healthz"):
            return 200, {"ok": True}
        if url.endswith("/v1/network/control-plane"):
            return 200, {
                "health": {"status": "healthy"},
                "nodes": {"ready": 1},
                "queue": {"queued": 0, "running": 0},
            }
        if "/gallery/browse" in url:
            return 200, {"total": gallery_total, "listings": [] if gallery_total == 0 else [{"id": 15}]}
        if url.endswith("/music/discover"):
            return 200, {"total": 0, "publications": []}
        raise AssertionError(f"unexpected url {url}")
    return fetch


def _status_fixture(url, timeout):
    return 200


def _text_fixture(body="No access code needed"):
    def fetch(url, timeout):
        return 200, body
    return fetch


def test_launch_smoke_passes_when_public_surfaces_are_clean():
    checks = run_smoke(
        api_base="https://api.example",
        web_base="https://web.example",
        timeout=1,
        json_fetcher=_json_fixture(gallery_total=0),
        status_fetcher=_status_fixture,
        text_fetcher=_text_fixture(),
    )

    assert all(check.ok for check in checks)
    assert {check.name for check in checks} >= {
        "legacy_gallery_hidden",
        "api_health",
        "web/create",
        "create_no_invite_copy",
    }


def test_launch_smoke_fails_nonzero_legacy_gallery_by_default():
    checks = run_smoke(
        api_base="https://api.example",
        web_base="https://web.example",
        timeout=1,
        json_fetcher=_json_fixture(gallery_total=7),
        status_fetcher=_status_fixture,
        text_fetcher=_text_fixture(),
    )

    failures = {check.name: check for check in checks if not check.ok}
    assert set(failures) == {"legacy_gallery_hidden"}
    assert "total=7" in failures["legacy_gallery_hidden"].detail


def test_launch_smoke_can_record_explicit_legacy_gallery_waiver():
    checks = run_smoke(
        api_base="https://api.example",
        web_base="https://web.example",
        timeout=1,
        allow_legacy_gallery=True,
        json_fetcher=_json_fixture(gallery_total=7),
        status_fetcher=_status_fixture,
        text_fetcher=_text_fixture(),
    )

    assert all(check.ok for check in checks)
    gallery = next(check for check in checks if check.name == "legacy_gallery_hidden")
    assert "allow_legacy_gallery=True" in gallery.detail


def test_launch_smoke_fails_when_create_exposes_legacy_access_code_copy():
    checks = run_smoke(
        api_base="https://api.example",
        web_base="https://web.example",
        timeout=1,
        json_fetcher=_json_fixture(gallery_total=0),
        status_fetcher=_status_fixture,
        text_fetcher=_text_fixture("Legacy alpha access code"),
    )

    failures = {check.name: check for check in checks if not check.ok}
    assert set(failures) == {"create_no_invite_copy"}
    assert "Legacy alpha access code" in failures["create_no_invite_copy"].detail

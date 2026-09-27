import json

from scripts.backup_evidence_audit import audit_backup_evidence
from scripts.rollback_evidence_audit import audit_rollback_evidence


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_backup_audit_requires_remote_or_waiver(tmp_path):
    manifest = _write_json(tmp_path, "manifest.json", {
        "integrity_check": "ok",
        "backup_size_bytes": 1024,
        "local_mode_octal": "0o600",
        "local_retention_count": 5,
        "backup_sha256": "abc123",
        "created_at": "20260927T000000Z",
        "remote": {"configured": False},
    })
    restore = _write_json(tmp_path, "restore.json", {
        "verified": True,
        "integrity": "ok",
        "foreign_key_violations": 0,
        "table_count": 84,
    })
    media = _write_json(tmp_path, "media.json", {
        "verified": True,
        "source_sha256": "abc",
        "restored_sha256": "abc",
    })

    report = audit_backup_evidence(
        backup_manifest=manifest,
        restore_report=restore,
        media_reports=[media],
        remote_waiver=None,
        require_remote=True,
    )

    assert report["passed"] is False
    assert report["blockers"] == [
        "remote/offsite backup is not configured and no dated waiver was supplied"
    ]
    assert all(check["ok"] for check in report["checks"])


def test_backup_audit_accepts_dated_remote_waiver(tmp_path):
    manifest = _write_json(tmp_path, "manifest.json", {
        "integrity_check": "ok",
        "backup_size_bytes": 1024,
        "local_mode_octal": "0o600",
        "local_retention_count": 5,
        "backup_sha256": "abc123",
        "remote": {"configured": False},
    })
    restore = _write_json(tmp_path, "restore.json", {
        "verified": True,
        "integrity_check": "ok",
        "foreign_key_violations": [],
        "tables": {"accounts": 1},
    })
    waiver = _write_json(tmp_path, "waiver.json", {
        "approved_by": "Marcus Little",
        "expires_at": "2026-10-01",
        "mitigation": "Daily local encrypted copies retained until remote target is configured.",
    })

    report = audit_backup_evidence(
        backup_manifest=manifest,
        restore_report=restore,
        media_reports=[],
        remote_waiver=waiver,
        require_remote=True,
    )

    assert report["passed"] is True
    assert report["summary"]["remote_configured"] is False


def test_rollback_audit_requires_all_surfaces_or_waivers(tmp_path):
    coordinator = _write_json(tmp_path, "coordinator.json", {
        "environment": "production-like",
        "operator": "Codex",
        "utc_start": "2026-09-27T00:00:00Z",
        "utc_end": "2026-09-27T00:10:00Z",
        "source_version": "old",
        "target_version": "new",
        "health_checks": {"local_healthz": True, "public_healthz": 200},
        "outcome": "restored previous version",
        "rollback_safe": True,
    })

    report = audit_rollback_evidence(
        coordinator_packet=coordinator,
        web_packet=None,
        node_packet=None,
        coordinator_waiver=None,
        web_waiver=None,
        node_waiver=None,
    )

    assert report["passed"] is False
    assert report["blockers"] == [
        "web rollback packet or dated waiver is missing",
        "node rollback packet or dated waiver is missing",
    ]


def test_rollback_audit_accepts_web_and_node_waivers(tmp_path):
    coordinator = _write_json(tmp_path, "coordinator.json", {
        "environment": "production-like",
        "operator": "Codex",
        "utc_start": "2026-09-27T00:00:00Z",
        "utc_end": "2026-09-27T00:10:00Z",
        "source_version": "old",
        "target_version": "new",
        "health_checks": {"local_healthz": True},
        "outcome": "forward fix selected",
        "forward_fix_rationale": "Schema change is not backward-safe.",
    })
    waiver = {
        "approved_by": "Marcus Little",
        "expires_at": "2026-10-01",
        "mitigation": "Keep previous deployment available and smoke manually.",
        "reason": "No production rollback action approved during this window.",
    }
    web = _write_json(tmp_path, "web-waiver.json", waiver)
    node = _write_json(tmp_path, "node-waiver.json", waiver)

    report = audit_rollback_evidence(
        coordinator_packet=coordinator,
        web_packet=None,
        node_packet=None,
        coordinator_waiver=None,
        web_waiver=web,
        node_waiver=node,
    )

    assert report["passed"] is True
    assert report["blockers"] == []

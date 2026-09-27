from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rollback_evidence_audit as audit  # type: ignore


def write_json(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def coordinator(path: Path) -> Path:
    return write_json(path, {
        "source_version": "2f8c8e38cd98da991e71a64de89ba9a2b85060b1",
        "rollback_version": "ad7164dbacc5a0351546b88ca7b1558a0b34f36e",
        "restored_version": "2f8c8e38cd98da991e71a64de89ba9a2b85060b1",
        "health_checks": {"before_ok": True, "rollback_ok": True, "restored_ok": True},
        "rollback_safety_reviewed": True,
    })


def web(path: Path) -> Path:
    return write_json(path, {
        "current_deployment_id": "dpl_current_secret",
        "rollback_deployment_id": "dpl_previous_secret",
        "rollback_action": "inventory_only",
        "rollback_safety_reviewed": True,
        "smoke_routes": [
            {"path": "/", "status": 200},
            {"path": "/create", "status": 200},
        ],
    })


def node(path: Path) -> Path:
    return write_json(path, {
        "node_id": "node-secret",
        "previous_runtime_present": True,
        "rollback_action": "exercised",
        "heartbeat_ok": True,
        "model_capabilities": ["image", "video"],
        "rollback_safety_reviewed": True,
    })


def test_full_rollback_evidence_passes_and_redacts_identifiers(tmp_path: Path) -> None:
    report = audit.collect(audit.parse_args([
        "--coordinator-evidence", str(coordinator(tmp_path / "coordinator.json")),
        "--web-evidence", str(web(tmp_path / "web.json")),
        "--node-evidence", str(node(tmp_path / "node.json")),
    ]))
    encoded = json.dumps(report)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["surfaces"]["coordinator"]["source_version"] == "2f8c8e38cd98"
    assert report["surfaces"]["web"]["smoke_routes_ok"] is True
    assert report["surfaces"]["node"]["heartbeat_ok"] is True
    assert "dpl_current_secret" not in encoded
    assert "dpl_previous_secret" not in encoded
    assert "node-secret" not in encoded


def test_missing_web_and_node_require_evidence_or_waiver(tmp_path: Path) -> None:
    report = audit.collect(audit.parse_args([
        "--coordinator-evidence", str(coordinator(tmp_path / "coordinator.json")),
    ]))

    assert report["passed"] is False
    assert report["missing"] == ["web", "node"]


def test_explicit_waivers_can_satisfy_unexercised_surfaces(tmp_path: Path) -> None:
    report = audit.collect(audit.parse_args([
        "--coordinator-evidence", str(coordinator(tmp_path / "coordinator.json")),
        "--web-waiver-id", "HAVN-46-web-waiver",
        "--web-waiver-expires", "2026-10-10",
        "--node-waiver-id", "HAVN-46-node-waiver",
        "--node-waiver-expires", "2026-10-10",
    ]))

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["waivers"]["web"]["present"] is True
    assert report["waivers"]["node"]["present"] is True

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_coordinator_deploy_defaults_match_live_service_layout():
    for name in ("deploy_coordinator_release.sh", "deploy_flagship.sh"):
        body = _script(name)
        assert "COORDINATOR_REPO=\"${COORDINATOR_REPO:-/home/marcus/Downloads/source-code/havnai-core}\"" in body
        assert "COORDINATOR_DB_PATH=\"${COORDINATOR_DB_PATH:-/home/marcus/Downloads/source-code/havnai-core/db/ledger.db}\"" in body
        assert ".venv/bin/python -m pip install -r server/requirements.txt" in body
        assert 'HAVNAI_DB_PATH="$COORDINATOR_DB_PATH" HAVNAI_BACKUP_DIR="$COORDINATOR_BACKUP_DIR" .venv/bin/python scripts/backup_coordinator.py' in body
        assert "sudo -u havnai" not in body
        assert "/var/lib/havnai/ledger.db" not in body


def test_coordinator_deploy_preserves_nodes_json_across_forward_and_rollback_paths():
    body = _script("deploy_coordinator_release.sh")
    assert 'nodes_backup="/tmp/havnai-$SHA.nodes.json"' in body
    assert 'if [ -f nodes.json ]; then cp nodes.json "$nodes_backup"; fi' in body
    assert "git checkout -- nodes.json 2>/dev/null || true" in body
    assert 'if [ -f "$nodes_backup" ]; then cp "$nodes_backup" nodes.json; fi' in body
    assert 'git switch --detach "$previous"' in body
    assert "sudo systemctl restart havnai-coordinator.service" in body
    assert "curl -fsS http://127.0.0.1:5001/healthz" in body
    assert "curl -fsS https://api.joinhavn.io/healthz" in body


def test_flagship_rolls_back_coordinator_when_gpu_deploy_fails():
    body = _script("deploy_flagship.sh")
    assert "rollback_pi()" in body
    assert "if ! deploy_gpu; then" in body
    assert "rollback_pi" in body
    assert 'echo "GPU deploy failed; both hosts were rolled back" >&2' in body
    assert 'ssh "$PI_HOST" "sudo rm -f \'/tmp/havnai-$SHA.previous\'"' in body

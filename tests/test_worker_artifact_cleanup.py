from pathlib import Path
import pytest
from client.artifact_cleanup import candidates, cleanup_job, run_batch
from tests.test_account_auth import keys
from tests.test_account_music import music
import app
import artifact_lifecycle as lifecycle


def put(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_worker_cleanup_is_scoped_and_idempotent(tmp_path):
    names = ["outputs/job-1.png", "outputs/originals/job-1.png", "outputs/manifests/job-1.json",
             "outputs/video_job-1.mp4", "outputs/music/job-1/music.wav", "assets/job-1/source.png"]
    owned = [put(tmp_path, name) for name in names]
    other = put(tmp_path, "outputs/job-2.png")
    model = put(tmp_path, "models/model.safetensors")
    assert candidates(tmp_path) == ["job-1", "job-2"]
    assert cleanup_job(tmp_path, "job-1") == len(owned)
    assert cleanup_job(tmp_path, "job-1") == 0
    assert other.exists() and model.exists()
    assert all(not path.exists() for path in owned)


def test_worker_rejects_traversal_and_symlinks_before_unlink(tmp_path):
    image = put(tmp_path, "outputs/job-1.png")
    model = put(tmp_path, "models/model.bin")
    original = tmp_path / "outputs/originals/job-1.png"
    original.parent.mkdir()
    original.symlink_to(model)
    for job in ("../models", "job-1"):
        with pytest.raises(ValueError):
            cleanup_job(tmp_path, job)
    assert image.exists() and model.exists()


def test_worker_requires_explicit_bounded_authorization(tmp_path):
    image = put(tmp_path, "outputs/job-1.png")
    class Session:
        approved = []
        def post(self, url, **kwargs):
            assert kwargs["json"] == {"node_id": "node-test", "job_ids": ["job-1"]}
            return self
        def raise_for_status(self):
            pass
        def json(self):
            return {"job_ids": self.approved}
    session = Session()
    assert run_batch(session, "https://coordinator", {}, "node-test", tmp_path) == ("job-1", 0)
    assert image.exists()
    session.approved = ["job-2"]
    with pytest.raises(ValueError):
        run_batch(session, "https://coordinator", {}, "node-test", tmp_path)
    assert image.exists()
    session.approved = ["job-1"]
    run_batch(session, "https://coordinator", {}, "node-test", tmp_path)
    assert not image.exists()


def test_coordinator_only_authorizes_completed_purge_for_assigned_node(music, monkeypatch, tmp_path):
    harness, _, account = music
    conn = app.get_db()
    conn.execute("UPDATE jobs SET node_id='node-test' WHERE id='job-1'")
    conn.commit()
    body = {"node_id": "node-test", "job_ids": ["job-1"]}
    route = "/v1/node/artifact-purges"
    headers = {"X-HavnAI-Token": "node-test"}
    monkeypatch.setattr(app, "NODE_API_TOKEN", "node-test")
    assert harness.client.post(route, json=body).status_code in {401, 403}
    def allowed():
        response = harness.client.post(route, json=body, headers=headers)
        assert response.status_code == 200, response.json
        return response.json["job_ids"]
    assert allowed() == []
    lifecycle.delete(conn, account, "job-1", now=1)
    assert allowed() == []
    lifecycle.set_hold(conn, "job-1", "cleanup-test", actor="ops", reason="legal")
    assert allowed() == []
    lifecycle.set_hold(conn, "job-1", "cleanup-test", actor="ops", release=True)
    lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=1 + lifecycle.RECOVERY_SECONDS)
    assert allowed() == ["job-1"]
    local = put(tmp_path, "outputs/originals/job-1.png")
    class Adapter:
        def post(self, url, **kwargs):
            self.response = harness.client.post(route, json=kwargs["json"], headers=kwargs["headers"])
            return self
        def raise_for_status(self):
            assert self.response.status_code == 200
        def json(self):
            return self.response.json
    run_batch(Adapter(), "https://coordinator", headers, "node-test", tmp_path)
    assert not local.exists()
    body["node_id"] = "different-node"
    assert allowed() == []
    for jobs in (["../job-1"], ["job-1"] * 26, []):
        assert harness.client.post(route, json={"node_id": "node-test", "job_ids": jobs}, headers=headers).status_code == 422

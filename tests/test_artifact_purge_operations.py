from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
import sys
import json
import os

import pytest
from tests.test_account_auth import keys
from tests.test_account_music import music
import app
import artifact_lifecycle as lifecycle


def test_legacy_retention_timer_obeys_deletion_holds_and_keeps_records(music, tmp_path):
    _, _, account = music
    conn = app.get_db()
    original = Path(conn.execute("SELECT path FROM artifacts WHERE id='artifact-1'").fetchone()[0])
    conn.execute("UPDATE artifacts SET created_at=0")
    conn.commit()
    env = {**os.environ, "HAVNAI_DB_PATH": str(app.DB_PATH),
           "HAVNAI_OUTPUTS_DIR": str(app.OUTPUTS_DIR), "HAVNAI_ASSETS_DIR": str(tmp_path / "assets"),
           "HAVNAI_ARTIFACT_RETENTION_DAYS": "1"}
    command = [sys.executable, "-B", str(Path(lifecycle.__file__).parents[1] / "scripts" / "retain_artifacts.py")]
    def run():
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert run().returncode == 0
    assert original.exists()
    lifecycle.delete(conn, account, "job-1")
    assert run().returncode == 0
    assert original.exists()
    conn.execute("UPDATE artifact_lifecycle SET recover_until=1")
    conn.commit()
    lifecycle.set_hold(conn, "job-1", "timer-hold", actor="ops", reason="legal")
    assert run().returncode == 0
    assert original.exists()
    lifecycle.set_hold(conn, "job-1", "timer-hold", actor="ops", release=True)
    assert run().returncode == 0
    assert not original.exists()
    assert conn.execute("SELECT count(*) FROM artifacts WHERE id='artifact-1'").fetchone()[0] == 1
    assert conn.execute("SELECT purged_at FROM artifact_lifecycle WHERE job_id='job-1'").fetchone()[0] is not None
    del env["HAVNAI_OUTPUTS_DIR"]
    assert run().returncode == 2


def derived_frame(conn, account, root):
    root.mkdir(parents=True, exist_ok=True)
    frame = root / "last-frame.png"
    frame.write_bytes(b"derived frame")
    conn.execute("""INSERT INTO assets(id,owner,kind,filename,content_type,path,size_bytes,sha256,created_at,owner_account_id)
        VALUES ('frame-1','','image','last-frame.png','image/png',?,13,'digest',0,?)""", (str(frame), account))
    conn.execute("INSERT INTO account_video_frames VALUES (?,'artifact-1','frame-1')", (account,))
    conn.commit()
    return frame


def test_purge_removes_derived_frame_and_retains_audit_rows(music, tmp_path):
    _, _, account = music
    conn = app.get_db()
    root = tmp_path / "assets"
    frame = derived_frame(conn, account, root)
    original = Path(conn.execute("SELECT path FROM artifacts WHERE id='artifact-1'").fetchone()[0])
    lifecycle.delete(conn, account, "job-1", now=1)
    due = 1 + lifecycle.RECOVERY_SECONDS
    with pytest.raises(lifecycle.LifecycleError, match="assets_root_required"):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=due)
    assert original.exists() and frame.exists()
    for _ in range(2):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, assets_dir=root, now=due)
    assert not original.exists() and not frame.exists()
    assert conn.execute("SELECT count(*) FROM assets WHERE id='frame-1'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM artifact_lifecycle_events WHERE action='purge'").fetchone()[0] == 1


def test_partial_frame_purge_can_retry_without_false_completion(music, tmp_path, monkeypatch):
    _, _, account = music
    conn = app.get_db()
    root = tmp_path / "assets"
    frame = derived_frame(conn, account, root)
    original = Path(conn.execute("SELECT path FROM artifacts WHERE id='artifact-1'").fetchone()[0])
    lifecycle.delete(conn, account, "job-1", now=1)
    unlink = Path.unlink
    calls = []
    def fail_second(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise PermissionError("fixture storage failure")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_second)
        result = lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, assets_dir=root, now=1 + lifecycle.RECOVERY_SECONDS)
    assert result == [{"job_id": "job-1", "outcome": "storage_error"}]
    assert original.exists() != frame.exists()
    assert conn.execute("SELECT purged_at FROM artifact_lifecycle WHERE job_id='job-1'").fetchone()[0] is None
    assert lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, assets_dir=root, now=2 + lifecycle.RECOVERY_SECONDS) == [{"job_id": "job-1", "outcome": "purged"}]
    assert not original.exists() and not frame.exists()


@pytest.mark.parametrize("problem", ["outside", "shared", "active"])
def test_derived_frame_validation_precedes_any_unlink(music, tmp_path, problem):
    _, _, account = music
    conn = app.get_db()
    root = tmp_path / "assets"
    frame = derived_frame(conn, account, root)
    original = Path(conn.execute("SELECT path FROM artifacts WHERE id='artifact-1'").fetchone()[0])
    if problem == "shared":
        conn.execute("""INSERT INTO assets SELECT 'frame-alias',owner,kind,filename,content_type,path,
            size_bytes,sha256,created_at,owner_account_id FROM assets WHERE id='frame-1'""")
    elif problem == "active":
        conn.execute("""INSERT INTO jobs(id,wallet,model,data,task_type,weight,status,timestamp,owner_account_id)
            VALUES ('consumer','','model',?,'VIDEO_GEN',1,'running',2,?)""",
            (json.dumps({"source_asset_id": "frame-1"}), account))
    conn.commit()
    lifecycle.delete(conn, account, "job-1", now=1)
    allowed = tmp_path / "other-assets" if problem == "outside" else root
    expected = {"outside": "unsafe_artifact_path", "shared": "shared_artifact_storage", "active": "derived_asset_in_use"}[problem]
    with pytest.raises(lifecycle.LifecycleError, match=expected):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, assets_dir=allowed, now=1 + lifecycle.RECOVERY_SECONDS)
    assert original.exists() and frame.exists()
    assert conn.execute("SELECT purged_at FROM artifact_lifecycle WHERE job_id='job-1'").fetchone()[0] is None
    if problem == "active":
        conn.execute("UPDATE jobs SET status='completed' WHERE id='consumer'")
        conn.commit()
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, assets_dir=root, now=1 + lifecycle.RECOVERY_SECONDS)
        assert not frame.exists()


def test_operator_holds_are_audited_idempotent_and_do_not_reactivate(music):
    _, _, account = music
    conn = app.get_db()
    lifecycle.delete(conn, account, "job-1", now=1)
    for _ in range(2):
        lifecycle.set_hold(conn, "job-1", "case-1", actor="ops", reason="support", now=2)
    assert conn.execute("SELECT count(*) FROM artifact_lifecycle_events WHERE action='hold:case-1'").fetchone()[0] == 1
    assert lifecycle.deleted(conn, "job-1")
    assert lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, now=1 + lifecycle.RECOVERY_SECONDS) == []
    for _ in range(2):
        lifecycle.set_hold(conn, "job-1", "case-1", actor="ops", release=True, now=3)
    with pytest.raises(lifecycle.LifecycleError, match="hold_id_conflict"):
        lifecycle.set_hold(conn, "job-1", "case-1", actor="ops", reason="support", now=4)
    result = lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, now=1 + lifecycle.RECOVERY_SECONDS)
    assert result == [{"job_id": "job-1", "outcome": "purged"}]
    with pytest.raises(lifecycle.LifecycleError, match="artifact_already_purged"):
        lifecycle.set_hold(conn, "job-1", "case-2", actor="ops", reason="legal")


def test_failed_oldest_purge_does_not_starve_other_generations(music, tmp_path):
    _, _, account = music
    conn = app.get_db()
    conn.execute("""INSERT INTO jobs(id,wallet,model,data,task_type,weight,status,timestamp,owner_account_id)
        VALUES ('job-2','','model','{}','MUSIC_GEN',1,'completed',2,?)""", (account,))
    conn.commit()
    outside = tmp_path / "keep.mp3"
    outside.write_bytes(b"outside")
    conn.execute("UPDATE artifacts SET path=? WHERE job_id='job-1'", (str(outside),))
    conn.commit()
    lifecycle.delete(conn, account, "job-1", now=1)
    lifecycle.delete(conn, account, "job-2", now=2)
    due = 3 + lifecycle.RECOVERY_SECONDS
    first = lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, limit=1, now=due)
    assert first == [{"job_id": "job-1", "outcome": "unsafe_artifact_path"}]
    assert lifecycle.purge_batch(conn, outputs_dir=app.OUTPUTS_DIR, limit=1, now=due + 1) == [{"job_id": "job-2", "outcome": "purged"}]
    assert outside.exists()


def test_asset_reference_protects_shared_artifact_file(music):
    _, _, account = music
    conn = app.get_db()
    row = conn.execute("SELECT * FROM artifacts WHERE id='artifact-1'").fetchone()
    conn.execute("""INSERT INTO assets(id,owner,kind,filename,content_type,path,size_bytes,sha256,created_at,owner_account_id)
        VALUES ('asset-shared','','audio',?,?,?,?,?,0,?)""",
        (row["filename"], row["content_type"], row["path"], row["size_bytes"], row["sha256"], account))
    conn.commit()
    lifecycle.delete(conn, account, "job-1", now=1)
    with pytest.raises(lifecycle.LifecycleError, match="shared_artifact_storage"):
        lifecycle.purge(conn, "job-1", outputs_dir=app.OUTPUTS_DIR, now=1 + lifecycle.RECOVERY_SECONDS)
    assert Path(row["path"]).exists()


def test_concurrent_hold_and_purge_have_one_serialized_outcome(music):
    _, _, account = music
    lifecycle.delete(app.get_db(), account, "job-1", now=1)
    barrier = threading.Barrier(2)
    database = app.DB_PATH
    root = app.OUTPUTS_DIR
    def run(hold):
        conn = sqlite3.connect(database, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            barrier.wait()
            if hold:
                lifecycle.set_hold(conn, "job-1", "case-race", actor="ops", reason="legal")
            else:
                lifecycle.purge(conn, "job-1", outputs_dir=root)
            return "held" if hold else "purged"
        except lifecycle.LifecycleError as exc:
            return str(exc)
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = set(executor.map(run, [True, False]))
    assert results in ({"held", "artifact_on_hold"}, {"purged", "artifact_already_purged"})


def test_operator_cli_hold_release_and_scheduled_purge(music):
    _, _, account = music
    lifecycle.delete(app.get_db(), account, "job-1", now=1)
    command = [sys.executable, "-B", lifecycle.__file__, "--database", str(app.DB_PATH)]
    def invoke(*args):
        return subprocess.run([*command, *args], capture_output=True, text=True, timeout=15)
    assert invoke("--hold", "job-1", "--hold-id", "case-cli", "--actor", "operator", "--reason", "legal").returncode == 0
    assert invoke("--outputs-dir", str(app.OUTPUTS_DIR)).stdout == ""
    assert invoke("--release-hold", "job-1", "--hold-id", "case-cli", "--actor", "operator").returncode == 0
    purged = invoke("--outputs-dir", str(app.OUTPUTS_DIR))
    assert purged.returncode == 0, purged.stderr
    assert purged.stdout.strip() == "job-1 purged"
    assert invoke("--outputs-dir", str(app.OUTPUTS_DIR)).stdout == ""

import json
import sqlite3

import pytest

from scripts.repair_orphan_job_attempts import run


def _db_with_orphan_attempt(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys=OFF;
            CREATE TABLE jobs(id TEXT PRIMARY KEY, status TEXT);
            CREATE TABLE job_settlement(job_id TEXT PRIMARY KEY);
            CREATE TABLE job_attempts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 1,
                claim_time REAL NOT NULL,
                finish_time REAL,
                status TEXT NOT NULL DEFAULT 'claimed',
                error_message TEXT,
                execution_metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (job_id) REFERENCES job_settlement(job_id)
            );
            INSERT INTO job_settlement(job_id) VALUES ('job-ok');
            INSERT INTO job_attempts(job_id,node_id,claim_time,created_at,updated_at)
                VALUES ('job-ok','node',1,1,1);
            INSERT INTO job_attempts(job_id,node_id,claim_time,created_at,updated_at)
                VALUES ('job-orphan','node',2,2,2);
            """
        )
    return db


def test_audit_exports_orphans_without_mutating_database(tmp_path):
    db = _db_with_orphan_attempt(tmp_path)
    report = run(db, tmp_path / "audit.json")
    assert report["mode"] == "audit"
    assert report["orphan_attempt_count"] == 1
    assert report["deleted_attempt_count"] == 0
    assert report["orphan_attempts"][0]["job_id"] == "job-orphan"
    assert len(report["foreign_key_violations_before"]) == 1
    assert json.loads((tmp_path / "audit.json").read_text())["orphan_attempt_count"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_attempts").fetchone()[0] == 2


def test_apply_deletes_only_double_orphan_attempts_and_clears_fk(tmp_path):
    db = _db_with_orphan_attempt(tmp_path)
    report = run(db, tmp_path / "repair.json", apply=True)
    assert report["mode"] == "apply"
    assert report["orphan_attempt_count"] == 1
    assert report["deleted_attempt_count"] == 1
    assert len(report["foreign_key_violations_before"]) == 1
    assert report["foreign_key_violations_after"] == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        rows = conn.execute("SELECT job_id FROM job_attempts ORDER BY id").fetchall()
    assert rows == [("job-ok",)]


def test_refuses_to_overwrite_existing_report(tmp_path):
    db = _db_with_orphan_attempt(tmp_path)
    report = tmp_path / "report.json"
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        run(db, report)

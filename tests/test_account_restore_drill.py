import json
import sqlite3

import pytest

from scripts.account_restore_drill import run_drill


def test_restores_committed_wal_and_preserves_source(tmp_path):
    source = tmp_path / "live.sqlite3"
    with sqlite3.connect(source) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE accounts(id TEXT PRIMARY KEY)")
        live.execute("INSERT INTO accounts VALUES ('account')")
        live.commit()
        live.execute("INSERT INTO accounts VALUES ('uncommitted')")
        output = tmp_path / "drill"
        report = run_drill(source, output)
        assert report["verified"] is True
        assert report["table_counts"] == {"accounts": 1}
        assert report["snapshot_sha256"] == report["restored_sha256"]
        assert json.loads((output / "report.json").read_text()) == report
        with sqlite3.connect(output / "restored.sqlite3") as restored:
            assert restored.execute("SELECT id FROM accounts").fetchall() == [("account",)]
            restored.execute("DELETE FROM accounts")
        assert live.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 2
        live.rollback()


def test_never_overwrites_existing_destination_or_creates_missing_source(tmp_path):
    source = tmp_path / "live.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE example(id)")
    with pytest.raises(FileExistsError):
        run_drill(source, tmp_path)
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        run_drill(missing, tmp_path / "drill")
    assert not missing.exists()


def test_corruption_or_broken_foreign_keys_never_produces_success_report(tmp_path):
    source = tmp_path / "broken.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.executescript("CREATE TABLE parent(id PRIMARY KEY); CREATE TABLE child(id REFERENCES parent(id)); INSERT INTO child VALUES(1);")
    with pytest.raises(RuntimeError, match="foreign-key"):
        run_drill(source, tmp_path / "bad-fk")
    assert not (tmp_path / "bad-fk" / "report.json").exists()
    source.write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        run_drill(source, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt" / "report.json").exists()

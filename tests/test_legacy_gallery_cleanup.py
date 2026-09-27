import sqlite3
import time

from scripts import legacy_gallery_cleanup as cleanup


def _db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE gallery_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            seller_wallet TEXT NOT NULL DEFAULT '',
            owner_wallet TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            price_credits REAL NOT NULL DEFAULT 1,
            asset_type TEXT DEFAULT 'image',
            model TEXT DEFAULT '',
            listed INTEGER NOT NULL DEFAULT 1,
            sold INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            owner_account_id TEXT
        )
    """)
    return conn


def _listing(conn, job_id, title, *, listed=1, sold=0):
    now = time.time()
    return conn.execute(
        """
        INSERT INTO gallery_listings
            (job_id, title, listed, sold, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, title, listed, sold, now, now),
    ).lastrowid


def test_audit_reports_legacy_rows_without_mutating(tmp_path):
    db_path = tmp_path / "havn.db"
    conn = _db(db_path)
    legacy_id = _listing(conn, "legacy-job", "Bad launch preview")
    _listing(conn, "already-delisted", "Hidden", listed=0)
    conn.commit()

    with cleanup.connect(str(db_path)) as audit_conn:
        candidates = cleanup.find_candidates(audit_conn)
        report = cleanup.build_report(
            db_path=str(db_path),
            candidates=candidates,
            applied=False,
            delisted_count=0,
            include_job_ids=False,
        )

    assert [candidate.id for candidate in candidates] == [legacy_id]
    assert report["schema"] == cleanup.SCHEMA
    assert report["mode"] == "audit"
    assert report["candidate_count"] == 1
    assert report["deleted_rows"] == 0
    assert report["candidates"][0]["job_id_hash"]
    assert "job_id" not in report["candidates"][0]
    assert conn.execute("SELECT listed FROM gallery_listings WHERE id=?", (legacy_id,)).fetchone()[0] == 1


def test_apply_delists_only_active_legacy_rows(tmp_path):
    db_path = tmp_path / "havn.db"
    conn = _db(db_path)
    legacy_id = _listing(conn, "legacy-job", "Bad launch preview")
    sold_id = _listing(conn, "sold-job", "Sold", sold=1)
    account_id = _listing(conn, "account-job", "Account marketplace")
    conn.execute("INSERT INTO jobs(id, owner_account_id) VALUES ('account-job', 'acct_1')")
    conn.commit()

    with cleanup.connect(str(db_path)) as apply_conn:
        candidates = cleanup.find_candidates(apply_conn)
        delisted = cleanup.apply_delist(apply_conn, candidates, now=12345.0)

    rows = {
        row["id"]: row
        for row in conn.execute("SELECT id, listed, sold, updated_at FROM gallery_listings")
    }
    assert delisted == 1
    assert rows[legacy_id]["listed"] == 0
    assert rows[legacy_id]["updated_at"] == 12345.0
    assert rows[sold_id]["listed"] == 1
    assert rows[account_id]["listed"] == 1


def test_only_id_limits_cleanup_scope(tmp_path):
    db_path = tmp_path / "havn.db"
    conn = _db(db_path)
    first_id = _listing(conn, "first", "First")
    second_id = _listing(conn, "second", "Second")
    conn.commit()

    with cleanup.connect(str(db_path)) as apply_conn:
        candidates = cleanup.find_candidates(apply_conn, only_ids=[second_id])
        delisted = cleanup.apply_delist(apply_conn, candidates, now=12345.0)

    assert [candidate.id for candidate in candidates] == [second_id]
    assert delisted == 1
    assert conn.execute("SELECT listed FROM gallery_listings WHERE id=?", (first_id,)).fetchone()[0] == 1
    assert conn.execute("SELECT listed FROM gallery_listings WHERE id=?", (second_id,)).fetchone()[0] == 0


def test_missing_db_path_exits_cleanly(capsys):
    assert cleanup.main(["--json"]) == 2
    assert "missing --db-path" in capsys.readouterr().err

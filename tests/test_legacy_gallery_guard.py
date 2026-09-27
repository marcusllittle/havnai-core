import sqlite3
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import gallery


def _setup(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    gallery.get_db = lambda: conn
    gallery.log_event = lambda *args, **kwargs: None
    gallery.init_gallery_tables(conn)
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, owner_account_id TEXT)")
    return conn


def _listing(conn):
    now = time.time()
    conn.execute(
        """
        INSERT INTO gallery_listings
            (job_id, seller_wallet, owner_wallet, title, price_credits, listed, sold, created_at, updated_at)
        VALUES ('legacy-job', '0x1111111111111111111111111111111111111111',
                '0x1111111111111111111111111111111111111111', 'Legacy row', 1, 1, 0, ?, ?)
        """,
        (now, now),
    )
    conn.commit()


def test_legacy_gallery_is_default_closed(tmp_path, monkeypatch):
    conn = _setup(tmp_path / "havn.db")
    _listing(conn)
    monkeypatch.delenv("HAVNAI_LEGACY_GALLERY_PUBLIC_ENABLED", raising=False)

    assert gallery.legacy_public_gallery_enabled() is False
    assert gallery.browse_gallery()["total"] == 0
    assert gallery.get_listing(1) is None


def test_legacy_gallery_can_be_explicitly_enabled(tmp_path, monkeypatch):
    conn = _setup(tmp_path / "havn.db")
    _listing(conn)
    monkeypatch.setenv("HAVNAI_LEGACY_GALLERY_PUBLIC_ENABLED", "1")

    result = gallery.browse_gallery()
    assert result["total"] == 1
    assert result["listings"][0]["job_id"] == "legacy-job"

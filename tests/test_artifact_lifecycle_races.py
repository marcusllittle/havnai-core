import concurrent.futures
import sqlite3

from tests.test_account_auth import keys
from tests.test_account_music import music
import app
import artifact_lifecycle as lifecycle


def test_concurrent_deletes_have_one_event_and_keep_storage(music):
    _, _, account = music
    db = app.DB_PATH
    def run():
        conn = sqlite3.connect(db, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            return lifecycle.delete(conn, account, "job-1")
        finally:
            conn.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert results[0] == results[1]
    conn = app.get_db()
    assert conn.execute("SELECT COUNT(*) FROM artifact_lifecycle_events").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM artifacts WHERE job_id='job-1'").fetchone()[0] > 0

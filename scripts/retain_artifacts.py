#!/usr/bin/env python3
"""Delete expired artifacts only after removing their database records."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path


def main() -> int:
    database = Path(os.environ["HAVNAI_DB_PATH"])
    retention_days = max(1, int(os.getenv("HAVNAI_ARTIFACT_RETENTION_DAYS", "30")))
    cutoff = time.time() - retention_days * 86400
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, path FROM artifacts WHERE created_at < ?",
        (cutoff,),
    ).fetchall()
    removed = 0
    for row in rows:
        path = Path(str(row["path"]))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        conn.execute("DELETE FROM artifacts WHERE id=?", (row["id"],))
        removed += 1
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    conn.close()
    print(f"removed {removed} expired artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

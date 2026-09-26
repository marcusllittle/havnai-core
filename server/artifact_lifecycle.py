"""Account generation lifecycle (HAVN-26). Financial records are never mutated."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

RECOVERY_SECONDS = 30 * 86400


class LifecycleError(ValueError):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.status = status


def initialize(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS artifact_lifecycle (
            job_id TEXT PRIMARY KEY REFERENCES jobs(id),
            deleted_at REAL NOT NULL, recover_until REAL NOT NULL,
            purged_at REAL, restored_at REAL
        );
        CREATE TABLE IF NOT EXISTS artifact_lifecycle_holds (
            job_id TEXT NOT NULL REFERENCES jobs(id), hold_id TEXT NOT NULL,
            reason TEXT NOT NULL, created_at REAL NOT NULL, released_at REAL,
            PRIMARY KEY(job_id,hold_id)
        );
        CREATE TABLE IF NOT EXISTS artifact_lifecycle_events (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
            account_id TEXT NOT NULL, action TEXT NOT NULL, created_at REAL NOT NULL
        );
    """)


def deleted(conn, job_id):
    return bool(conn.execute("SELECT 1 FROM artifact_lifecycle WHERE job_id=? AND restored_at IS NULL", (job_id,)).fetchone())


def derived_asset_deleted(conn, asset_id):
    return bool(conn.execute("""SELECT 1 FROM account_video_frames f
        JOIN artifacts a ON a.id=f.artifact_id
        JOIN artifact_lifecycle d ON d.job_id=a.job_id
        WHERE f.asset_id=? AND d.restored_at IS NULL""", (asset_id,)).fetchone())


def chain_deleted(conn, chain_id):
    return bool(conn.execute("""SELECT 1 FROM artifact_lifecycle d
        WHERE d.restored_at IS NULL AND d.job_id IN (
            SELECT job_id FROM account_video_chain_clips WHERE chain_id=?
            UNION SELECT job_id FROM account_video_chain_outputs WHERE chain_id=?)""",
        (chain_id, chain_id)).fetchone())


def _owner(conn, account, job_id):
    job = conn.execute("SELECT * FROM jobs WHERE id=? AND owner_account_id=?", (job_id, account)).fetchone()
    if not job:
        raise LifecycleError("job_not_found", 404)
    return job


def _event(conn, account, job_id, action, now):
    conn.execute("INSERT INTO artifact_lifecycle_events VALUES (?,?,?,?,?)",
                 (uuid.uuid4().hex, job_id, account, action, now))


def delete(conn, account, job_id, *, now=None):
    now = time.time() if now is None else now
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        job = _owner(conn, account, job_id)
        old = conn.execute("SELECT * FROM artifact_lifecycle WHERE job_id=? AND restored_at IS NULL", (job_id,)).fetchone()
        if old:
            return dict(old)
        import platform_v1
        if platform_v1.canonical_job_state(job["status"]) not in platform_v1.FINAL_JOB_STATES:
            raise LifecycleError("cancel_job_before_deleting")
        conn.execute("""INSERT INTO artifact_lifecycle VALUES (?,?,?,NULL,NULL)
            ON CONFLICT(job_id) DO UPDATE SET deleted_at=excluded.deleted_at,
            recover_until=excluded.recover_until,purged_at=NULL,restored_at=NULL""",
            (job_id, now, now + RECOVERY_SECONDS))
        # Unpublish and detach playlist placements atomically. Counts and all
        # receipts remain intact. Restore never reactivates these references.
        conn.execute("DELETE FROM music_playlist_items WHERE publication_id IN (SELECT id FROM music_publications WHERE job_id=?)", (job_id,))
        conn.execute("UPDATE music_publications SET state='unpublished',updated_at=? WHERE job_id=? AND state='published'", (now, job_id))
        conn.execute("UPDATE gallery_listings SET listed=0,updated_at=? WHERE job_id=? AND listed=1", (now, job_id))
        _event(conn, account, job_id, "delete", now)
        return dict(conn.execute("SELECT * FROM artifact_lifecycle WHERE job_id=?", (job_id,)).fetchone())


def restore(conn, account, job_id, *, now=None):
    now = time.time() if now is None else now
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _owner(conn, account, job_id)
        row = conn.execute("SELECT * FROM artifact_lifecycle WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise LifecycleError("deleted_artifact_not_found", 404)
        if row["restored_at"] is not None:
            return dict(row)
        if row["purged_at"] is not None or now >= row["recover_until"]:
            raise LifecycleError("recovery_window_expired")
        conn.execute("UPDATE artifact_lifecycle SET restored_at=? WHERE job_id=?", (now, job_id))
        _event(conn, account, job_id, "restore", now)
        return dict(conn.execute("SELECT * FROM artifact_lifecycle WHERE job_id=?", (job_id,)).fetchone())


def purge(conn, job_id, *, outputs_dir, now=None):
    """Scheduled purge. A write lock serializes holds/restore and publication.

    A crash between unlink and commit leaves a deleted, non-restorable item for
    an idempotent retry. Never follow an artifact outside the configured root.
    """
    now = time.time() if now is None else now
    root = Path(outputs_dir).resolve()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        row = conn.execute("""SELECT d.*,j.owner_account_id FROM artifact_lifecycle d
            JOIN jobs j ON j.id=d.job_id WHERE d.job_id=?""", (job_id,)).fetchone()
        if not row or row["restored_at"] is not None or now < row["recover_until"]:
            raise LifecycleError("purge_not_eligible")
        if row["purged_at"] is not None:
            return
        if conn.execute("SELECT 1 FROM artifact_lifecycle_holds WHERE job_id=? AND released_at IS NULL", (job_id,)).fetchone():
            raise LifecycleError("artifact_on_hold")
        artifacts = conn.execute("SELECT path FROM artifacts WHERE job_id=?", (job_id,)).fetchall()
        paths = set()
        for artifact in artifacts:
            path = Path(artifact["path"]).resolve()
            if not path.is_relative_to(root) or path == root or (path.exists() and not path.is_file()):
                raise LifecycleError("unsafe_artifact_path")
            # Do not destroy bytes another generation still references, even if
            # a legacy writer used an aliased path spelling.
            for other in conn.execute("SELECT path FROM artifacts WHERE job_id<>?", (job_id,)):
                if Path(other["path"]).resolve() == path:
                    raise LifecycleError("shared_artifact_storage")
            paths.add(path)
        for path in paths:
            path.unlink(missing_ok=True)
        conn.execute("UPDATE artifact_lifecycle SET purged_at=? WHERE job_id=?", (now, job_id))
        _event(conn, row["owner_account_id"], job_id, "purge", now)


def main():
    import argparse
    import sqlite3
    parser = argparse.ArgumentParser(description="Purge expired soft-deleted generation artifacts")
    parser.add_argument("--database", required=True)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("limit must be between 1 and 100")
    conn = sqlite3.connect(f"file:{Path(args.database).resolve().as_posix()}?mode=rw", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    failed = False
    try:
        rows = conn.execute("""SELECT d.job_id FROM artifact_lifecycle d
            WHERE d.restored_at IS NULL AND d.purged_at IS NULL AND d.recover_until<=?
            AND NOT EXISTS (SELECT 1 FROM artifact_lifecycle_holds h WHERE h.job_id=d.job_id AND h.released_at IS NULL)
            ORDER BY d.recover_until LIMIT ?""", (time.time(), args.limit)).fetchall()
        for row in rows:
            try:
                purge(conn, row[0], outputs_dir=args.outputs_dir)
                print(row[0], "purged")
            except (LifecycleError, OSError):
                failed = True
                print(row[0], "purge_failed")
    finally:
        conn.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

"""Account generation lifecycle (HAVN-26). Financial records are never mutated."""
from __future__ import annotations

import time
import uuid
import re
import base64
import binascii
import json
import math
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
        CREATE TABLE IF NOT EXISTS artifact_purge_checks (
            job_id TEXT PRIMARY KEY REFERENCES jobs(id),
            checked_at REAL NOT NULL, outcome TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS artifact_lifecycle_deleted_order
            ON artifact_lifecycle(deleted_at DESC,job_id DESC) WHERE restored_at IS NULL;
    """)


def deleted(conn, job_id):
    return bool(conn.execute("SELECT 1 FROM artifact_lifecycle WHERE job_id=? AND restored_at IS NULL", (job_id,)).fetchone())


def list_deleted(conn, account, *, before=None, limit=25):
    """Owner-scoped keyset pages; restoring the previous page's last row is safe.

    The cursor is a position, not an authorization credential. Every page still
    filters the authenticated owner. Never put prompt/settings/media in it.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise LifecycleError("invalid_recovery_limit", 422)
    params = [account]
    position = ""
    if before is not None:
        try:
            if not isinstance(before, str) or not 1 <= len(before) <= 1024:
                raise ValueError()
            cursor = json.loads(base64.b64decode(before.encode("ascii"), altchars=b"-_", validate=True))
            if (not isinstance(cursor, list) or len(cursor) != 4 or type(cursor[0]) is not int or cursor[0] != 1 or cursor[1] != account
                    or type(cursor[2]) not in (int, float) or not math.isfinite(cursor[2]) or cursor[2] < 0
                    or not isinstance(cursor[3], str) or not 1 <= len(cursor[3]) <= 200):
                raise ValueError()
        except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError):
            raise LifecycleError("invalid_recovery_cursor", 422) from None
        position = " AND (d.deleted_at<? OR (d.deleted_at=? AND d.job_id<?))"
        params.extend([cursor[2], cursor[2], cursor[3]])
    rows = conn.execute("""SELECT d.job_id,d.deleted_at,d.recover_until,d.purged_at
        FROM artifact_lifecycle d JOIN jobs j ON j.id=d.job_id
        WHERE j.owner_account_id=? AND d.restored_at IS NULL""" + position +
        " ORDER BY d.deleted_at DESC,d.job_id DESC LIMIT ?", (*params, limit + 1)).fetchall()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = base64.urlsafe_b64encode(json.dumps(
            [1, account, last["deleted_at"], last["job_id"]], separators=(",", ":")).encode()).decode("ascii")
    return {"generations": [dict(row) for row in page], "next_cursor": next_cursor}


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
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        now = time.time() if now is None else now
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
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        now = time.time() if now is None else now
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


def set_hold(conn, job_id, hold_id, *, actor, reason=None, release=False, now=None):
    """Operator-only entry point; deliberately not exposed by customer routes."""
    if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value) for value in (hold_id, actor)):
        raise LifecycleError("invalid_hold_identity", 422)
    if not release and reason not in {"admin", "legal", "support", "dispute", "settlement"}:
        raise LifecycleError("invalid_hold_reason", 422)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        now = time.time() if now is None else now
        if not conn.execute("SELECT 1 FROM jobs WHERE id=? AND owner_account_id IS NOT NULL", (job_id,)).fetchone():
            raise LifecycleError("job_not_found", 404)
        old = conn.execute("SELECT * FROM artifact_lifecycle_holds WHERE job_id=? AND hold_id=?", (job_id, hold_id)).fetchone()
        if release:
            if not old:
                raise LifecycleError("hold_not_found", 404)
            if old["released_at"] is not None:
                return
            conn.execute("UPDATE artifact_lifecycle_holds SET released_at=? WHERE job_id=? AND hold_id=?", (now, job_id, hold_id))
        else:
            if old:
                if old["released_at"] is not None or old["reason"] != reason:
                    raise LifecycleError("hold_id_conflict")
                return
            if conn.execute("SELECT 1 FROM artifact_lifecycle WHERE job_id=? AND purged_at IS NOT NULL", (job_id,)).fetchone():
                raise LifecycleError("artifact_already_purged")
            conn.execute("INSERT INTO artifact_lifecycle_holds VALUES (?,?,?,?,NULL)", (job_id, hold_id, reason, now))
        _event(conn, "operator:" + actor, job_id, ("release_hold:" if release else "hold:") + hold_id, now)


def purge(conn, job_id, *, outputs_dir, assets_dir=None, now=None):
    """Scheduled purge. A write lock serializes holds/restore and publication.

    A crash between unlink and commit leaves a deleted, non-restorable item for
    an idempotent retry. Never follow an artifact outside the configured root.
    """
    root = Path(outputs_dir).resolve()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        now = time.time() if now is None else now
        row = conn.execute("""SELECT d.*,j.owner_account_id FROM artifact_lifecycle d
            JOIN jobs j ON j.id=d.job_id WHERE d.job_id=?""", (job_id,)).fetchone()
        if not row or row["restored_at"] is not None or now < row["recover_until"]:
            raise LifecycleError("purge_not_eligible")
        if row["purged_at"] is not None:
            return
        if conn.execute("SELECT 1 FROM artifact_lifecycle_holds WHERE job_id=? AND released_at IS NULL", (job_id,)).fetchone():
            raise LifecycleError("artifact_on_hold")
        artifacts = conn.execute("SELECT path FROM artifacts WHERE job_id=?", (job_id,)).fetchall()
        frames = conn.execute("""SELECT DISTINCT s.id,s.path,s.owner_account_id
            FROM account_video_frames f JOIN artifacts a ON a.id=f.artifact_id
            JOIN assets s ON s.id=f.asset_id WHERE a.job_id=?""", (job_id,)).fetchall()
        frame_ids = {frame["id"] for frame in frames}
        if frames and assets_dir is None:
            raise LifecycleError("assets_root_required")
        asset_root = Path(assets_dir).resolve() if assets_dir is not None else None
        for frame in frames:
            if frame["owner_account_id"] != row["owner_account_id"] or conn.execute("""
                SELECT 1 FROM account_video_frames f JOIN artifacts a ON a.id=f.artifact_id
                WHERE f.asset_id=? AND a.job_id<>?""", (frame["id"], job_id)).fetchone():
                raise LifecycleError("shared_artifact_storage")
        # An accepted continuation may still need its source frame. Enqueue uses
        # this same write lock and rejects newly submitted deleted inputs.
        if frame_ids:
            import platform_v1
            for consumer in conn.execute("SELECT status,data FROM jobs WHERE id<>?", (job_id,)):
                if platform_v1.canonical_job_state(consumer["status"]) in platform_v1.FINAL_JOB_STATES:
                    continue
                try:
                    pending = [json.loads(consumer["data"])]
                except (ValueError, TypeError):
                    raise LifecycleError("unreadable_active_job_inputs") from None
                while pending:
                    value = pending.pop()
                    if isinstance(value, dict):
                        pending.extend(value.values())
                    elif isinstance(value, list):
                        pending.extend(value)
                    elif isinstance(value, str) and value in frame_ids:
                        raise LifecycleError("derived_asset_in_use")
        referenced_paths = {Path(other[0]).resolve() for other in conn.execute(
            "SELECT path FROM artifacts WHERE job_id<>?", (job_id,))}
        referenced_paths.update(Path(other["path"]).resolve() for other in conn.execute(
            "SELECT id,path FROM assets") if other["id"] not in frame_ids)
        paths = set()
        for artifact, allowed_root in [(item, root) for item in artifacts] + [(item, asset_root) for item in frames]:
            path = Path(artifact["path"]).resolve()
            if not path.is_relative_to(allowed_root) or path == allowed_root or (path.exists() and not path.is_file()):
                raise LifecycleError("unsafe_artifact_path")
            # Do not destroy bytes another generation still references, even if
            # a legacy writer used an aliased path spelling.
            if path in referenced_paths:
                raise LifecycleError("shared_artifact_storage")
            paths.add(path)
        for path in paths:
            path.unlink(missing_ok=True)
        conn.execute("UPDATE artifact_lifecycle SET purged_at=? WHERE job_id=?", (now, job_id))
        _event(conn, row["owner_account_id"], job_id, "purge", now)


def purge_batch(conn, *, outputs_dir, assets_dir=None, limit=25, now=None):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise LifecycleError("invalid_purge_limit", 422)
    now = time.time() if now is None else now
    rows = conn.execute("""SELECT d.job_id FROM artifact_lifecycle d
        LEFT JOIN artifact_purge_checks c ON c.job_id=d.job_id
        WHERE d.restored_at IS NULL AND d.purged_at IS NULL AND d.recover_until<=?
        AND NOT EXISTS (SELECT 1 FROM artifact_lifecycle_holds h WHERE h.job_id=d.job_id AND h.released_at IS NULL)
        ORDER BY COALESCE(c.checked_at,0),d.recover_until,d.job_id LIMIT ?""", (now, limit)).fetchall()
    results = []
    for row in rows:
        outcome = "purged"
        try:
            purge(conn, row[0], outputs_dir=outputs_dir, assets_dir=assets_dir, now=now)
        except LifecycleError as exc:
            outcome = str(exc)
        except OSError:
            outcome = "storage_error"
        with conn:
            conn.execute("""INSERT INTO artifact_purge_checks VALUES (?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET checked_at=excluded.checked_at,outcome=excluded.outcome""", (row[0], now, outcome))
        results.append({"job_id": row[0], "outcome": outcome})
    return results


def main():
    import argparse
    import sqlite3
    parser = argparse.ArgumentParser(description="Purge expired soft-deleted generation artifacts")
    parser.add_argument("--database", required=True)
    parser.add_argument("--outputs-dir")
    parser.add_argument("--assets-dir")
    parser.add_argument("--limit", type=int, default=25)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--hold", metavar="JOB_ID")
    actions.add_argument("--release-hold", metavar="JOB_ID")
    parser.add_argument("--hold-id")
    parser.add_argument("--actor")
    parser.add_argument("--reason", choices=["admin", "legal", "support", "dispute", "settlement"])
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("limit must be between 1 and 100")
    if args.hold or args.release_hold:
        if not args.hold_id or not args.actor or (args.hold and not args.reason):
            parser.error("hold operations require --hold-id, --actor and, for a new hold, --reason")
    elif not args.outputs_dir or args.hold_id or args.actor or args.reason:
        parser.error("purge requires --outputs-dir and does not accept hold options")
    conn = sqlite3.connect(Path(args.database).resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    failed = False
    try:
        if args.hold or args.release_hold:
            try:
                set_hold(conn, args.hold or args.release_hold, args.hold_id, actor=args.actor,
                         reason=args.reason, release=bool(args.release_hold))
                print("hold_updated")
            except LifecycleError as exc:
                failed = True
                print(str(exc))
        else:
            for result in purge_batch(conn, outputs_dir=args.outputs_dir, assets_dir=args.assets_dir, limit=args.limit):
                print(result["job_id"], result["outcome"])
                failed |= result["outcome"] not in {"purged", "artifact_on_hold", "purge_not_eligible"}
    finally:
        conn.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

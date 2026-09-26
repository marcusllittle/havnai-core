"""Private derived inputs for account-owned video continuation."""
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import platform_v1
import artifact_lifecycle


class VideoInputError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS account_video_frames (
        account_id TEXT NOT NULL REFERENCES accounts(id),
        artifact_id TEXT NOT NULL REFERENCES artifacts(id),
        asset_id TEXT NOT NULL REFERENCES assets(id),
        PRIMARY KEY(account_id,artifact_id))""")
    conn.commit()


def source(conn, account, job_id, outputs):
    job = conn.execute("SELECT status FROM jobs WHERE id=? AND owner_account_id=?", (job_id, account)).fetchone()
    if not job or artifact_lifecycle.deleted(conn, job_id):
        raise VideoInputError("job_not_found", 404)
    if platform_v1.canonical_job_state(job["status"]) != "succeeded":
        raise VideoInputError("video_not_ready", 409)
    artifact = conn.execute("SELECT id,path,sha256 FROM artifacts WHERE job_id=? AND kind='video' ORDER BY created_at DESC,id DESC LIMIT 1", (job_id,)).fetchone()
    if not artifact:
        raise VideoInputError("video_not_found", 404)
    path = Path(artifact["path"]).resolve()
    if not path.is_relative_to(outputs.resolve()) or not path.is_file():
        raise VideoInputError("video_missing", 410)
    return dict(artifact)


def cached(conn, account, artifact_id, assets):
    row = conn.execute("""SELECT a.* FROM account_video_frames f JOIN assets a ON a.id=f.asset_id
        WHERE f.account_id=? AND f.artifact_id=? AND a.owner_account_id=?""", (account, artifact_id, account)).fetchone()
    if not row:
        return None
    path = Path(row["path"]).resolve()
    if not path.is_relative_to(assets.resolve()) or not path.is_file():
        raise VideoInputError("asset_missing", 410)
    return {"id": row["id"], "kind": "image", "filename": row["filename"], "sha256": row["sha256"]}


def extract_frame(video, output):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise VideoInputError("video_processing_unavailable", 503)
    # Decode only a local MP4/MOV container. A playlist cannot cause network fetches.
    try:
        subprocess.run([executable, "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
            "-f", "mov", "-sseof", "-1", "-i", str(video), "-an", "-vf", "reverse",
            "-frames:v", "1", "-threads", "1", str(output)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except (subprocess.SubprocessError, OSError) as exc:
        raise VideoInputError("video_frame_extraction_failed", 422) from exc
    if not output.is_file() or not output.stat().st_size:
        raise VideoInputError("video_frame_extraction_failed", 422)


def last_frame(conn, account, job_id, *, outputs, assets, max_bytes):
    original = source(conn, account, job_id, outputs)
    existing = cached(conn, account, original["id"], assets)
    if existing:
        return existing
    assets.mkdir(parents=True, exist_ok=True)
    asset_id = platform_v1.new_asset_id()
    destination = assets / asset_id / "last-frame.png"
    persisted = False
    try:
        with tempfile.TemporaryDirectory(prefix=".video-frame-", dir=assets) as temporary:
            output = Path(temporary) / "frame.png"
            extract_frame(Path(original["path"]), output)
            with output.open("rb") as handle:
                size, digest = platform_v1.atomic_stream_write(handle, destination, max_bytes)
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            if source(conn, account, job_id, outputs) != original:
                raise VideoInputError("video_changed", 409)
            # Concurrent requests may have finished the same extraction already.
            existing = cached(conn, account, original["id"], assets)
            if existing:
                return existing
            conn.execute("""INSERT INTO assets (id,owner,kind,filename,content_type,path,size_bytes,sha256,created_at,owner_account_id)
                VALUES (?,'','image','last-frame.png','image/png',?,?,?,?,?)""",
                (asset_id, str(destination), size, digest, time.time(), account))
            conn.execute("INSERT INTO account_video_frames VALUES (?,?,?)", (account, original["id"], asset_id))
        persisted = True
        return {"id": asset_id, "kind": "image", "filename": "last-frame.png", "sha256": digest}
    finally:
        if not persisted:
            destination.unlink(missing_ok=True)
            if destination.parent.is_dir():
                destination.parent.rmdir()

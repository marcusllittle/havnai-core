"""Private, repeatable clip concatenation with source provenance."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import account_video
import account_video_chains
import platform_v1
from account_video import VideoInputError


def concatenate(paths, output):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise VideoInputError("video_processing_unavailable", 503)
    signatures = []
    try:
        for path in paths:
            probe = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-f", "mov",
                "-show_streams", "-of", "json", str(path)], check=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=15)
            streams = json.loads(probe.stdout)["streams"]
            selected = [stream for stream in streams if stream.get("codec_type") in {"video", "audio"}]
            if sum(stream.get("codec_type") == "video" for stream in selected) != 1 or sum(stream.get("codec_type") == "audio" for stream in selected) > 1:
                raise VideoInputError("incompatible_video_clips")
            signatures.append([{key: stream.get(key) for key in ("codec_type", "codec_name", "profile", "level",
                "width", "height", "pix_fmt", "time_base", "avg_frame_rate", "sample_rate", "channels", "channel_layout")}
                for stream in selected])
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise VideoInputError("incompatible_video_clips")
        with tempfile.TemporaryDirectory(prefix="havnai-stitch-") as temporary:
            directory = Path(temporary)
            lines = []
            for index, path in enumerate(paths):
                local = directory / f"clip-{index}.mp4"
                try:
                    os.link(path, local)
                except OSError:
                    shutil.copyfile(path, local)
                lines.append(f"file '{local.name}'")
            listing = directory / "clips.txt"
            listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
                "-f", "concat", "-safe", "1", "-i", str(listing), "-map", "0:v:0", "-map", "0:a?",
                "-c", "copy", "-movflags", "+faststart", str(output)], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (subprocess.SubprocessError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise VideoInputError("video_stitch_failed") from exc
    if not output.is_file() or not output.stat().st_size:
        raise VideoInputError("video_stitch_failed")


def stitch(conn, account, chain_id, *, outputs, max_bytes):
    chain = account_video_chains.read(conn, account, chain_id)
    if chain["result_job_id"]:
        account_video.source(conn, account, chain["result_job_id"], outputs)
        return chain["result_job_id"]
    if len(chain["jobs"]) != chain["total"] or any(job["status"] != "succeeded" for job in chain["jobs"]):
        raise VideoInputError("video_chain_not_ready", 409)
    sources = [account_video.source(conn, account, job["id"], outputs) for job in chain["jobs"]]
    job_id = "job-" + uuid.uuid4().hex
    artifact_id = platform_v1.new_artifact_id()
    destination = outputs / "private-chains" / job_id / "result.mp4"
    persisted = False
    try:
        with tempfile.TemporaryDirectory(prefix="havnai-stitch-output-") as temporary:
            merged = Path(temporary) / "result.mp4"
            concatenate([Path(source["path"]) for source in sources], merged)
            with merged.open("rb") as handle:
                size, digest = platform_v1.atomic_stream_write(handle, destination, max_bytes)
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            current = account_video_chains.read(conn, account, chain_id)
            if current["result_job_id"]:
                account_video.source(conn, account, current["result_job_id"], outputs)
                return current["result_job_id"]
            if [account_video.source(conn, account, job["id"], outputs) for job in current["jobs"]] != sources:
                raise VideoInputError("video_chain_changed", 409)
            now = time.time()
            params = {"prompt": chain["template"]["prompt"], "chain_id": chain_id,
                "source_job_ids": [job["id"] for job in chain["jobs"]], "source_artifact_ids": [source["id"] for source in sources]}
            conn.execute("""INSERT INTO jobs (id,wallet,creator_account_id,owner_account_id,model,data,task_type,weight,
                status,timestamp,completed_at,progress,stage,updated_at,resolved_spec)
                VALUES (?,'',?,?,?,?,'VIDEO_STITCH',0,'completed',?,?,100,'succeeded',?,?)""",
                (job_id, account, account, chain["template"]["model"], json.dumps({"v1_type": "video_stitch", **params}),
                 now, now, now, json.dumps({"schema_version": 1, "task_type": "video_stitch", "parameters": params})))
            conn.execute("""INSERT INTO artifacts (id,job_id,kind,filename,content_type,path,size_bytes,sha256,metadata,created_at)
                VALUES (?,?,'video','result.mp4','video/mp4',?,?,?,?,?)""",
                (artifact_id, job_id, str(destination), size, digest, json.dumps(params), now))
            conn.execute("INSERT INTO account_video_chain_outputs VALUES (?,?,?)", (chain_id, job_id, artifact_id))
        persisted = True
        return job_id
    finally:
        if not persisted:
            destination.unlink(missing_ok=True)
            if destination.parent.is_dir():
                destination.parent.rmdir()

"""Durable account clip plans; clip records and credit reservations commit together."""
import hashlib
import json
import time
import uuid

import platform_v1
from account_video import VideoInputError


def initialize(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_video_chains (
            id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
            request_key TEXT NOT NULL, payload_hash TEXT NOT NULL,
            template TEXT NOT NULL, total INTEGER NOT NULL, auto_stitch INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'active', created_at REAL NOT NULL,
            UNIQUE(account_id,request_key));
        CREATE TABLE IF NOT EXISTS account_video_chain_clips (
            chain_id TEXT NOT NULL REFERENCES account_video_chains(id),
            clip_index INTEGER NOT NULL, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
            PRIMARY KEY(chain_id,clip_index));
        CREATE TABLE IF NOT EXISTS account_video_chain_outputs (
            chain_id TEXT PRIMARY KEY REFERENCES account_video_chains(id),
            job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
            artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts(id));
    """)
    conn.commit()


def read(conn, account, chain_id):
    row = conn.execute("SELECT * FROM account_video_chains WHERE id=? AND account_id=?", (chain_id, account)).fetchone()
    if not row:
        raise VideoInputError("video_chain_not_found", 404)
    clips = conn.execute("""SELECT c.clip_index,j.id,j.status,j.owner_account_id FROM account_video_chain_clips c
        JOIN jobs j ON j.id=c.job_id WHERE c.chain_id=? ORDER BY c.clip_index""", (chain_id,)).fetchall()
    if any(clip["owner_account_id"] != account for clip in clips):
        raise VideoInputError("video_chain_not_found", 404)
    jobs = [{"index": clip["clip_index"], "id": clip["id"], "status": platform_v1.canonical_job_state(clip["status"])} for clip in clips]
    state = row["state"]
    if state == "active":
        if any(job["status"] in platform_v1.FINAL_JOB_STATES - {"succeeded"} for job in jobs):
            state = "failed"
        elif len(jobs) == row["total"] and all(job["status"] == "succeeded" for job in jobs):
            state = "rendered"
    output = conn.execute("SELECT o.job_id,o.artifact_id,j.owner_account_id FROM account_video_chain_outputs o JOIN jobs j ON j.id=o.job_id WHERE o.chain_id=?", (chain_id,)).fetchone()
    if output and output["owner_account_id"] != account:
        raise VideoInputError("video_chain_not_found", 404)
    if output:
        state = "complete"
    return {"id": row["id"], "template": json.loads(row["template"]), "total": row["total"],
            "auto_stitch": bool(row["auto_stitch"]), "state": state, "created_at": row["created_at"], "jobs": jobs,
            "result_job_id": output["job_id"] if output else None, "result_artifact_id": output["artifact_id"] if output else None}


def create(conn, account, request_key, body):
    if not request_key or len(request_key) > 128:
        raise VideoInputError("idempotency_key_required")
    try:
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        raise VideoInputError("invalid_payload")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        old = conn.execute("SELECT id,payload_hash FROM account_video_chains WHERE account_id=? AND request_key=?", (account, request_key)).fetchone()
        if old:
            if old["payload_hash"] != digest:
                raise VideoInputError("idempotency_conflict", 409)
            return read(conn, account, old["id"])
        if (not isinstance(body, dict) or set(body) != {"template", "total", "auto_stitch"}
            or type(body["total"]) is not int or not 2 <= body["total"] <= 7 or type(body["auto_stitch"]) is not bool):
            raise VideoInputError("invalid_video_chain")
        template = body["template"]
        allowed = {"type", "model", "prompt", "negative_prompt", "source_asset_id", "audio_asset_id", "seed",
                   "width", "height", "frames", "fps", "steps", "guidance", "strength", "motion_strength",
                   "sfw_mode", "preset", "aspect_ratio", "duration_seconds"}
        if (not isinstance(template, dict) or set(template) - allowed or template.get("type") not in {"image_to_video", "text_to_video"}
            or not isinstance(template.get("prompt"), str) or not template["prompt"].strip()
            or not isinstance(template.get("model"), str) or not template["model"].strip()):
            raise VideoInputError("invalid_video_chain")
        if ("sfw_mode" in template and type(template["sfw_mode"]) is not bool) or ("negative_prompt" in template and not isinstance(template["negative_prompt"], str)):
            raise VideoInputError("invalid_video_chain")
        spec = platform_v1.resolve_video_spec(template, model=template["model"])
        if (template["type"] == "image_to_video") != bool(template.get("source_asset_id")):
            raise VideoInputError("invalid_video_source")
        for field, kind in (("source_asset_id", "image"), ("audio_asset_id", "audio")):
            value = template.get(field)
            if value is not None and not isinstance(value, str):
                raise VideoInputError("invalid_video_chain")
            if value and not conn.execute(
                "SELECT 1 FROM assets WHERE id=? AND owner_account_id=? AND kind=?", (value, account, kind)).fetchone():
                raise VideoInputError("asset_not_found", 404)
        template = {**template, "seed": spec["parameters"]["seed"]}
        chain_id = "chain-" + uuid.uuid4().hex
        conn.execute("INSERT INTO account_video_chains VALUES (?,?,?,?,?,?,?,'active',?)", (chain_id, account,
            request_key, digest, json.dumps(template), body["total"], int(body["auto_stitch"]), time.time()))
        return read(conn, account, chain_id)


def check_enqueue(conn, account, chain_id, index):
    chain = read(conn, account, chain_id)
    if chain["state"] != "active" or len(chain["jobs"]) != index or index >= chain["total"]:
        raise VideoInputError("video_chain_changed", 409)
    if index and chain["jobs"][-1]["status"] != "succeeded":
        raise VideoInputError("video_not_ready", 409)


def stop(conn, account, chain_id):
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        read(conn, account, chain_id)
        conn.execute("UPDATE account_video_chains SET state='stopped' WHERE id=? AND account_id=?", (chain_id, account))
    return read(conn, account, chain_id)

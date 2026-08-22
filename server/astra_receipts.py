"""Deterministic provenance receipts for Astra reward artifacts.

Receipts bind the final output bytes to the coordinator's durable job,
routing, and settlement records. They are intentionally not described as
on-chain proofs: current node payouts may still use the simulated HAI ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional


RECEIPT_SCHEMA = "havnai.astra.artifact-receipt"
RECEIPT_VERSION = 1
FINAL_JOB_STATUSES = {"completed", "done", "succeeded", "success"}
VIDEO_TASK_TYPES = {"VIDEO_GEN", "ANIMATEDIFF", "LTX_VIDEO_GEN"}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_FILE_HASH_CACHE: Dict[tuple[str, int, int], str] = {}
_FILE_HASH_LOCK = threading.Lock()


class ReceiptUnavailable(Exception):
    """A stable, route-safe reason why a receipt cannot be issued."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def init_receipt_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS astra_artifact_receipts (
            job_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_astra_receipts_artifact_sha "
        "ON astra_artifact_receipts(artifact_sha256)"
    )
    conn.commit()


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _FILE_HASH_LOCK:
        cached = _FILE_HASH_CACHE.get(cache_key)
        if cached:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        if len(_FILE_HASH_CACHE) >= 256:
            _FILE_HASH_CACHE.pop(next(iter(_FILE_HASH_CACHE)))
        _FILE_HASH_CACHE[cache_key] = value
        return value


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _normalized_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _safe_output_path(raw_path: Any, outputs_dir: Path) -> Optional[Path]:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = outputs_dir / path
    try:
        resolved = path.resolve()
        resolved.relative_to(outputs_dir.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _owner_commitment(wallet: str) -> str:
    normalized = str(wallet or "").strip().lower()
    return f"sha256:{sha256_text(f'havnai-owner:v1:{normalized}')}"


def _canonical_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return "succeeded" if normalized in FINAL_JOB_STATUSES else normalized or "unknown"


def _find_astra_record(conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT run_id, job_id AS image_job_id, video_job_id, wallet, pilot_id,
               outfit_id, map_id, grade, created_at
        FROM astra_reward_images
        WHERE job_id = ? OR video_job_id = ?
        LIMIT 1
        """,
        (job_id, job_id),
    ).fetchone()
    return dict(row) if row else None


def _indexed_artifact(
    conn: sqlite3.Connection,
    job_id: str,
    preferred_kind: str,
    outputs_dir: Path,
) -> Optional[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at DESC",
        (job_id,),
    ).fetchall()
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: str(row.get("kind") or "").lower() != preferred_kind,
    )
    for row in ordered:
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in {"image", "video"}:
            continue
        path = _safe_output_path(row.get("path"), outputs_dir)
        if path is None:
            continue
        stored_digest = str(row.get("sha256") or "").strip().lower()
        digest = stored_digest if SHA256_RE.fullmatch(stored_digest) else _file_sha256(path)
        return {
            "id": str(row.get("id") or f"indexed:{job_id}"),
            "kind": kind,
            "content_type": str(row.get("content_type") or "application/octet-stream"),
            "size_bytes": int(row.get("size_bytes") or path.stat().st_size),
            "sha256": digest,
            "digest_source": "node_upload" if SHA256_RE.fullmatch(stored_digest) else "coordinator_scan",
            "created_at": _normalized_float(row.get("created_at")),
        }
    return None


def _legacy_artifact(job_id: str, kind: str, outputs_dir: Path) -> Optional[Dict[str, Any]]:
    if kind == "video":
        path = outputs_dir / "videos" / f"{job_id}.mp4"
        content_type = "video/mp4"
    else:
        path = outputs_dir / f"{job_id}.png"
        content_type = "image/png"
    if not path.is_file():
        return None
    return {
        "id": f"legacy:{job_id}",
        "kind": kind,
        "content_type": content_type,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "digest_source": "coordinator_scan",
        "created_at": _normalized_float(path.stat().st_mtime),
    }


def _stored_receipt(conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT canonical_json, receipt_sha256, created_at FROM astra_artifact_receipts WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    if not row:
        return None
    canonical = str(row["canonical_json"])
    expected = str(row["receipt_sha256"])
    if sha256_text(canonical) != expected:
        raise ReceiptUnavailable("receipt_corrupt", 500)
    try:
        receipt = json.loads(canonical)
    except json.JSONDecodeError as exc:
        raise ReceiptUnavailable("receipt_corrupt", 500) from exc
    if not isinstance(receipt, dict):
        raise ReceiptUnavailable("receipt_corrupt", 500)
    return {
        "receipt": receipt,
        "canonical_json": canonical,
        "receipt_sha256": f"sha256:{expected}",
        "issued_at": _normalized_float(row["created_at"]),
    }


def get_or_issue_receipt(
    conn: sqlite3.Connection,
    job_id: str,
    outputs_dir: Path,
) -> Dict[str, Any]:
    """Return the persisted receipt, issuing it once after output finalization."""
    stored = _stored_receipt(conn, job_id)
    if stored:
        return stored

    astra = _find_astra_record(conn, job_id)
    if astra is None:
        raise ReceiptUnavailable("astra_artifact_not_found", 404)
    job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_row:
        raise ReceiptUnavailable("job_not_found", 404)
    job = dict(job_row)
    if _canonical_status(job.get("status")) != "succeeded":
        raise ReceiptUnavailable("artifact_not_ready", 409)

    settlement_row = conn.execute(
        "SELECT * FROM job_settlement WHERE job_id = ?", (job_id,)
    ).fetchone()
    settlement = dict(settlement_row) if settlement_row else {}
    if not settlement or str(settlement.get("settlement_outcome") or "pending") == "pending":
        raise ReceiptUnavailable("settlement_not_ready", 409)

    task_type = str(job.get("task_type") or "IMAGE_GEN").strip().upper()
    preferred_kind = "video" if task_type in VIDEO_TASK_TYPES else "image"
    artifact = _indexed_artifact(conn, job_id, preferred_kind, outputs_dir)
    if artifact is None:
        artifact = _legacy_artifact(job_id, preferred_kind, outputs_dir)
    if artifact is None:
        raise ReceiptUnavailable("artifact_not_ready", 409)

    payout_row = conn.execute(
        "SELECT * FROM node_payouts WHERE job_id = ? ORDER BY created_at DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    payout = dict(payout_row) if payout_row else {}
    reward_row = conn.execute(
        "SELECT reward_hai, timestamp FROM rewards WHERE task_id = ?", (job_id,)
    ).fetchone()
    reward = dict(reward_row) if reward_row else {}

    job_data = _json_object(job.get("data"))
    model_metadata = _json_object(settlement.get("input_metadata"))
    creator_node_id = str(
        settlement.get("assigned_node_id") or job.get("node_id") or payout.get("node_id") or ""
    ).strip()
    preferred_node_id = str(job_data.get("preferred_node_id") or "").strip() or None
    routing_strategy = (
        "player_affinity"
        if preferred_node_id or job_data.get("routing_source") == "player_affinity"
        else "automatic"
    )
    payout_amount = payout.get("reward_amount", reward.get("reward_hai", 0.0))
    payout_timestamp = payout.get("created_at", reward.get("timestamp"))

    receipt: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "job_id": job_id,
        "run_id": str(astra.get("run_id") or ""),
        "owner_commitment": _owner_commitment(str(astra.get("wallet") or "")),
        "artifact": artifact,
        "execution": {
            "status": _canonical_status(job.get("status")),
            "task_type": task_type,
            "creator_node_id": creator_node_id,
            "queued_at": _normalized_float(job.get("timestamp")),
            "completed_at": _normalized_float(job.get("completed_at")),
            "attempt_count": int(settlement.get("attempt_count") or 0),
            "model": {
                "key": str(model_metadata.get("model_key") or job.get("model") or ""),
                "name": str(model_metadata.get("model_name") or job.get("model") or ""),
                "pipeline": str(model_metadata.get("pipeline") or ""),
                "tier": str(model_metadata.get("tier") or ""),
            },
        },
        "routing": {
            "strategy": routing_strategy,
            "preferred_node_id": preferred_node_id,
            "preference_honored": (
                creator_node_id == preferred_node_id if preferred_node_id and creator_node_id else None
            ),
        },
        "settlement": {
            "execution_status": str(settlement.get("execution_status") or "unavailable"),
            "quality_status": str(settlement.get("quality_status") or "unavailable"),
            "outcome": str(settlement.get("settlement_outcome") or "unavailable"),
            "credits_spent": _normalized_float(settlement.get("spent_amount")),
            "node_reward": _normalized_float(payout_amount),
            "reward_asset_type": str(payout.get("reward_asset_type") or "simulated_hai"),
            "transaction_hash": str(payout.get("tx_hash") or "") or None,
            "settled_at": _normalized_float(payout_timestamp or settlement.get("updated_at")),
        },
        "game": {
            "pilot_id": str(astra.get("pilot_id") or ""),
            "outfit_id": str(astra.get("outfit_id") or ""),
            "map_id": str(astra.get("map_id") or ""),
            "grade": str(astra.get("grade") or ""),
        },
    }

    canonical = canonical_json(receipt)
    digest = sha256_text(canonical)
    issued_at = _normalized_float(job.get("completed_at") or artifact.get("created_at"))
    conn.execute(
        """
        INSERT OR IGNORE INTO astra_artifact_receipts
            (job_id, schema_version, artifact_sha256, canonical_json, receipt_sha256, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, RECEIPT_VERSION, artifact["sha256"], canonical, digest, issued_at),
    )
    conn.commit()
    return _stored_receipt(conn, job_id) or {
        "receipt": receipt,
        "canonical_json": canonical,
        "receipt_sha256": f"sha256:{digest}",
        "issued_at": issued_at,
    }

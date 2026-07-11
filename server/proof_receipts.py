"""Tamper-evident Proof of Creation receipts."""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


get_db: Callable[[], sqlite3.Connection]
SIGNING_KEY: str = ""
ED25519_PRIVATE_KEY: str = ""


def init_receipt_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS proof_of_creation_receipts (
            job_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            payload TEXT NOT NULL,
            receipt_hash TEXT NOT NULL,
            signature TEXT,
            signature_algorithm TEXT NOT NULL,
            artifact_path TEXT,
            artifact_sha256 TEXT,
            key_id TEXT,
            public_key TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_proof_receipt_hash ON proof_of_creation_receipts (receipt_hash)"
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(proof_of_creation_receipts)").fetchall()}
    if "key_id" not in columns:
        conn.execute("ALTER TABLE proof_of_creation_receipts ADD COLUMN key_id TEXT")
    if "public_key" not in columns:
        conn.execute("ALTER TABLE proof_of_creation_receipts ADD COLUMN public_key TEXT")
    conn.commit()


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_ed25519_private_key() -> Optional[bytes]:
    value = ED25519_PRIVATE_KEY.strip()
    if not value:
        return None
    try:
        raw = bytes.fromhex(value[2:] if value.startswith("0x") else value)
    except ValueError:
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("HAVNAI_RECEIPT_ED25519_PRIVATE_KEY must be 32-byte hex or base64") from exc
    if len(raw) != 32:
        raise ValueError("HAVNAI_RECEIPT_ED25519_PRIVATE_KEY must decode to exactly 32 bytes")
    return raw


def public_signing_key() -> Optional[Dict[str, str]]:
    private_bytes = _decode_ed25519_private_key()
    if not private_bytes:
        return None
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "algorithm": "ed25519",
        "key_id": hashlib.sha256(public_bytes).hexdigest()[:16],
        "public_key": base64.b64encode(public_bytes).decode("ascii"),
    }


def _sign(receipt_hash: str) -> tuple[Optional[str], str, Optional[str], Optional[str]]:
    private_bytes = _decode_ed25519_private_key()
    if private_bytes:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key_meta = public_signing_key() or {}
        signature = Ed25519PrivateKey.from_private_bytes(private_bytes).sign(
            receipt_hash.encode("ascii")
        )
        return (
            base64.b64encode(signature).decode("ascii"),
            "ed25519",
            key_meta.get("key_id"),
            key_meta.get("public_key"),
        )
    if not SIGNING_KEY:
        return None, "sha256", None, None
    signature = hmac.new(
        SIGNING_KEY.encode("utf-8"), receipt_hash.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return signature, "hmac-sha256", None, None


def create_receipt(
    job_id: str,
    payload: Dict[str, Any],
    *,
    artifact_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create an immutable receipt, returning an existing one idempotently."""
    existing = get_receipt(job_id)
    if existing:
        return existing

    artifact_sha256 = sha256_file(artifact_path) if artifact_path and artifact_path.exists() else None
    receipt_payload = dict(payload)
    receipt_payload["schema_version"] = "proof-of-creation.v1"
    receipt_payload["job_id"] = job_id
    receipt_payload["artifact"] = {
        **(receipt_payload.get("artifact") or {}),
        "sha256": artifact_sha256,
    }
    encoded = canonical_json(receipt_payload)
    receipt_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    signature, algorithm, key_id, public_key = _sign(receipt_hash)
    created_at = time.time()
    conn = get_db()
    conn.execute(
        """
        INSERT OR IGNORE INTO proof_of_creation_receipts (
            job_id, schema_version, payload, receipt_hash, signature,
            signature_algorithm, artifact_path, artifact_sha256, key_id,
            public_key, created_at
        ) VALUES (?, 'proof-of-creation.v1', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            encoded,
            receipt_hash,
            signature,
            algorithm,
            str(artifact_path) if artifact_path else None,
            artifact_sha256,
            key_id,
            public_key,
            created_at,
        ),
    )
    conn.commit()
    return get_receipt(job_id) or {}


def get_receipt(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM proof_of_creation_receipts WHERE job_id=?", (job_id,)
    ).fetchone()
    if not row:
        return None
    receipt = dict(row)
    receipt["canonical_payload"] = receipt["payload"]
    receipt["payload"] = json.loads(receipt["payload"])
    receipt["signed"] = bool(receipt.get("signature"))
    return receipt


def verify_receipt(job_id: str) -> Dict[str, Any]:
    receipt = get_receipt(job_id)
    if not receipt:
        return {"job_id": job_id, "valid": False, "error": "receipt_not_found"}
    encoded = canonical_json(receipt["payload"])
    calculated_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    hash_valid = hmac.compare_digest(calculated_hash, str(receipt["receipt_hash"]))

    artifact_path = Path(receipt["artifact_path"]) if receipt.get("artifact_path") else None
    artifact_exists = bool(artifact_path and artifact_path.exists())
    artifact_valid = False
    if artifact_exists and receipt.get("artifact_sha256"):
        artifact_valid = hmac.compare_digest(
            sha256_file(artifact_path), str(receipt["artifact_sha256"])
        )

    signature = receipt.get("signature")
    if signature and receipt.get("signature_algorithm") == "ed25519" and receipt.get("public_key"):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            Ed25519PublicKey.from_public_bytes(base64.b64decode(receipt["public_key"])).verify(
                base64.b64decode(signature), str(receipt["receipt_hash"]).encode("ascii")
            )
            signature_valid = True
        except Exception:
            signature_valid = False
    elif signature and SIGNING_KEY:
        expected, _, _, _ = _sign(str(receipt["receipt_hash"]))
        signature_valid = bool(expected and hmac.compare_digest(expected, signature))
    elif signature:
        signature_valid = None
    else:
        signature_valid = None

    integrity_valid = hash_valid and artifact_valid
    authenticity = "verified" if signature_valid is True else "unverifiable" if signature else "unsigned"
    return {
        "job_id": job_id,
        "valid": integrity_valid and signature_valid is not False,
        "integrity_valid": integrity_valid,
        "hash_valid": hash_valid,
        "artifact_exists": artifact_exists,
        "artifact_valid": artifact_valid,
        "signature_valid": signature_valid,
        "authenticity": authenticity,
        "receipt_hash": receipt["receipt_hash"],
        "schema_version": receipt["schema_version"],
    }

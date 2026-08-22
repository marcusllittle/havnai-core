"""Immutable Merkle batches for Astra artifact receipts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


get_db: Callable[[], sqlite3.Connection]

BATCH_SCHEMA = "receipt-merkle-batch.v1"


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def init_merkle_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS receipt_merkle_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version TEXT NOT NULL,
            merkle_root TEXT NOT NULL UNIQUE,
            leaf_count INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            anchor_network TEXT,
            anchor_chain_id INTEGER,
            anchor_tx_hash TEXT UNIQUE,
            anchor_block INTEGER,
            anchor_from TEXT,
            anchor_to TEXT,
            anchor_calldata TEXT,
            anchored_at REAL,
            created_at REAL NOT NULL
        )
        """
    )
    migrations = {
        "anchor_chain_id": "INTEGER",
        "anchor_block": "INTEGER",
        "anchor_from": "TEXT",
        "anchor_to": "TEXT",
        "anchor_calldata": "TEXT",
    }
    existing = _column_names(conn, "receipt_merkle_batches")
    for column, declaration in migrations.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE receipt_merkle_batches ADD COLUMN {column} {declaration}"
            )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS receipt_merkle_leaves (
            batch_id INTEGER NOT NULL,
            job_id TEXT NOT NULL UNIQUE,
            leaf_index INTEGER NOT NULL,
            receipt_hash TEXT NOT NULL,
            leaf_hash TEXT NOT NULL,
            proof TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(batch_id, leaf_index),
            FOREIGN KEY(batch_id) REFERENCES receipt_merkle_batches(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_receipt_merkle_leaves_job "
        "ON receipt_merkle_leaves(job_id)"
    )
    conn.commit()


def _hash_leaf(receipt_hash: str) -> bytes:
    return hashlib.sha256(b"\x00" + bytes.fromhex(receipt_hash)).digest()


def _hash_parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def build_tree(receipt_hashes: Sequence[str]) -> Tuple[str, List[List[Dict[str, str]]]]:
    if not receipt_hashes:
        raise ValueError("at least one receipt hash is required")
    leaves = [_hash_leaf(value) for value in receipt_hashes]
    proofs: List[List[Dict[str, str]]] = [[] for _ in leaves]
    positions = [[index] for index in range(len(leaves))]
    level = leaves

    while len(level) > 1:
        next_level: List[bytes] = []
        next_positions: List[List[int]] = []
        for offset in range(0, len(level), 2):
            left = level[offset]
            right = level[offset + 1] if offset + 1 < len(level) else left
            left_positions = positions[offset]
            right_positions = (
                positions[offset + 1] if offset + 1 < len(level) else positions[offset]
            )
            for index in left_positions:
                proofs[index].append({"position": "right", "hash": right.hex()})
            if offset + 1 < len(level):
                for index in right_positions:
                    proofs[index].append({"position": "left", "hash": left.hex()})
            next_level.append(_hash_parent(left, right))
            next_positions.append(
                left_positions + (right_positions if offset + 1 < len(level) else [])
            )
        level = next_level
        positions = next_positions
    return level[0].hex(), proofs


def verify_proof(
    receipt_hash: str,
    proof: Sequence[Dict[str, str]],
    merkle_root: str,
) -> bool:
    try:
        current = _hash_leaf(receipt_hash)
        for step in proof:
            sibling = bytes.fromhex(str(step["hash"]))
            if step["position"] == "left":
                current = _hash_parent(sibling, current)
            elif step["position"] == "right":
                current = _hash_parent(current, sibling)
            else:
                return False
        return current.hex() == merkle_root.lower()
    except (KeyError, TypeError, ValueError):
        return False


def create_batch(limit: int = 100, min_count: int = 1) -> Optional[Dict[str, Any]]:
    conn = get_db()
    safe_limit = max(1, min(int(limit), 10000))
    safe_min = max(1, min(int(min_count), safe_limit))
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT r.job_id, r.receipt_sha256 AS receipt_hash
            FROM astra_artifact_receipts r
            LEFT JOIN receipt_merkle_leaves l ON l.job_id = r.job_id
            WHERE l.job_id IS NULL
            ORDER BY r.created_at ASC, r.job_id ASC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        if len(rows) < safe_min:
            conn.rollback()
            return None
        receipt_hashes = [str(row["receipt_hash"]) for row in rows]
        merkle_root, proofs = build_tree(receipt_hashes)
        created_at = time.time()
        cursor = conn.execute(
            """
            INSERT INTO receipt_merkle_batches
                (schema_version, merkle_root, leaf_count, status, created_at)
            VALUES (?, ?, ?, 'ready', ?)
            """,
            (BATCH_SCHEMA, merkle_root, len(rows), created_at),
        )
        batch_id = int(cursor.lastrowid)
        for index, row in enumerate(rows):
            receipt_hash = str(row["receipt_hash"])
            conn.execute(
                """
                INSERT INTO receipt_merkle_leaves
                    (batch_id, job_id, leaf_index, receipt_hash, leaf_hash, proof, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    row["job_id"],
                    index,
                    receipt_hash,
                    _hash_leaf(receipt_hash).hex(),
                    json.dumps(proofs[index], separators=(",", ":")),
                    created_at,
                ),
            )
        conn.commit()
        return get_batch(batch_id)
    except Exception:
        conn.rollback()
        raise


def get_batch(batch_id: int) -> Optional[Dict[str, Any]]:
    row = get_db().execute(
        "SELECT * FROM receipt_merkle_batches WHERE id=?", (batch_id,)
    ).fetchone()
    return dict(row) if row else None


def list_batches(limit: int = 50) -> List[Dict[str, Any]]:
    rows = get_db().execute(
        "SELECT * FROM receipt_merkle_batches ORDER BY id DESC LIMIT ?",
        (max(1, min(int(limit), 500)),),
    ).fetchall()
    return [dict(row) for row in rows]


def count_unbatched_receipts() -> int:
    row = get_db().execute(
        """
        SELECT COUNT(*) AS count
        FROM astra_artifact_receipts r
        LEFT JOIN receipt_merkle_leaves l ON l.job_id = r.job_id
        WHERE l.job_id IS NULL
        """
    ).fetchone()
    return int(row["count"] if row else 0)


def get_inclusion_proof(job_id: str) -> Optional[Dict[str, Any]]:
    row = get_db().execute(
        """
        SELECT l.*, b.schema_version, b.merkle_root, b.leaf_count, b.status,
               b.anchor_network, b.anchor_chain_id, b.anchor_tx_hash,
               b.anchor_block, b.anchor_from, b.anchor_to, b.anchored_at,
               b.created_at AS batch_created_at
        FROM receipt_merkle_leaves l
        JOIN receipt_merkle_batches b ON b.id = l.batch_id
        WHERE l.job_id=?
        """,
        (job_id,),
    ).fetchone()
    if not row:
        return None
    payload = dict(row)
    payload["proof"] = json.loads(payload["proof"])
    payload["valid"] = verify_proof(
        payload["receipt_hash"], payload["proof"], payload["merkle_root"]
    )
    return payload


def mark_anchored(batch_id: int, anchor: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db()
    current = get_batch(batch_id)
    if current is None:
        raise KeyError("batch_not_found")
    tx_hash = str(anchor["tx_hash"]).lower()
    current_tx_hash = str(current.get("anchor_tx_hash") or "").lower()
    if current["status"] == "anchored":
        if str(current.get("anchor_tx_hash") or "").lower() != tx_hash:
            raise ValueError("batch_already_anchored")
        return current
    if current_tx_hash and current_tx_hash != tx_hash:
        raise ValueError("batch_anchor_transaction_mismatch")
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE receipt_merkle_batches
            SET status='anchored', anchor_network=?, anchor_chain_id=?,
                anchor_tx_hash=?, anchor_block=?, anchor_from=?, anchor_to=?,
                anchor_calldata=?, anchored_at=?
            WHERE id=? AND status IN ('ready', 'pending')
            """,
            (
                anchor["network"],
                anchor["chain_id"],
                tx_hash,
                anchor["block_number"],
                anchor["from"],
                anchor["to"],
                anchor["calldata"],
                time.time(),
                batch_id,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("batch_anchor_conflict")
        conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    anchored = get_batch(batch_id)
    if anchored is None:
        raise KeyError("batch_not_found")
    return anchored


def mark_anchor_pending(batch_id: int, anchor: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a structurally valid anchor while it accumulates confirmations."""
    conn = get_db()
    current = get_batch(batch_id)
    if current is None:
        raise KeyError("batch_not_found")
    tx_hash = str(anchor["tx_hash"]).lower()
    current_tx_hash = str(current.get("anchor_tx_hash") or "").lower()
    if current["status"] == "anchored":
        if current_tx_hash != tx_hash:
            raise ValueError("batch_already_anchored")
        return current
    if current_tx_hash and current_tx_hash != tx_hash:
        raise ValueError("batch_anchor_transaction_mismatch")
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE receipt_merkle_batches
            SET status='pending', anchor_network=?, anchor_chain_id=?,
                anchor_tx_hash=?, anchor_block=?, anchor_from=?, anchor_to=?,
                anchor_calldata=?
            WHERE id=? AND status IN ('ready', 'pending')
            """,
            (
                anchor["network"],
                anchor["chain_id"],
                tx_hash,
                anchor["block_number"],
                anchor["from"],
                anchor["to"],
                anchor["calldata"],
                batch_id,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("batch_anchor_conflict")
        conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    pending = get_batch(batch_id)
    if pending is None:
        raise KeyError("batch_not_found")
    return pending

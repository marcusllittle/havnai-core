"""Immutable Ethereum-compatible Merkle batches for node operator payouts."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from eth_utils import keccak


get_db: Callable[[], sqlite3.Connection]

SCHEMA_VERSION = "havnai-node-payout-claims.v1"
LEAF_DOMAIN = keccak(text="HAVNAI_NODE_PAYOUT_CLAIM_V1")
TOKEN_SCALE = 10**18
AMOUNT_QUANTUM = Decimal("0.000001")
WALLET_RE = re.compile(r"^0x[0-9a-f]{40}$")


class PayoutClaimError(RuntimeError):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code)
        self.code = code
        self.status = status


def init_payout_claim_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS node_payout_claim_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version TEXT NOT NULL,
            merkle_root TEXT,
            leaf_count INTEGER NOT NULL DEFAULT 0,
            payout_count INTEGER NOT NULL DEFAULT 0,
            total_amount_wei TEXT NOT NULL DEFAULT '0',
            status TEXT NOT NULL DEFAULT 'building',
            publish_network TEXT,
            publish_chain_id INTEGER,
            publish_contract TEXT,
            publish_tx_hash TEXT UNIQUE,
            publish_block INTEGER,
            publish_from TEXT,
            created_at REAL NOT NULL,
            published_at REAL
        );

        CREATE TABLE IF NOT EXISTS node_payout_claim_leaves (
            batch_id INTEGER NOT NULL,
            leaf_index INTEGER NOT NULL,
            wallet TEXT NOT NULL,
            amount_wei TEXT NOT NULL,
            leaf_hash TEXT NOT NULL,
            node_ids_json TEXT NOT NULL,
            payout_count INTEGER NOT NULL,
            claimed_tx_hash TEXT UNIQUE,
            claimed_at REAL,
            PRIMARY KEY (batch_id, leaf_index),
            UNIQUE (batch_id, wallet),
            FOREIGN KEY (batch_id) REFERENCES node_payout_claim_batches(id)
        );

        CREATE TABLE IF NOT EXISTS node_payout_claim_items (
            payout_id INTEGER PRIMARY KEY,
            batch_id INTEGER NOT NULL,
            leaf_index INTEGER NOT NULL,
            FOREIGN KEY (batch_id, leaf_index)
                REFERENCES node_payout_claim_leaves(batch_id, leaf_index)
        );

        CREATE INDEX IF NOT EXISTS idx_node_payout_claim_batches_status
            ON node_payout_claim_batches(status);
        CREATE INDEX IF NOT EXISTS idx_node_payout_claim_leaves_wallet
            ON node_payout_claim_leaves(wallet);
        CREATE INDEX IF NOT EXISTS idx_node_payout_claim_items_batch
            ON node_payout_claim_items(batch_id, leaf_index);
        """
    )
    conn.commit()


def _uint256(value: int) -> bytes:
    if value < 0 or value >= 2**256:
        raise PayoutClaimError("uint256_out_of_range", 400)
    return value.to_bytes(32, "big")


def _address_word(wallet: str) -> bytes:
    normalized = str(wallet or "").strip().lower()
    if not WALLET_RE.fullmatch(normalized):
        raise PayoutClaimError("invalid_operator_wallet", 400)
    return bytes(12) + bytes.fromhex(normalized[2:])


def amount_to_wei(amount: Any) -> int:
    try:
        normalized = Decimal(str(amount)).quantize(AMOUNT_QUANTUM, rounding=ROUND_DOWN)
    except Exception as exc:
        raise PayoutClaimError("invalid_payout_amount", 400) from exc
    if normalized <= 0:
        raise PayoutClaimError("invalid_payout_amount", 400)
    return int(normalized * TOKEN_SCALE)


def format_wei(amount_wei: int) -> str:
    value = Decimal(amount_wei) / Decimal(TOKEN_SCALE)
    return format(value.normalize(), "f")


def payout_leaf(batch_id: int, index: int, wallet: str, amount_wei: int) -> bytes:
    """Match keccak256(abi.encode(domain,batchId,index,account,amount))."""
    return keccak(
        LEAF_DOMAIN
        + _uint256(batch_id)
        + _uint256(index)
        + _address_word(wallet)
        + _uint256(amount_wei)
    )


def _hash_pair(left: bytes, right: bytes) -> bytes:
    return keccak(left + right) if left <= right else keccak(right + left)


def _tree_levels(leaves: Sequence[bytes]) -> List[List[bytes]]:
    if not leaves:
        raise PayoutClaimError("empty_payout_batch")
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            [
                _hash_pair(current[index], current[index + 1] if index + 1 < len(current) else current[index])
                for index in range(0, len(current), 2)
            ]
        )
    return levels


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    return _tree_levels(leaves)[-1][0]


def merkle_proof(leaves: Sequence[bytes], index: int) -> List[bytes]:
    levels = _tree_levels(leaves)
    if index < 0 or index >= len(leaves):
        raise PayoutClaimError("leaf_index_out_of_range", 400)
    proof: List[bytes] = []
    cursor = index
    for level in levels[:-1]:
        sibling = cursor ^ 1
        proof.append(level[sibling] if sibling < len(level) else level[cursor])
        cursor //= 2
    return proof


def verify_proof(leaf: bytes, proof: Iterable[bytes], root: bytes) -> bool:
    computed = leaf
    for sibling in proof:
        computed = _hash_pair(computed, sibling)
    return computed == root


def _eligible_payouts(conn: sqlite3.Connection, limit: int) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT p.id, p.node_id, p.job_id, p.reward_amount,
                  LOWER(TRIM(w.wallet)) AS operator_wallet
             FROM node_payouts p
             JOIN node_wallets w ON w.node_id = p.node_id
             LEFT JOIN node_payout_claim_items i ON i.payout_id = p.id
            WHERE p.status = 'completed'
              AND p.reward_asset_type = 'simulated_hai'
              AND p.tx_hash IS NULL
              AND i.payout_id IS NULL
            ORDER BY p.created_at ASC, p.id ASC
            LIMIT ?""",
        (max(1, min(int(limit), 2000)),),
    ).fetchall()


def count_unbatched_payouts() -> int:
    conn = get_db()
    row = conn.execute(
        """SELECT COUNT(*)
             FROM node_payouts p
             JOIN node_wallets w ON w.node_id = p.node_id
             LEFT JOIN node_payout_claim_items i ON i.payout_id = p.id
            WHERE p.status = 'completed'
              AND p.reward_asset_type = 'simulated_hai'
              AND p.tx_hash IS NULL
              AND i.payout_id IS NULL
              AND LOWER(TRIM(w.wallet)) GLOB '0x[0-9a-f]*'
              AND LENGTH(TRIM(w.wallet)) = 42"""
    ).fetchone()
    return int(row[0] if row else 0)


def create_batch(limit: int = 500, min_count: int = 1) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = [
            row for row in _eligible_payouts(conn, limit)
            if WALLET_RE.fullmatch(str(row["operator_wallet"] or ""))
        ]
        if len(rows) < max(1, int(min_count)):
            conn.rollback()
            return None

        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            wallet = str(row["operator_wallet"])
            entry = grouped.setdefault(
                wallet,
                {"amount_wei": 0, "payout_ids": [], "node_ids": set()},
            )
            entry["amount_wei"] += amount_to_wei(row["reward_amount"])
            entry["payout_ids"].append(int(row["id"]))
            entry["node_ids"].add(str(row["node_id"]))

        now = time.time()
        cursor = conn.execute(
            """INSERT INTO node_payout_claim_batches
               (schema_version, status, created_at)
               VALUES (?, 'building', ?)""",
            (SCHEMA_VERSION, now),
        )
        batch_id = int(cursor.lastrowid)
        ordered = sorted(grouped.items())
        leaves: List[bytes] = []
        total_wei = 0

        for index, (wallet, entry) in enumerate(ordered):
            amount_wei = int(entry["amount_wei"])
            leaf = payout_leaf(batch_id, index, wallet, amount_wei)
            leaves.append(leaf)
            total_wei += amount_wei
            conn.execute(
                """INSERT INTO node_payout_claim_leaves
                   (batch_id, leaf_index, wallet, amount_wei, leaf_hash,
                    node_ids_json, payout_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    index,
                    wallet,
                    str(amount_wei),
                    leaf.hex(),
                    json.dumps(sorted(entry["node_ids"]), separators=(",", ":")),
                    len(entry["payout_ids"]),
                ),
            )
            conn.executemany(
                """INSERT INTO node_payout_claim_items
                   (payout_id, batch_id, leaf_index) VALUES (?, ?, ?)""",
                [(payout_id, batch_id, index) for payout_id in entry["payout_ids"]],
            )

        root = merkle_root(leaves).hex()
        conn.execute(
            """UPDATE node_payout_claim_batches
                  SET merkle_root=?, leaf_count=?, payout_count=?,
                      total_amount_wei=?, status='ready'
                WHERE id=? AND status='building'""",
            (root, len(leaves), len(rows), str(total_wei), batch_id),
        )
        conn.commit()
        return get_batch(batch_id)
    except Exception:
        conn.rollback()
        raise


def _batch_payload(row: sqlite3.Row) -> Dict[str, Any]:
    payload = dict(row)
    payload["batch_id"] = int(payload.pop("id"))
    payload["total_amount_wei"] = str(payload["total_amount_wei"])
    payload["total_amount_hai"] = format_wei(int(payload["total_amount_wei"]))
    return payload


def get_batch(batch_id: int) -> Optional[Dict[str, Any]]:
    row = get_db().execute(
        "SELECT * FROM node_payout_claim_batches WHERE id = ?", (int(batch_id),)
    ).fetchone()
    return _batch_payload(row) if row else None


def list_batches(limit: int = 50) -> List[Dict[str, Any]]:
    rows = get_db().execute(
        "SELECT * FROM node_payout_claim_batches ORDER BY id DESC LIMIT ?",
        (max(1, min(int(limit), 200)),),
    ).fetchall()
    return [_batch_payload(row) for row in rows]


def _batch_leaves(batch_id: int) -> List[sqlite3.Row]:
    return get_db().execute(
        """SELECT * FROM node_payout_claim_leaves
            WHERE batch_id = ? ORDER BY leaf_index ASC""",
        (int(batch_id),),
    ).fetchall()


def get_claim(batch_id: int, index: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    batch = get_batch(batch_id)
    if not batch:
        return None
    rows = _batch_leaves(batch_id)
    if index < 0 or index >= len(rows):
        return None
    row = rows[index]
    leaves = [bytes.fromhex(str(item["leaf_hash"])) for item in rows]
    proof = merkle_proof(leaves, index)
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": int(batch_id),
        "leaf_index": int(index),
        "wallet": str(row["wallet"]),
        "amount_wei": str(row["amount_wei"]),
        "amount_hai": format_wei(int(row["amount_wei"])),
        "leaf_hash": str(row["leaf_hash"]),
        "merkle_root": str(batch["merkle_root"]),
        "proof": [item.hex() for item in proof],
        "node_ids": json.loads(str(row["node_ids_json"])),
        "payout_count": int(row["payout_count"]),
        "batch_status": str(batch["status"]),
        "publish_tx_hash": batch.get("publish_tx_hash"),
        "claimed": bool(row["claimed_tx_hash"]),
        "claimed_tx_hash": row["claimed_tx_hash"],
        "claimed_at": row["claimed_at"],
        "valid": verify_proof(
            payout_leaf(batch_id, index, str(row["wallet"]), int(row["amount_wei"])),
            proof,
            bytes.fromhex(str(batch["merkle_root"])),
        ),
    }


def get_wallet_claims(wallet: str, limit: int = 100) -> List[Dict[str, Any]]:
    normalized = str(wallet or "").strip().lower()
    if not WALLET_RE.fullmatch(normalized):
        raise PayoutClaimError("invalid_operator_wallet", 400)
    rows = get_db().execute(
        """SELECT batch_id, leaf_index
             FROM node_payout_claim_leaves
            WHERE wallet = ?
            ORDER BY batch_id DESC
            LIMIT ?""",
        (normalized, max(1, min(int(limit), 200))),
    ).fetchall()
    return [
        claim
        for row in rows
        if (claim := get_claim(int(row["batch_id"]), int(row["leaf_index"]))) is not None
    ]


def mark_publish_pending(batch_id: int, verification: Dict[str, Any]) -> Dict[str, Any]:
    return _mark_published(batch_id, verification, pending=True)


def mark_published(batch_id: int, verification: Dict[str, Any]) -> Dict[str, Any]:
    return _mark_published(batch_id, verification, pending=False)


def _mark_published(
    batch_id: int,
    verification: Dict[str, Any],
    *,
    pending: bool,
) -> Dict[str, Any]:
    conn = get_db()
    current = get_batch(batch_id)
    if not current:
        raise PayoutClaimError("payout_batch_not_found", 404)
    tx_hash = str(verification["tx_hash"]).lower()
    existing = str(current.get("publish_tx_hash") or "").lower()
    if existing and existing != tx_hash:
        raise PayoutClaimError("payout_batch_transaction_conflict")
    if current["status"] == "published":
        return current
    status = "pending" if pending else "published"
    conn.execute(
        """UPDATE node_payout_claim_batches
              SET status=?, publish_network=?, publish_chain_id=?,
                  publish_contract=?, publish_tx_hash=?, publish_block=?,
                  publish_from=?, published_at=?
            WHERE id=?""",
        (
            status,
            verification.get("network"),
            verification.get("chain_id"),
            verification.get("contract"),
            tx_hash,
            verification.get("block_number"),
            verification.get("from"),
            None if pending else time.time(),
            int(batch_id),
        ),
    )
    conn.commit()
    return get_batch(batch_id) or current


def mark_claimed(batch_id: int, index: int, verification: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db()
    claim = get_claim(batch_id, index)
    if not claim:
        raise PayoutClaimError("payout_claim_not_found", 404)
    tx_hash = str(verification["tx_hash"]).lower()
    existing = str(claim.get("claimed_tx_hash") or "").lower()
    if existing and existing != tx_hash:
        raise PayoutClaimError("payout_claim_transaction_conflict")
    if existing == tx_hash:
        return claim
    conn.execute(
        """UPDATE node_payout_claim_leaves
              SET claimed_tx_hash=?, claimed_at=?
            WHERE batch_id=? AND leaf_index=? AND claimed_tx_hash IS NULL""",
        (tx_hash, time.time(), int(batch_id), int(index)),
    )
    conn.execute(
        """UPDATE node_payouts
              SET reward_asset_type='onchain_hai', tx_hash=?, updated_at=?
            WHERE id IN (
                SELECT payout_id FROM node_payout_claim_items
                 WHERE batch_id=? AND leaf_index=?
            )""",
        (tx_hash, time.time(), int(batch_id), int(index)),
    )
    conn.commit()
    return get_claim(batch_id, index) or claim

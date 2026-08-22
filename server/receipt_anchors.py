"""Sepolia transaction payloads and verification for receipt Merkle roots."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Optional


ANCHOR_SCHEMA = "havnai.receipt-root.v1"
ANCHOR_MAGIC = b"HAVNAI_RECEIPT_ROOT_V1"
SEPOLIA_CHAIN_ID = 11155111
SEPOLIA_NETWORK = "sepolia"
MIN_CONFIRMATIONS = max(
    1, int(os.getenv("HAVNAI_RECEIPT_ANCHOR_CONFIRMATIONS", "2"))
)
TREASURY_WALLET = os.getenv("HAVNAI_HAI_TREASURY_WALLET", "").strip().lower()
TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
ROOT_RE = re.compile(r"^[a-fA-F0-9]{64}$")
WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def build_calldata(batch_id: int, merkle_root: str) -> str:
    if batch_id <= 0:
        raise ValueError("batch_id must be positive")
    if batch_id >= 2**64:
        raise ValueError("batch_id exceeds uint64")
    if not ROOT_RE.fullmatch(merkle_root):
        raise ValueError("invalid_merkle_root")
    payload = ANCHOR_MAGIC + batch_id.to_bytes(8, "big") + bytes.fromhex(merkle_root)
    return "0x" + payload.hex()


def build_anchor_payload(batch_id: int, merkle_root: str) -> Dict[str, Any]:
    return {
        "schema": ANCHOR_SCHEMA,
        "network": SEPOLIA_NETWORK,
        "chain_id": SEPOLIA_CHAIN_ID,
        "from": TREASURY_WALLET or None,
        "to": TREASURY_WALLET or None,
        "value": "0x0",
        "calldata": build_calldata(batch_id, merkle_root),
    }


def _hex_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = str(value or "0x0").strip()
    return int(raw, 16 if raw.lower().startswith("0x") else 10)


def _invalid(error: str, **extra: Any) -> Dict[str, Any]:
    return {"verified": False, "pending": False, "error": error, **extra}


def _pending(error: str, **extra: Any) -> Dict[str, Any]:
    return {"verified": False, "pending": True, "error": error, **extra}


def verify_anchor_transaction(
    tx_hash: str,
    batch_id: int,
    merkle_root: str,
    rpc_call: Callable[[str, list], Any],
    *,
    treasury_wallet: Optional[str] = None,
    min_confirmations: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify that a treasury self-transaction permanently commits one root."""
    normalized_hash = str(tx_hash or "").strip().lower()
    if not TX_HASH_RE.fullmatch(normalized_hash):
        return _invalid("invalid_transaction_hash")
    treasury = str(treasury_wallet or TREASURY_WALLET).strip().lower()
    if not WALLET_RE.fullmatch(treasury):
        return _invalid("treasury_wallet_not_configured")
    required_confirmations = max(1, int(min_confirmations or MIN_CONFIRMATIONS))
    expected_calldata = build_calldata(batch_id, merkle_root).lower()

    try:
        chain_id = _hex_int(rpc_call("eth_chainId", []))
    except Exception as exc:
        return _pending("rpc_unavailable", detail=str(exc))
    if chain_id != SEPOLIA_CHAIN_ID:
        return _invalid("wrong_chain", expected=SEPOLIA_CHAIN_ID, actual=chain_id)

    try:
        transaction = rpc_call("eth_getTransactionByHash", [normalized_hash])
        receipt = rpc_call("eth_getTransactionReceipt", [normalized_hash])
    except Exception as exc:
        return _pending("rpc_unavailable", detail=str(exc))
    if transaction is None or receipt is None:
        return _pending("transaction_pending")
    if _hex_int(receipt.get("status")) != 1:
        return _invalid("transaction_failed")

    transaction_hash = str(
        receipt.get("transactionHash") or transaction.get("hash") or ""
    ).lower()
    if transaction_hash and transaction_hash != normalized_hash:
        return _invalid("transaction_hash_mismatch")
    sender = str(transaction.get("from") or "").lower()
    recipient = str(transaction.get("to") or "").lower()
    if sender != treasury:
        return _invalid("wrong_sender")
    if recipient != treasury:
        return _invalid("wrong_recipient")
    if _hex_int(transaction.get("value")) != 0:
        return _invalid("nonzero_value")
    calldata = str(transaction.get("input") or transaction.get("data") or "0x").lower()
    if calldata != expected_calldata:
        return _invalid("anchor_payload_mismatch")

    try:
        block_number = _hex_int(receipt.get("blockNumber"))
        current_block = _hex_int(rpc_call("eth_blockNumber", []))
    except Exception as exc:
        return _pending("confirmation_check_failed", detail=str(exc))
    confirmations = max(0, current_block - block_number + 1)
    if confirmations < required_confirmations:
        return _pending(
            "insufficient_confirmations",
            confirmations=confirmations,
            required_confirmations=required_confirmations,
            network=SEPOLIA_NETWORK,
            chain_id=chain_id,
            tx_hash=normalized_hash,
            block_number=block_number,
            **{"from": sender, "to": recipient, "calldata": calldata},
        )

    return {
        "verified": True,
        "pending": False,
        "network": SEPOLIA_NETWORK,
        "chain_id": chain_id,
        "tx_hash": normalized_hash,
        "block_number": block_number,
        "confirmations": confirmations,
        "from": sender,
        "to": recipient,
        "calldata": calldata,
    }

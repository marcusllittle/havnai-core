"""Strict Sepolia verification for node payout root publication and claims."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Optional

from eth_utils import keccak


SEPOLIA_CHAIN_ID = 11155111
SEPOLIA_NETWORK = "sepolia"
MIN_CONFIRMATIONS = max(1, int(os.getenv("HAVNAI_NODE_CLAIM_CONFIRMATIONS", "2")))
TREASURY_WALLET = os.getenv("HAVNAI_HAI_TREASURY_WALLET", "").strip().lower()
CLAIM_CONTRACT = os.getenv("HAVNAI_NODE_CLAIM_CONTRACT", "").strip().lower()
HAI_TOKEN_ADDRESS = os.getenv("HAVNAI_HAI_TOKEN_ADDRESS", "").strip().lower()

TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
ROOT_RE = re.compile(r"^(?:0x)?[a-fA-F0-9]{64}$")
WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
PUBLISH_SELECTOR = keccak(text="publishRoot(uint256,bytes32,uint256)")[:4]
CLAIM_EVENT_TOPIC = keccak(text="RewardClaimed(uint256,uint256,address,uint256)")


def _uint256(value: int) -> bytes:
    if value < 0 or value >= 2**256:
        raise ValueError("uint256_out_of_range")
    return value.to_bytes(32, "big")


def build_publish_calldata(batch_id: int, merkle_root: str, total_amount_wei: int) -> str:
    normalized_root = str(merkle_root or "").removeprefix("0x")
    if batch_id <= 0 or total_amount_wei <= 0:
        raise ValueError("invalid_publish_values")
    if not ROOT_RE.fullmatch(normalized_root):
        raise ValueError("invalid_merkle_root")
    return "0x" + (
        PUBLISH_SELECTOR
        + _uint256(batch_id)
        + bytes.fromhex(normalized_root)
        + _uint256(total_amount_wei)
    ).hex()


def publish_payload(batch_id: int, merkle_root: str, total_amount_wei: int) -> Dict[str, Any]:
    return {
        "network": SEPOLIA_NETWORK,
        "chain_id": SEPOLIA_CHAIN_ID,
        "from": TREASURY_WALLET or None,
        "to": CLAIM_CONTRACT or None,
        "value": "0x0",
        "calldata": build_publish_calldata(batch_id, merkle_root, total_amount_wei),
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


def _base_transaction(
    tx_hash: str,
    rpc_call: Callable[[str, list], Any],
    *,
    min_confirmations: Optional[int] = None,
) -> Dict[str, Any]:
    normalized_hash = str(tx_hash or "").strip().lower()
    if not TX_HASH_RE.fullmatch(normalized_hash):
        return _invalid("invalid_transaction_hash")
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
    if _hex_int(transaction.get("value")) != 0:
        return _invalid("nonzero_value")
    try:
        block_number = _hex_int(receipt.get("blockNumber"))
        current_block = _hex_int(rpc_call("eth_blockNumber", []))
    except Exception as exc:
        return _pending("confirmation_check_failed", detail=str(exc))
    required = max(1, int(min_confirmations or MIN_CONFIRMATIONS))
    confirmations = max(0, current_block - block_number + 1)
    base = {
        "network": SEPOLIA_NETWORK,
        "chain_id": chain_id,
        "tx_hash": normalized_hash,
        "block_number": block_number,
        "confirmations": confirmations,
        "required_confirmations": required,
        "from": str(transaction.get("from") or "").lower(),
        "to": str(transaction.get("to") or "").lower(),
        "calldata": str(transaction.get("input") or transaction.get("data") or "0x").lower(),
        "receipt": receipt,
    }
    return {
        "verified": True,
        "pending": False,
        "confirmed": confirmations >= required,
        **base,
    }


def _confirmation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.pop("confirmed", False):
        result.pop("receipt", None)
        return result
    pending_payload = {
        key: value
        for key, value in result.items()
        if key not in {"verified", "pending", "receipt"}
    }
    return _pending("insufficient_confirmations", **pending_payload)


def verify_publish_transaction(
    tx_hash: str,
    batch_id: int,
    merkle_root: str,
    total_amount_wei: int,
    rpc_call: Callable[[str, list], Any],
    *,
    treasury_wallet: Optional[str] = None,
    claim_contract: Optional[str] = None,
    min_confirmations: Optional[int] = None,
) -> Dict[str, Any]:
    treasury = str(treasury_wallet or TREASURY_WALLET).strip().lower()
    contract = str(claim_contract or CLAIM_CONTRACT).strip().lower()
    if not WALLET_RE.fullmatch(treasury):
        return _invalid("treasury_wallet_not_configured")
    if not WALLET_RE.fullmatch(contract):
        return _invalid("claim_contract_not_configured")
    result = _base_transaction(tx_hash, rpc_call, min_confirmations=min_confirmations)
    if not result.get("verified"):
        return result
    if result["from"] != treasury:
        return _invalid("wrong_sender")
    if result["to"] != contract:
        return _invalid("wrong_contract")
    expected = build_publish_calldata(batch_id, merkle_root, total_amount_wei).lower()
    if result["calldata"] != expected:
        return _invalid("publish_payload_mismatch")
    result["contract"] = contract
    return _confirmation_result(result)


def verify_claim_transaction(
    tx_hash: str,
    batch_id: int,
    index: int,
    wallet: str,
    amount_wei: int,
    rpc_call: Callable[[str, list], Any],
    *,
    claim_contract: Optional[str] = None,
    min_confirmations: Optional[int] = None,
) -> Dict[str, Any]:
    contract = str(claim_contract or CLAIM_CONTRACT).strip().lower()
    account = str(wallet or "").strip().lower()
    if not WALLET_RE.fullmatch(contract):
        return _invalid("claim_contract_not_configured")
    if not WALLET_RE.fullmatch(account):
        return _invalid("invalid_operator_wallet")
    result = _base_transaction(tx_hash, rpc_call, min_confirmations=min_confirmations)
    if not result.get("verified"):
        return result
    if result["to"] != contract:
        return _invalid("wrong_contract")

    expected_topics = [
        "0x" + CLAIM_EVENT_TOPIC.hex(),
        "0x" + _uint256(batch_id).hex(),
        "0x" + _uint256(index).hex(),
        "0x" + (bytes(12) + bytes.fromhex(account[2:])).hex(),
    ]
    matched = False
    for entry in result["receipt"].get("logs", []):
        topics = [str(topic).lower() for topic in entry.get("topics", [])]
        if (
            str(entry.get("address") or "").lower() == contract
            and topics == expected_topics
            and _hex_int(entry.get("data")) == amount_wei
        ):
            matched = True
            break
    if not matched:
        return _invalid("claim_event_mismatch")
    result["contract"] = contract
    result["wallet"] = account
    result["batch_id"] = batch_id
    result["leaf_index"] = index
    result["amount_wei"] = str(amount_wei)
    return _confirmation_result(result)

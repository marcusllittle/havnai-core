"""HAI token funding — on-chain transfer verification and credit grant.

Follows the same pattern as stripe_payments.py:
  - Injected dependencies (get_db, log_event, deposit_credits)
  - Idempotent processing via UNIQUE tx_hash
  - Atomic status transitions to prevent double-credits

v0.1 uses direct ERC-20 transfer() + backend RPC verification.
No approve/transferFrom. No payment contract.
Standard ERC-20 ABI is sufficient.

Environment variables:
    HAVNAI_SEPOLIA_RPC_URL       - Sepolia JSON-RPC endpoint (Infura/Alchemy)
    HAVNAI_HAI_TOKEN_ADDRESS     - HAI ERC-20 contract address on Sepolia
    HAVNAI_HAI_TREASURY_WALLET   - Treasury wallet receiving HAI payments
    HAVNAI_HAI_FUNDING_ENABLED   - Set to "1"/"true" to enable HAI funding routes
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

import requests

# Will be injected by app.py (same pattern as stripe_payments.py)
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
deposit_credits: Callable[[str, float, str], float]

# Config from environment
SEPOLIA_RPC_URL: str = os.getenv("HAVNAI_SEPOLIA_RPC_URL", "").strip()
HAI_TOKEN_ADDRESS: str = os.getenv("HAVNAI_HAI_TOKEN_ADDRESS", "").strip().lower()
HAI_TREASURY_WALLET: str = os.getenv("HAVNAI_HAI_TREASURY_WALLET", "").strip().lower()
HAI_FUNDING_ENABLED: bool = os.getenv("HAVNAI_HAI_FUNDING_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# ERC-20 Transfer event topic: keccak256("Transfer(address,address,uint256)")
TRANSFER_EVENT_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Minimum confirmations before accepting a transaction
MIN_CONFIRMATIONS = 2

# Conversion rate: 1 HAI = 1 credit (for v0.1)
HAI_TO_CREDITS_RATE = 1.0


def init_hai_funding_tables(conn: sqlite3.Connection) -> None:
    """Create the hai_fundings tracking table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hai_fundings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            amount REAL NOT NULL,
            tx_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            credits_granted REAL NOT NULL DEFAULT 0.0,
            verified_at REAL,
            created_at REAL NOT NULL,
            error TEXT
        )
        """
    )
    conn.commit()


def _rpc_call(method: str, params: list) -> Any:
    """Make a JSON-RPC call to the Sepolia node."""
    if not SEPOLIA_RPC_URL:
        raise ValueError("HAVNAI_SEPOLIA_RPC_URL not configured. Cannot verify on-chain transactions.")

    resp = requests.post(
        SEPOLIA_RPC_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"RPC error: {data['error']}")
    return data.get("result")


def _decode_address(hex_value: str) -> str:
    """Decode a 32-byte padded address from event log topic/data."""
    # Remove 0x prefix, take last 40 chars (20 bytes)
    return "0x" + hex_value[-40:].lower()


def _decode_uint256(hex_value: str) -> int:
    """Decode a uint256 from hex."""
    return int(hex_value, 16)


def verify_hai_transfer(
    tx_hash: str,
    expected_wallet: str,
    expected_amount: float,
) -> Dict[str, Any]:
    """Verify a HAI token transfer on Sepolia.

    Checks:
    1. Transaction exists and succeeded (status 0x1)
    2. Contains a Transfer event from the HAI token contract
    3. Sender matches expected wallet
    4. Recipient matches treasury
    5. Amount matches expected amount (within rounding tolerance)
    6. Has sufficient confirmations

    Returns dict with 'verified' bool and optional 'error' string.
    """
    expected_wallet_lower = expected_wallet.lower()

    if not HAI_TOKEN_ADDRESS:
        return {"verified": False, "error": "HAI token address not configured on server."}
    if not HAI_TREASURY_WALLET:
        return {"verified": False, "error": "Treasury wallet not configured on server."}

    try:
        receipt = _rpc_call("eth_getTransactionReceipt", [tx_hash])
    except Exception as exc:
        return {
            "verified": False,
            "pending": True,
            "error": f"Failed to fetch tx receipt: {exc}",
        }

    if receipt is None:
        return {
            "verified": False,
            "pending": True,
            "error": "Transaction not found. It may still be pending.",
        }

    # Check tx succeeded
    tx_status = receipt.get("status", "0x0")
    if tx_status != "0x1":
        return {"verified": False, "error": "Transaction failed on-chain (reverted)."}

    # Check confirmations
    try:
        tx_block = int(receipt["blockNumber"], 16)
        current_block_hex = _rpc_call("eth_blockNumber", [])
        current_block = int(current_block_hex, 16)
        # Include the tx block itself as the first confirmation.
        confirmations = max(0, current_block - tx_block + 1)
        if confirmations < MIN_CONFIRMATIONS:
            return {
                "verified": False,
                "pending": True,
                "confirmations": confirmations,
                "error": f"Insufficient confirmations ({confirmations}/{MIN_CONFIRMATIONS}). Try again shortly.",
            }
    except Exception as exc:
        return {
            "verified": False,
            "pending": True,
            "error": f"Failed to check confirmations: {exc}",
        }

    # Find the Transfer event from the HAI token contract
    logs = receipt.get("logs", [])
    transfer_found = False

    for log_entry in logs:
        log_address = (log_entry.get("address") or "").lower()
        topics = log_entry.get("topics", [])

        # Must be from HAI token contract
        if log_address != HAI_TOKEN_ADDRESS:
            continue

        # Must be a Transfer event
        if len(topics) < 3 or topics[0].lower() != TRANSFER_EVENT_TOPIC:
            continue

        # Decode sender (topic[1]) and recipient (topic[2])
        sender = _decode_address(topics[1])
        recipient = _decode_address(topics[2])

        # Decode amount from data field
        raw_amount = _decode_uint256(log_entry.get("data", "0x0"))

        # Check sender matches expected wallet
        if sender != expected_wallet_lower:
            continue  # might be a different Transfer in the same tx

        # Check recipient matches treasury
        if recipient != HAI_TREASURY_WALLET:
            continue

        # Check amount — convert raw to human units (assume 18 decimals)
        # Allow small rounding tolerance
        human_amount = raw_amount / (10 ** 18)
        if abs(human_amount - expected_amount) > 0.001:
            return {
                "verified": False,
                "error": f"Amount mismatch: expected {expected_amount} HAI, got {human_amount:.6f} HAI.",
            }

        transfer_found = True
        break

    if not transfer_found:
        return {
            "verified": False,
            "error": "No matching HAI Transfer event found in transaction logs.",
        }

    return {"verified": True, "confirmations": confirmations}


def fund_credits_with_hai(
    wallet: str,
    amount: float,
    tx_hash: str,
) -> Dict[str, Any]:
    """Process a HAI funding request.

    1. Record the funding attempt (tx_hash UNIQUE prevents duplicates)
    2. Verify the transfer on-chain
    3. If valid, deposit credits
    4. Return result

    Idempotent: duplicate tx_hash returns 'already_processed'.
    """
    conn = get_db()
    now = time.time()
    credits_amount = amount * HAI_TO_CREDITS_RATE

    # Insert funding record — UNIQUE tx_hash prevents duplicates
    try:
        conn.execute(
            """
            INSERT INTO hai_fundings (wallet, amount, tx_hash, status, credits_granted, created_at)
            VALUES (?, ?, ?, 'pending', 0.0, ?)
            """,
            (wallet, amount, tx_hash, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # tx_hash already exists — check its status
        existing = conn.execute(
            "SELECT status, credits_granted FROM hai_fundings WHERE tx_hash = ?",
            (tx_hash,),
        ).fetchone()
        if existing:
            existing_status = str(existing["status"] or "").lower()
            if existing_status == "completed":
                return {
                    "status": "already_processed",
                    "tx_hash": tx_hash,
                    "credits_granted": float(existing["credits_granted"]),
                    "message": "This transaction has already been processed.",
                }
            # Non-completed records are retryable; reopen the tx for verification.
            conn.execute(
                "UPDATE hai_fundings SET status = 'pending', error = NULL WHERE tx_hash = ? AND status != 'completed'",
                (tx_hash,),
            )
            conn.commit()
        else:
            return {
                "status": "error",
                "error": "Duplicate transaction hash.",
            }

    log_event("HAI funding initiated", wallet=wallet, amount=amount, tx_hash=tx_hash)

    # Verify the transfer on-chain
    verification = verify_hai_transfer(tx_hash, wallet, amount)

    if not verification.get("verified"):
        error_msg = verification.get("error", "Verification failed.")
        is_pending = bool(verification.get("pending"))
        next_status = "pending" if is_pending else "failed"
        conn.execute(
            "UPDATE hai_fundings SET status = ?, error = ? WHERE tx_hash = ?",
            (next_status, error_msg, tx_hash),
        )
        conn.commit()
        if is_pending:
            log_event(
                "HAI funding pending verification",
                wallet=wallet,
                tx_hash=tx_hash,
                error=error_msg,
                confirmations=verification.get("confirmations"),
            )
        else:
            log_event("HAI funding verification failed", wallet=wallet, tx_hash=tx_hash, error=error_msg)
        return {
            "status": next_status,
            "tx_hash": tx_hash,
            "error": error_msg,
            "confirmations": verification.get("confirmations"),
        }

    # Verification passed — deposit credits
    # Use atomic UPDATE to prevent double-crediting the same tx hash.
    cur = conn.execute(
        """
        UPDATE hai_fundings
        SET status = 'completed', credits_granted = ?, verified_at = ?, error = NULL
        WHERE tx_hash = ? AND status != 'completed'
        """,
        (credits_amount, time.time(), tx_hash),
    )
    conn.commit()

    if cur.rowcount == 0:
        # Race: another request already processed this tx
        return {
            "status": "already_processed",
            "tx_hash": tx_hash,
            "message": "This transaction was already processed by another request.",
        }

    new_balance = deposit_credits(wallet, credits_amount, reason=f"hai:{tx_hash}")

    log_event(
        "HAI funding completed",
        wallet=wallet,
        amount=amount,
        credits_granted=credits_amount,
        new_balance=new_balance,
        tx_hash=tx_hash,
    )

    return {
        "status": "completed",
        "tx_hash": tx_hash,
        "credits_granted": credits_amount,
        "balance": new_balance,
    }


def get_funding_history(wallet: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent HAI funding records for a wallet."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, wallet, amount, tx_hash, status, credits_granted, verified_at, created_at, error
        FROM hai_fundings
        WHERE wallet = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (wallet, limit),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "wallet": row["wallet"],
            "amount": float(row["amount"]),
            "tx_hash": row["tx_hash"],
            "status": row["status"],
            "credits_granted": float(row["credits_granted"]),
            "verified_at": row["verified_at"],
            "created_at": row["created_at"],
            "error": row["error"],
        }
        for row in rows
    ]

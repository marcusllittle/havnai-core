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

# Test-token requests are a manual-review queue, separate from verified funding.
TESTER_DISTRIBUTION_ENABLED = os.getenv("HAVNAI_TESTER_DISTRIBUTION_ENABLED", "1").lower() in {"1", "true", "yes"}
TESTER_DISTRIBUTION_ALLOWED_WALLETS = {value.strip().lower() for value in os.getenv("HAVNAI_TESTER_DISTRIBUTION_ALLOWLIST", "").split(",") if value.strip()}
TESTER_DISTRIBUTION_DEFAULT_HAI = 100.0
TESTER_DISTRIBUTION_COOLDOWN_HOURS = 24

def tester_distribution_config() -> Dict[str, Any]:
    return {
        "enabled": os.getenv("HAVNAI_TESTER_DISTRIBUTION_ENABLED", str(int(TESTER_DISTRIBUTION_ENABLED))).lower() in {"1", "true", "yes"},
        "allowlist_enforced": bool(TESTER_DISTRIBUTION_ALLOWED_WALLETS),
        "default_request_hai": TESTER_DISTRIBUTION_DEFAULT_HAI,
        "cooldown_hours": TESTER_DISTRIBUTION_COOLDOWN_HOURS,
    }


def _init_tester_requests(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS tester_distribution_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, wallet TEXT NOT NULL,
        requested_hai REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        request_note TEXT NOT NULL DEFAULT '', admin_note TEXT NOT NULL DEFAULT '',
        tx_hash TEXT, credits_granted REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL, resolved_at REAL
    )""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tester_distribution_requests)")}
    if "credits_applied" not in columns:
        conn.execute("ALTER TABLE tester_distribution_requests ADD COLUMN credits_applied REAL NOT NULL DEFAULT 0")
    conn.commit()


def create_tester_distribution_request(*, wallet: str, requested_hai: Any = None,
                                       request_note: str = "") -> Dict[str, Any]:
    import math
    config = tester_distribution_config()
    if not config["enabled"]:
        return {"error": "disabled", "message": "Test HAI requests are currently disabled."}
    wallet = wallet.strip().lower()
    allowed = TESTER_DISTRIBUTION_ALLOWED_WALLETS
    if allowed and wallet not in allowed:
        return {"error": "wallet_not_allowed", "message": "This wallet is not on the test HAI allowlist."}
    try:
        amount = float(config["default_request_hai"] if requested_hai is None else requested_hai)
    except (TypeError, ValueError):
        amount = 0
    if not math.isfinite(amount) or amount <= 0:
        return {"error": "invalid_amount", "message": "Enter a positive, finite HAI amount."}
    conn = get_db()
    _init_tester_requests(conn)
    now = time.time()
    cooldown = max(0, config["cooldown_hours"]) * 3600
    conn.execute("BEGIN IMMEDIATE")
    try:
        previous = conn.execute("SELECT * FROM tester_distribution_requests WHERE wallet=? ORDER BY id DESC LIMIT 1", (wallet,)).fetchone()
        if previous and previous["status"] == "pending":
            conn.rollback()
            return {"status": "pending_exists", "request": dict(previous), "request_id": previous["id"], "message": "Your existing request is awaiting manual review."}
        if previous and now - previous["created_at"] < cooldown:
            conn.rollback()
            return {"error": "cooldown_active", "message": "Please wait before requesting more test HAI.", "retry_after_seconds": int(cooldown - (now - previous["created_at"]))}
        cursor = conn.execute("INSERT INTO tester_distribution_requests (wallet, requested_hai, request_note, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                              (wallet, amount, request_note[:2000], now, now))
        row = conn.execute("SELECT * FROM tester_distribution_requests WHERE id=?", (cursor.lastrowid,)).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"status": "pending", "request": dict(row), "request_id": row["id"], "message": "Request received for manual review. No tokens or credits have been granted yet."}


def list_tester_distribution_requests(*, wallet: Optional[str] = None, status: Optional[str] = None,
                                     limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db()
    _init_tester_requests(conn)
    clauses, params = [], []
    if wallet:
        clauses.append("wallet=?")
        params.append(wallet.strip().lower())
    if status:
        clauses.append("status=?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return [dict(row) for row in conn.execute("SELECT * FROM tester_distribution_requests" + where + " ORDER BY id DESC LIMIT ?", (*params, max(1, min(limit, 200))))]


def resolve_tester_distribution_request(*, request_id: int, new_status: str, admin_note: str = "",
                                        tx_hash: Optional[str] = None, credits_granted: Any = None) -> Dict[str, Any]:
    """Apply an explicit admin credit grant atomically; never transfer HAI tokens."""
    import math
    if new_status not in {"approved", "rejected", "fulfilled", "completed"}:
        return {"error": "invalid_status"}
    try:
        granted = float(credits_granted or 0)
    except (ValueError, TypeError):
        return {"error": "invalid_credits_granted"}
    if not math.isfinite(granted) or granted < 0:
        return {"error": "invalid_credits_granted"}
    conn = get_db()
    _init_tester_requests(conn)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM tester_distribution_requests WHERE id=?", (request_id,)).fetchone()
        if not row:
            conn.rollback()
            return {"error": "not_found"}
        applied = float(row["credits_applied"])
        if new_status == "rejected" and granted > 0:
            conn.rollback()
            return {"error": "invalid_credits_granted"}
        target = max(applied, granted)
        delta = target - applied
        if delta > 0:
            conn.execute("""INSERT INTO credits (wallet, balance, total_deposited, total_spent, updated_at)
                VALUES (?, ?, ?, 0, ?) ON CONFLICT(wallet) DO UPDATE SET
                balance=balance+excluded.balance, total_deposited=total_deposited+excluded.total_deposited,
                updated_at=excluded.updated_at""", (row["wallet"], delta, delta, now))
        conn.execute("""UPDATE tester_distribution_requests SET status=?, admin_note=?,
            tx_hash=COALESCE(?, tx_hash), credits_granted=?, credits_applied=?, updated_at=?, resolved_at=? WHERE id=?""",
            (new_status, admin_note[:2000], tx_hash, target, target, now, now, request_id))
        updated = dict(conn.execute("SELECT * FROM tester_distribution_requests WHERE id=?", (request_id,)).fetchone())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"status": new_status, "request": updated}

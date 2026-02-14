"""Stripe payment integration for HavnAI credits.

Handles Checkout Session creation and webhook processing.
Credits are deposited only after Stripe confirms payment via webhook.

Environment variables:
    STRIPE_SECRET_KEY       – Stripe API secret key (sk_test_... or sk_live_...)
    STRIPE_WEBHOOK_SECRET   – Webhook signing secret (whsec_...)
    STRIPE_ENABLED          – Set to "1" / "true" to enable Stripe routes
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

# Will be injected by app.py (same pattern as credits.py)
get_db: Callable[[], "sqlite3.Connection"]
log_event: Callable[..., None]
deposit_credits: Callable[[str, float, str], float]

STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_ENABLED: bool = os.getenv("STRIPE_ENABLED", "").strip().lower() in {"1", "true", "yes"}

# Credit packages – price in USD cents, credits granted.
CREDIT_PACKAGES: List[Dict[str, Any]] = [
    {"id": "starter", "name": "Starter Pack", "credits": 50, "price_cents": 500, "description": "50 credits"},
    {"id": "creator", "name": "Creator Pack", "credits": 150, "price_cents": 1200, "description": "150 credits – 20% bonus"},
    {"id": "pro", "name": "Pro Pack", "credits": 500, "price_cents": 3500, "description": "500 credits – 30% bonus"},
]


def _get_package(package_id: str) -> Optional[Dict[str, Any]]:
    for pkg in CREDIT_PACKAGES:
        if pkg["id"] == package_id:
            return pkg
    return None


def init_stripe_tables(conn: "sqlite3.Connection") -> None:
    """Create the payments tracking table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stripe_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session_id TEXT UNIQUE NOT NULL,
            wallet TEXT NOT NULL,
            package_id TEXT NOT NULL,
            credits_amount REAL NOT NULL,
            price_cents INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            completed_at REAL
        )
        """
    )
    conn.commit()


def create_checkout_session(wallet: str, package_id: str, success_url: str, cancel_url: str) -> Dict[str, Any]:
    """Create a Stripe Checkout Session and record it locally.

    Returns dict with ``session_id`` and ``checkout_url`` for the frontend redirect.
    """
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    pkg = _get_package(package_id)
    if not pkg:
        raise ValueError(f"Unknown package: {package_id}")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": pkg["price_cents"],
                    "product_data": {
                        "name": f"HavnAI {pkg['name']}",
                        "description": pkg["description"],
                    },
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "wallet": wallet,
            "package_id": package_id,
            "credits": str(pkg["credits"]),
        },
    )

    # Record pending payment
    conn = get_db()
    conn.execute(
        """
        INSERT INTO stripe_payments (stripe_session_id, wallet, package_id, credits_amount, price_cents, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?)
        """,
        (session.id, wallet, package_id, pkg["credits"], pkg["price_cents"], time.time()),
    )
    conn.commit()

    log_event(
        "Stripe checkout created",
        wallet=wallet,
        package_id=package_id,
        session_id=session.id,
    )

    return {
        "session_id": session.id,
        "checkout_url": session.url,
    }


def handle_webhook_event(payload: bytes, sig_header: str) -> Dict[str, Any]:
    """Verify and process a Stripe webhook event.

    On ``checkout.session.completed``, deposits credits to the wallet
    recorded in the session metadata.  Idempotent — duplicate webhooks
    for an already-completed session are safely ignored.
    """
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        log_event("Stripe webhook: invalid signature", level="warning")
        raise

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session["id"]

        # Verify the payment actually succeeded
        payment_status = session.get("payment_status", "")
        if payment_status != "paid":
            log_event(
                "Stripe webhook: session completed but payment_status is not 'paid'",
                level="warning",
                session_id=session_id,
                payment_status=payment_status,
            )
            return {"status": "ignored", "reason": f"payment_status={payment_status}"}

        metadata = session.get("metadata", {})
        wallet = metadata.get("wallet", "")
        package_id = metadata.get("package_id", "")
        credits_amount = float(metadata.get("credits", "0"))

        if not wallet or credits_amount <= 0:
            log_event(
                "Stripe webhook: missing wallet or credits in metadata",
                level="warning",
                session_id=session_id,
            )
            return {"status": "ignored", "reason": "missing metadata"}

        # Idempotency: only process if this session is still pending.
        # Use atomic UPDATE ... WHERE status='pending' to prevent double-deposits.
        conn = get_db()
        cur = conn.execute(
            """
            UPDATE stripe_payments
            SET status = 'completed', completed_at = ?
            WHERE stripe_session_id = ? AND status = 'pending'
            """,
            (time.time(), session_id),
        )
        conn.commit()

        if cur.rowcount == 0:
            # Already processed or unknown session — skip deposit
            log_event(
                "Stripe webhook: duplicate or unknown session, skipping deposit",
                level="info",
                session_id=session_id,
            )
            return {"status": "already_processed", "session_id": session_id}

        # Deposit credits (only reached once per session)
        new_balance = deposit_credits(wallet, credits_amount, reason=f"stripe:{package_id}:{session_id}")

        log_event(
            "Stripe payment completed",
            wallet=wallet,
            package_id=package_id,
            credits=credits_amount,
            new_balance=new_balance,
            session_id=session_id,
        )

        return {"status": "credits_deposited", "credits": credits_amount, "balance": new_balance}

    return {"status": "ignored", "event_type": event["type"]}


def get_payment_history(wallet: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent payment records for a wallet."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT stripe_session_id, package_id, credits_amount, price_cents, status, created_at, completed_at
        FROM stripe_payments
        WHERE wallet = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (wallet, limit),
    ).fetchall()
    return [
        {
            "session_id": row["stripe_session_id"],
            "package_id": row["package_id"],
            "credits": row["credits_amount"],
            "price_cents": row["price_cents"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]

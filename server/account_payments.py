"""Account-owned Stripe Checkout and reconciled, append-only payment receipts.

External reads happen outside SQLite write transactions. A revision fence prevents
an older read overwriting a newer reconciliation; conflicts request a webhook retry.
Neither redirect parameters nor webhook object snapshots can grant credits.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

import stripe

import account_ledger as ledger

PACKAGES = (
    {"id": "starter", "name": "Starter Pack", "units": 50_000, "price_cents": 500},
    {"id": "creator", "name": "Creator Pack", "units": 150_000, "price_cents": 1200},
    {"id": "pro", "name": "Pro Pack", "units": 500_000, "price_cents": 3500},
)


class PaymentError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


@dataclass(frozen=True)
class Config:
    secret_key: str
    webhook_secret: str
    origin: str
    terms_version: str
    enabled: bool = False

    @classmethod
    def from_environment(cls):
        return cls(os.getenv("STRIPE_SECRET_KEY", "").strip(),
                   os.getenv("STRIPE_ACCOUNT_WEBHOOK_SECRET", "").strip(),
                   os.getenv("HAVNAI_CHECKOUT_ORIGIN", "").strip().rstrip("/"),
                   os.getenv("HAVNAI_CREDIT_TERMS_VERSION", "").strip(),
                   os.getenv("HAVNAI_ACCOUNT_CHECKOUT_ENABLED", "").lower() in {"1", "true"})

    @property
    def livemode(self):
        return self.secret_key.startswith("sk_live_") or self.secret_key.startswith("rk_live_")

    def require(self, *, checkout=False):
        if not self.secret_key.startswith(("sk_test_", "sk_live_", "rk_test_", "rk_live_")) or not self.webhook_secret:
            raise PaymentError("account_payments_not_configured", 503)
        if checkout:
            parts = urlsplit(self.origin)
            local = parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
            if (not self.enabled or not self.terms_version or not parts.hostname or parts.path
                    or parts.query or parts.fragment or parts.username or parts.password
                    or not (parts.scheme == "https" or (local and not self.livemode))):
                raise PaymentError("account_checkout_not_configured", 503)


def initialize(conn):
    if conn.in_transaction:
        raise RuntimeError("payment initialization requires an idle connection")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_purchases (
            id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
            request_key TEXT NOT NULL, package_id TEXT NOT NULL,
            units INTEGER NOT NULL CHECK(units>0), price_cents INTEGER NOT NULL CHECK(price_cents>0),
            currency TEXT NOT NULL, terms_version TEXT NOT NULL, livemode INTEGER NOT NULL,
            checkout_params TEXT NOT NULL, session_id TEXT UNIQUE, checkout_url TEXT,
            payment_id TEXT UNIQUE, state TEXT NOT NULL DEFAULT 'pending',
            revision INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            UNIQUE(account_id, request_key)
        );
        CREATE TABLE IF NOT EXISTS account_payment_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, purchase_id TEXT NOT NULL REFERENCES account_purchases(id),
            payment_id TEXT NOT NULL UNIQUE, account_id TEXT NOT NULL REFERENCES accounts(id),
            price_cents INTEGER NOT NULL, currency TEXT NOT NULL, units INTEGER NOT NULL,
            terms_version TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account_payment_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, purchase_id TEXT NOT NULL REFERENCES account_purchases(id),
            revision INTEGER NOT NULL, refunded_cents INTEGER NOT NULL, disputed_cents INTEGER NOT NULL,
            retained_units INTEGER NOT NULL, settled_delta INTEGER NOT NULL,
            evidence TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(purchase_id,revision)
        );
        CREATE TABLE IF NOT EXISTS account_payment_events (
            event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, payment_id TEXT, processed_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS account_purchases_owner ON account_purchases(account_id,created_at);
        CREATE TRIGGER IF NOT EXISTS account_receipts_no_update BEFORE UPDATE ON account_payment_receipts
            BEGIN SELECT RAISE(ABORT,'payment receipts are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_receipts_no_delete BEFORE DELETE ON account_payment_receipts
            BEGIN SELECT RAISE(ABORT,'payment receipts are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_adjustments_no_update BEFORE UPDATE ON account_payment_adjustments
            BEGIN SELECT RAISE(ABORT,'payment adjustments are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_adjustments_no_delete BEFORE DELETE ON account_payment_adjustments
            BEGIN SELECT RAISE(ABORT,'payment adjustments are append-only'); END;
    """)


def _row(conn, purchase_id):
    cursor = conn.execute("SELECT * FROM account_purchases WHERE id=?", (purchase_id,))
    values = cursor.fetchone()
    return dict(zip([column[0] for column in cursor.description], values)) if values else None


def _plain(value):
    if isinstance(value, stripe.StripeObject):
        convert = getattr(value, "to_dict", None)
        return convert() if callable(convert) else value.to_dict_recursive()
    return value


def _object_id(value):
    value = _plain(value)
    return value.get("id") if isinstance(value, dict) else value


def _require_database_mode(conn, config):
    if conn.execute("SELECT 1 FROM account_purchases WHERE livemode!=? LIMIT 1", (int(config.livemode),)).fetchone():
        raise PaymentError("payment_database_mode_mismatch", 503)


def create_checkout(conn, account_id, *, package_id, request_key, terms_version, config):
    config.require(checkout=True)
    if not isinstance(request_key, str) or not 16 <= len(request_key) <= 128:
        raise PaymentError("invalid_idempotency_key")
    package = next((p for p in PACKAGES if p["id"] == package_id), None)
    if not package or terms_version != config.terms_version:
        raise PaymentError("invalid_package_or_terms")
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _require_database_mode(conn, config)
        existing = conn.execute("SELECT id FROM account_purchases WHERE account_id=? AND request_key=?",
                                (account_id, request_key)).fetchone()
        if existing:
            purchase = _row(conn, existing[0])
            if (purchase["package_id"] != package_id or purchase["terms_version"] != terms_version
                    or bool(purchase["livemode"]) != config.livemode):
                raise PaymentError("idempotency_conflict", 409)
        else:
            purchase_id = "pur_" + uuid.uuid4().hex
            metadata = {"havnai_purchase": purchase_id, "surface": "account_v2"}
            params = {"mode": "payment", "payment_method_types": ["card"],
                      "line_items": [{"price_data": {"currency": "usd", "unit_amount": package["price_cents"],
                                      "product_data": {"name": "HavnAI " + package["name"]}}, "quantity": 1}],
                      "client_reference_id": purchase_id, "metadata": metadata,
                      "payment_intent_data": {"metadata": metadata},
                      "success_url": config.origin + "/account?purchase=" + purchase_id,
                      "cancel_url": config.origin + "/pricing?purchase=" + purchase_id}
            conn.execute("""INSERT INTO account_purchases
                (id,account_id,request_key,package_id,units,price_cents,currency,terms_version,livemode,
                 checkout_params,created_at,updated_at) VALUES (?,?,?,?,?,?,'usd',?,?,?,?,?)""",
                (purchase_id, account_id, request_key, package_id, package["units"], package["price_cents"],
                 terms_version, int(config.livemode), json.dumps(params), now, now))
            purchase = _row(conn, purchase_id)
    if purchase["session_id"]:
        return {"purchase_id": purchase["id"], "checkout_url": purchase["checkout_url"], "state": purchase["state"]}
    # Stripe may prune idempotency keys after 24h. An ambiguous old request must
    # be reconciled by an operator instead of creating a second charge session.
    if now - purchase["created_at"] >= 23 * 3600:
        raise PaymentError("checkout_requires_reconciliation", 409)
    session = _plain(stripe.checkout.Session.create(**json.loads(purchase["checkout_params"]),
        api_key=config.secret_key, idempotency_key="havnai-account:" + purchase["id"]))
    if bool(session.get("livemode")) != config.livemode:
        raise PaymentError("payment_mode_mismatch", 409)
    checkout_url = session.get("url")
    if not isinstance(checkout_url, str) or urlsplit(checkout_url).scheme != "https" or urlsplit(checkout_url).hostname != "checkout.stripe.com":
        raise PaymentError("invalid_checkout_response", 502)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        current = _row(conn, purchase["id"])
        if current["session_id"] not in {None, session["id"]}:
            raise PaymentError("checkout_session_conflict", 409)
        conn.execute("UPDATE account_purchases SET session_id=?,checkout_url=?,updated_at=? WHERE id=?",
                     (session["id"], checkout_url, time.time(), purchase["id"]))
    return {"purchase_id": purchase["id"], "checkout_url": checkout_url, "state": current["state"]}


def reconcile(conn, purchase_id, payment_id, *, config):
    """Retrieve current provider state, then atomically reconcile credits + receipts."""
    config.require()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _require_database_mode(conn, config)
        purchase = _row(conn, purchase_id)
        if not purchase:
            raise PaymentError("purchase_not_found", 503)
        if purchase["payment_id"] not in {None, payment_id} or bool(purchase["livemode"]) != config.livemode:
            raise PaymentError("payment_binding_mismatch", 409)
        conn.execute("UPDATE account_purchases SET revision=revision+1 WHERE id=?", (purchase_id,))
        revision = purchase["revision"] + 1
    intent = _plain(stripe.PaymentIntent.retrieve(payment_id, expand=["latest_charge"], api_key=config.secret_key))
    if (intent.get("metadata", {}).get("havnai_purchase") != purchase_id
            or intent.get("metadata", {}).get("surface") != "account_v2"
            or intent.get("id") != payment_id or bool(intent.get("livemode")) != config.livemode
            or intent.get("currency") != purchase["currency"] or intent.get("amount") != purchase["price_cents"]):
        raise PaymentError("payment_binding_mismatch", 409)
    paid = intent.get("status") == "succeeded"
    refunded = disputed = 0
    evidence = []
    if paid:
        charge = intent.get("latest_charge")
        if (intent.get("amount_received") != purchase["price_cents"] or not isinstance(charge, dict)
                or not charge.get("paid") or charge.get("currency") != purchase["currency"]
                or charge.get("amount") != purchase["price_cents"]
                or _object_id(charge.get("payment_intent")) != payment_id):
            raise PaymentError("payment_amount_mismatch", 409)
        refunded = charge.get("amount_refunded", 0)
        if type(refunded) is not int or not 0 <= refunded <= purchase["price_cents"]:
            raise PaymentError("invalid_refund_state", 409)
        disputes = stripe.Dispute.list(charge=charge["id"], limit=100, api_key=config.secret_key)
        for dispute in disputes.auto_paging_iter():
            dispute = _plain(dispute)
            status = dispute["status"]
            amount = dispute["amount"]
            if type(amount) is not int or amount < 0 or dispute["currency"] != purchase["currency"]:
                raise PaymentError("invalid_dispute_state", 409)
            if status in {"needs_response", "under_review", "lost"}:
                disputed += amount
            elif status not in {"won", "prevented", "warning_closed", "warning_needs_response", "warning_under_review"}:
                raise PaymentError("unknown_dispute_state", 503)
            evidence.append({"id": dispute["id"], "status": status, "amount": amount})
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        current = _row(conn, purchase_id)
        if current["revision"] != revision:
            raise PaymentError("payment_reconciliation_retry", 503)
        try:
            conn.execute("UPDATE account_purchases SET payment_id=? WHERE id=?", (payment_id, purchase_id))
        except sqlite3.IntegrityError:
            raise PaymentError("payment_binding_mismatch", 409) from None
        receipt = conn.execute("SELECT id FROM account_payment_receipts WHERE purchase_id=?", (purchase_id,)).fetchone()
        if not paid:
            if receipt:  # A paid intent cannot become unpaid; never erase a receipt.
                raise PaymentError("payment_state_conflict", 503)
            state = "cancelled" if intent.get("status") == "canceled" else "pending"
        else:
            if not receipt:
                ledger.fund_in_transaction(conn, purchase["account_id"], purchase["units"], payment_id=payment_id)
                conn.execute("""INSERT INTO account_payment_receipts
                    (purchase_id,payment_id,account_id,price_cents,currency,units,terms_version,created_at)
                    VALUES (?,?,?,?,?,?,?,?)""", (purchase_id, payment_id, purchase["account_id"], purchase["price_cents"],
                    purchase["currency"], purchase["units"], purchase["terms_version"], time.time()))
            reversed_cents = min(purchase["price_cents"], refunded + disputed)
            retained = purchase["units"] * (purchase["price_cents"] - reversed_cents) // purchase["price_cents"]
            previous = conn.execute("""SELECT retained_units,refunded_cents,disputed_cents,evidence
                FROM account_payment_adjustments WHERE purchase_id=? ORDER BY id DESC LIMIT 1""", (purchase_id,)).fetchone()
            evidence_json = json.dumps(sorted(evidence, key=lambda d: d["id"]), sort_keys=True)
            before = previous[0] if previous else purchase["units"]
            delta = retained - before
            if delta:
                ledger.adjust_payment_in_transaction(conn, purchase["account_id"], payment_id=payment_id,
                    target_units=retained, adjustment_id=f"{purchase_id}:{revision}")
            if not previous or tuple(previous) != (retained, refunded, disputed, evidence_json):
                conn.execute("""INSERT INTO account_payment_adjustments
                    (purchase_id,revision,refunded_cents,disputed_cents,retained_units,settled_delta,evidence,created_at)
                    VALUES (?,?,?,?,?,?,?,?)""", (purchase_id, revision, refunded, disputed, retained, delta, evidence_json, time.time()))
            state = "disputed" if disputed else "refunded" if refunded == purchase["price_cents"] else "partially_refunded" if refunded else "paid"
        conn.execute("UPDATE account_purchases SET state=?,updated_at=? WHERE id=?", (state, time.time(), purchase_id))
    return state


EVENT_TYPES = {"checkout.session.completed", "checkout.session.async_payment_succeeded", "checkout.session.async_payment_failed",
               "checkout.session.expired", "payment_intent.succeeded", "payment_intent.payment_failed", "payment_intent.canceled",
               "charge.refunded", "charge.dispute.created", "charge.dispute.updated", "charge.dispute.closed",
               "charge.dispute.funds_withdrawn", "charge.dispute.funds_reinstated", "refund.created", "refund.updated", "refund.failed"}


def webhook(conn, payload, signature, *, config):
    config.require()
    try:
        event = _plain(stripe.Webhook.construct_event(payload, signature, config.webhook_secret))
    except (ValueError, stripe.SignatureVerificationError):
        raise PaymentError("invalid_webhook_signature", 400) from None
    if event["type"] not in EVENT_TYPES:
        return {"received": True}
    if bool(event.get("livemode")) != config.livemode or event.get("account"):
        raise PaymentError("payment_mode_mismatch", 400)
    if conn.execute("SELECT 1 FROM account_payment_events WHERE event_id=?", (event["id"],)).fetchone():
        return {"received": True}
    obj = event["data"]["object"]
    kind = obj.get("object")
    payment_id = obj.get("id") if kind == "payment_intent" else _object_id(obj.get("payment_intent"))
    if not payment_id and obj.get("charge"):
        charge = _plain(stripe.Charge.retrieve(_object_id(obj["charge"]), api_key=config.secret_key))
        payment_id = _object_id(charge.get("payment_intent"))
    if payment_id:
        intent = _plain(stripe.PaymentIntent.retrieve(payment_id, api_key=config.secret_key))
        metadata = intent.get("metadata", {})
        if metadata.get("surface") == "account_v2":
            reconcile(conn, metadata.get("havnai_purchase"), payment_id, config=config)
    elif kind == "checkout.session" and obj.get("metadata", {}).get("surface") == "account_v2":
        # A Checkout cancellation redirect is not a cancellation of payment.
        # Only a current, expired provider session can mark a no-intent order expired.
        recover_purchase(conn, obj["metadata"].get("havnai_purchase"), config=config, session_id=obj["id"])
    with conn:
        conn.execute("INSERT OR IGNORE INTO account_payment_events VALUES (?,?,?,?)", (event["id"], event["type"], payment_id, time.time()))
    return {"received": True}


def receipt(conn, account_id, purchase_id):
    purchase = _row(conn, purchase_id)
    if not purchase or purchase["account_id"] != account_id:
        raise PaymentError("purchase_not_found", 404)
    paid = conn.execute("""SELECT id,price_cents,currency,units,terms_version,created_at FROM account_payment_receipts
        WHERE purchase_id=? AND account_id=?""", (purchase_id, account_id)).fetchone()
    adjustments = conn.execute("""SELECT refunded_cents,disputed_cents,retained_units,settled_delta,created_at
        FROM account_payment_adjustments WHERE purchase_id=? ORDER BY id""", (purchase_id,)).fetchall()
    return {"purchase_id": purchase_id, "state": purchase["state"], "scale": ledger.SCALE,
            "receipt": dict(zip(["id", "price_cents", "currency", "units", "terms_version", "created_at"], paid)) if paid else None,
            "adjustments": [dict(zip(["refunded_cents", "disputed_cents", "retained_units", "settled_delta", "created_at"], row)) for row in adjustments]}


def recover_purchase(conn, purchase_id, *, config, session_id=None):
    """Operator recovery for a missed webhook or lost Checkout creation response.

    An optional session id is a lookup hint, never payment evidence. Stripe must
    return the exact persisted purchase binding, amount, currency, and mode.
    """
    config.require()
    purchase = _row(conn, purchase_id)
    if not purchase:
        raise PaymentError("purchase_not_found", 404)
    candidate = session_id or purchase["session_id"]
    if not candidate:
        raise PaymentError("checkout_session_required", 409)
    session = _plain(stripe.checkout.Session.retrieve(candidate, api_key=config.secret_key))
    if (session.get("id") != candidate or session.get("metadata", {}).get("havnai_purchase") != purchase_id
            or session.get("metadata", {}).get("surface") != "account_v2"
            or session.get("client_reference_id") != purchase_id or session.get("mode") != "payment"
            or session.get("amount_total") != purchase["price_cents"] or session.get("currency") != purchase["currency"]
            or bool(session.get("livemode")) != config.livemode):
        raise PaymentError("payment_binding_mismatch", 409)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _require_database_mode(conn, config)
        current = _row(conn, purchase_id)
        if current["session_id"] not in {None, candidate}:
            raise PaymentError("checkout_session_conflict", 409)
        checkout_url = session.get("url")
        if checkout_url and (urlsplit(checkout_url).scheme != "https" or urlsplit(checkout_url).hostname != "checkout.stripe.com"):
            raise PaymentError("invalid_checkout_response", 502)
        conn.execute("UPDATE account_purchases SET session_id=?,checkout_url=COALESCE(?,checkout_url),updated_at=? WHERE id=?",
                     (candidate, checkout_url, time.time(), purchase_id))
    payment_id = _object_id(session.get("payment_intent"))
    if payment_id:
        return reconcile(conn, purchase_id, payment_id, config=config)
    if session.get("status") == "expired":
        with conn:
            conn.execute("UPDATE account_purchases SET state='expired',updated_at=? WHERE id=? AND payment_id IS NULL AND state='pending'",
                         (time.time(), purchase_id))
    return _row(conn, purchase_id)["state"]


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from urllib.parse import quote

    parser = argparse.ArgumentParser(description="Reconcile an account purchase against Stripe without creating a charge.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--purchase", required=True)
    parser.add_argument("--session", help="Stripe Checkout session id; required only if its creation response was lost")
    args = parser.parse_args()
    connection = sqlite3.connect("file:" + quote(str(Path(args.database).resolve())) + "?mode=rw", uri=True)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        print(recover_purchase(connection, args.purchase, session_id=args.session, config=Config.from_environment()))
    except (PaymentError, stripe.StripeError) as exc:
        parser.exit(1, (str(exc) if isinstance(exc, PaymentError) else "payment_provider_unavailable") + "\n")
    finally:
        connection.close()

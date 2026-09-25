"""Commercial account API. Does not share legacy wallet or owner authorization."""
from __future__ import annotations

import sqlite3
import uuid
from functools import wraps
from typing import Callable

from flask import Blueprint, g, jsonify, request

import account_auth
import account_identity
import account_ledger
import account_payments
import account_lifecycle
import stripe


def create_blueprint(get_db: Callable[[], sqlite3.Connection], rate_limit: Callable[..., bool]) -> Blueprint:
    api = Blueprint("commercial_accounts", __name__, url_prefix="/v2")

    def authenticate(*, recent=False):
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                if not rate_limit(f"account:{request.remote_addr}", limit=120):
                    return fail("rate_limited", 429)
                g.account_principal = account_auth.verify_bearer(
                    request.headers.get("Authorization", ""),
                    config=account_auth.AuthConfig.from_environment(), recent=recent,
                )
                g.account_id = account_identity.ensure_account(get_db(), g.account_principal)
                return function(*args, **kwargs)
            return wrapped
        return decorate

    def fail(code, status):
        return jsonify({"error": {"code": code, "message": code.replace("_", " ").capitalize()},
                        "request_id": uuid.uuid4().hex}), status

    @api.errorhandler(account_auth.AccountAuthError)
    def auth_error(exc):
        return fail(str(exc), exc.status)

    @api.errorhandler(account_identity.IdentityError)
    def identity_error(exc):
        code = str(exc)
        status = 401 if code == "account_session_revoked" else 403 if code == "account_suspended" else 409 if code == "wallet_already_linked" else 422
        return fail(code, status)

    @api.errorhandler(account_payments.PaymentError)
    def payment_error(exc):
        return fail(str(exc), exc.status)

    @api.errorhandler(account_lifecycle.LifecycleError)
    def lifecycle_error(exc):
        return fail(str(exc), exc.status)

    @api.post("/auth/clerk/webhook")
    def clerk_webhook():
        if request.content_length is not None and request.content_length > 1024 * 1024:
            return fail("payload_too_large", 413)
        return jsonify(account_lifecycle.webhook(get_db(), request.get_data(), request.headers,
            config=account_lifecycle.Config.from_environment()))

    @api.errorhandler(stripe.StripeError)
    def provider_error(exc):
        # Provider exceptions can contain request data. Keep it out of responses.
        return fail("payment_provider_unavailable", 503)

    @api.after_request
    def private(response):
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Authorization"
        return response

    @api.get("/account")
    @authenticate()
    def account():
        rows = get_db().execute("""SELECT id,wallet,namespace,verified_at,linked_at
            FROM wallet_links WHERE account_id=? AND unlinked_at IS NULL ORDER BY linked_at""", (g.account_id,)).fetchall()
        return jsonify({"id": g.account_id, "status": "active", "wallets": [dict(row) for row in rows],
                        "wallet_capabilities": ["rewards", "token_transfers", "blockchain_ownership"]})

    @api.get("/account/credits")
    @authenticate()
    def credits():
        return jsonify(account_ledger.balance(get_db(), g.account_id))

    @api.get("/account/ledger")
    @authenticate()
    def ledger():
        try:
            cursor = int(request.args.get("before", str(2**63 - 1)))
            limit = min(100, max(1, int(request.args.get("limit", "30"))))
        except ValueError:
            return fail("invalid_pagination", 422)
        if not 0 < cursor <= 2**63 - 1:
            return fail("invalid_pagination", 422)
        rows = get_db().execute("""SELECT id,operation,resource_id,settled_delta,reserved_delta,
            settled_after,reserved_after,reason,created_at FROM account_credit_ledger
            WHERE account_id=? AND id<? ORDER BY id DESC LIMIT ?""", (g.account_id, cursor, limit)).fetchall()
        return jsonify({"entries": [dict(row) for row in rows], "next_cursor": rows[-1]["id"] if len(rows) == limit else None})

    @api.get("/credit-packages")
    def credit_packages():
        config = account_payments.Config.from_environment()
        try:
            config.require(checkout=True)
            available = True
        except account_payments.PaymentError:
            available = False
        return jsonify({"packages": account_payments.PACKAGES, "currency": "usd", "scale": account_ledger.SCALE,
                        "checkout_available": available, "terms_version": config.terms_version,
                        "catalog_version": account_payments.catalog_version(config),
                        "terms_url": config.terms_url if config.valid_url(config.terms_url) else None,
                        "refund_url": config.refund_url if config.valid_url(config.refund_url) else None})

    @api.post("/account/checkout")
    @authenticate()
    def checkout():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"package_id", "terms_version", "catalog_version"}:
            return fail("invalid_payload", 422)
        if not rate_limit(f"checkout:{g.account_id}", limit=10):
            return fail("rate_limited", 429)
        result = account_payments.create_checkout(get_db(), g.account_id, package_id=data["package_id"],
            terms_version=data["terms_version"], request_key=request.headers.get("Idempotency-Key", ""),
            quote_version=data["catalog_version"],
            config=account_payments.Config.from_environment())
        return jsonify(result), 201

    @api.get("/account/purchases")
    @authenticate()
    def purchases():
        # Opaque purchase id cursor preserves ties in creation timestamps.
        conn = get_db()
        cursor = request.args.get("before")
        params = [g.account_id]
        clause = ""
        if cursor:
            previous = conn.execute("SELECT created_at,id FROM account_purchases WHERE id=? AND account_id=?", (cursor, g.account_id)).fetchone()
            if not previous:
                return fail("invalid_pagination", 422)
            clause = " AND (created_at,id)<(?,?)"
            params.extend(previous)
        rows = conn.execute("""SELECT id,package_id,units,price_cents,currency,terms_version,state,created_at,updated_at
            FROM account_purchases WHERE account_id=?""" + clause + " ORDER BY created_at DESC,id DESC LIMIT 30", params).fetchall()
        return jsonify({"purchases": [dict(row) for row in rows], "scale": account_ledger.SCALE,
                        "next_cursor": rows[-1]["id"] if len(rows) == 30 else None})

    @api.get("/account/purchases/<purchase_id>")
    @authenticate()
    def purchase_receipt(purchase_id):
        return jsonify(account_payments.receipt(get_db(), g.account_id, purchase_id))

    @api.post("/payments/stripe/webhook")
    def stripe_webhook():
        if request.content_length is not None and request.content_length > 1024 * 1024:
            return fail("payload_too_large", 413)
        payload = request.get_data()
        if len(payload) > 1024 * 1024:
            return fail("payload_too_large", 413)
        return jsonify(account_payments.webhook(get_db(), payload, request.headers.get("Stripe-Signature", ""),
            config=account_payments.Config.from_environment()))

    @api.post("/account/wallet-challenges")
    @authenticate(recent=True)
    def wallet_challenge():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("wallet"), str):
            return fail("invalid_payload", 422)
        # The allowlisted Origin is request context, never an arbitrary body field.
        config = account_auth.AuthConfig.from_environment()
        origin = request.headers.get("Origin", "")
        if origin not in config.authorized_parties:
            return fail("invalid_origin", 403)
        chain = data.get("chain_id")
        if chain not in {1, 11155111} or type(chain) is not int:
            return fail("unsupported_chain", 422)
        result = account_identity.issue_wallet_challenge(get_db(), g.account_principal,
            wallet=data["wallet"], origin=origin, chain_id=chain,
            purpose=data.get("purpose", "wallet_link"), link_id=data.get("link_id"))
        return jsonify(result), 201

    def finish_wallet(purpose, expected_link=None):
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or not isinstance(data.get("challenge_id"), str)
                or not isinstance(data.get("signature"), str) or len(data["signature"]) > 1024):
            return fail("invalid_payload", 422)
        if expected_link is not None:
            proof = get_db().execute("SELECT link_id FROM account_wallet_challenges WHERE id=? AND account_id=?",
                                     (data["challenge_id"], g.account_id)).fetchone()
            if not proof or proof[0] != expected_link:
                return fail("invalid_challenge", 422)
        link_id = account_identity.complete_wallet_challenge(get_db(), g.account_principal,
            challenge_id=data["challenge_id"], signature=data["signature"], purpose=purpose)
        return jsonify({"link_id": link_id, "status": "linked" if purpose == "wallet_link" else "unlinked"})

    @api.post("/account/wallet-links")
    @authenticate(recent=True)
    def link_wallet():
        return finish_wallet("wallet_link")

    @api.delete("/account/wallet-links/<link_id>")
    @authenticate(recent=True)
    def unlink_wallet(link_id):
        return finish_wallet("wallet_unlink", link_id)

    return api

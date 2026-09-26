"""Commercial account API. Does not share legacy wallet or owner authorization."""
from __future__ import annotations

import sqlite3
import os
import uuid
from functools import wraps
from typing import Callable

from flask import Blueprint, Response, g, jsonify, request

import account_auth
import account_identity
import account_ledger
import account_payments
import account_lifecycle
import account_marketplace
import account_import
import account_workflows
import stripe

ERROR_DETAILS = {
    "insufficient_credits": {
        "message": "Not enough credits for this purchase. Add credits and try again.",
        "action": "fund_credits",
        "retryable": False,
    },
    "listing_price_changed": {
        "message": "This listing's price changed. Review the listing before buying.",
        "action": "reload_listing",
        "retryable": False,
    },
    "cannot_buy_own_listing": {
        "message": "You already own this listing.",
        "action": "open_collection",
        "retryable": False,
    },
    "listing_not_found": {
        "message": "This listing is no longer available.",
        "action": "reload_marketplace",
        "retryable": False,
    },
    "marketplace_artifact_unavailable": {
        "message": "This creation is no longer available to buy.",
        "action": "reload_marketplace",
        "retryable": False,
    },
    "idempotency_conflict": {
        "message": "A saved marketplace request does not match this action. Review it before retrying.",
        "action": "review_pending_request",
        "retryable": False,
    },
}


def create_blueprint(get_db: Callable[[], sqlite3.Connection], rate_limit: Callable[..., bool], *, outputs_dir=None) -> Blueprint:
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
        detail = ERROR_DETAILS.get(code, {})
        error = {"code": code, "message": detail.get("message", code.replace("_", " ").capitalize())}
        if detail:
            error["action"] = detail["action"]
            error["retryable"] = detail["retryable"]
        return jsonify({"error": error,
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

    @api.errorhandler(account_marketplace.MarketplaceError)
    def marketplace_error(exc):
        return fail(str(exc), exc.status)

    @api.errorhandler(account_import.MigrationError)
    def import_error(exc):
        return fail(str(exc), exc.status)

    def imports_enabled():
        return os.environ.get("HAVNAI_ACCOUNT_IMPORT_ENABLED", "") == "1"

    @api.errorhandler(account_workflows.WorkflowError)
    def workflow_error(exc):
        return fail(str(exc), exc.status)

    @api.get("/workflows")
    @api.get("/workflows/<int:workflow_id>")
    def public_workflows(workflow_id=None):
        if not rate_limit(f"workflow-read:{request.remote_addr}", limit=120):
            return fail("rate_limited", 429)
        try:
            limit, offset = int(request.args.get("limit", "50")), int(request.args.get("offset", "0"))
        except ValueError:
            return fail("invalid_pagination", 422)
        return jsonify(account_workflows.public(get_db(), workflow_id, limit=limit, offset=offset,
            search=request.args.get("search", "").strip(), category=request.args.get("category", "").strip()))

    @api.route("/account/workflows", methods=["GET", "POST"])
    @authenticate()
    def account_workflow_collection():
        if request.method == "POST":
            return jsonify(account_workflows.create(get_db(), g.account_principal,
                request.headers.get("Idempotency-Key", ""), request.get_json(silent=True))), 201
        try:
            limit, offset = int(request.args.get("limit", "50")), int(request.args.get("offset", "0"))
        except ValueError:
            return fail("invalid_pagination", 422)
        return jsonify(account_workflows.listing(get_db(), g.account_id, limit, offset))

    @api.route("/account/workflows/<int:workflow_id>", methods=["GET", "PATCH", "DELETE"])
    @authenticate()
    def account_workflow_item(workflow_id):
        if request.method == "PATCH":
            return jsonify(account_workflows.update(get_db(), g.account_principal, workflow_id, request.get_json(silent=True)))
        if request.method == "DELETE":
            account_workflows.delete(get_db(), g.account_principal, workflow_id)
            return Response(status=204)
        return jsonify(account_workflows.owned(get_db(), g.account_id, workflow_id))

    @api.get("/account/import-capabilities")
    @authenticate()
    def wallet_import_capabilities():
        return jsonify({"execution_enabled": imports_enabled(),
                        "scopes": ["generation_history", "available_credits", "music_publications", "music_playlists", "workflows", "music_likes", "music_saves"],
                        "legacy_stripe_balance_import": False})

    @api.post("/account/wallet-links/<link_id>/import-snapshots")
    @authenticate(recent=True)
    def wallet_import_prepare(link_id):
        return jsonify(account_import.prepare(get_db(), g.account_principal, link_id,
            request.headers.get("Idempotency-Key", ""), request.get_json(silent=True))), 201

    @api.get("/account/import-snapshots/<snapshot_id>")
    @authenticate()
    def wallet_import_snapshot(snapshot_id):
        return jsonify(account_import.load(get_db(), g.account_principal, snapshot_id))

    @api.get("/account/import-receipts/<snapshot_id>")
    @authenticate()
    def wallet_import_receipt(snapshot_id):
        return jsonify(account_import.receipt(get_db(), g.account_principal, snapshot_id))

    @api.get("/account/import-receipts")
    @authenticate()
    def wallet_import_receipts():
        try:
            limit, offset = int(request.args.get("limit", "50")), int(request.args.get("offset", "0"))
        except ValueError:
            return fail("invalid_pagination", 422)
        return jsonify(account_import.receipts(get_db(), g.account_principal, limit=limit, offset=offset))

    @api.post("/account/import-snapshots/<snapshot_id>/challenge")
    @authenticate(recent=True)
    def wallet_import_challenge(snapshot_id):
        if not imports_enabled():
            return fail("import_execution_unavailable", 503)
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"chain_id"}:
            return fail("invalid_payload", 422)
        origin = request.headers.get("Origin", "")
        if origin not in account_auth.AuthConfig.from_environment().authorized_parties:
            return fail("invalid_origin", 403)
        return jsonify(account_import.issue_challenge(get_db(), g.account_principal, snapshot_id,
            origin=origin, chain_id=data["chain_id"])), 201

    @api.post("/account/import-snapshots/<snapshot_id>/execute")
    @authenticate(recent=True)
    def wallet_import_execute(snapshot_id):
        if not imports_enabled():
            return fail("import_execution_unavailable", 503)
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"challenge_id", "signature", "chain_id"}:
            return fail("invalid_payload", 422)
        origin = request.headers.get("Origin", "")
        if origin not in account_auth.AuthConfig.from_environment().authorized_parties:
            return fail("invalid_origin", 403)
        if type(data["chain_id"]) is not int or data["chain_id"] not in {1, 11155111}:
            return fail("unsupported_chain", 422)
        try:
            result = account_import.execute(get_db(), g.account_principal, snapshot_id,
                challenge_id=data["challenge_id"], signature=data["signature"], origin=origin, chain_id=data["chain_id"])
        except account_ledger.LedgerError as exc:
            return fail(str(exc), 409)
        return jsonify({"receipt": result, "scale": account_ledger.SCALE})

    @api.get("/account/wallet-links/<link_id>/import-preview")
    @authenticate()
    def wallet_import_preview(link_id):
        try:
            limit, offset = int(request.args.get("limit", "50")), int(request.args.get("offset", "0"))
            return jsonify(account_import.preview(get_db(), g.account_id, link_id, limit=limit, offset=offset))
        except account_import.MigrationError as exc:
            return fail(str(exc), exc.status)
        except ValueError:
            return fail("invalid_pagination", 422)

    @api.post("/marketplace/listings")
    @authenticate()
    def marketplace_list():
        result = account_marketplace.create(get_db(), g.account_id,
            request.headers.get("Idempotency-Key", ""), request.get_json(silent=True))
        return jsonify(result), 201

    @api.get("/marketplace/listings")
    def marketplace_browse():
        if not rate_limit(f"marketplace-read:{request.remote_addr}", limit=120):
            return fail("rate_limited", 429)
        try:
            limit, offset = int(request.args.get("limit", "24")), int(request.args.get("offset", "0"))
        except ValueError:
            return fail("invalid_pagination", 422)
        return jsonify(account_marketplace.browse(get_db(), search=request.args.get("search", ""),
            category=request.args.get("category", ""), sort=request.args.get("sort", "newest"), limit=limit, offset=offset))

    @api.get("/marketplace/listings/<int:listing_id>")
    def marketplace_detail(listing_id):
        if not rate_limit(f"marketplace-read:{request.remote_addr}", limit=120):
            return fail("rate_limited", 429)
        return jsonify(account_marketplace.detail(get_db(), listing_id))

    @api.get("/marketplace/listings/<int:listing_id>/preview")
    def marketplace_preview(listing_id):
        if not rate_limit(f"marketplace-preview:{request.remote_addr}", limit=60):
            return fail("rate_limited", 429)
        if outputs_dir is None:
            return fail("preview_unavailable", 503)
        response = Response(account_marketplace.preview(get_db(), listing_id, outputs_dir=outputs_dir()), mimetype="image/jpeg")
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @api.post("/marketplace/listings/<int:listing_id>/purchase")
    @authenticate()
    def marketplace_purchase(listing_id):
        try:
            return jsonify(account_marketplace.purchase(get_db(), g.account_id, listing_id,
                request.headers.get("Idempotency-Key", ""), request.get_json(silent=True)))
        except account_ledger.LedgerError as exc:
            return fail(str(exc), 409)

    @api.delete("/marketplace/listings/<int:listing_id>")
    @authenticate()
    def marketplace_delist(listing_id):
        account_marketplace.delist(get_db(), g.account_id, listing_id)
        return "", 204

    @api.get("/account/marketplace/listings")
    @api.get("/account/marketplace/receipts")
    @authenticate()
    def account_marketplace_history():
        try:
            limit, offset = int(request.args.get("limit", "24")), int(request.args.get("offset", "0"))
        except ValueError:
            return fail("invalid_pagination", 422)
        read = account_marketplace.receipts if request.path.endswith("/receipts") else account_marketplace.account_listings
        return jsonify(read(get_db(), g.account_id, limit=limit, offset=offset))

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

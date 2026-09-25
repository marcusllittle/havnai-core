"""Account payment contracts using signed webhooks and a controllable provider.

No Stripe API calls or charges are made by this suite. SQLite transactions and
webhook HMAC verification are real; provider retrievals are deliberately mocked.
"""
import copy
import hashlib
import hmac
import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_identity as identity
import account_ledger as ledger
import account_payments as payments
import account_routes
from tests.test_account_auth import keys, token, config as auth_config


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / "payments.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    identity.initialize(conn)
    ledger.initialize(conn)
    payments.initialize(conn)
    alice = identity.ensure_account(conn, identity.VerifiedPrincipal("issuer", "alice", "sess"))
    bob = identity.ensure_account(conn, identity.VerifiedPrincipal("issuer", "bob", "sess"))
    cfg = payments.Config("sk_test_fake", "whsec_test", "https://joinhavn.io", "credits-v1", True)
    create = Mock(return_value={"id": "cs_1", "url": "https://checkout.stripe.com/c/pay/cs_1", "livemode": False})
    monkeypatch.setattr(payments.stripe.checkout.Session, "create", create)
    intent = {"id": "pi_1", "object": "payment_intent", "metadata": {}, "livemode": False,
              "currency": "usd", "amount": 500, "amount_received": 500, "status": "succeeded",
              "latest_charge": {"id": "ch_1", "amount": 500, "amount_refunded": 0,
                                "currency": "usd", "paid": True, "payment_intent": "pi_1"}}
    retrieve = Mock(side_effect=lambda *a, **kw: payments.stripe.StripeObject.construct_from(copy.deepcopy(intent), "sk_test_fake"))
    monkeypatch.setattr(payments.stripe.PaymentIntent, "retrieve", retrieve)
    disputes = []
    listing = Mock()
    listing.auto_paging_iter.side_effect = lambda: iter(copy.deepcopy(disputes))
    monkeypatch.setattr(payments.stripe.Dispute, "list", Mock(return_value=listing))
    monkeypatch.setattr(payments.Config, "from_environment", lambda: cfg)
    yield {"conn": conn, "path": path, "alice": alice, "bob": bob, "cfg": cfg, "create": create,
           "intent": intent, "retrieve": retrieve, "disputes": disputes}
    conn.close()


def checkout(env, **overrides):
    args = {"package_id": "starter", "request_key": "request-key-alice-1", "terms_version": "credits-v1", "config": env["cfg"]}
    args.update(overrides)
    result = payments.create_checkout(env["conn"], env["alice"], **args)
    env["intent"]["metadata"] = {"havnai_purchase": result["purchase_id"], "surface": "account_v2"}
    return result["purchase_id"]


def reconcile(env, purchase):
    return payments.reconcile(env["conn"], purchase, "pi_1", config=env["cfg"])


def send_event(env, event_id="evt_1", event_type="payment_intent.succeeded", obj=None, signature=None):
    payload = json.dumps({"id": event_id, "object": "event", "livemode": False,
                          "type": event_type, "data": {"object": obj or env["intent"]}}).encode()
    stamp = str(int(time.time()))
    digest = hmac.new(b"whsec_test", stamp.encode() + b"." + payload, hashlib.sha256).hexdigest()
    return payments.webhook(env["conn"], payload, signature or f"t={stamp},v1={digest}", config=env["cfg"])


def test_checkout_is_persisted_before_provider_and_retry_is_stable(env):
    def provider(**kwargs):
        assert env["conn"].execute("SELECT COUNT(*) FROM account_purchases").fetchone()[0] == 1
        assert not env["conn"].in_transaction
        raise TimeoutError("response lost")
    env["create"].side_effect = provider
    with pytest.raises(TimeoutError):
        checkout(env)
    original_params = env["create"].call_args
    env["create"].side_effect = None
    purchase = checkout(env)
    assert env["create"].call_args == original_params
    assert checkout(env) == purchase
    assert env["create"].call_count == 2
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0
    with pytest.raises(payments.PaymentError, match="idempotency_conflict"):
        checkout(env, package_id="pro")


def test_old_ambiguous_checkout_never_creates_another_session(env):
    env["create"].side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        checkout(env)
    with env["conn"]:
        env["conn"].execute("UPDATE account_purchases SET created_at=?", (time.time() - 24 * 3600,))
    with pytest.raises(payments.PaymentError, match="requires_reconciliation"):
        checkout(env)
    assert env["create"].call_count == 1


def test_signed_duplicate_and_different_events_fund_once(env):
    purchase = checkout(env)
    send_event(env)
    send_event(env)
    send_event(env, "evt_2")
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 50_000
    assert env["conn"].execute("SELECT COUNT(*) FROM account_payment_receipts").fetchone()[0] == 1
    assert env["conn"].execute("SELECT COUNT(*) FROM account_credit_ledger").fetchone()[0] == 1
    assert payments.receipt(env["conn"], env["alice"], purchase)["receipt"]["price_cents"] == 500
    with pytest.raises(payments.PaymentError, match="purchase_not_found"):
        payments.receipt(env["conn"], env["bob"], purchase)


@pytest.mark.parametrize("status", ["requires_payment_method", "processing", "canceled"])
def test_unpaid_failed_cancelled_do_not_fund(env, status):
    purchase = checkout(env)
    env["intent"]["status"] = status
    # A stale success event cannot fund an unpaid intent.
    send_event(env)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0
    assert payments.receipt(env["conn"], env["alice"], purchase)["receipt"] is None


@pytest.mark.parametrize("field,value", [("amount", 1), ("amount_received", 1), ("currency", "eur"), ("livemode", True), ("metadata", {})])
def test_wrong_payment_binding_never_funds(env, field, value):
    purchase = checkout(env)
    env["intent"][field] = value
    with pytest.raises(payments.PaymentError):
        reconcile(env, purchase)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0


def test_invalid_signature_has_no_provider_calls_or_funding(env):
    checkout(env)
    with pytest.raises(payments.PaymentError, match="invalid_webhook_signature"):
        send_event(env, signature="t=1,v1=bad")
    env["retrieve"].assert_not_called()


def test_partial_refund_then_dispute_then_won_restores_only_unrefunded_credits(env):
    purchase = checkout(env)
    reconcile(env, purchase)
    env["conn"].execute("BEGIN IMMEDIATE")
    with env["conn"]:
        ledger.reserve_in_transaction(env["conn"], env["alice"], 40_000, job_id="job1")
        ledger.finish_in_transaction(env["conn"], env["alice"], job_id="job1", succeeded=True)
    env["intent"]["latest_charge"]["amount_refunded"] = 100
    assert reconcile(env, purchase) == "partially_refunded"
    assert ledger.balance(env["conn"], env["alice"])["settled_units"] == 0
    env["disputes"].append({"id": "du_1", "amount": 500, "currency": "usd", "status": "needs_response"})
    assert reconcile(env, purchase) == "disputed"
    assert ledger.balance(env["conn"], env["alice"])["debt_units"] == 40_000
    env["disputes"][0]["status"] = "won"
    reconcile(env, purchase)
    reconcile(env, purchase)
    assert ledger.balance(env["conn"], env["alice"])["settled_units"] == 0
    receipt = payments.receipt(env["conn"], env["alice"], purchase)
    assert [a["settled_delta"] for a in receipt["adjustments"]] == [0, -10_000, -40_000, 40_000]
    # A late old event reads the current provider state, rather than reapplying loss.
    send_event(env, obj={"id": "du_1", "object": "dispute", "payment_intent": "pi_1", "status": "lost"}, event_type="charge.dispute.closed")
    assert ledger.balance(env["conn"], env["alice"])["settled_units"] == 0


def test_full_refund_before_success_event_nets_zero_atomically(env):
    purchase = checkout(env)
    env["intent"]["latest_charge"]["amount_refunded"] = 500
    send_event(env, event_type="charge.refunded", obj={"object": "charge", "payment_intent": "pi_1"})
    assert ledger.balance(env["conn"], env["alice"])["settled_units"] == 0
    assert payments.receipt(env["conn"], env["alice"], purchase)["state"] == "refunded"


def test_older_snapshot_cannot_overwrite_newer_reconciliation(env):
    purchase = checkout(env)
    original = copy.deepcopy(env["intent"])
    def interleaved(*args, **kwargs):
        env["retrieve"].side_effect = lambda *a, **k: copy.deepcopy(env["intent"])
        env["intent"]["latest_charge"]["amount_refunded"] = 500
        other = sqlite3.connect(env["path"])
        try:
            payments.reconcile(other, purchase, "pi_1", config=env["cfg"])
        finally:
            other.close()
        return original
    env["retrieve"].side_effect = interleaved
    with pytest.raises(payments.PaymentError, match="reconciliation_retry"):
        reconcile(env, purchase)
    assert ledger.balance(env["conn"], env["alice"])["settled_units"] == 0


def test_receipt_failure_rolls_back_funding_and_event_can_retry(env, monkeypatch):
    purchase = checkout(env)
    env["conn"].execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON account_payment_receipts BEGIN SELECT RAISE(ABORT,'disk test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        send_event(env)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0
    assert not env["conn"].execute("SELECT 1 FROM account_payment_events").fetchone()
    env["conn"].execute("DROP TRIGGER fail_receipt")
    send_event(env)
    assert payments.receipt(env["conn"], env["alice"], purchase)["state"] == "paid"


def test_receipts_and_adjustments_are_immutable(env):
    purchase = checkout(env)
    reconcile(env, purchase)
    for table in ("account_payment_receipts", "account_payment_adjustments"):
        for sql in (f"DELETE FROM {table}", f"UPDATE {table} SET created_at=0"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"), env["conn"]:
                env["conn"].execute(sql)


def test_lost_event_ack_retries_without_duplicate_funding(env):
    checkout(env)
    env["conn"].execute("CREATE TRIGGER fail_event BEFORE INSERT ON account_payment_events BEGIN SELECT RAISE(ABORT,'lost ack'); END")
    with pytest.raises(sqlite3.IntegrityError):
        send_event(env)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 50_000
    env["conn"].execute("DROP TRIGGER fail_event")
    send_event(env)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 50_000
    assert env["conn"].execute("SELECT COUNT(*) FROM account_credit_ledger").fetchone()[0] == 1


def test_test_and_live_purchases_cannot_share_balances(env):
    checkout(env)
    live = payments.Config("sk_live_fake", "whsec_test", "https://joinhavn.io", "credits-v1", True)
    with pytest.raises(payments.PaymentError, match="database_mode_mismatch"):
        checkout(env, request_key="another-request-key", config=live)


def test_recovery_checks_provider_binding_before_settlement(env, monkeypatch):
    purchase = checkout(env)
    session = {"id": "cs_1", "client_reference_id": purchase, "metadata": env["intent"]["metadata"],
               "amount_total": 500, "currency": "usd", "livemode": False, "mode": "payment", "payment_intent": "pi_1"}
    monkeypatch.setattr(payments.stripe.checkout.Session, "retrieve", Mock(return_value=session))
    with env["conn"]:
        env["conn"].execute("UPDATE account_purchases SET session_id=NULL,checkout_url=NULL")
    assert payments.recover_purchase(env["conn"], purchase, config=env["cfg"], session_id="cs_1") == "paid"
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 50_000
    session["client_reference_id"] = "another-purchase"
    with pytest.raises(payments.PaymentError, match="binding_mismatch"):
        payments.recover_purchase(env["conn"], purchase, config=env["cfg"], session_id="cs_1")


def test_expired_checkout_webhook_does_not_fund(env, monkeypatch):
    purchase = checkout(env)
    session = {"id": "cs_1", "object": "checkout.session", "client_reference_id": purchase,
               "metadata": env["intent"]["metadata"], "amount_total": 500, "currency": "usd", "livemode": False,
               "mode": "payment", "payment_intent": None, "status": "expired"}
    monkeypatch.setattr(payments.stripe.checkout.Session, "retrieve", Mock(return_value=session))
    send_event(env, event_type="checkout.session.expired", obj=session)
    assert payments.receipt(env["conn"], env["alice"], purchase)["state"] == "expired"
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0


def test_http_checkout_uses_verified_account_and_receipts_are_private(env, keys, monkeypatch):
    monkeypatch.setattr(account_routes.account_auth.AuthConfig, "from_environment", lambda: auth_config(keys))
    app = Flask(__name__)
    app.register_blueprint(account_routes.create_blueprint(lambda: env["conn"], lambda *a, **kw: True))
    client = app.test_client()
    body = {"package_id": "starter", "terms_version": "credits-v1"}
    assert client.post("/v2/account/checkout", json=body).status_code == 401
    headers = {"Authorization": token(keys), "Idempotency-Key": "http-checkout-key-1"}
    assert client.post("/v2/account/checkout", json={**body, "account_id": env["bob"]}, headers=headers).status_code == 422
    response = client.post("/v2/account/checkout", json=body, headers=headers)
    assert response.status_code == 201
    purchase = response.json["purchase_id"]
    assert client.get(f"/v2/account/purchases/{purchase}", headers=headers).status_code == 200
    bob = {"Authorization": token(keys, sub="other-user")}
    assert client.get(f"/v2/account/purchases/{purchase}", headers=bob).status_code == 404
    assert client.get("/v2/account/purchases", headers=bob).json["purchases"] == []
    assert client.get("/v2/account/purchases", headers=headers).headers["Cache-Control"] == "private, no-store"
    assert client.get("/v2/account/ledger?before=" + "9" * 100, headers=headers).status_code == 422

from unittest.mock import Mock
import pytest
from tests.test_account_payments import env, checkout, payments, ledger


def test_batch_recovers_paid_refund_and_never_creates_checkout(env, monkeypatch):
    purchase = checkout(env)
    session = {"id": "cs_1", "client_reference_id": purchase, "metadata": env["intent"]["metadata"],
               "amount_total": 500, "currency": "usd", "livemode": False, "mode": "payment", "payment_intent": "pi_1"}
    monkeypatch.setattr(payments.stripe.checkout.Session, "retrieve", Mock(return_value=session))
    env["create"].reset_mock()
    first = payments.reconcile_batch(env["conn"], config=env["cfg"])
    assert first == [{"purchase_id": purchase, "outcome": "reconciled", "state": "paid"}]
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 50000
    env["intent"]["latest_charge"]["amount_refunded"] = 500
    assert payments.reconcile_batch(env["conn"], config=env["cfg"])[0]["state"] == "refunded"
    assert payments.reconcile_batch(env["conn"], config=env["cfg"])[0]["state"] == "refunded"
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0
    assert env["conn"].execute("SELECT COUNT(*) FROM account_payment_receipts").fetchone()[0] == 1
    assert env["conn"].execute("SELECT COUNT(*) FROM account_payment_reconcile_checks").fetchone()[0] == 1
    env["create"].assert_not_called()


def test_failed_purchase_does_not_starve_later_batches(env, monkeypatch):
    first = checkout(env)
    env["create"].return_value = {"id": "cs_2", "url": "https://checkout.stripe.com/c/pay/cs_2", "livemode": False}
    second = checkout(env, request_key="second-request-key")
    recover = Mock(side_effect=payments.stripe.APIConnectionError("secret provider diagnostic"))
    monkeypatch.setattr(payments, "recover_purchase", recover)
    one = payments.reconcile_batch(env["conn"], config=env["cfg"], limit=1)
    two = payments.reconcile_batch(env["conn"], config=env["cfg"], limit=1)
    assert {one[0]["purchase_id"], two[0]["purchase_id"]} == {first, second}
    assert one[0]["outcome"] == two[0]["outcome"] == "provider_unavailable"
    assert "secret" not in str(one + two)
    assert ledger.balance(env["conn"], env["alice"])["available_units"] == 0


def test_missing_session_is_reported_without_creating_one(env):
    checkout(env)
    with env["conn"]:
        env["conn"].execute("UPDATE account_purchases SET session_id=NULL")
    env["create"].reset_mock()
    assert payments.reconcile_batch(env["conn"], config=env["cfg"])[0]["outcome"] == "needs_session"
    env["create"].assert_not_called()


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_batch_limit_is_bounded(env, limit):
    with pytest.raises(payments.PaymentError, match="invalid_reconciliation_limit"):
        payments.reconcile_batch(env["conn"], config=env["cfg"], limit=limit)

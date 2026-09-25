import concurrent.futures
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_identity as identity
import account_ledger as ledger


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "ledger.db"
    conn = sqlite3.connect(path)
    identity.initialize(conn)
    ledger.initialize(conn)
    account = identity.ensure_account(conn, identity.VerifiedPrincipal("issuer", "user", "session"))
    yield conn, path, account
    conn.close()


def transact(conn, function, *args, **kwargs):
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        return function(conn, *args, **kwargs)


def fund(conn, account, units=1000, payment="pay-1"):
    return transact(conn, ledger.fund_in_transaction, account, units, payment_id=payment)


def test_duplicate_funding_is_one_ledger_entry(database):
    conn, _, account = database
    first = fund(conn, account)
    second = fund(conn, account)
    assert first["entry_id"] == second["entry_id"]
    assert second["replayed"]
    assert ledger.balance(conn, account)["available_units"] == 1000
    assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger").fetchone()[0] == 1
    with pytest.raises(ledger.LedgerError, match="idempotency_conflict"):
        fund(conn, account, 2000)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), float("inf"), "1000", 10**16])
def test_only_bounded_integer_units_allowed(database, value):
    with pytest.raises(ledger.LedgerError, match="invalid_credit_units"):
        fund(database[0], database[2], value)


@pytest.mark.parametrize("success", [True, False])
def test_job_is_reserved_and_finalized_exactly_once(database, success):
    conn, _, account = database
    fund(conn, account)
    for _ in range(2):
        transact(conn, ledger.reserve_in_transaction, account, 600, job_id="job-1")
    assert ledger.balance(conn, account)["available_units"] == 400
    for _ in range(2):
        transact(conn, ledger.finish_in_transaction, account, job_id="job-1", succeeded=success)
    balance = ledger.balance(conn, account)
    assert balance["reserved_units"] == 0
    assert balance["available_units"] == (400 if success else 1000)
    with pytest.raises(ledger.LedgerError, match="already_finalized"):
        transact(conn, ledger.finish_in_transaction, account, job_id="job-1", succeeded=not success)
    assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger").fetchone()[0] == 3


def test_real_concurrent_spends_never_oversubscribe(database):
    conn, path, account = database
    fund(conn, account, 1000)
    barrier = threading.Barrier(8)

    def spend(index):
        db = sqlite3.connect(path, timeout=10)
        try:
            barrier.wait(timeout=10)
            return transact(db, ledger.reserve_in_transaction, account, 300, job_id=f"job-{index}")
        except ledger.LedgerError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(spend, range(8)))
    assert sum(isinstance(result, dict) for result in results) == 3
    assert results.count("insufficient_credits") == 5
    assert ledger.balance(conn, account)["available_units"] == 100
    assert ledger.balance(conn, account)["reserved_units"] == 900


def test_refund_after_spending_records_debt_and_retry_is_safe(database):
    conn, _, account = database
    fund(conn, account)
    transact(conn, ledger.reserve_in_transaction, account, 800, job_id="job")
    transact(conn, ledger.finish_in_transaction, account, job_id="job", succeeded=True)
    for _ in range(2):
        transact(conn, ledger.reverse_funding_in_transaction, account, 1000,
                 payment_id="pay-1", adjustment_id="refund-1", reason="refund")
    assert ledger.balance(conn, account)["debt_units"] == 800
    assert ledger.balance(conn, account)["available_units"] == 0
    with pytest.raises(ledger.LedgerError, match="insufficient_credits"):
        transact(conn, ledger.reserve_in_transaction, account, 1, job_id="more")
    fund(conn, account, 1000, "pay-2")
    assert ledger.balance(conn, account)["available_units"] == 200
    assert ledger.balance(conn, account)["debt_units"] == 0


def test_release_cannot_recreate_refunded_credit(database):
    conn, _, account = database
    fund(conn, account)
    transact(conn, ledger.reserve_in_transaction, account, 1000, job_id="job")
    transact(conn, ledger.reverse_funding_in_transaction, account, 1000,
             payment_id="pay-1", adjustment_id="refund-1", reason="refund")
    transact(conn, ledger.finish_in_transaction, account, job_id="job", succeeded=False)
    assert ledger.balance(conn, account)["available_units"] == 0


def test_partial_refund_cannot_exceed_original_funding(database):
    conn, _, account = database
    fund(conn, account)
    transact(conn, ledger.reverse_funding_in_transaction, account, 700,
             payment_id="pay-1", adjustment_id="refund-1", reason="refund")
    with pytest.raises(ledger.LedgerError, match="exceeds_purchase"):
        transact(conn, ledger.reverse_funding_in_transaction, account, 400,
                 payment_id="pay-1", adjustment_id="dispute-1", reason="chargeback")
    assert ledger.balance(conn, account)["available_units"] == 300


def test_business_record_and_ledger_rollback_together(database):
    conn, _, account = database
    conn.execute("CREATE TABLE test_jobs(id TEXT PRIMARY KEY)")
    fund(conn, account)
    with pytest.raises(RuntimeError):
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            conn.execute("INSERT INTO test_jobs VALUES ('job')")
            ledger.reserve_in_transaction(conn, account, 1000, job_id="job")
            raise RuntimeError("simulated durable job failure")
    assert conn.execute("SELECT COUNT(*) FROM test_jobs").fetchone()[0] == 0
    assert ledger.balance(conn, account)["available_units"] == 1000
    assert conn.execute("SELECT COUNT(*) FROM account_credit_reservations").fetchone()[0] == 0


def test_mutations_require_transaction_and_ledger_is_immutable(database):
    conn, _, account = database
    with pytest.raises(RuntimeError):
        ledger.fund_in_transaction(conn, account, 1000, payment_id="pay")
    fund(conn, account)
    for sql in ["DELETE FROM account_credit_ledger", "UPDATE account_credit_ledger SET settled_delta=999"]:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with conn:
                conn.execute(sql)


def seller_account(conn):
    return identity.ensure_account(conn, identity.VerifiedPrincipal("issuer", "seller", "session"))


def test_sale_receipt_binds_both_accounts_and_price(database):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    first = transact(conn, ledger.settle_sale_in_transaction, buyer, seller, 700, sale_id="sale-1")
    retry = transact(conn, ledger.settle_sale_in_transaction, buyer, seller, 700, sale_id="sale-1")
    assert retry == {**first, "replayed": True}
    assert ledger.balance(conn, buyer)["available_units"] == 300
    assert ledger.balance(conn, seller)["available_units"] == 700
    for args in [(buyer, seller, 701), (seller, buyer, 700)]:
        with pytest.raises(ledger.LedgerError, match="idempotency_conflict"):
            transact(conn, ledger.settle_sale_in_transaction, *args, sale_id="sale-1")
    assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == 1
    entries = conn.execute("SELECT settled_delta FROM account_credit_ledger WHERE resource_id='sale-1'").fetchall()
    assert sorted(row[0] for row in entries) == [-700, 700]
    for sql in ["DELETE FROM account_credit_sales", "UPDATE account_credit_sales SET units=1"]:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with conn:
                conn.execute(sql)


def test_sale_cannot_spend_reserved_generation_credits(database):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    transact(conn, ledger.reserve_in_transaction, buyer, 600, job_id="running")
    with pytest.raises(ledger.LedgerError, match="insufficient_credits"):
        transact(conn, ledger.settle_sale_in_transaction, buyer, seller, 500, sale_id="sale")
    assert ledger.balance(conn, buyer)["available_units"] == 400
    assert ledger.balance(conn, seller)["settled_units"] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == 0


def test_failed_seller_credit_undoes_debit_even_when_caller_catches(database):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    fund(conn, seller, ledger.MAX_UNITS, "seller-funding")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        with pytest.raises(ledger.LedgerError, match="ledger_limit_exceeded"):
            ledger.settle_sale_in_transaction(conn, buyer, seller, 700, sale_id="sale")
        assert conn.in_transaction
    assert ledger.balance(conn, buyer)["settled_units"] == 1000
    assert ledger.balance(conn, seller)["settled_units"] == ledger.MAX_UNITS
    assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger WHERE resource_id='sale'").fetchone()[0] == 0


def test_sale_rolls_back_with_ownership_failure(database):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    with pytest.raises(RuntimeError, match="ownership"):
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            ledger.settle_sale_in_transaction(conn, buyer, seller, 700, sale_id="sale")
            raise RuntimeError("ownership write failed")
    assert ledger.balance(conn, buyer)["settled_units"] == 1000
    assert ledger.balance(conn, seller)["settled_units"] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["same_sale", "different_sales"])
def test_concurrent_sales_are_idempotent_and_cannot_overspend(database, mode):
    conn, path, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    barrier = threading.Barrier(6)

    def buy(index):
        db = sqlite3.connect(path, timeout=10)
        try:
            barrier.wait(timeout=10)
            return transact(db, ledger.settle_sale_in_transaction, buyer, seller, 300,
                            sale_id="shared" if mode == "same_sale" else f"sale-{index}")
        except ledger.LedgerError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(buy, range(6)))
    sales = 1 if mode == "same_sale" else 3
    assert ledger.balance(conn, buyer)["available_units"] == 1000 - sales * 300
    assert ledger.balance(conn, seller)["available_units"] == sales * 300
    assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == sales
    if mode == "same_sale":
        assert sum(result["replayed"] for result in results) == 5
    else:
        assert results.count("insufficient_credits") == 3


def test_sale_rejects_self_unknown_and_suspended_accounts(database):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    with pytest.raises(RuntimeError, match="transaction"):
        ledger.settle_sale_in_transaction(conn, buyer, seller, 10, sale_id="sale")
    with pytest.raises(ledger.LedgerError, match="cannot_buy_own"):
        transact(conn, ledger.settle_sale_in_transaction, buyer, buyer, 10, sale_id="self")
    with conn:
        conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (seller,))
    for target in (seller, "missing"):
        with pytest.raises(ledger.LedgerError, match="sale_account_unavailable"):
            transact(conn, ledger.settle_sale_in_transaction, buyer, target, 10, sale_id="sale")
    assert ledger.balance(conn, buyer)["settled_units"] == 1000


@pytest.mark.parametrize("units", [0, -1, True, 1.5, float("nan"), "1000", 10**16])
def test_sale_rejects_invalid_prices_without_moving_credits(database, units):
    conn, _, buyer = database
    seller = seller_account(conn)
    fund(conn, buyer)
    with pytest.raises(ledger.LedgerError, match="invalid_credit_units"):
        transact(conn, ledger.settle_sale_in_transaction, buyer, seller, units, sale_id="invalid")
    assert ledger.balance(conn, buyer)["settled_units"] == 1000
    assert ledger.balance(conn, seller)["settled_units"] == 0

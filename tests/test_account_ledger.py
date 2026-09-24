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

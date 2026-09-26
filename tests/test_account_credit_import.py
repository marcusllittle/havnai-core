import concurrent.futures
import sqlite3
import threading

import pytest

from tests.test_account_ledger import database, transact, fund
import account_identity as identity
import account_ledger as ledger

WALLET = "0x" + "a" * 40


@pytest.fixture
def legacy(database):
    conn, path, account = database
    conn.executescript("""CREATE TABLE credits (
        wallet TEXT PRIMARY KEY, balance REAL NOT NULL CHECK(balance>=0),
        total_deposited REAL NOT NULL, total_spent REAL NOT NULL, updated_at REAL NOT NULL);
    """)
    with conn:
        conn.execute("INSERT INTO credits VALUES (?,2.125,5,2.875,1)", (WALLET,))
        conn.execute("INSERT INTO wallet_links(id,account_id,wallet,verified_at,linked_at) VALUES ('link',?,?,1,1)", (account, WALLET))
    return conn, path, account


def move(conn, account, units=2125, migration="migration-one", wallet=WALLET):
    return transact(conn, ledger.import_legacy_in_transaction, account, wallet, units, migration_id=migration)


def test_import_has_paired_immutable_receipt_and_exactly_once_credit(legacy):
    conn, _, account = legacy
    receipt = move(conn, account)
    assert not receipt["replayed"]
    assert ledger.balance(conn, account)["settled_units"] == 2125
    assert conn.execute("SELECT balance,total_deposited,total_spent FROM credits").fetchone() == (0, 5, 2.875)
    paired = conn.execute("""SELECT i.legacy_delta_units,l.settled_delta,l.operation
        FROM account_credit_imports i JOIN account_credit_ledger l ON l.id=i.credit_entry_id""").fetchone()
    assert paired == (-2125, 2125, "legacy_import")
    # Later legacy funds must not be swept by recovery of an old successful import.
    with conn:
        conn.execute("UPDATE credits SET balance=7")
    retry = move(conn, account)
    assert retry == {**receipt, "replayed": True}
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 7
    assert ledger.balance(conn, account)["settled_units"] == 2125
    for query in ("UPDATE account_credit_imports SET units=1", "DELETE FROM account_credit_imports"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"), conn:
            conn.execute(query)
    for kwargs in ({"units": 1}, {"wallet": "0x" + "b" * 40}):
        with pytest.raises(ledger.LedgerError, match="idempotency_conflict"):
            move(conn, account, **kwargs)


@pytest.mark.parametrize("balance", [0, 2, 3, 2.12501, float("inf")])
def test_import_refuses_any_changed_or_unrepresentable_balance(legacy, balance):
    conn, _, account = legacy
    with conn:
        conn.execute("UPDATE credits SET balance=?", (balance,))
    with pytest.raises(ledger.LedgerError, match="import_balance_changed"):
        move(conn, account)
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == balance
    assert ledger.balance(conn, account)["settled_units"] == 0


def test_import_refuses_ambiguous_wallet_case_and_unlinked_account(legacy):
    conn, _, account = legacy
    with conn:
        conn.execute("INSERT INTO credits VALUES (?,2.125,5,2.875,1)", (WALLET.upper(),))
    with pytest.raises(ledger.LedgerError, match="import_balance_changed"):
        move(conn, account)
    with conn:
        conn.execute("DELETE FROM credits WHERE wallet=?", (WALLET.upper(),))
        conn.execute("UPDATE wallet_links SET unlinked_at=2")
    with pytest.raises(ledger.LedgerError, match="import_wallet_unavailable"):
        move(conn, account)


def test_import_cannot_credit_other_or_suspended_account(legacy):
    conn, _, account = legacy
    other = identity.ensure_account(conn, identity.VerifiedPrincipal("issuer", "other", "session"))
    with pytest.raises(ledger.LedgerError, match="import_wallet_unavailable"):
        move(conn, other)
    with conn:
        conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (account,))
    with pytest.raises(ledger.LedgerError, match="import_wallet_unavailable"):
        move(conn, account)


def test_failure_rolls_back_debit_even_if_caller_catches_error(legacy):
    conn, _, account = legacy
    fund(conn, account, ledger.MAX_UNITS)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        with pytest.raises(ledger.LedgerError, match="ledger_limit_exceeded"):
            ledger.import_legacy_in_transaction(conn, account, WALLET, 2125, migration_id="migration")
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 2.125
    assert ledger.balance(conn, account)["settled_units"] == ledger.MAX_UNITS
    assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 0


def test_receipt_failure_rolls_back_both_sides_and_outer_failure_rolls_back_all(legacy):
    conn, _, account = legacy
    conn.execute("""CREATE TRIGGER fail_import BEFORE INSERT ON account_credit_imports
        BEGIN SELECT RAISE(ABORT,'simulated receipt failure'); END""")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        with pytest.raises(sqlite3.IntegrityError, match="simulated receipt failure"):
            ledger.import_legacy_in_transaction(conn, account, WALLET, 2125, migration_id="migration")
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 2.125
    assert ledger.balance(conn, account)["settled_units"] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger").fetchone()[0] == 0
    conn.execute("DROP TRIGGER fail_import")
    with pytest.raises(RuntimeError, match="later ownership failure"):
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            ledger.import_legacy_in_transaction(conn, account, WALLET, 2125, migration_id="migration")
            raise RuntimeError("later ownership failure")
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 2.125
    assert ledger.balance(conn, account)["settled_units"] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 0


@pytest.mark.parametrize("same_id", [True, False])
def test_real_concurrent_imports_never_duplicate_wallet_credit(legacy, same_id):
    conn, path, account = legacy
    barrier = threading.Barrier(6)

    def attempt(index):
        db = sqlite3.connect(path, timeout=10)
        try:
            barrier.wait(timeout=10)
            return move(db, account, migration="same" if same_id else f"import-{index}")
        except ledger.LedgerError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt, range(6)))
    assert sum(isinstance(result, dict) and not result["replayed"] for result in results) == 1
    if same_id:
        assert sum(isinstance(result, dict) and result["replayed"] for result in results) == 5
    else:
        assert results.count("import_balance_changed") == 5
    assert ledger.balance(conn, account)["settled_units"] == 2125
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 1


def test_import_and_legacy_spend_serialize_without_double_spend(legacy):
    conn, path, account = legacy
    barrier = threading.Barrier(2)

    def attempt(importing):
        db = sqlite3.connect(path, timeout=10)
        try:
            barrier.wait(timeout=10)
            db.execute("BEGIN IMMEDIATE")
            with db:
                if importing:
                    return ledger.import_legacy_in_transaction(db, account, WALLET, 2125, migration_id="race")
                return db.execute("UPDATE credits SET balance=balance-1 WHERE wallet=? AND balance>=1", (WALLET,)).rowcount
        except ledger.LedgerError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        imported, spent = list(pool.map(attempt, [True, False]))
    if isinstance(imported, dict):
        assert spent == 0 and ledger.balance(conn, account)["settled_units"] == 2125
        assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 0
    else:
        assert imported == "import_balance_changed" and spent == 1
        assert ledger.balance(conn, account)["settled_units"] == 0
        assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 1.125


def test_stray_ledger_key_cannot_authorize_a_second_wallet_debit(legacy):
    conn, _, account = legacy
    transact(conn, ledger._apply, account, operation="legacy_import", key="legacy_import:migration-one",
        resource="migration-one", settled_delta=2125, reserved_delta=0, actor=account,
        reason="explicit_wallet_credit_import")
    with pytest.raises(ledger.LedgerError, match="import_ledger_inconsistent"):
        move(conn, account)
    assert conn.execute("SELECT balance FROM credits").fetchone()[0] == 2.125
    assert ledger.balance(conn, account)["settled_units"] == 2125
    assert conn.execute("SELECT COUNT(*) FROM account_credit_imports").fetchone()[0] == 0


@pytest.mark.parametrize("units", [0, -1, True, 1.5, "2125", 10**16])
def test_import_requires_bounded_positive_integer_units(legacy, units):
    with pytest.raises(ledger.LedgerError, match="invalid_credit_units"):
        move(legacy[0], legacy[2], units=units)


def test_import_requires_outer_transaction(legacy):
    with pytest.raises(RuntimeError, match="caller-owned transaction"):
        ledger.import_legacy_in_transaction(legacy[0], legacy[2], WALLET, 2125, migration_id="missing-transaction")

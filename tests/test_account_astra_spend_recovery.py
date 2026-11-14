"""Fault injection stays inside disposable account HTTP fixtures."""
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from tests.test_account_astra import platform, keys
import app
import account_ledger


@pytest.mark.parametrize('window', ['before_debit', 'after_committed_debit'])
def test_account_pending_retry_requires_committed_matching_debit(platform, monkeypatch, window):
    harness, headers, account_id = platform
    original = app.astra_rewards.process_spend

    def interrupted(**kwargs):
        debit = kwargs['deduct_fn']

        def fault(*args):
            if window == 'after_committed_debit':
                debit(*args)
            raise RuntimeError('disposable interruption')

        return original(**{**kwargs, 'deduct_fn': fault})

    body = {'action': 'gacha_1', 'idempotency_key': 'interrupted-key'}
    with monkeypatch.context() as patch:
        patch.setattr(app.astra_rewards, 'process_spend', interrupted)
        assert harness.client.post('/v2/account/astra/spend', headers=headers, json=body).status_code == 500
    with app.app.app_context():
        conn = app.get_db()
        expected_count = int(window == 'after_committed_debit')
        assert conn.execute("SELECT status FROM astra_spends WHERE idempotency_key='interrupted-key'").fetchone()[0] == 'pending'
        assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=? AND operation='spend'", (account_id,)).fetchone()[0] == expected_count
        before = account_ledger.balance(conn, account_id)['settled_units']
    for _ in range(2):
        retried = harness.client.post('/v2/account/astra/spend', headers=headers, json=body)
        if window == 'after_committed_debit':
            assert retried.status_code == 200
            assert retried.json['ok'] and retried.json['replayed']
        else:
            assert retried.status_code == 422
            assert retried.json['error']['code'] == 'duplicate_request'
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account_id)['settled_units'] == before
        assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=? AND operation='spend'", (account_id,)).fetchone()[0] == expected_count
        state = conn.execute("SELECT status FROM astra_spends WHERE idempotency_key='interrupted-key'").fetchone()[0]
        assert state == ('completed' if expected_count else 'pending')


@pytest.mark.parametrize('ambiguity', ['different_action', 'different_amount', 'other_account', 'unknown_status', 'intent_amount'])
def test_account_recovery_never_uses_ambiguous_ledger_evidence(platform, keys, ambiguity):
    harness, headers, account_id = platform
    with app.app.app_context():
        conn = app.get_db()
        ledger_account = account_id
        if ambiguity == 'other_account':
            import account_identity
            ledger_account = account_identity.ensure_account(conn,
                account_identity.VerifiedPrincipal('https://auth.example', 'other', 'other-session'))
        conn.execute('BEGIN IMMEDIATE')
        with conn:
            if ledger_account != account_id:
                account_ledger.fund_in_transaction(conn, ledger_account, 100000, payment_id='other-funding')
            account_ledger.astra_spend_in_transaction(conn, ledger_account,
                20000 if ambiguity == 'different_amount' else 10000,
                action='gacha_10' if ambiguity == 'different_action' else 'gacha_1',
                idempotency_key='ambiguous-key')
            conn.execute('INSERT INTO astra_spends(wallet,action,amount,created_at,idempotency_key,status) VALUES (?,?,?,?,?,?)',
                (app._astra_account_principal(account_id), 'gacha_1', 20 if ambiguity == 'intent_amount' else 10, app.time.time(), 'ambiguous-key',
                 'unknown' if ambiguity == 'unknown_status' else 'pending'))
        before = [tuple(row) for row in conn.execute('SELECT * FROM account_credit_ledger')]
        balances = [tuple(row) for row in conn.execute('SELECT * FROM account_credit_balances')]
    retry = harness.client.post('/v2/account/astra/spend', headers=headers,
        json={'action': 'gacha_1', 'idempotency_key': 'ambiguous-key'})
    assert retry.status_code == 422
    assert retry.json['error']['code'] == 'duplicate_request'
    with app.app.app_context():
        conn = app.get_db()
        assert [tuple(row) for row in conn.execute('SELECT * FROM account_credit_ledger')] == before
        assert [tuple(row) for row in conn.execute('SELECT * FROM account_credit_balances')] == balances
        assert conn.execute("SELECT status FROM astra_spends WHERE idempotency_key='ambiguous-key'").fetchone()[0] == ('unknown' if ambiguity == 'unknown_status' else 'pending')


def test_concurrent_account_recovery_never_charges_again(platform):
    _harness, headers, account_id = platform
    with app.app.app_context():
        conn = app.get_db()
        conn.execute('BEGIN IMMEDIATE')
        with conn:
            account_ledger.astra_spend_in_transaction(conn, account_id, 10000,
                action='gacha_1', idempotency_key='concurrent-recovery')
            conn.execute("INSERT INTO astra_spends(wallet,action,amount,created_at,idempotency_key,status) VALUES (?,?,?,?,?,'pending')",
                (app._astra_account_principal(account_id), 'gacha_1', 10, app.time.time(), 'concurrent-recovery'))
    barrier = Barrier(2)

    def retry(_index):
        client = app.app.test_client()
        barrier.wait(timeout=5)
        response = client.post('/v2/account/astra/spend', headers=headers,
            json={'action': 'gacha_1', 'idempotency_key': 'concurrent-recovery'})
        if response.status_code == 200:
            assert response.json['replayed']
        else:
            assert response.status_code == 422
            assert response.json['error']['code'] == 'duplicate_request'
        return response.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert 200 in list(pool.map(retry, range(2)))
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account_id)['settled_units'] == 90000
        assert conn.execute("SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=? AND operation='spend'", (account_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT status FROM astra_spends WHERE idempotency_key='concurrent-recovery'").fetchone()[0] == 'completed'

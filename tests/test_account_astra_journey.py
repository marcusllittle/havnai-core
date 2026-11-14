"""Disposable account journey through real HTTP handlers; no live Clerk/browser."""
import pytest

from tests.test_account_astra import platform, keys, token
from tests.test_account_import import inventory, WALLET, PREPARE, challenge_request
from eth_account import Account
from eth_account.messages import encode_defunct
import app
import account_ledger
import account_lifecycle
from tests.test_account_lifecycle import signed, CONFIG as lifecycle_config


def test_account_refresh_reload_reward_and_spend_journey(platform, keys, monkeypatch):
    harness, headers, account_id = platform
    assert harness.client.get('/v2/account/astra/stats').status_code == 401
    expired = {'Authorization': token(keys, iat=1, nbf=1, exp=2)}
    assert harness.client.get('/v2/account/astra/stats', headers=expired).status_code == 401
    # A refreshed token identifies the same account; no wallet proof is used.
    refreshed = {'Authorization': token(keys, iat=int(app.time.time()) - 1)}
    assert harness.client.get('/v2/account', headers=refreshed).json['id'] == account_id
    start = harness.client.post('/v2/account/astra/run/start', headers=refreshed,
                                 json={'map_id': 'nebula-runway'})
    assert start.status_code == 200
    run = start.json['run_token']
    with app.app.app_context():
        conn = app.get_db()
        conn.execute('UPDATE astra_run_tokens SET started_at=? WHERE token_hash=?',
                     (app.time.time() - 300, app.astra_rewards._hash_token(run)))
        conn.commit()
    body = {'map_id': 'nebula-runway', 'run_token': run, 'score': 100000,
            'grade': 'S', 'duration_s': 9999}
    assert harness.client.post('/v2/account/astra/reward', headers=refreshed, json=body).status_code == 200
    assert harness.client.post('/v2/account/astra/reward', headers=refreshed, json=body).json['error']['code'] == 'duplicate_run'
    spend = {'action': 'gacha_1', 'idempotency_key': 'journey-purchase'}
    assert harness.client.post('/v2/account/astra/spend', headers=refreshed, json=spend).status_code == 200
    # New HTTP client has no cookies/session cache; current bearer restores access.
    reloaded = app.app.test_client()
    stats = reloaded.get('/v2/account/astra/stats', headers=refreshed)
    assert stats.status_code == 200
    assert stats.json['account_id'] == account_id
    assert stats.json['total_runs'] == 1
    assert stats.json['balance']['available_units'] == 105000
    replay = reloaded.post('/v2/account/astra/spend', headers=refreshed, json=spend)
    assert replay.status_code == 200 and replay.json['replayed']
    assert reloaded.post('/v2/account/astra/run/start', headers=refreshed,
                          json={'map_id': 'nebula-runway'}).status_code == 200
    assert reloaded.get('/v2/account/astra/stats').status_code == 401
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account_id)['settled_units'] == 105000
        for operation in ('reward', 'spend'):
            assert conn.execute('SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=? AND operation=?',
                                (account_id, operation)).fetchone()[0] == 1
    monkeypatch.setattr(account_lifecycle.Config, 'from_environment', lambda: lifecycle_config)
    event, event_headers = signed('session.revoked', {'id': 'sess_alice'})
    assert reloaded.post('/v2/auth/clerk/webhook', data=event, headers=event_headers).status_code == 200
    revoked = reloaded.get('/v2/account/astra/stats', headers=refreshed)
    assert revoked.status_code == 403
    assert revoked.json['error']['code'] == 'account_session_revoked'
    signed_in_again = {'Authorization': token(keys, sid='sess_alice_new')}
    restored = reloaded.get('/v2/account/astra/stats', headers=signed_in_again)
    assert restored.status_code == 200
    assert restored.json['account_id'] == account_id
    assert restored.json['balance']['available_units'] == 105000


@pytest.mark.parametrize('after_deduction', [False, True])
def test_pending_account_spend_never_reports_purchase_success(platform, after_deduction):
    harness, headers, account_id = platform
    with app.app.app_context():
        conn = app.get_db()
        if after_deduction:
            conn.execute('BEGIN IMMEDIATE')
            with conn:
                account_ledger.astra_spend_in_transaction(conn, account_id, 10000,
                    action='gacha_1', idempotency_key='interrupted-purchase')
        conn.execute("INSERT INTO astra_spends(wallet,action,amount,created_at,idempotency_key,status) VALUES (?,?,?,?,?,'pending')",
                     (app._astra_account_principal(account_id), 'gacha_1', 10, app.time.time(), 'interrupted-purchase'))
        conn.commit()
    retry = harness.client.post('/v2/account/astra/spend', headers=headers,
        json={'action': 'gacha_1', 'idempotency_key': 'interrupted-purchase'})
    assert retry.status_code == 422
    assert retry.json['error']['code'] == 'duplicate_request'
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account_id)['settled_units'] == (90000 if after_deduction else 100000)
        assert conn.execute("SELECT status FROM astra_spends WHERE idempotency_key='interrupted-purchase'").fetchone()[0] == 'pending'


def test_linked_legacy_import_restores_balance_and_generation_history(inventory, monkeypatch):
    harness, headers, account_id, _other = inventory
    # Reuse existing synthetic signed wallet-link/import fixture, never a real key.
    signer = Account.create()
    with app.app.app_context():
        conn = app.get_db()
        for table in ('jobs', 'credits', 'wallet_links'):
            conn.execute(f'UPDATE {table} SET wallet=? WHERE lower(wallet)=?',
                         (signer.address.lower(), WALLET))
        conn.commit()
    prepared = harness.client.post(PREPARE, headers=headers,
        json={'job_ids': ['01-ready'], 'include_credits': True})
    assert prepared.status_code == 201, prepared.json
    snapshot = prepared.json
    challenged = challenge_request(harness, headers, snapshot)
    assert challenged.status_code == 201, challenged.json
    challenge = challenged.json
    signature = signer.sign_message(encode_defunct(text=challenge['message'])).signature.hex()
    monkeypatch.setattr(app, 'ASTRA_SPEND_ENABLED', True)
    result = harness.client.post('/v2/account/import-snapshots/' + snapshot['id'] + '/execute',
        headers={**headers, 'Origin': 'https://joinhavn.io'},
        json={'challenge_id': challenge['challenge_id'], 'signature': signature, 'chain_id': 11155111})
    assert result.status_code == 200
    assert result.json['receipt']['credit_units'] == 2125
    reloaded = app.app.test_client()
    stats = reloaded.get('/v2/account/astra/stats', headers=headers)
    assert stats.status_code == 200
    assert stats.json['balance']['available_units'] == 102125
    assert reloaded.get('/v2/jobs/01-ready', headers=headers).status_code == 200
    assert reloaded.get('/v2/account/ledger', headers=headers).status_code == 200

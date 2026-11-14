"""A spend key identifies one action in either authenticated economy mode."""
import pytest

from tests.test_account_astra import platform, keys
import app
import account_ledger


@pytest.mark.parametrize('mode', ['account', 'wallet'])
@pytest.mark.parametrize('state', ['pending', 'failed', 'completed'])
@pytest.mark.parametrize('changed_action', [False, True])
def test_spend_key_binds_action_without_rewriting_original_outcome(platform, mode, state, changed_action):
    harness, account_headers, account_id = platform
    wallet = '0x' + '7' * 40
    with app.app.app_context():
        conn = app.get_db()
        if mode == 'account':
            principal, headers = app._astra_account_principal(account_id), account_headers
            path = '/v2/account/astra/spend'
        else:
            principal = wallet
            app.settlement.init_settlement_tables(conn)
            app.credits.deposit_credits(wallet, 100)
            headers = {'Authorization': 'Bearer ' + app.astra_rewards.create_session(wallet, 3600)['token']}
            path = '/astra/spend'
    original = {'action': 'gacha_1', 'idempotency_key': 'bound-purchase'}
    if state == 'completed':
        # A real completed charge, rather than an invented success row.
        charged = harness.client.post(path, headers=headers, json=original)
        assert charged.status_code == 200
    else:
        with app.app.app_context():
            conn = app.get_db()
            conn.execute('INSERT INTO astra_spends(wallet,action,amount,created_at,idempotency_key,status) VALUES (?,?,?,?,?,?)',
                         (principal, 'gacha_1', 10, app.time.time(), 'bound-purchase', state))
            conn.commit()

    def snapshot():
        with app.app.app_context():
            conn = app.get_db()
            if mode == 'account':
                balance = account_ledger.balance(conn, account_id)['settled_units']
                entries = conn.execute('SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=?', (account_id,)).fetchone()[0]
            else:
                balance = app.credits.get_credit_balance(wallet)
                entries = conn.execute('SELECT COUNT(*) FROM credit_ledger WHERE wallet=?', (wallet,)).fetchone()[0]
            rows = [tuple(row) for row in conn.execute('SELECT * FROM astra_spends WHERE wallet=?', (principal,)).fetchall()]
            return balance, entries, rows

    before = snapshot()
    result = harness.client.post(path, headers=headers,
        json={**original, 'action': 'gacha_10' if changed_action else 'gacha_1'})
    if not changed_action and state == 'completed':
        assert result.status_code == 200
        assert result.json['ok'] and result.json['replayed']
        assert result.json['action'] == 'gacha_1'
    else:
        assert result.status_code == 422
        expected = 'idempotency_conflict' if changed_action else (
            'duplicate_request' if state == 'pending' else 'insufficient_credits')
        code = result.json['error']['code'] if mode == 'account' else result.json['reason']
        assert code == expected
    assert snapshot() == before

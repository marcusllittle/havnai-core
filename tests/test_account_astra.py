import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from tests import test_platform_v1 as platform_fixture
from tests.test_account_auth import keys, token, config
import app
import account_auth
import account_ledger
import account_astra


@pytest.fixture
def platform(keys, monkeypatch):
    harness = platform_fixture.PlatformApiContractTests(methodName="runTest")
    harness.setUp()
    monkeypatch.setattr(app, "RATE_LIMIT_BUCKETS", {})
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: config(keys))
    headers = {"Authorization": token(keys)}
    account = harness.client.get("/v2/account", headers=headers).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, account, 100000, payment_id="astra-paid")
    yield harness, headers, account
    harness.tearDown()


def _start_run(harness, headers, *, map_id="nebula-runway"):
    response = harness.client.post("/v2/astra/run/start", headers=headers, json={"map_id": map_id})
    assert response.status_code == 201, response.json
    return response.json["run_token"]


def _backdate_token(account, seconds=120):
    with app.app.app_context():
        app.get_db().execute(
            "UPDATE account_astra_run_tokens SET started_at=? WHERE account_id=? AND consumed_at IS NULL",
            (time.time() - seconds, account),
        )
        app.get_db().commit()


def test_account_astra_session_uses_bearer_without_wallet(platform):
    harness, headers, account = platform
    response = harness.client.get("/v2/astra/session", headers=headers)
    assert response.status_code == 200
    assert response.json["mode"] == "account"
    assert response.json["account_id"] == account
    assert response.json["wallet_required"] is False
    assert harness.client.get("/v2/astra/session").status_code == 401


def test_legacy_astra_session_token_cannot_authorize_account_routes(platform):
    harness, _, _ = platform
    with app.app.app_context():
        app.astra_rewards.init_astra_tables(app.get_db())
    legacy = app.astra_rewards.create_session("0x" + "1" * 40, 3600)

    response = harness.client.get(
        "/v2/astra/session",
        headers={"Authorization": f"Bearer {legacy['token']}"},
    )

    assert response.status_code == 401


def test_account_reward_is_server_timed_capped_and_never_uses_legacy_wallet(platform):
    harness, headers, account = platform
    token_value = _start_run(harness, headers)
    _backdate_token(account, seconds=300)
    response = harness.client.post("/v2/astra/reward", headers=headers, json={
        "score": 100_000,
        "grade": "S",
        "duration_s": 1,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert response.status_code == 200, response.json
    assert response.json["ok"] is True
    assert response.json["reward_units"] == account_astra._credits_to_units(app.astra_rewards.MAX_CREDITS_PER_RUN)
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account)["available_units"] == 100000 + response.json["reward_units"]
        legacy = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='astra_runs'").fetchone()
        if legacy:
            assert conn.execute("SELECT COUNT(*) FROM astra_runs").fetchone()[0] == 0
        row = conn.execute(
            "SELECT account_id, reward_units FROM account_astra_runs WHERE run_id=?",
            (response.json["run_id"],),
        ).fetchone()
        assert tuple(row) == (account, response.json["reward_units"])


def test_account_reward_rejects_fake_duration_and_burns_token(platform):
    harness, headers, account = platform
    token_value = _start_run(harness, headers)
    first = harness.client.post("/v2/astra/reward", headers=headers, json={
        "score": 100_000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert first.status_code == 422
    assert first.json["error"]["code"] == "run_too_short"
    second = harness.client.post("/v2/astra/reward", headers=headers, json={
        "score": 100_000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert second.status_code == 422
    assert second.json["error"]["code"] == "run_token_used"
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_astra_runs WHERE account_id=?", (account,)).fetchone()[0] == 0


def test_account_run_token_is_bound_to_account(platform, keys):
    harness, headers, account = platform
    token_value = _start_run(harness, headers)
    other_headers = {"Authorization": token(keys, sub="astra_other", sid="astra_other")}
    response = harness.client.post("/v2/astra/reward", headers=other_headers, json={
        "score": 100_000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert response.status_code == 422
    assert response.json["error"]["code"] == "run_token_account_mismatch"
    with app.app.app_context():
        token_row = app.get_db().execute(
            "SELECT consumed_at FROM account_astra_run_tokens WHERE account_id=?",
            (account,),
        ).fetchone()
        assert token_row["consumed_at"] is None


def test_account_spend_is_idempotent_and_uses_account_ledger(platform):
    harness, headers, account = platform
    spend_headers = {**headers, "Idempotency-Key": "astra-spend-one"}
    first = harness.client.post("/v2/astra/spend", headers=spend_headers, json={"action": "gacha_10"})
    second = harness.client.post("/v2/astra/spend", headers=spend_headers, json={"action": "gacha_10"})
    assert first.status_code == 200, first.json
    assert second.status_code == 200, second.json
    assert second.json["replayed"] is True
    assert first.json["cost_units"] == 80000
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account)["available_units"] == 20000
        rows = conn.execute(
            "SELECT operation, settled_delta, reason FROM account_credit_ledger WHERE account_id=? AND operation='astra_spend'",
            (account,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["settled_delta"] == -80000


def test_failed_account_spend_is_audited_but_not_replayed_as_success(platform):
    harness, headers, account = platform
    spend_headers = {**headers, "Idempotency-Key": "too-expensive"}
    response = harness.client.post("/v2/astra/spend", headers=spend_headers, json={"action": "gacha_10"})
    assert response.status_code == 200
    response = harness.client.post("/v2/astra/spend", headers={**headers, "Idempotency-Key": "too-expensive-2"}, json={"action": "gacha_10"})
    assert response.status_code == 422
    assert response.json["error"]["code"] == "insufficient_credits"
    retry = harness.client.post("/v2/astra/spend", headers={**headers, "Idempotency-Key": "too-expensive-2"}, json={"action": "gacha_10"})
    assert retry.status_code == 422
    assert retry.json["error"]["code"] == "insufficient_credits"
    with app.app.app_context():
        conn = app.get_db()
        assert account_ledger.balance(conn, account)["available_units"] == 20000
        assert conn.execute(
            "SELECT status FROM account_astra_spends WHERE idempotency_key='too-expensive-2'"
        ).fetchone()["status"] == "failed"

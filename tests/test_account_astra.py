import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from tests import test_platform_v1 as platform_fixture
from tests.test_account_auth import config, keys, token
import account_auth
import account_ledger
import app


@pytest.fixture
def platform(keys, monkeypatch):
    harness = platform_fixture.PlatformApiContractTests(methodName="runTest")
    harness.setUp()
    monkeypatch.setattr(app, "RATE_LIMIT_BUCKETS", {})
    monkeypatch.setattr(app, "ASTRA_SPEND_ENABLED", True)
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: config(keys))
    headers = {"Authorization": token(keys), "Idempotency-Key": "astra-spend-one"}
    account_id = harness.client.get("/v2/account", headers=headers).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        app.astra_rewards.init_astra_tables(conn)
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, account_id, 100000, payment_id="astra-test-fund")
    yield harness, headers, account_id
    harness.tearDown()


def test_account_astra_requires_account_auth(platform):
    harness, _headers, _account_id = platform
    response = harness.client.post("/v2/account/astra/run/start", json={"map_id": "nebula-runway"})
    assert response.status_code == 401
    assert response.json["error"]["code"] == "account_required"


def test_account_astra_reward_uses_server_run_and_account_ledger(platform):
    harness, headers, account_id = platform
    start = harness.client.post("/v2/account/astra/run/start", headers=headers, json={"map_id": "nebula-runway"})
    assert start.status_code == 200
    assert start.json["account_id"] == account_id
    token_value = start.json["run_token"]

    now = app.astra_rewards.time.time()
    with app.app.app_context():
        app.get_db().execute(
            "UPDATE astra_run_tokens SET started_at=? WHERE token_hash=?",
            (now - 300.0, app.astra_rewards._hash_token(token_value)),
        )
        app.get_db().commit()

    reward = harness.client.post("/v2/account/astra/reward", headers=headers, json={
        "score": 100000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert reward.status_code == 200, reward.json
    assert reward.json["reward"] == app.astra_rewards.MAX_CREDITS_PER_RUN
    with app.app.app_context():
        balance = account_ledger.balance(app.get_db(), account_id)
        entry = app.get_db().execute(
            "SELECT operation, settled_delta, reason FROM account_credit_ledger "
            "WHERE account_id=? AND operation='reward'",
            (account_id,),
        ).fetchone()
    assert balance["settled_units"] == 115000
    assert tuple(entry) == ("reward", 15000, "astra_game_reward")

    replay = harness.client.post("/v2/account/astra/reward", headers=headers, json={
        "score": 100000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": token_value,
    })
    assert replay.status_code == 422
    assert replay.json["error"]["code"] == "run_token_used"


def test_account_astra_rejects_client_faked_duration(platform):
    harness, headers, account_id = platform
    start = harness.client.post("/v2/account/astra/run/start", headers=headers, json={"map_id": "nebula-runway"})
    reward = harness.client.post("/v2/account/astra/reward", headers=headers, json={
        "score": 100000,
        "grade": "S",
        "duration_s": 9999,
        "map_id": "nebula-runway",
        "run_token": start.json["run_token"],
    })
    assert reward.status_code == 422, reward.json
    assert reward.json["error"]["code"] == "run_too_short"
    with app.app.app_context():
        assert account_ledger.balance(app.get_db(), account_id)["settled_units"] == 100000


def test_account_astra_spend_deducts_account_credits_once(platform):
    harness, headers, account_id = platform
    body = {"action": "gacha_1", "idempotency_key": "pull-one"}
    first = harness.client.post("/v2/account/astra/spend", headers=headers, json=body)
    assert first.status_code == 200, first.json
    assert first.json["cost"] == app.astra_rewards.SPEND_COSTS["gacha_1"]
    replay = harness.client.post("/v2/account/astra/spend", headers=headers, json=body)
    assert replay.status_code == 200, replay.json
    assert replay.json["replayed"] is True
    with app.app.app_context():
        balance = account_ledger.balance(app.get_db(), account_id)
        spend_count = app.get_db().execute(
            "SELECT COUNT(*) FROM account_credit_ledger WHERE account_id=? AND operation='spend'",
            (account_id,),
        ).fetchone()[0]
    assert balance["settled_units"] == 90000
    assert spend_count == 1


def test_account_astra_payload_cannot_replace_identity(platform):
    harness, headers, _account_id = platform
    response = harness.client.post("/v2/account/astra/run/start", headers=headers,
                                   json={"map_id": "nebula-runway", "account_id": "acct_other"})
    assert response.status_code == 422
    assert response.json["error"]["code"] == "invalid_payload"

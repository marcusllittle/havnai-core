"""Real Svix signatures, durable revocation, ordering, and account API enforcement."""
import base64
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from flask import Flask
from svix.webhooks import Webhook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_auth
import account_identity as identity
import account_ledger as ledger
import account_lifecycle as lifecycle
import account_routes
from tests.test_account_auth import keys, token, config as auth_config

SECRET = "whsec_" + base64.b64encode(b"fixture-signing-secret-32-bytes!!").decode()
CONFIG = lifecycle.Config("https://auth.example", "ins_fixture", SECRET)
ALICE = identity.VerifiedPrincipal(CONFIG.issuer, "user_alice", "sess_alice")


@pytest.fixture
def database(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db")
    conn.row_factory = sqlite3.Row
    identity.initialize(conn)
    ledger.initialize(conn)
    yield conn
    conn.close()


def signed(kind, data=None, *, event_id="msg_fixture", timestamp=1000, instance="ins_fixture", age=0):
    raw = json.dumps({"object": "event", "type": kind, "data": data or {"id": "user_alice"},
                      "instance_id": instance, "timestamp": timestamp})
    when = datetime.now(timezone.utc) - timedelta(seconds=age)
    return raw.encode(), {"svix-id": event_id, "svix-timestamp": str(int(when.timestamp())),
                          "svix-signature": Webhook(SECRET).sign(event_id, when, raw)}


def deliver(conn, kind, data=None, **kwargs):
    payload, headers = signed(kind, data, **kwargs)
    return lifecycle.webhook(conn, payload, headers, config=CONFIG)


def test_deleted_user_cannot_provision_and_cannot_be_revived(database):
    deliver(database, "user.deleted")
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        identity.ensure_account(database, ALICE)
    assert database.execute("SELECT count(*) FROM accounts").fetchone()[0] == 0
    deliver(database, "user.created", {"id": ALICE.subject, "banned": False, "updated_at": 2000}, event_id="msg_later")
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        identity.ensure_account(database, ALICE)


def test_bans_order_by_user_version_and_preserve_credits(database):
    account = identity.ensure_account(database, ALICE)
    database.execute("BEGIN IMMEDIATE")
    with database:
        ledger.fund_in_transaction(database, account, 5000, payment_id="paid_fixture")
    def update(version, banned, event_id):
        return deliver(database, "user.updated", {"id": ALICE.subject, "banned": banned, "updated_at": version}, event_id=event_id)
    update(2000, True, "msg_ban")
    update(1000, False, "msg_stale_unban")
    update(2000, False, "msg_tied_unban")
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        identity.ensure_account(database, ALICE)
    update(3000, False, "msg_unban")
    update(2500, True, "msg_stale_ban")
    assert identity.ensure_account(database, ALICE) == account
    assert ledger.balance(database, account)["settled_units"] == 5000
    database.execute("UPDATE accounts SET status='suspended' WHERE id=?", (account,))
    database.commit()
    update(4000, False, "msg_local_suspension")
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        identity.ensure_account(database, ALICE)


@pytest.mark.parametrize("kind", ["session.ended", "session.removed", "session.revoked"])
def test_session_revocation_keeps_other_sessions_and_provider_separate(database, kind):
    account = identity.ensure_account(database, ALICE)
    deliver(database, kind, {"id": ALICE.session_id})
    with pytest.raises(identity.IdentityError, match="account_session_revoked"):
        identity.ensure_account(database, ALICE)
    assert identity.ensure_account(database, identity.VerifiedPrincipal(ALICE.issuer, ALICE.subject, "sess_other")) == account
    assert identity.ensure_account(database, identity.VerifiedPrincipal("https://other.example", ALICE.subject, ALICE.session_id)) != account


def test_duplicate_delivery_and_mismatched_replay(database):
    assert deliver(database, "user.deleted")["status"] == "processed"
    assert deliver(database, "user.deleted")["status"] == "duplicate"
    with pytest.raises(lifecycle.LifecycleError, match="replay_mismatch"):
        deliver(database, "user.deleted", {"id": "user_other"})
    assert database.execute("SELECT count(*) FROM account_lifecycle_events").fetchone()[0] == 1
    assert identity.ensure_account(database, identity.VerifiedPrincipal(ALICE.issuer, "user_other", "sess_other"))


@pytest.mark.parametrize("bad", ["missing_signature", "altered_body", "expired", "wrong_instance", "malformed"])
def test_bad_webhooks_cannot_change_access(database, bad):
    payload, headers = signed("user.deleted", instance="ins_wrong" if bad == "wrong_instance" else "ins_fixture", age=600 if bad == "expired" else 0)
    if bad == "missing_signature":
        headers.pop("svix-signature")
    if bad == "altered_body":
        payload += b" "
    if bad == "malformed":
        payload, headers = signed("user.updated", {"id": ALICE.subject, "banned": "false", "updated_at": 1000})
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.webhook(database, payload, headers, config=CONFIG)
    assert database.execute("SELECT count(*) FROM account_lifecycle_events").fetchone()[0] == 0
    assert identity.ensure_account(database, ALICE)


def test_write_failure_rolls_back_revocation_and_allows_retry(database):
    database.execute("CREATE TRIGGER fail_ack BEFORE INSERT ON account_lifecycle_events BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        deliver(database, "user.deleted")
    assert database.execute("SELECT count(*) FROM account_provider_users").fetchone()[0] == 0
    database.execute("DROP TRIGGER fail_ack")
    assert deliver(database, "user.deleted")["status"] == "processed"


def test_http_revokes_real_bearer_access(database, keys, monkeypatch):
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: auth_config(keys))
    monkeypatch.setattr(lifecycle.Config, "from_environment", lambda: CONFIG)
    app = Flask(__name__)
    app.register_blueprint(account_routes.create_blueprint(lambda: database, lambda *a, **k: True))
    client = app.test_client()
    auth = {"Authorization": token(keys)}
    assert client.get("/v2/account", headers=auth).status_code == 200
    payload, headers = signed("session.revoked", {"id": ALICE.session_id})
    assert client.post("/v2/auth/clerk/webhook", data=payload, headers=headers).status_code == 200
    assert client.get("/v2/account/credits", headers=auth).status_code == 401
    payload, headers = signed("user.deleted", event_id="msg_delete")
    assert client.post("/v2/auth/clerk/webhook", data=payload, headers=headers).status_code == 200
    assert client.get("/v2/account", headers={"Authorization": token(keys, sid="sess_new")}).status_code == 403


def test_missing_config_and_oversize_payload(database):
    payload, headers = signed("user.deleted")
    with pytest.raises(lifecycle.LifecycleError) as error:
        lifecycle.webhook(database, payload, headers, config=lifecycle.Config("", "", ""))
    assert error.value.status == 503
    with pytest.raises(lifecycle.LifecycleError) as error:
        lifecycle.webhook(database, b"x" * (1024 * 1024 + 1), headers, config=CONFIG)
    assert error.value.status == 413

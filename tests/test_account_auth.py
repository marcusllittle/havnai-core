import sqlite3
import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_auth
import account_identity
import account_ledger
import account_routes


@pytest.fixture
def keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return key, public


def token(keys, **overrides):
    now = int(time.time())
    claims = {"iss": "https://auth.example", "sub": "user_alice", "sid": "sess_alice",
              "aud": "havnai-api", "azp": "https://joinhavn.io", "iat": now,
              "nbf": now, "exp": now + 60, "fva": [0, -1]}
    claims.update(overrides)
    return "Bearer " + jwt.encode(claims, keys[0], algorithm="RS256")


def config(keys):
    return account_auth.AuthConfig("https://auth.example", "havnai-api", ("https://joinhavn.io",), jwt_key=keys[1])


def test_real_signature_validates_without_network(keys):
    principal = account_auth.verify_bearer(token(keys), config=config(keys), recent=True)
    assert principal.subject == "user_alice"


@pytest.mark.parametrize("claims", [
    {"iss": "https://other.example"}, {"aud": "other-api"}, {"azp": "https://evil.example"},
    {"exp": 1}, {"sid": ""}, {"sub": ""}, {"exp": int(time.time()) + 3600},
    {"nbf": int(time.time()) + 100}, {"iat": int(time.time()) + 100}, {"azp": None}, {"sts": "pending"},
])
def test_invalid_session_claims_rejected(keys, claims):
    with pytest.raises(account_auth.AccountAuthError):
        account_auth.verify_bearer(token(keys, **claims), config=config(keys))


def test_another_signing_key_rejected(keys):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(account_auth.AccountAuthError):
        account_auth.verify_bearer(token((other, keys[1])), config=config(keys))


@pytest.mark.parametrize("header", ["", "Bearer owner-key", "Bearer 0x" + "1" * 40, "Basic ignored"])
def test_legacy_owner_and_wallet_are_not_sessions(keys, header):
    with pytest.raises(account_auth.AccountAuthError):
        account_auth.verify_bearer(header, config=config(keys))


@pytest.mark.parametrize("fva", [None, [], [-1, -1], [6, -1], ["0", -1]])
def test_recent_verification_is_not_token_refresh(keys, fva):
    with pytest.raises(account_auth.AccountAuthError, match="reauthentication_required"):
        account_auth.verify_bearer(token(keys, fva=fva), config=config(keys), recent=True)


def test_missing_configuration_fails_closed():
    with pytest.raises(account_auth.AccountAuthError, match="not_configured") as exc:
        account_auth.verify_bearer("Bearer anything", config=account_auth.AuthConfig("", "", ()))
    assert exc.value.status == 503


def test_http_account_is_bound_to_verified_subject(keys, monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    account_identity.initialize(conn)
    account_ledger.initialize(conn)
    app = Flask(__name__)
    app.register_blueprint(account_routes.create_blueprint(lambda: conn, lambda *a, **k: True))
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: config(keys))
    client = app.test_client()
    try:
        assert client.get("/v2/account", headers={"X-Account-ID": "forged"}).status_code == 401
        alice = client.get("/v2/account?wallet=forged", headers={"Authorization": token(keys)})
        assert alice.status_code == 200
        assert alice.json["id"].startswith("acct_")
        assert alice.json["wallets"] == []
        assert alice.headers["Cache-Control"] == "private, no-store"
        bob = client.get("/v2/account", headers={"Authorization": token(keys, sub="user_bob", sid="sess_bob")})
        assert bob.json["id"] != alice.json["id"]
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, alice.json["id"], 5000, payment_id="test")
        assert client.get("/v2/account/credits", headers={"Authorization": token(keys)}).json["available_units"] == 5000
        assert client.get("/v2/account/credits", headers={"Authorization": token(keys, sub="user_bob")}).json["available_units"] == 0
    finally:
        conn.close()

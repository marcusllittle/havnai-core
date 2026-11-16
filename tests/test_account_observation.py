"""Offline HTTP/SQLite contract for genuinely read-only restart observation."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from flask import Flask
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_auth
import account_identity
import account_observation

JOB = "job-" + "1" * 32
ALICE = account_identity.VerifiedPrincipal("https://synthetic.invalid", "alice", "session-a")
BOB = account_identity.VerifiedPrincipal("https://synthetic.invalid", "bob", "session-b")


def setup_observation(tmp_path, monkeypatch, scenario="same", *, deny_writes=True, raw_status="running"):
    path = tmp_path / "synthetic.sqlite"
    connect = sqlite3.connect
    with connect(path) as conn:
        account_identity.initialize(conn)
        alice = account_identity.ensure_account(conn, ALICE)
        account_identity.ensure_account(conn, BOB)
        conn.executescript("""
            CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,owner_account_id TEXT);
            CREATE TABLE artifact_lifecycle(job_id TEXT PRIMARY KEY,restored_at REAL);
        """)
        conn.execute("INSERT INTO jobs VALUES(?,?,?)", (JOB, raw_status, alice))
        if scenario == "revoked": conn.execute("INSERT INTO account_revoked_sessions VALUES(?,?,0)", (ALICE.issuer, ALICE.session_id))
        if scenario == "suspended": conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (alice,))
        if scenario in {"banned", "provider-deleted"}:
            conn.execute("INSERT INTO account_provider_users VALUES(?,?,?,?,0)",
                         (ALICE.issuer, ALICE.subject, int(scenario == "banned"), int(scenario == "provider-deleted")))
        if scenario in {"soft-deleted", "restored"}:
            conn.execute("INSERT INTO artifact_lifecycle VALUES(?,?)", (JOB, 1 if scenario == "restored" else None))
    before = path.read_bytes()
    calls, sql = [], []

    def read_only_connect(database_uri, **kwargs):
        assert database_uri == path.resolve().as_uri() + "?mode=ro"
        assert kwargs == {"uri": True}
        calls.append(database_uri)
        conn = connect(database_uri, **kwargs)
        conn.set_trace_callback(sql.append)
        if deny_writes:
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_TRANSACTION}
            conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        return conn

    def verify(authorization, *, config):
        assert authorization == "Bearer synthetic-session"
        if scenario == "expired": raise account_auth.AccountAuthError("invalid_account_session")
        if scenario == "unknown": return account_identity.VerifiedPrincipal(ALICE.issuer, "unknown", "session-unknown")
        return BOB if scenario == "other-owner" else ALICE

    def forbidden(*args, **kwargs): pytest.fail("provisioning/transaction setup forbidden")
    monkeypatch.setattr(account_observation.sqlite3, "connect", read_only_connect)
    monkeypatch.setattr(account_auth, "verify_bearer", verify)
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: object())
    monkeypatch.setattr(account_identity, "ensure_account", forbidden)
    monkeypatch.setattr(account_identity, "initialize", forbidden)
    monkeypatch.setattr(account_identity, "_begin", forbidden)
    monkeypatch.setattr(account_observation.platform_v1, "configure_connection", forbidden)
    monkeypatch.setattr(account_observation.platform_v1, "migrate", forbidden)
    app = Flask(__name__)
    app.register_blueprint(account_observation.create_blueprint(lambda: path, lambda *a, **k: True))
    return app.test_client(), path, before, alice, calls, sql


@pytest.mark.parametrize("scenario,expected", [
    ("same", 200), ("restored", 200), ("unknown", 401), ("revoked", 403),
    ("suspended", 403), ("banned", 403), ("provider-deleted", 403),
    ("other-owner", 404), ("soft-deleted", 404), ("expired", 401),
])
def test_observation_visibility_without_database_mutation(tmp_path, monkeypatch, scenario, expected):
    client, path, before, alice, calls, sql = setup_observation(tmp_path, monkeypatch, scenario)
    response = client.get(f"/v2/jobs/{JOB}/observation", headers={"Authorization": "Bearer synthetic-session"})
    assert response.status_code == expected
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["Vary"] == "Authorization"
    assert path.read_bytes() == before
    assert len(calls) == int(scenario != "expired")
    assert not any(statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "PRAGMA", "CREATE", "ALTER")) for statement in sql)
    if expected == 200:
        assert response.json == {"id": JOB, "status": "running", "owner_account_hash": hashlib.sha256(alice.encode()).hexdigest()[:12]}
    else:
        assert "owner_account_hash" not in response.json


def test_observation_connection_rejects_writes_itself(tmp_path, monkeypatch):
    client, path, before, _, _, _ = setup_observation(tmp_path, monkeypatch, deny_writes=False)
    def attempted_write(conn, principal):
        conn.execute("DELETE FROM jobs")
        pytest.fail("mode=ro unexpectedly allowed a write")
    monkeypatch.setattr(account_identity, "lookup_existing_account", attempted_write)
    response = client.get(f"/v2/jobs/{JOB}/observation", headers={"Authorization": "Bearer synthetic-session"})
    assert response.status_code == 503
    assert path.read_bytes() == before


@pytest.mark.parametrize("job_id", [".", "..", "job-1", "job-" + "A" * 32, "job-" + "2" * 32])
def test_invalid_or_missing_job_never_provisions(tmp_path, monkeypatch, job_id):
    client, path, before, _, calls, _ = setup_observation(tmp_path, monkeypatch)
    response = client.get(f"/v2/jobs/{job_id}/observation", headers={"Authorization": "Bearer synthetic-session"})
    assert response.status_code == 404
    assert path.read_bytes() == before
    assert len(calls) == int(job_id == "job-" + "2" * 32)


def test_non_get_method_cannot_enter_observer(tmp_path, monkeypatch):
    client, path, before, _, calls, _ = setup_observation(tmp_path, monkeypatch)
    assert client.post(f"/v2/jobs/{JOB}/observation").status_code == 405
    assert not calls and path.read_bytes() == before


def test_database_missing_fails_without_creating_it(tmp_path, monkeypatch):
    path = tmp_path / "absent.sqlite"
    monkeypatch.setattr(account_auth, "verify_bearer", lambda *a, **k: ALICE)
    monkeypatch.setattr(account_auth.AuthConfig, "from_environment", lambda: object())
    app = Flask(__name__)
    app.register_blueprint(account_observation.create_blueprint(lambda: path, lambda *a, **k: True))
    response = app.test_client().get(f"/v2/jobs/{JOB}/observation")
    assert response.status_code == 503 and not path.exists()


def test_resume_client_to_read_only_route_without_network_or_mutation(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import account_restart_recovery_drill as drill
    client, path, before, alice, _, _ = setup_observation(tmp_path, monkeypatch)
    evidence = {"schema": drill.RESUME_SCHEMA, "base_url": "https://api.joinhavn.io", "job_id": JOB,
                "owner_account_hash": hashlib.sha256(alice.encode()).hexdigest()[:12],
                "idempotency_key": "original-synthetic-key", "restart_phase": "restart_returned"}
    evidence_path = tmp_path / "evidence.json"; evidence_path.write_text(json.dumps(evidence))
    requests = []
    class Opener:
        def open(self, req, timeout):
            assert req.get_method() == "GET" and req.data is None
            assert req.full_url == f"https://api.joinhavn.io/v2/jobs/{JOB}/observation"
            requests.append(req.full_url)
            response = client.get(f"/v2/jobs/{JOB}/observation", headers={"Authorization": req.get_header("Authorization")})
            assert response.status_code == 200
            class Response:
                status = response.status_code
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def geturl(self): return req.full_url
                def read(self, size): return response.data
            return Response()
    def forbidden(*args, **kwargs): pytest.fail("submit/restart forbidden")
    monkeypatch.setattr(drill, "AccountClient", forbidden)
    monkeypatch.setattr(drill, "run_restart", forbidden)
    monkeypatch.setattr(drill.request, "build_opener", lambda *a: Opener())
    assert drill.main(["--resume", str(evidence_path), "--account-token", "synthetic-session"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["observation_passed"] is True
    assert not report["acceptance_passed"] and not report["settlement_evidence_passed"]
    assert len(requests) == 1 and path.read_bytes() == before


@pytest.mark.parametrize("raw_status", [None, "", "   ", "corrupt-status", 42, b"running"])
def test_corrupt_status_is_never_projected_as_pending_or_terminal(tmp_path, monkeypatch, raw_status):
    client, path, before, _, _, _ = setup_observation(tmp_path, monkeypatch, raw_status=raw_status)
    response = client.get(f"/v2/jobs/{JOB}/observation", headers={"Authorization": "Bearer synthetic-session"})
    assert response.status_code == 503
    assert response.json == {"error": "invalid_observation_state"}
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["Vary"] == "Authorization"
    assert path.read_bytes() == before


@pytest.mark.parametrize("raw_status,canonical", [
    *[(value, value) for value in account_observation.platform_v1.CANONICAL_JOB_STATES],
    *account_observation.platform_v1.LEGACY_STATE_MAP.items(),
    (" RUNNING ", "running"),
])
def test_recognized_status_is_projected_without_mutation(tmp_path, monkeypatch, raw_status, canonical):
    client, path, before, _, _, _ = setup_observation(tmp_path, monkeypatch, raw_status=raw_status)
    response = client.get(f"/v2/jobs/{JOB}/observation", headers={"Authorization": "Bearer synthetic-session"})
    assert response.status_code == 200
    assert response.json["status"] == canonical
    assert path.read_bytes() == before

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import account_restart_recovery_drill as drill  # type: ignore


def test_dry_run_redacts_account_token_and_prompt() -> None:
    script = ROOT / "scripts" / "account_restart_recovery_drill.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--account-token",
            "secret-token",
            "--prompt",
            "private prompt",
            "--restart-command",
            "systemctl restart havnai-coordinator.service",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema"] == "havn-45-account-restart-recovery-drill-plan.v1"
    assert payload["execute"] is False
    assert payload["acceptance_passed"] is False
    assert payload["has_account_token"] is True
    assert payload["account_token_hash"].startswith("sha256:")
    assert "secret-token" not in completed.stdout
    assert "private prompt" not in completed.stdout


def test_execute_requires_account_token() -> None:
    script = ROOT / "scripts" / "account_restart_recovery_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--execute", "--restart-command", "true"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires --account-token" in completed.stderr


def test_execute_requires_restart_command() -> None:
    script = ROOT / "scripts" / "account_restart_recovery_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--execute", "--account-token", "secret-token"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires --restart-command" in completed.stderr


def test_preflight_requires_account_token() -> None:
    script = ROOT / "scripts" / "account_restart_recovery_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--preflight"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    decoder = json.JSONDecoder()
    output_chunks: list[dict[str, object]] = []
    position = 0
    while position < len(completed.stdout):
        stripped = completed.stdout[position:].lstrip()
        if not stripped:
            break
        skipped = len(completed.stdout[position:]) - len(stripped)
        chunk, offset = decoder.raw_decode(stripped)
        output_chunks.append(chunk)
        position += skipped + offset
    preflight = output_chunks[-1]
    assert preflight["schema"] == "havn-45-account-restart-recovery-drill-preflight.v1"
    assert preflight["account_token_present"] is False
    assert preflight["passed"] is False
    assert preflight["blockers"] == [
        "HAVNAI_DRILL_ACCOUNT_TOKEN is not configured; funded private account token required."
    ]


def test_preflight_reports_funded_account_without_printing_token(monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self, base_url: str, account_token: str) -> None:
            assert base_url == "https://api.example.test"
            assert account_token == "secret-token"

        def account_preflight(self) -> dict[str, dict[str, object]]:
            return {
                "account": {"id": "acct-secret", "status": "active"},
                "credits": {"available_units": 1200, "reserved_units": 50},
            }

    monkeypatch.setattr(drill, "AccountClient", FakeClient)
    decoder = json.JSONDecoder()
    exit_code = drill.main([
        "--preflight",
        "--base-url",
        "https://api.example.test",
        "--account-token",
        "secret-token",
        "--restart-command",
        "systemctl restart havnai-coordinator.service",
    ])
    stdout = capsys.readouterr().out
    output_chunks: list[dict[str, object]] = []
    position = 0
    while position < len(stdout):
        stripped = stdout[position:].lstrip()
        if not stripped:
            break
        skipped = len(stdout[position:]) - len(stripped)
        chunk, offset = decoder.raw_decode(stripped)
        output_chunks.append(chunk)
        position += skipped + offset

    assert exit_code == 0
    plan, preflight = output_chunks
    assert plan["schema"] == "havn-45-account-restart-recovery-drill-plan.v1"
    assert preflight["schema"] == "havn-45-account-restart-recovery-drill-preflight.v1"
    assert preflight["account_id_hash"]
    assert preflight["account_present"] is True
    assert preflight["account_status"] == "active"
    assert preflight["available_units"] == 1200
    assert preflight["reserved_units"] == 50
    assert preflight["restart_command_configured"] is True
    assert preflight["passed"] is True
    assert preflight["acceptance_passed"] is False
    assert "secret-token" not in json.dumps(output_chunks)
    assert "acct-secret" not in json.dumps(output_chunks)


def test_readonly_db_checks_report_duplicate_charge_like_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT, owner_account_id TEXT);
            CREATE TABLE job_settlement (
                job_id TEXT PRIMARY KEY,
                attempt_count INTEGER,
                execution_status TEXT,
                settlement_outcome TEXT
            );
            CREATE TABLE account_credit_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                event_type TEXT,
                amount INTEGER,
                reason TEXT,
                purchase_id TEXT
            );
            CREATE TABLE account_credit_reservations (job_id TEXT);
            CREATE TABLE account_payment_receipts (purchase_id TEXT);
            CREATE TABLE node_payouts (job_id TEXT);
            """
        )
        conn.execute("INSERT INTO jobs VALUES ('job-1', 'succeeded', 'acct-secret')")
        conn.execute("INSERT INTO job_settlement VALUES ('job-1', 2, 'completed', 'captured')")
        conn.execute("INSERT INTO account_credit_ledger (job_id, event_type, amount, reason, purchase_id) VALUES "
                     "('job-1', 'capture', -10, 'generation', 'pur-1')")
        conn.execute("INSERT INTO account_credit_ledger (job_id, event_type, amount, reason, purchase_id) VALUES "
                     "('job-1', 'capture', -10, 'generation', 'pur-1')")
        conn.execute("INSERT INTO account_payment_receipts VALUES ('pur-1')")
        conn.execute("INSERT INTO node_payouts VALUES ('job-1')")
        conn.commit()

    checks = drill.readonly_db_checks(db_path, "job-1")

    assert checks["job"]["status"] == "succeeded"
    assert checks["job"]["owner_account_hash"]
    assert "owner_account_id" not in checks["job"]
    assert checks["settlement"]["attempt_count"] == 2
    assert checks["receipt_count"] == 1
    assert checks["node_payout_count"] == 1
    assert checks["charge_like_ledger_rows"] == 2
    assert checks["duplicate_charge_like_rows"] is True


def test_readonly_db_checks_report_single_charge_like_row_as_not_duplicate(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT, owner_account_id TEXT);
            CREATE TABLE job_settlement (
                job_id TEXT PRIMARY KEY,
                attempt_count INTEGER,
                execution_status TEXT,
                settlement_outcome TEXT
            );
            CREATE TABLE account_credit_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                event_type TEXT,
                amount INTEGER,
                reason TEXT,
                purchase_id TEXT
            );
            CREATE TABLE account_credit_reservations (job_id TEXT);
            CREATE TABLE account_payment_receipts (purchase_id TEXT);
            CREATE TABLE node_payouts (job_id TEXT);
            """
        )
        conn.execute("INSERT INTO jobs VALUES ('job-1', 'succeeded', 'acct-secret')")
        conn.execute("INSERT INTO job_settlement VALUES ('job-1', 1, 'completed', 'captured')")
        conn.execute("INSERT INTO account_credit_ledger (job_id, event_type, amount, reason, purchase_id) VALUES "
                     "('job-1', 'capture', -10, 'generation', 'pur-1')")
        conn.execute("INSERT INTO account_credit_ledger (job_id, event_type, amount, reason, purchase_id) VALUES "
                     "('job-1', 'release', 0, 'reservation_release', 'pur-1')")
        conn.execute("INSERT INTO account_payment_receipts VALUES ('pur-1')")
        conn.execute("INSERT INTO node_payouts VALUES ('job-1')")
        conn.commit()

    checks = drill.readonly_db_checks(db_path, "job-1")

    assert checks["settlement"]["attempt_count"] == 1
    assert checks["receipt_count"] == 1
    assert checks["node_payout_count"] == 1
    assert checks["charge_like_ledger_rows"] == 1
    assert checks["duplicate_charge_like_rows"] is False
    assert checks["acceptance_eligible"] is False
    assert drill.evidence_errors(checks, "job-1", completed=True)


def current_fixture(path: Path) -> None:
    # Match account_ledger.initialize/account_payments.initialize and rewards;
    # omit uniqueness constraints so corrupt/duplicate evidence can be exercised.
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE jobs(id TEXT,status TEXT,owner_account_id TEXT);
        CREATE TABLE account_credit_ledger(id INTEGER PRIMARY KEY,account_id TEXT,operation TEXT,
            resource_id TEXT,settled_delta INTEGER,reserved_delta INTEGER);
        CREATE TABLE account_credit_reservations(job_id TEXT,account_id TEXT,units INTEGER,state TEXT);
        CREATE TABLE account_payment_receipts(account_id TEXT,purchase_id TEXT,payment_id TEXT);
        CREATE TABLE node_payouts(job_id TEXT);
        CREATE TABLE rewards(task_id TEXT);
        INSERT INTO jobs VALUES('job-1','running','acct-test');
        INSERT INTO account_credit_reservations VALUES('job-1','acct-test',10,'reserved');
        INSERT INTO account_credit_ledger VALUES(1,'acct-test','reserve','job-1',0,10);
        """)


def finish_fixture(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        UPDATE jobs SET status='succeeded';
        UPDATE account_credit_reservations SET state='captured';
        INSERT INTO account_credit_ledger VALUES(2,'acct-test','capture','job-1',-10,-10);
        """)


@pytest.mark.parametrize("payout_count,reward_count", [(0, 0), (1, 1)])
def test_current_account_contract_allows_non_applicable_legacy_counts(tmp_path, payout_count, reward_count):
    path = tmp_path / "current.sqlite"
    current_fixture(path)
    assert drill.evidence_errors(drill.readonly_db_checks(path, "job-1"), "job-1", completed=False) == []
    finish_fixture(path)
    with sqlite3.connect(path) as conn:
        for _ in range(payout_count): conn.execute("INSERT INTO node_payouts VALUES('job-1')")
        for _ in range(reward_count): conn.execute("INSERT INTO rewards VALUES('job-1')")
        # Two different funding purchases are legitimate, not two job receipts.
        conn.executescript("""
        INSERT INTO account_payment_receipts VALUES('acct-test','purchase-1','payment-1');
        INSERT INTO account_payment_receipts VALUES('acct-test','purchase-2','payment-2');
        """)
    before = path.read_bytes()
    checks = drill.readonly_db_checks(path, "job-1")
    assert path.read_bytes() == before
    assert drill.evidence_errors(checks, "job-1", completed=True) == []
    assert checks["receipt_applicability"].startswith("funding purchases only")


@pytest.mark.parametrize("mutation", [
    "INSERT INTO account_credit_ledger VALUES(3,'acct-test','capture','job-1',-10,-10)",
    "INSERT INTO account_credit_ledger VALUES(3,'acct-test','charge','job-1',-10,0)",
    "INSERT INTO account_credit_ledger VALUES(3,'acct-test','debit','job-1',-10,0)",
    "INSERT INTO node_payouts VALUES('job-1'); INSERT INTO node_payouts VALUES('job-1')",
    "INSERT INTO rewards VALUES('job-1'); INSERT INTO rewards VALUES('job-1')",
    "INSERT INTO account_payment_receipts VALUES('acct-test','p','a'); INSERT INTO account_payment_receipts VALUES('acct-test','p','b')",
    "INSERT INTO account_payment_receipts VALUES('acct-test','p','a'); INSERT INTO account_payment_receipts VALUES('acct-test','q','a')",
    "UPDATE account_credit_reservations SET state='reserved'",
    "UPDATE account_credit_ledger SET settled_delta=-9 WHERE operation='capture'",
    "UPDATE account_credit_ledger SET account_id='other' WHERE operation='capture'",
    "DELETE FROM account_credit_ledger WHERE operation='capture'",
    "DELETE FROM account_credit_reservations",
    "DELETE FROM jobs",
    "UPDATE jobs SET status='failed'",
])
def test_current_evidence_rejects_invalid_settlement(tmp_path, mutation):
    path = tmp_path / "current.sqlite"
    current_fixture(path)
    finish_fixture(path)
    with sqlite3.connect(path) as conn: conn.executescript(mutation)
    assert drill.evidence_errors(drill.readonly_db_checks(path, "job-1"), "job-1", completed=True)


@pytest.mark.parametrize("status", ["queued", "leased", "running", "uploading"])
def test_accepted_states_require_exact_job_and_owner(status):
    owner = drill.account_hash("acct-test")
    state = {"id": "job-1", "status": status, "owner_account_id": "acct-test"}
    assert drill.accepted_state_errors(state, "job-1", owner) == []
    assert drill.accepted_state_errors({**state, "id": "other"}, "job-1", owner)
    assert drill.accepted_state_errors({**state, "owner_account_id": "other"}, "job-1", owner)


@pytest.mark.parametrize("initial_status,mutation", [
    ("running", ""), ("succeeded", ""), ("failed", ""), ("mystery", ""), (None, ""),
    ("running", "INSERT INTO node_payouts VALUES('job-1'); INSERT INTO node_payouts VALUES('job-1')"),
    ("running", "INSERT INTO rewards VALUES('job-1'); INSERT INTO rewards VALUES('job-1')"),
    ("running", "INSERT INTO account_payment_receipts VALUES('acct-test','p','a'); INSERT INTO account_payment_receipts VALUES('acct-test','p','b')"),
    ("running", "UPDATE jobs SET owner_account_id='other'; UPDATE account_credit_reservations SET account_id='other'; UPDATE account_credit_ledger SET account_id='other'"),
    ("running", "UPDATE account_credit_reservations SET units=20; UPDATE account_credit_ledger SET reserved_delta=20 WHERE operation='reserve'; UPDATE account_credit_ledger SET reserved_delta=-20,settled_delta=-20 WHERE operation='capture'"),
])
def test_disposable_execute_only_restarts_nonterminal_job(monkeypatch, capsys, tmp_path, initial_status, mutation):
    path = tmp_path / "current.sqlite"
    current_fixture(path)
    events = []

    class FakeClient:
        def __init__(self, *args): pass
        def submit(self, payload, **kwargs):
            events.append("submit")
            return "job-1"
        def job(self, job_id):
            events.append("observe")
            return {"id": job_id, "status": "succeeded" if "restart" in events else initial_status,
                    "owner_account_id": "acct-test"}

    def restart(*args):
        events.append("restart")
        # The accepted observation and marker must precede the restart callback.
        assert events[-2] == "observe"
        finish_fixture(path)
        with sqlite3.connect(path) as conn: conn.executescript(mutation)
        return {"returncode": 0}

    monkeypatch.setattr(drill, "AccountClient", FakeClient)
    monkeypatch.setattr(drill, "run_restart", restart)
    monkeypatch.setattr(drill.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    code = drill.main(["--execute", "--account-token", "synthetic", "--restart-command", "SYNTHETIC",
                       "--db-path", str(path), "--poll-seconds", "0"])
    output = capsys.readouterr().out
    if initial_status == "running" and not mutation:
        assert code == 2
        assert '"settlement_evidence_passed": true' in output
        assert '"acceptance_passed": true' not in output
        assert '"passed": true' not in output
        assert "independent service/worker restart verification" in output
        assert "authenticated owner-access verification" in output
        assert output.index('"event": "accepted_state"') < output.index('"event": "restart_marker"')
    elif initial_status == "running":
        assert code == 2 and '"acceptance_passed": false' in output
        assert '"settlement_evidence_passed": true' not in output
        assert "restart" in events
    else:
        assert code == 2 and "restart" not in events
        assert '"settlement_evidence_passed": true' not in output
        assert '"event": "restart_marker"' not in output


def test_execute_requires_usable_current_db_before_submission(monkeypatch, tmp_path):
    monkeypatch.setattr(drill, "AccountClient", lambda *args: pytest.fail("must not submit"))
    args = ["--execute", "--account-token", "synthetic", "--restart-command", "SYNTHETIC"]
    monkeypatch.delenv("HAVNAI_DB_PATH", raising=False)
    with pytest.raises(SystemExit, match="requires --db-path"):
        drill.main(args)
    with pytest.raises(sqlite3.OperationalError):
        drill.main(args + ["--db-path", str(tmp_path / "missing.sqlite")])
    path = tmp_path / "legacy.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="current read-only evidence schema"):
        drill.main(args + ["--db-path", str(path)])


def test_historical_evidence_is_never_accepting():
    assert drill.evidence_errors({"acceptance_eligible": False}, "job-1", completed=True)


def resume_evidence():
    return {"schema": drill.RESUME_SCHEMA, "base_url": "https://api.joinhavn.io", "job_id": "job-11111111111111111111111111111111",
            "owner_account_hash": drill.account_hash("acct-original"), "idempotency_key": "original-key",
            "restart_phase": "restart_returned"}


@pytest.fixture
def observation_only(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("submission/restart/process forbidden in resume")
    monkeypatch.setattr(drill, "AccountClient", forbidden)
    monkeypatch.setattr(drill, "run_restart", forbidden)
    monkeypatch.setattr(drill.subprocess, "run", forbidden)


def mock_observer_http(monkeypatch, response=None, http_error=None):
    calls = []

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def geturl(self): return "https://api.joinhavn.io/v2/jobs/job-11111111111111111111111111111111/observation"
        def read(self, size): return json.dumps(response).encode()

    class Opener:
        def open(self, req, timeout):
            assert req.get_method() == "GET"
            assert req.full_url == "https://api.joinhavn.io/v2/jobs/job-11111111111111111111111111111111/observation"
            assert req.get_header("Authorization") == "Bearer synthetic-fresh-session"
            assert req.data is None and req.get_header("Idempotency-key") is None
            calls.append(req.full_url)
            if http_error:
                import io
                raise drill.error.HTTPError(req.full_url, http_error, "untrusted-message", {},
                                            io.BytesIO(b'private-error-body synthetic-fresh-session'))
            return Response()

    def build_opener(handler):
        assert isinstance(handler, drill.NoCredentialRedirects)
        return Opener()
    monkeypatch.setattr(drill.request, "build_opener", build_opener)
    return calls


@pytest.mark.parametrize("status", ["running", "succeeded", "failed"])
@pytest.mark.parametrize("phase", ["not_restarted", "restart_started", "restart_returned", "restart_failed"])
def test_resume_same_owner_is_one_get_never_acceptance(tmp_path, monkeypatch, capsys, observation_only, status, phase):
    evidence = resume_evidence(); evidence["restart_phase"] = phase
    path = tmp_path / "resume.json"; path.write_text(json.dumps(evidence))
    calls = mock_observer_http(monkeypatch, {"id": "job-11111111111111111111111111111111", "owner_account_hash": drill.account_hash("acct-original"),
                                             "status": status})
    assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["observation_passed"] is True
    assert report["acceptance_passed"] is False and report["passed"] is False
    assert report["settlement_evidence_passed"] is False
    assert report["original_evidence"] == evidence and report["status"] == status
    assert len(calls) == 1
    for secret in ("synthetic-fresh-session", "acct-original", "private prompt", "private URL"):
        assert secret not in json.dumps(report)


@pytest.mark.parametrize("http_error", [401, 403, 404, 302, 307, 500])
def test_resume_auth_errors_and_redirects_are_sanitized(tmp_path, monkeypatch, capsys, observation_only, http_error):
    path = tmp_path / "resume.json"; path.write_text(json.dumps(resume_evidence()))
    calls = mock_observer_http(monkeypatch, http_error=http_error)
    assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session"]) == 2
    output = capsys.readouterr().out; report = json.loads(output)
    assert report["observation_passed"] is False and report["acceptance_passed"] is False
    assert len(calls) == 1
    assert "private-error-body" not in output and "synthetic-fresh-session" not in output
    assert "untrusted-message" not in output
    if http_error == 401: assert report["error"] == "authentication_expired_or_invalid"
    if 300 <= http_error < 400: assert report["error"] == "redirect_rejected"


@pytest.mark.parametrize("mutation", [
    {"owner_account_hash": drill.account_hash("acct-other")}, {"owner_account_hash": None}, {"id": "job-other"}, {"status": "unknown"},
    {"status": []}, {"status": {}}, {"status": 1}, {"status": None}, {"status": True},
])
def test_resume_rejects_wrong_job_or_owner(tmp_path, monkeypatch, capsys, observation_only, mutation):
    path = tmp_path / "resume.json"; path.write_text(json.dumps(resume_evidence()))
    mock_observer_http(monkeypatch, {"id": "job-11111111111111111111111111111111", "owner_account_hash": drill.account_hash("acct-original"), "status": "running", **mutation})
    assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session"]) == 2
    assert json.loads(capsys.readouterr().out)["observation_passed"] is False


@pytest.mark.parametrize("mutation", [
    {"owner_account_hash": ""}, {"restart_phase": "unknown"}, {"job_id": "../other"},
    {"base_url": "https://untrusted.invalid"}, {"token": "must-not-be-emitted"}, {"idempotency_key": "bad\nkey"},
    {"job_id": "."}, {"job_id": ".."}, {"job_id": "job-1"}, {"job_id": "job-" + "A" * 32},
    {"job_id": "job-" + "1" * 32 + "%2Fother"},
])
def test_resume_invalid_evidence_never_requests(tmp_path, monkeypatch, capsys, observation_only, mutation):
    path = tmp_path / "resume.json"; path.write_text(json.dumps({**resume_evidence(), **mutation}))
    monkeypatch.setattr(drill.request, "build_opener", lambda *a: pytest.fail("no HTTP for invalid evidence"))
    assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session"]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error"] == "invalid_resume_evidence"
    assert "must-not-be-emitted" not in output


@pytest.mark.parametrize("extra", [["--execute"], ["--preflight"]])
def test_resume_conflicting_flags_never_execute(tmp_path, monkeypatch, capsys, observation_only, extra):
    path = tmp_path / "resume.json"; path.write_text(json.dumps(resume_evidence()))
    monkeypatch.setattr(drill.request, "build_opener", lambda *a: pytest.fail("no HTTP for conflicting flags"))
    assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session", *extra]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "resume_conflicts_with_execution_or_preflight"


def test_redirect_handler_never_forwards_credentials():
    req = drill.request.Request("https://api.joinhavn.io/v2/jobs/job-11111111111111111111111111111111/observation", headers={"Authorization": "Bearer synthetic"})
    with pytest.raises(drill.ObservationError, match="redirect_rejected"):
        drill.NoCredentialRedirects().redirect_request(req, None, 302, "found", {}, "https://untrusted.invalid")


def test_execute_emits_only_sanitized_resume_record(capsys):
    args = drill.parse_args(["--account-token", "synthetic-secret", "--idempotency-key", "original-key"])
    for phase in drill.RESTART_PHASES:
        drill.emit_resume_evidence(args, "job-11111111111111111111111111111111", drill.account_hash("acct-original"), phase)
    output = capsys.readouterr().out
    assert "synthetic-secret" not in output and "acct-original" not in output
    for line in output.splitlines():
        record = json.loads(line)
        assert drill.validate_resume_evidence(record["evidence"]) == record["evidence"]


@pytest.mark.parametrize("before_restart", [True, False])
def test_execute_expiry_preserves_original_job_key_and_phase(tmp_path, monkeypatch, capsys, before_restart):
    path = tmp_path / "current.sqlite"; current_fixture(path)
    job_id = "job-" + "1" * 32
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE jobs SET id=?", (job_id,))
        conn.execute("UPDATE account_credit_reservations SET job_id=?", (job_id,))
        conn.execute("UPDATE account_credit_ledger SET resource_id=?", (job_id,))
    events = []
    class Client:
        def __init__(self, *args): pass
        def submit(self, *args, **kwargs): events.append("submit"); return job_id
        def job(self, job_id):
            if before_restart or "restart" in events: raise RuntimeError("account_request_http_401")
            return {"id": job_id, "status": "running", "owner_account_id": "acct-test"}
    def restart(*args): events.append("restart"); return {"returncode": 0}
    monkeypatch.setattr(drill, "AccountClient", Client)
    monkeypatch.setattr(drill, "run_restart", restart)
    with pytest.raises(RuntimeError, match="account_request_http_401"):
        drill.main(["--execute", "--account-token", "synthetic-secret", "--restart-command", "SYNTHETIC",
                    "--db-path", str(path), "--idempotency-key", "known-original-key"])
    output = capsys.readouterr().out
    # Plan is pretty-printed; resume records are individual flushed JSON lines.
    records = [json.loads(line)["evidence"] for line in output.splitlines() if '"event": "resume_evidence"' in line]
    assert records[-1]["restart_phase"] == ("not_restarted" if before_restart else "restart_returned")
    assert records[-1]["job_id"] == job_id and records[-1]["idempotency_key"] == "known-original-key"
    assert events.count("submit") == 1 and events.count("restart") == int(not before_restart)
    assert "synthetic-secret" not in output


def test_account_http_error_never_emits_body(monkeypatch):
    def rejected(req, **kwargs):
        import io
        raise drill.error.HTTPError(req.full_url, 401, "untrusted", {}, io.BytesIO(b"secret response body"))
    monkeypatch.setattr(drill.request, "urlopen", rejected)
    with pytest.raises(RuntimeError, match="^account_request_http_401$"):
        drill.AccountClient("https://api.joinhavn.io", "synthetic").job("job-11111111111111111111111111111111")


def test_resume_missing_session_and_incomplete_file_fail_closed(tmp_path, monkeypatch, capsys, observation_only):
    monkeypatch.delenv("HAVNAI_DRILL_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setattr(drill.request, "build_opener", lambda *a: pytest.fail("no HTTP"))
    path = tmp_path / "resume.json"; path.write_text(json.dumps(resume_evidence()))
    assert drill.main(["--resume", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "fresh_account_session_required"
    for value in ('{', '{}', ' ' * 16385):
        path.write_text(value)
        assert drill.main(["--resume", str(path), "--account-token", "synthetic-fresh-session"]) == 2
        assert json.loads(capsys.readouterr().out)["error"] == "invalid_resume_evidence"

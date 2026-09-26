from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


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

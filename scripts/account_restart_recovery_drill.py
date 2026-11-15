"""Run the HAVN-45/HAVN-49 private accepted-work restart drill.

The drill submits one account-owned generation through ``/v2/jobs``, executes an
operator-provided restart command while the job is accepted, polls the private
account job route to a terminal state, and requires read-only SQLite evidence
for settlement evidence. Independent restart and recovered-artifact verification
are not supported, so full acceptance always remains pending.

By default this script prints a dry-run plan and performs no network writes.
Live execution requires an account bearer token and an explicit restart command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_TIMEOUT_SECONDS = 60 * 60
TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled", "expired"}
ACCEPTED_STATUSES = {"queued", "leased", "running", "uploading"}
CURRENT_COLUMNS = {
    "jobs": {"id", "status", "owner_account_id"},
    "account_credit_ledger": {"id", "account_id", "operation", "resource_id", "settled_delta", "reserved_delta"},
    "account_credit_reservations": {"job_id", "account_id", "units", "state"},
    "account_payment_receipts": {"account_id", "purchase_id", "payment_id"},
    "node_payouts": {"job_id"},
    "rewards": {"task_id"},
}
DEFAULT_PROMPT = (
    "HAVN-45 private restart recovery drill: polished commercial-quality HavnAI "
    "operations scene, coherent lighting, premium product-ready composition"
)
MISSING_ACCEPTANCE_EVIDENCE = [
    "independent service/worker restart verification (an exit-zero command is insufficient)",
    "recovered artifact ownership and authenticated owner-access verification",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def redact_token(value: str) -> str:
    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def account_hash(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def result_status(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").strip().lower()
    if status:
        return status
    if result.get("output_url") or result.get("image_url") or result.get("video_url") or result.get("audio_url"):
        return "succeeded"
    if result.get("error"):
        return "failed"
    return "pending"


class AccountClient:
    def __init__(self, base_url: str, account_token: str) -> None:
        self.base_url = clean_base_url(base_url)
        self.account_token = account_token.strip()
        if not self.account_token:
            raise ValueError("account token required")

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None,
              idempotency_key: str | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.account_token}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(self.base_url + path, data=body, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail[:500]}") from exc
        return json.loads(data or "{}")

    def submit(self, payload: dict[str, Any], *, idempotency_key: str) -> str:
        response = self._json("POST", "/v2/jobs", payload, idempotency_key=idempotency_key)
        job_id = str(response.get("id") or "").strip()
        if not job_id:
            raise RuntimeError(f"account submit response did not include id: {response}")
        return job_id

    def job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v2/jobs/{parse.quote(job_id)}")

    def account_preflight(self) -> dict[str, Any]:
        account = self._json("GET", "/v2/account")
        credits = self._json("GET", "/v2/account/credits")
        return {"account": account, "credits": credits}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": args.job_type,
        "model": args.model,
        "prompt": args.prompt,
        "seed": args.seed,
        "sfw_mode": True,
    }
    if args.job_type == "image":
        payload.update({"steps": args.steps, "width": args.width, "height": args.height, "guidance": args.guidance})
    elif args.job_type == "text_to_video":
        payload.update({"steps": args.video_steps, "frames": args.frames, "fps": args.fps,
                        "duration_seconds": args.duration_seconds})
    elif args.job_type == "text_to_music":
        payload.update({"style": args.music_style, "instrumental": True, "duration": args.music_duration})
    return payload


def run_restart(command: str, timeout_seconds: float) -> dict[str, Any]:
    started = utc_now()
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "command": command,
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1200:],
        "stderr_tail": completed.stderr[-1200:],
    }


def readonly_db_checks(db_path: Path, job_id: str) -> dict[str, Any]:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")  # one consistent read-only snapshot
        if "operation" in {row[1] for row in conn.execute("PRAGMA table_info(account_credit_ledger)")}:
            return current_db_checks(conn, job_id)
        job = conn.execute("SELECT id, status, owner_account_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        settlement = conn.execute(
            "SELECT job_id, attempt_count, execution_status, settlement_outcome "
            "FROM job_settlement WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        ledger = conn.execute(
            "SELECT event_type, amount, reason FROM account_credit_ledger WHERE job_id = ? ORDER BY id",
            (job_id,),
        ).fetchall()
        reservations = conn.execute(
            "SELECT COUNT(*) AS count FROM account_credit_reservations WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        receipts = conn.execute(
            "SELECT COUNT(*) AS count FROM account_payment_receipts WHERE purchase_id IN ("
            "SELECT purchase_id FROM account_credit_ledger WHERE job_id = ?)",
            (job_id,),
        ).fetchone()
        payouts = conn.execute("SELECT COUNT(*) AS count FROM node_payouts WHERE job_id = ?", (job_id,)).fetchone()

    ledger_rows = [dict(row) for row in ledger]
    charge_rows = [row for row in ledger_rows if row.get("event_type") in {"capture", "charge", "debit"}]
    job_row = dict(job) if job else None
    if job_row:
        owner_account_id = str(job_row.pop("owner_account_id") or "")
        job_row["owner_account_hash"] = account_hash(owner_account_id)
    return {
        "acceptance_eligible": False,
        "evidence_mode": "legacy-schema-only",
        "job": job_row,
        "settlement": dict(settlement) if settlement else None,
        "ledger": ledger_rows,
        "reservation_count": int(reservations["count"] if reservations else 0),
        "receipt_count": int(receipts["count"] if receipts else 0),
        "node_payout_count": int(payouts["count"] if payouts else 0),
        "charge_like_ledger_rows": len(charge_rows),
        "duplicate_charge_like_rows": len(charge_rows) > 1,
    }


def require_current_schema(conn: sqlite3.Connection) -> None:
    for table, required in CURRENT_COLUMNS.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not required <= columns:
            raise ValueError(f"acceptance requires current read-only evidence schema: {table}")


def current_db_checks(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    require_current_schema(conn)
    job = conn.execute("SELECT id,status,owner_account_id FROM jobs WHERE id=?", (job_id,)).fetchone()
    owner = job["owner_account_id"] if job else None
    ledger = [dict(row) for row in conn.execute(
        "SELECT account_id,operation,settled_delta,reserved_delta FROM account_credit_ledger WHERE resource_id=? ORDER BY id",
        (job_id,))]
    reservations = [dict(row) for row in conn.execute(
        "SELECT account_id,units,state FROM account_credit_reservations WHERE job_id=?", (job_id,))]
    # Payment receipts are funding-purchase receipts, not generation receipts.
    # Check duplicate groups for the owner; never invent a job/purchase join.
    receipt_duplicates = {}
    for key in ("purchase_id", "payment_id"):
        receipt_duplicates[key] = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT {key} FROM account_payment_receipts WHERE account_id=? GROUP BY {key} HAVING COUNT(*)>1)",
            (owner,)).fetchone()[0]
    result = {
        "acceptance_eligible": True,
        "job": {"id": job["id"], "status": job["status"], "owner_account_hash": account_hash(owner)} if job else None,
        "ledger": [{key: value for key, value in row.items() if key != "account_id"} for row in ledger],
        "reservation_count": len(reservations),
        "reservation_units": reservations[0]["units"] if len(reservations) == 1 else None,
        "reservation_state": reservations[0]["state"] if len(reservations) == 1 else None,
        "owner_matches": bool(owner) and all(row["account_id"] == owner for row in ledger + reservations),
        "receipt_applicability": "funding purchases only; zero per-job receipts is expected",
        "duplicate_receipt_groups": receipt_duplicates,
        "node_payout_count": conn.execute("SELECT COUNT(*) FROM node_payouts WHERE job_id=?", (job_id,)).fetchone()[0],
        "reward_count": conn.execute("SELECT COUNT(*) FROM rewards WHERE task_id=?", (job_id,)).fetchone()[0],
    }
    return result


def evidence_errors(checks: dict[str, Any], job_id: str, *, completed: bool) -> list[str]:
    errors = []
    if not checks.get("acceptance_eligible"):
        return ["current read-only account-ledger evidence required"]
    job = checks.get("job") or {}
    if job.get("id") != job_id or not job.get("owner_account_hash") or not checks.get("owner_matches"):
        errors.append("job/reservation/ledger ownership mismatch or missing job")
    if job.get("status") not in ({"succeeded", "completed", "success", "done"} if completed else ACCEPTED_STATUSES):
        errors.append("database job is not in the required state")
    units = checks.get("reservation_units")
    if checks.get("reservation_count") != 1 or not isinstance(units, int) or units <= 0:
        errors.append("exactly one positive account reservation required")
    if checks.get("reservation_state") != ("captured" if completed else "reserved"):
        errors.append("reservation not in required state")
    ledger = checks.get("ledger", [])
    reserve = [row for row in ledger if row.get("operation") == "reserve"]
    capture = [row for row in ledger if row.get("operation") in {"capture", "charge", "debit"}]
    if len(reserve) != 1 or reserve[0].get("settled_delta") != 0 or reserve[0].get("reserved_delta") != units:
        errors.append("exactly one matching reserve entry required")
    if len(capture) != int(completed):
        errors.append("unexpected capture/charge/debit count")
    elif completed and isinstance(units, int) and (capture[0].get("settled_delta") != -units or capture[0].get("reserved_delta") != -units):
        errors.append("capture amounts do not match reservation")
    if any(row.get("operation") not in {"reserve", "capture"} for row in ledger):
        errors.append("unexpected generation ledger operation")
    if any(checks.get("duplicate_receipt_groups", {}).values()):
        errors.append("duplicate funding receipt")
    # v2 account jobs need not create legacy rewards/payouts. If present,
    # settlement.py and rewards.record_reward permit at most one per job.
    for field in ("node_payout_count", "reward_count"):
        if checks.get(field) not in ({0, 1} if completed else {0}):
            errors.append(f"unexpected {field}")
    return errors


def accepted_state_errors(state: dict[str, Any], job_id: str, owner_hash: str | None) -> list[str]:
    if (state.get("id") != job_id or state.get("status") not in ACCEPTED_STATUSES
            or not owner_hash or account_hash(state.get("owner_account_id")) != owner_hash):
        return ["same account-owned job must be explicitly nonterminal immediately before restart"]
    return []


def summarize(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "havn-45-account-restart-recovery-drill-plan.v1",
        "generated_at": utc_now(),
        "base_url": clean_base_url(args.base_url),
        "execute": bool(args.execute),
        "acceptance_passed": False,
        "settlement_evidence_passed": False,
        "missing_acceptance_evidence": MISSING_ACCEPTANCE_EVIDENCE,
        "job_type": args.job_type,
        "model": args.model,
        "idempotency_key": args.idempotency_key,
        "has_account_token": bool(args.account_token.strip()),
        "account_token_hash": redact_token(args.account_token.strip()),
        "restart_command_configured": bool(args.restart_command.strip()),
        "db_checks_configured": bool(args.db_path),
        "payload": {key: value for key, value in payload.items() if key != "prompt"},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--account-token", default=os.environ.get("HAVNAI_DRILL_ACCOUNT_TOKEN", ""),
                        help="commercial account bearer token for private /v2/jobs drill")
    parser.add_argument("--restart-command", default=os.environ.get("HAVNAI_RESTART_DRILL_COMMAND", ""),
                        help="operator command to run after job acceptance, for example: sudo systemctl restart havnai-coordinator.service")
    parser.add_argument("--restart-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--db-path", default=os.environ.get("HAVNAI_DB_PATH", ""),
                        help="required current coordinator SQLite DB for full read-only acceptance evidence")
    parser.add_argument("--idempotency-key", default=os.environ.get("HAVNAI_RESTART_DRILL_IDEMPOTENCY_KEY",
                                                                    f"havn-45-{int(time.time())}"))
    parser.add_argument("--job-type", choices=["image", "text_to_video", "text_to_music"], default="image")
    parser.add_argument("--model", default=os.environ.get("HAVNAI_RESTART_DRILL_MODEL", "juggernautxl_ragnarokby"))
    parser.add_argument("--prompt", default=os.environ.get("HAVNAI_RESTART_DRILL_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--seed", type=int, default=45049)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--guidance", type=float, default=6.5)
    parser.add_argument("--video-steps", type=int, default=4)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--duration-seconds", type=int, default=3)
    parser.add_argument("--music-style", default="cinematic synthwave")
    parser.add_argument("--music-duration", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--preflight", action="store_true",
                        help="check private account identity and credit balance without submitting a job")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    print(json.dumps(summarize(args, payload), indent=2, sort_keys=True), flush=True)
    if args.preflight:
        if not args.account_token.strip():
            print(json.dumps({
                "schema": "havn-45-account-restart-recovery-drill-preflight.v1",
                "generated_at": utc_now(),
                "base_url": clean_base_url(args.base_url),
                "account_token_present": False,
                "account_id_hash": None,
                "account_present": False,
                "account_status": None,
                "available_units": None,
                "reserved_units": None,
                "restart_command_configured": bool(args.restart_command.strip()),
                "passed": False,
                "acceptance_passed": False,
                "settlement_evidence_passed": False,
                "missing_acceptance_evidence": MISSING_ACCEPTANCE_EVIDENCE,
                "blockers": ["HAVNAI_DRILL_ACCOUNT_TOKEN is not configured; funded private account token required."],
            }, indent=2, sort_keys=True), flush=True)
            return 2
        client = AccountClient(args.base_url, args.account_token)
        preflight = client.account_preflight()
        account = preflight.get("account") or {}
        credits = preflight.get("credits") or {}
        account_id = str(account.get("id") or account.get("account_id") or "").strip()
        available_units = int(credits.get("available_units") or credits.get("available") or 0)
        result = {
            "schema": "havn-45-account-restart-recovery-drill-preflight.v1",
            "generated_at": utc_now(),
            "base_url": clean_base_url(args.base_url),
            "account_token_present": True,
            "account_id_hash": account_hash(account_id),
            "account_present": bool(account_id),
            "account_status": account.get("status"),
            "available_units": available_units,
            "reserved_units": int(credits.get("reserved_units") or credits.get("reserved") or 0),
            "restart_command_configured": bool(args.restart_command.strip()),
            "passed": bool(account_id) and available_units > 0,
            "acceptance_passed": False,
            "settlement_evidence_passed": False,
            "missing_acceptance_evidence": MISSING_ACCEPTANCE_EVIDENCE,
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["passed"] else 2
    if not args.execute:
        return 0
    if not args.account_token.strip():
        raise SystemExit("--execute requires --account-token or HAVNAI_DRILL_ACCOUNT_TOKEN")
    if not args.restart_command.strip():
        raise SystemExit("--execute requires --restart-command or HAVNAI_RESTART_DRILL_COMMAND")
    if not args.db_path:
        raise SystemExit("--execute requires --db-path for full acceptance")
    db_path = Path(args.db_path)
    # Fail before submitting or restarting if the evidence source is unusable.
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        require_current_schema(conn)

    client = AccountClient(args.base_url, args.account_token)
    job_id = client.submit(payload, idempotency_key=args.idempotency_key)
    print(json.dumps({"event": "submitted", "job_id": job_id, "submitted_at": utc_now()}), flush=True)

    before = readonly_db_checks(db_path, job_id)
    errors = evidence_errors(before, job_id, completed=False)
    first_state = client.job(job_id)  # last external observation before restart marker
    errors += accepted_state_errors(first_state, job_id, (before.get("job") or {}).get("owner_account_hash"))
    print(json.dumps({"event": "accepted_state", "job_id": job_id, "status": result_status(first_state),
                      "observed_at": utc_now(), "db_checks": before, "errors": errors}), flush=True)
    if errors:
        print(json.dumps({"passed": False, "acceptance_passed": False,
                          "settlement_evidence_passed": False,
                          "missing_acceptance_evidence": MISSING_ACCEPTANCE_EVIDENCE,
                          "errors": errors}), flush=True)
        return 2

    print(json.dumps({"event": "restart_marker", "job_id": job_id, "observed_at": utc_now()}), flush=True)
    restart = run_restart(args.restart_command, args.restart_timeout_seconds)
    print(json.dumps({"event": "restart", "job_id": job_id, **restart}), flush=True)
    if restart["returncode"] != 0:
        raise SystemExit(3)

    deadline = time.monotonic() + args.timeout_seconds
    final_state: dict[str, Any] = first_state
    while time.monotonic() < deadline:
        final_state = client.job(job_id)
        status = result_status(final_state)
        print(json.dumps({"event": "poll", "job_id": job_id, "status": status, "observed_at": utc_now()}), flush=True)
        if status in TERMINAL_STATUSES:
            break
        time.sleep(args.poll_seconds)

    status = result_status(final_state)
    db_checks = readonly_db_checks(db_path, job_id)
    errors = evidence_errors(db_checks, job_id, completed=True)
    if db_checks.get("reservation_units") != before.get("reservation_units"):
        errors.append("reservation amount changed across restart")
    if (final_state.get("id") != job_id
            or account_hash(final_state.get("owner_account_id")) != before["job"]["owner_account_hash"]
            or (db_checks.get("job") or {}).get("owner_account_hash") != before["job"]["owner_account_hash"]):
        errors.append("job identity/ownership changed across restart")
    if status not in {"succeeded", "completed"}:
        errors.append("terminal success not observed")
    summary = {
        "schema": "havn-45-account-restart-recovery-drill.v1",
        "generated_at": utc_now(),
        "job_id": job_id,
        "terminal_status": status,
        "terminal": status in TERMINAL_STATUSES,
        # No independent restart/artifact verifier exists in this harness.
        "passed": False,
        "acceptance_passed": False,
        "settlement_evidence_passed": not errors,
        "missing_acceptance_evidence": MISSING_ACCEPTANCE_EVIDENCE,
        "errors": errors,
        "db_checks": db_checks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

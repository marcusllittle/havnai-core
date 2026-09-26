"""Run the HAVN-45/HAVN-49 private accepted-work restart drill.

The drill submits one account-owned generation through ``/v2/jobs``, executes an
operator-provided restart command while the job is accepted, polls the private
account job route to a terminal state, and optionally runs read-only SQLite
checks for duplicate charges, receipts, and payouts for the same ``job_id``.

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
DEFAULT_PROMPT = (
    "HAVN-45 private restart recovery drill: polished commercial-quality HavnAI "
    "operations scene, coherent lighting, premium product-ready composition"
)


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
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
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
        "job": job_row,
        "settlement": dict(settlement) if settlement else None,
        "ledger": ledger_rows,
        "reservation_count": int(reservations["count"] if reservations else 0),
        "receipt_count": int(receipts["count"] if receipts else 0),
        "node_payout_count": int(payouts["count"] if payouts else 0),
        "charge_like_ledger_rows": len(charge_rows),
        "duplicate_charge_like_rows": len(charge_rows) > 1,
    }


def summarize(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "havn-45-account-restart-recovery-drill-plan.v1",
        "generated_at": utc_now(),
        "base_url": clean_base_url(args.base_url),
        "execute": bool(args.execute),
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
                        help="optional local coordinator SQLite DB path for read-only duplicate checks")
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
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    print(json.dumps(summarize(args, payload), indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if not args.account_token.strip():
        raise SystemExit("--execute requires --account-token or HAVNAI_DRILL_ACCOUNT_TOKEN")
    if not args.restart_command.strip():
        raise SystemExit("--execute requires --restart-command or HAVNAI_RESTART_DRILL_COMMAND")

    client = AccountClient(args.base_url, args.account_token)
    job_id = client.submit(payload, idempotency_key=args.idempotency_key)
    print(json.dumps({"event": "submitted", "job_id": job_id, "submitted_at": utc_now()}), flush=True)

    first_state = client.job(job_id)
    print(json.dumps({"event": "accepted_state", "job_id": job_id, "status": result_status(first_state),
                      "observed_at": utc_now()}), flush=True)

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
    db_checks = None
    if args.db_path:
        db_checks = readonly_db_checks(Path(args.db_path), job_id)
    summary = {
        "schema": "havn-45-account-restart-recovery-drill.v1",
        "generated_at": utc_now(),
        "job_id": job_id,
        "terminal_status": status,
        "terminal": status in TERMINAL_STATUSES,
        "passed": status in {"succeeded", "completed"} and not (db_checks or {}).get("duplicate_charge_like_rows", False),
        "db_checks": db_checks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

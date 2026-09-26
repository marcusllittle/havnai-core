"""Account-authenticated Astra economy routes.

This module deliberately does not write account IDs into legacy Astra wallet
columns. Account play gets its own run, token, and spend tables while preserving
the same reward validation semantics as the wallet-era Astra endpoints.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import uuid
from typing import Any

import account_ledger
import astra_rewards

RUN_TOKEN_TTL_SECONDS = astra_rewards.RUN_TOKEN_TTL_SECONDS


class AstraAccountError(ValueError):
    def __init__(self, code: str, status: int = 422):
        super().__init__(code)
        self.status = status


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_astra_run_tokens (
            token_hash TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            map_id TEXT,
            started_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            consumed_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_account_astra_run_tokens_account
            ON account_astra_run_tokens(account_id);

        CREATE TABLE IF NOT EXISTS account_astra_runs (
            run_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            score INTEGER NOT NULL,
            grade TEXT NOT NULL,
            duration_s REAL NOT NULL,
            map_id TEXT,
            reward_units INTEGER NOT NULL DEFAULT 0,
            run_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(account_id, run_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_account_astra_runs_account
            ON account_astra_runs(account_id);
        CREATE INDEX IF NOT EXISTS idx_account_astra_runs_created
            ON account_astra_runs(created_at);

        CREATE TABLE IF NOT EXISTS account_astra_spends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            action TEXT NOT NULL,
            units INTEGER NOT NULL CHECK(typeof(units)='integer' AND units>0),
            status TEXT NOT NULL CHECK(status IN ('pending','completed','failed')),
            idempotency_key TEXT,
            created_at REAL NOT NULL,
            UNIQUE(account_id, idempotency_key)
        );
        CREATE INDEX IF NOT EXISTS idx_account_astra_spends_account
            ON account_astra_spends(account_id);
    """)
    conn.commit()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _run_hash(account_id: str, run_token: str) -> str:
    return hashlib.sha256(f"{account_id}:{run_token}".encode()).hexdigest()[:16]


def _daily_earned_units(conn: sqlite3.Connection, account_id: str) -> int:
    day_start = float(int(time.time() // 86400) * 86400)
    row = conn.execute(
        "SELECT COALESCE(SUM(reward_units),0) FROM account_astra_runs WHERE account_id=? AND created_at>=?",
        (account_id, day_start),
    ).fetchone()
    return int(row[0] or 0)


def _last_reward_time(conn: sqlite3.Connection, account_id: str) -> float | None:
    row = conn.execute(
        "SELECT MAX(created_at) FROM account_astra_runs WHERE account_id=? AND reward_units>0",
        (account_id,),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _is_first_win_today(conn: sqlite3.Connection, account_id: str) -> bool:
    return _daily_earned_units(conn, account_id) == 0


def _recent_run_count(conn: sqlite3.Connection, account_id: str, now: float) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM account_astra_runs WHERE account_id=? AND created_at>=?",
        (account_id, now - astra_rewards.STREAK_WINDOW_SECONDS),
    ).fetchone()
    return int(row[0] or 0)


def _credits_to_units(value: float) -> int:
    scaled = round(float(value) * account_ledger.SCALE)
    if scaled < 0:
        raise AstraAccountError("invalid_reward")
    return int(scaled)


def _spend_units(action: str) -> int:
    cost = astra_rewards.SPEND_COSTS.get(action)
    if cost is None:
        raise AstraAccountError("invalid_action")
    return _credits_to_units(cost)


def session(account_id: str) -> dict[str, Any]:
    return {
        "mode": "account",
        "account_id": account_id,
        "auth": "bearer",
        "wallet_required": False,
        "endpoints": {
            "start_run": "/v2/astra/run/start",
            "reward": "/v2/astra/reward",
            "spend": "/v2/astra/spend",
            "stats": "/v2/astra/stats",
        },
    }


def start_run(conn: sqlite3.Connection, account_id: str, map_id: str) -> dict[str, Any]:
    now = time.time()
    token = secrets.token_urlsafe(24)
    conn.execute(
        """INSERT INTO account_astra_run_tokens
           (token_hash, account_id, map_id, started_at, expires_at, consumed_at)
           VALUES (?, ?, ?, ?, ?, NULL)""",
        (_hash_token(token), account_id, map_id, now, now + RUN_TOKEN_TTL_SECONDS),
    )
    conn.execute("DELETE FROM account_astra_run_tokens WHERE expires_at < ?", (now - 86400.0,))
    conn.commit()
    return {"run_token": token, "started_at": now, "map_id": map_id, "mode": "account"}


def consume_run_token(conn: sqlite3.Connection, account_id: str, token: str) -> tuple[float | None, str | None]:
    if not token:
        return None, "missing_run_token"
    token_hash = _hash_token(token)
    row = conn.execute(
        """SELECT account_id, started_at, expires_at, consumed_at
           FROM account_astra_run_tokens WHERE token_hash=?""",
        (token_hash,),
    ).fetchone()
    if row is None:
        return None, "unknown_run_token"
    if str(row["account_id"]) != account_id:
        return None, "run_token_account_mismatch"
    if row["consumed_at"] is not None:
        return None, "run_token_used"
    if float(row["expires_at"]) < time.time():
        return None, "run_token_expired"
    conn.execute(
        "UPDATE account_astra_run_tokens SET consumed_at=? WHERE token_hash=? AND consumed_at IS NULL",
        (time.time(), token_hash),
    )
    return float(row["started_at"]), None


def submit_reward(
    conn: sqlite3.Connection,
    account_id: str,
    *,
    score: int,
    grade: str,
    duration_s: float,
    map_id: str,
    run_token: str,
) -> dict[str, Any]:
    now = time.time()
    with conn:
        started_at, token_error = consume_run_token(conn, account_id, run_token)
        if token_error:
            return {"ok": False, "reason": token_error, "reward_units": 0, "scale": account_ledger.SCALE}

        server_duration = max(0.0, now - float(started_at))
        if score < astra_rewards.MIN_SCORE_THRESHOLD:
            return {"ok": False, "reason": "score_too_low", "reward_units": 0, "scale": account_ledger.SCALE}
        if server_duration < astra_rewards.MIN_RUN_DURATION_SECONDS:
            return {"ok": False, "reason": "run_too_short", "reward_units": 0, "scale": account_ledger.SCALE}
        last = _last_reward_time(conn, account_id)
        if last is not None and (now - last) < astra_rewards.REWARD_COOLDOWN_SECONDS:
            wait = int(astra_rewards.REWARD_COOLDOWN_SECONDS - (now - last)) + 1
            return {"ok": False, "reason": "cooldown", "wait_seconds": wait, "reward_units": 0, "scale": account_ledger.SCALE}

        earned_today = _daily_earned_units(conn, account_id)
        daily_cap_units = _credits_to_units(astra_rewards.DAILY_EARN_CAP)
        remaining_cap = max(0, daily_cap_units - earned_today)
        if remaining_cap <= 0:
            return {"ok": False, "reason": "daily_cap_reached", "reward_units": 0, "scale": account_ledger.SCALE}

        run_hash = _run_hash(account_id, run_token)
        if conn.execute(
            "SELECT 1 FROM account_astra_runs WHERE account_id=? AND run_hash=?",
            (account_id, run_hash),
        ).fetchone():
            return {"ok": False, "reason": "duplicate_run", "reward_units": 0, "scale": account_ledger.SCALE}

        raw_reward = astra_rewards._interpolate_reward(score)
        multiplier = 1.0
        bonuses: list[str] = []
        if _is_first_win_today(conn, account_id):
            multiplier *= astra_rewards.FIRST_WIN_DAILY_MULTIPLIER
            bonuses.append("first_win_of_day")
        if _recent_run_count(conn, account_id, now) >= astra_rewards.STREAK_THRESHOLD:
            multiplier *= astra_rewards.STREAK_MULTIPLIER
            bonuses.append("streak_bonus")
        reward_units = min(
            _credits_to_units(raw_reward * multiplier),
            _credits_to_units(astra_rewards.MAX_CREDITS_PER_RUN),
            remaining_cap,
        )
        run_id = f"acct_astra_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """INSERT INTO account_astra_runs
               (run_id, account_id, score, grade, duration_s, map_id, reward_units, run_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, account_id, score, grade, server_duration, map_id, reward_units, run_hash, now),
        )
        entry = None
        if reward_units > 0:
            entry = account_ledger._apply(
                conn,
                account_id,
                operation="astra_reward",
                key=f"astra_reward:{run_id}",
                resource=run_id,
                settled_delta=reward_units,
                reserved_delta=0,
                actor="astra",
                reason="astra_game_reward",
            )
    return {
        "ok": True,
        "run_id": run_id,
        "reward_units": reward_units,
        "reward": round(reward_units / account_ledger.SCALE, 3),
        "scale": account_ledger.SCALE,
        "daily_earned_units": earned_today + reward_units,
        "daily_cap_units": daily_cap_units,
        "bonuses": bonuses or None,
        "multiplier": multiplier if multiplier > 1.0 else None,
        "ledger_entry_id": entry["entry_id"] if entry else None,
    }


def process_spend(
    conn: sqlite3.Connection,
    account_id: str,
    *,
    action: str,
    idempotency_key: str,
) -> dict[str, Any]:
    if not idempotency_key or len(idempotency_key) > 128:
        raise AstraAccountError("idempotency_key_required", 400)
    units = _spend_units(action)
    now = time.time()
    with conn:
        prior = conn.execute(
            """SELECT action, units, status FROM account_astra_spends
               WHERE account_id=? AND idempotency_key=?""",
            (account_id, idempotency_key),
        ).fetchone()
        if prior:
            if prior["action"] != action or int(prior["units"]) != units:
                raise account_ledger.LedgerError("idempotency_conflict")
            balance = account_ledger.balance(conn, account_id)
            return {
                "ok": prior["status"] == "completed",
                "replayed": True,
                "reason": "insufficient_credits" if prior["status"] == "failed" else None,
                "action": prior["action"],
                "cost_units": int(prior["units"]),
                "balance": balance,
                "scale": account_ledger.SCALE,
            }
        cursor = conn.execute(
            """INSERT INTO account_astra_spends
               (account_id, action, units, status, idempotency_key, created_at)
               VALUES (?, ?, ?, 'pending', ?, ?)""",
            (account_id, action, units, idempotency_key, now),
        )
        try:
            entry = account_ledger._apply(
                conn,
                account_id,
                operation="astra_spend",
                key=f"astra_spend:{idempotency_key}",
                resource=str(cursor.lastrowid),
                settled_delta=-units,
                reserved_delta=0,
                actor=account_id,
                reason=f"astra:{action}",
                require_available=units,
            )
        except account_ledger.LedgerError as exc:
            if str(exc) != "insufficient_credits":
                raise
            conn.execute("UPDATE account_astra_spends SET status='failed' WHERE id=?", (cursor.lastrowid,))
            return {
                "ok": False,
                "reason": "insufficient_credits",
                "action": action,
                "cost_units": units,
                "balance": account_ledger.balance(conn, account_id),
                "scale": account_ledger.SCALE,
            }
        conn.execute("UPDATE account_astra_spends SET status='completed' WHERE id=?", (cursor.lastrowid,))
        balance = account_ledger.balance(conn, account_id)
    return {
        "ok": True,
        "action": action,
        "cost_units": units,
        "balance": balance,
        "scale": account_ledger.SCALE,
        "ledger_entry_id": entry["entry_id"],
    }


def stats(conn: sqlite3.Connection, account_id: str) -> dict[str, Any]:
    row = conn.execute(
        """SELECT COUNT(*) AS runs, MAX(score) AS best, COALESCE(SUM(reward_units),0) AS earned
           FROM account_astra_runs WHERE account_id=?""",
        (account_id,),
    ).fetchone()
    earned_today = _daily_earned_units(conn, account_id)
    last = _last_reward_time(conn, account_id)
    cooldown_remaining = 0
    if last is not None:
        cooldown_remaining = max(0, int(astra_rewards.REWARD_COOLDOWN_SECONDS - (time.time() - last)))
    return {
        "mode": "account",
        "account_id": account_id,
        "total_runs": int(row["runs"] or 0),
        "best_score": int(row["best"] or 0),
        "total_earned_units": int(row["earned"] or 0),
        "daily_earned_units": earned_today,
        "daily_cap_units": _credits_to_units(astra_rewards.DAILY_EARN_CAP),
        "cooldown_remaining": cooldown_remaining,
        "scale": account_ledger.SCALE,
    }

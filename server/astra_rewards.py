"""
Astra Valkyries game economy integration.

Isolated module for game-specific credit operations:
  - Bounded reward (earn credits from gameplay)
  - Validated spend (gacha pulls via shared credits)
  - Leaderboard (top Astra players by score)

All earn/spend logic is server-enforced. The game client submits
results and the server decides the reward amount.
"""

import hashlib
import sqlite3
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─── Injected dependencies (set by app.py) ──────────────────

get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]

# ─── Constants ───────────────────────────────────────────────

# Earn limits
MAX_CREDITS_PER_RUN = 15
DAILY_EARN_CAP = 50
REWARD_COOLDOWN_SECONDS = 60
MIN_SCORE_THRESHOLD = 5000        # roughly "Grade B" territory
MIN_RUN_DURATION_SECONDS = 30     # minimum plausible game run length

# Bonus multipliers
FIRST_WIN_DAILY_MULTIPLIER = 2.0  # 2x credits on first completed run of the day
STREAK_THRESHOLD = 3              # play 3 runs in a row → streak bonus
STREAK_MULTIPLIER = 1.5           # 1.5x credits during streak
STREAK_WINDOW_SECONDS = 600       # runs within 10 minutes count as streak

# Spend costs (shared credits — rebalanced from local economy)
SPEND_COSTS: Dict[str, float] = {
    "gacha_1":  10.0,   # single pull
    "gacha_10": 80.0,   # ten-pull (~17% discount)
    "continue": 15.0,   # revive once per run
    "boost_damage": 5.0,  # temporary +10% damage for one run
}

# Score → credit reward curve (linear interpolation between thresholds)
REWARD_TIERS: List[Tuple[int, float]] = [
    (5_000,   2.0),    # Grade B minimum
    (10_000,  5.0),    # Solid B
    (25_000,  8.0),    # Grade A
    (50_000, 12.0),    # Grade S
    (100_000, 15.0),   # Grade SS (cap)
]


def _interpolate_reward(score: int) -> float:
    """Map score to credit reward using linear interpolation between tiers."""
    if score < REWARD_TIERS[0][0]:
        return 0.0
    for i in range(len(REWARD_TIERS) - 1):
        lo_score, lo_reward = REWARD_TIERS[i]
        hi_score, hi_reward = REWARD_TIERS[i + 1]
        if score <= hi_score:
            t = (score - lo_score) / (hi_score - lo_score)
            return round(lo_reward + t * (hi_reward - lo_reward), 2)
    return float(REWARD_TIERS[-1][1])


# ─── Table setup ─────────────────────────────────────────────

def init_astra_tables(db: sqlite3.Connection) -> None:
    """Create Astra-specific tables."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS astra_runs (
            run_id      TEXT PRIMARY KEY,
            wallet      TEXT NOT NULL,
            score       INTEGER NOT NULL,
            grade       TEXT NOT NULL,
            duration_s  REAL NOT NULL,
            map_id      TEXT,
            reward      REAL NOT NULL DEFAULT 0.0,
            run_hash    TEXT NOT NULL,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_astra_runs_wallet
            ON astra_runs(wallet);
        CREATE INDEX IF NOT EXISTS idx_astra_runs_created
            ON astra_runs(created_at);

        CREATE TABLE IF NOT EXISTS astra_spends (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet      TEXT NOT NULL,
            action      TEXT NOT NULL,
            amount      REAL NOT NULL,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_astra_spends_wallet
            ON astra_spends(wallet);
    """)


# ─── Anti-abuse helpers ──────────────────────────────────────

def _run_hash(wallet: str, score: int, map_id: str, duration: float) -> str:
    """Deterministic hash for replay detection."""
    raw = f"{wallet}:{score}:{map_id}:{duration:.1f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _daily_earned(db: sqlite3.Connection, wallet: str) -> float:
    """Sum of rewards earned in the current UTC day."""
    day_start = float(int(time.time() // 86400) * 86400)
    row = db.execute(
        "SELECT COALESCE(SUM(reward), 0) FROM astra_runs WHERE wallet = ? AND created_at >= ?",
        (wallet, day_start),
    ).fetchone()
    return float(row[0])


def _last_reward_time(db: sqlite3.Connection, wallet: str) -> Optional[float]:
    """Timestamp of most recent rewarded run (or None)."""
    row = db.execute(
        "SELECT MAX(created_at) FROM astra_runs WHERE wallet = ? AND reward > 0",
        (wallet,),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _is_first_win_today(db: sqlite3.Connection, wallet: str) -> bool:
    """Check if this is the player's first rewarded run of the UTC day."""
    day_start = float(int(time.time() // 86400) * 86400)
    row = db.execute(
        "SELECT COUNT(*) FROM astra_runs WHERE wallet = ? AND created_at >= ? AND reward > 0",
        (wallet, day_start),
    ).fetchone()
    return row[0] == 0


def _recent_run_count(db: sqlite3.Connection, wallet: str, now: float) -> int:
    """Count runs within the streak window."""
    cutoff = now - STREAK_WINDOW_SECONDS
    row = db.execute(
        "SELECT COUNT(*) FROM astra_runs WHERE wallet = ? AND created_at >= ?",
        (wallet, cutoff),
    ).fetchone()
    return row[0]


# ─── Core operations ─────────────────────────────────────────

def submit_reward(
    wallet: str,
    score: int,
    grade: str,
    duration_s: float,
    map_id: str,
    deposit_fn: Callable[[str, float, str], float],
    ledger_fn: Callable[..., None],
) -> Dict[str, Any]:
    """
    Validate a game run and award bounded credits.

    Returns dict with run_id, reward amount, and any rejection reason.
    The deposit_fn is credits.deposit_credits; ledger_fn is settlement._record_ledger.
    """
    now = time.time()
    db = get_db()

    # ── Validation ───────────────────────────────────────────
    if score < MIN_SCORE_THRESHOLD:
        return {"ok": False, "reason": "score_too_low", "reward": 0}

    if duration_s < MIN_RUN_DURATION_SECONDS:
        return {"ok": False, "reason": "run_too_short", "reward": 0}

    # Cooldown check
    last = _last_reward_time(db, wallet)
    if last is not None and (now - last) < REWARD_COOLDOWN_SECONDS:
        wait = int(REWARD_COOLDOWN_SECONDS - (now - last)) + 1
        return {"ok": False, "reason": "cooldown", "wait_seconds": wait, "reward": 0}

    # Daily cap
    earned_today = _daily_earned(db, wallet)
    remaining_cap = max(0.0, DAILY_EARN_CAP - earned_today)
    if remaining_cap <= 0:
        return {"ok": False, "reason": "daily_cap_reached", "reward": 0}

    # Replay detection
    rh = _run_hash(wallet, score, map_id, duration_s)
    dup = db.execute(
        "SELECT 1 FROM astra_runs WHERE run_hash = ? AND wallet = ?", (rh, wallet)
    ).fetchone()
    if dup:
        return {"ok": False, "reason": "duplicate_run", "reward": 0}

    # ── Compute reward ───────────────────────────────────────
    raw_reward = _interpolate_reward(score)

    # Apply bonus multipliers
    multiplier = 1.0
    bonus_reasons = []
    if _is_first_win_today(db, wallet):
        multiplier *= FIRST_WIN_DAILY_MULTIPLIER
        bonus_reasons.append("first_win_of_day")
    recent = _recent_run_count(db, wallet, now)
    if recent >= STREAK_THRESHOLD:
        multiplier *= STREAK_MULTIPLIER
        bonus_reasons.append("streak_bonus")

    raw_reward *= multiplier
    reward = min(raw_reward, MAX_CREDITS_PER_RUN * multiplier, remaining_cap)
    reward = round(reward, 2)

    # ── Persist ──────────────────────────────────────────────
    run_id = f"astra_{uuid.uuid4().hex[:12]}"
    db.execute(
        """INSERT INTO astra_runs (run_id, wallet, score, grade, duration_s, map_id, reward, run_hash, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, wallet, score, grade, duration_s, map_id, reward, rh, now),
    )

    if reward > 0:
        new_balance = deposit_fn(wallet, reward, f"Astra game reward (run {run_id})")
        ledger_fn(wallet, "reward", reward, new_balance, None, f"astra:reward:{run_id}")
        log_event(f"Astra reward: {wallet[:10]}… earned {reward} credits (score={score})")

    db.commit()

    return {
        "ok": True,
        "run_id": run_id,
        "reward": reward,
        "daily_earned": round(earned_today + reward, 2),
        "daily_cap": DAILY_EARN_CAP,
        "bonuses": bonus_reasons if bonus_reasons else None,
        "multiplier": multiplier if multiplier > 1.0 else None,
    }


def process_spend(
    wallet: str,
    action: str,
    deduct_fn: Callable[[str, float, str], Tuple[bool, float]],
    ledger_fn: Callable[..., None],
) -> Dict[str, Any]:
    """
    Deduct shared credits for a gacha action.

    Returns dict with success, amount, remaining balance.
    """
    cost = SPEND_COSTS.get(action)
    if cost is None:
        return {"ok": False, "reason": "invalid_action"}

    success, remaining = deduct_fn(wallet, cost, f"astra:{action}")
    if not success:
        return {"ok": False, "reason": "insufficient_credits", "cost": cost}

    db = get_db()
    db.execute(
        "INSERT INTO astra_spends (wallet, action, amount, created_at) VALUES (?, ?, ?, ?)",
        (wallet, action, cost, time.time()),
    )
    ledger_fn(wallet, "spend", -cost, remaining, None, f"astra:{action}")
    db.commit()

    log_event(f"Astra spend: {wallet[:10]}… spent {cost} credits on {action}")
    return {"ok": True, "action": action, "cost": cost, "remaining": remaining}


def get_leaderboard(limit: int = 50) -> List[Dict[str, Any]]:
    """Top Astra players by highest single-run score."""
    db = get_db()
    rows = db.execute(
        """SELECT wallet,
                  MAX(score) AS best_score,
                  COUNT(*) AS total_runs,
                  COALESCE(SUM(reward), 0) AS total_earned
             FROM astra_runs
            GROUP BY wallet
            ORDER BY best_score DESC
            LIMIT ?""",
        (limit,),
    ).fetchall()

    return [
        {
            "rank": i + 1,
            "wallet": r[0],
            "wallet_short": f"{r[0][:6]}…{r[0][-4:]}",
            "best_score": r[1],
            "total_runs": r[2],
            "total_earned": round(r[3], 2),
        }
        for i, r in enumerate(rows)
    ]


def get_player_stats(wallet: str) -> Dict[str, Any]:
    """Stats for a single player."""
    db = get_db()
    row = db.execute(
        """SELECT COUNT(*) AS runs,
                  MAX(score) AS best,
                  COALESCE(SUM(reward), 0) AS earned
             FROM astra_runs WHERE wallet = ?""",
        (wallet,),
    ).fetchone()

    earned_today = _daily_earned(db, wallet)
    last = _last_reward_time(db, wallet)
    cooldown_remaining = 0
    if last is not None:
        cd = REWARD_COOLDOWN_SECONDS - (time.time() - last)
        cooldown_remaining = max(0, int(cd))

    return {
        "wallet": wallet,
        "total_runs": row[0],
        "best_score": row[1] or 0,
        "total_earned": round(row[2], 2),
        "daily_earned": round(earned_today, 2),
        "daily_cap": DAILY_EARN_CAP,
        "cooldown_remaining": cooldown_remaining,
    }

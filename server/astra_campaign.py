"""Shared Astra campaign derived from real game and creator-network records."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


get_db: Callable[[], sqlite3.Connection]

WEEK_SECONDS = 7 * 24 * 60 * 60
MONDAY_EPOCH = 4 * 24 * 60 * 60  # 1970-01-05 00:00:00 UTC
COMBAT_TARGET = 100
FORGE_TARGET = 12
CAMPAIGN_MAPS: Tuple[Dict[str, str], ...] = (
    {
        "map_id": "nebula-runway",
        "name": "Nebula Runway",
        "operation": "Break the convoy siege",
    },
    {
        "map_id": "solar-rift",
        "name": "Solar Rift",
        "operation": "Collapse the thermal blockade",
    },
    {
        "map_id": "abyss-crown",
        "name": "Abyss Crown",
        "operation": "Seal the void breach",
    },
)

SUCCESS_JOB_STATUSES = ("completed", "done", "succeeded", "success")
FINAL_SETTLEMENT_OUTCOMES = ("spent", "released")


def _campaign_window(now: Optional[float] = None) -> Tuple[int, int, int]:
    timestamp = int(time.time() if now is None else now)
    index = (timestamp - MONDAY_EPOCH) // WEEK_SECONDS
    start = MONDAY_EPOCH + index * WEEK_SECONDS
    return index, start, start + WEEK_SECONDS


def _campaign_identity(index: int, start: int) -> str:
    iso = datetime.fromtimestamp(start, timezone.utc).isocalendar()
    return f"astra-{iso.year}-w{iso.week:02d}-{index % len(CAMPAIGN_MAPS) + 1}"


def combat_points(score: int) -> int:
    """Bound one accepted run to 1-10 campaign points."""
    return max(1, min(10, max(0, int(score)) // 5_000))


def _owner_alias(wallet: str, campaign_id: str) -> str:
    digest = hashlib.sha256(f"{campaign_id}:{wallet.lower()}".encode()).hexdigest()
    return f"pilot-{digest[:8]}"


def _combat_rows(
    db: sqlite3.Connection,
    map_id: str,
    starts_at: int,
    ends_at: int,
) -> List[sqlite3.Row]:
    return db.execute(
        """SELECT run_id, wallet, score, grade, created_at
             FROM astra_runs
            WHERE map_id = ? AND created_at >= ? AND created_at < ? AND reward > 0
            ORDER BY created_at DESC""",
        (map_id, starts_at, ends_at),
    ).fetchall()


def _forge_rows(
    db: sqlite3.Connection,
    map_id: str,
    starts_at: int,
    ends_at: int,
) -> List[Dict[str, Any]]:
    placeholders_status = ",".join("?" for _ in SUCCESS_JOB_STATUSES)
    placeholders_outcome = ",".join("?" for _ in FINAL_SETTLEMENT_OUTCOMES)
    common_params: Tuple[Any, ...] = (
        map_id,
        starts_at,
        ends_at,
        *SUCCESS_JOB_STATUSES,
        *FINAL_SETTLEMENT_OUTCOMES,
    )

    def fetch(job_column: str, artifact_type: str, points: int) -> List[Dict[str, Any]]:
        rows = db.execute(
            f"""SELECT {job_column} AS job_id, r.wallet,
                       COALESCE(j.completed_at, s.updated_at, r.created_at) AS completed_at,
                       COALESCE(s.assigned_node_id, j.node_id) AS node_id
                  FROM astra_reward_images r
                  JOIN jobs j ON j.id = {job_column}
                  JOIN job_settlement s ON s.job_id = {job_column}
                 WHERE r.map_id = ?
                   AND COALESCE(j.completed_at, s.updated_at, r.created_at) >= ?
                   AND COALESCE(j.completed_at, s.updated_at, r.created_at) < ?
                   AND LOWER(j.status) IN ({placeholders_status})
                   AND LOWER(s.execution_status) = 'settled'
                   AND LOWER(s.quality_status) = 'valid'
                   AND LOWER(s.settlement_outcome) IN ({placeholders_outcome})
                   AND COALESCE(s.assigned_node_id, j.node_id) IS NOT NULL""",
            common_params,
        ).fetchall()
        return [
            {
                "job_id": str(row["job_id"]),
                "wallet": str(row["wallet"]),
                "created_at": float(row["completed_at"]),
                "node_id": str(row["node_id"]),
                "artifact_type": artifact_type,
                "points": points,
            }
            for row in rows
            if row["job_id"]
        ]

    return fetch("r.job_id", "image", 1) + fetch("r.video_job_id", "video", 2)


def _phase(combat_total: int, forge_total: int) -> str:
    combat_ready = combat_total >= COMBAT_TARGET
    forge_ready = forge_total >= FORGE_TARGET
    if combat_ready and forge_ready:
        return "secured"
    if combat_ready:
        return "awaiting_forge"
    if forge_ready:
        return "awaiting_victories"
    return "contested"


def get_campaign(
    wallet: Optional[str] = None,
    *,
    now: Optional[float] = None,
    event_limit: int = 8,
) -> Dict[str, Any]:
    """Aggregate this week's front from rewarded runs and final settlements."""
    db = get_db()
    index, starts_at, ends_at = _campaign_window(now)
    definition = CAMPAIGN_MAPS[index % len(CAMPAIGN_MAPS)]
    campaign_id = _campaign_identity(index, starts_at)
    combat = _combat_rows(db, definition["map_id"], starts_at, ends_at)
    forge = _forge_rows(db, definition["map_id"], starts_at, ends_at)

    combat_total = sum(combat_points(int(row["score"])) for row in combat)
    forge_total = sum(int(row["points"]) for row in forge)
    combat_ratio = min(1.0, combat_total / COMBAT_TARGET)
    forge_ratio = min(1.0, forge_total / FORGE_TARGET)

    events: List[Dict[str, Any]] = [
        {
            "kind": "combat",
            "id": str(row["run_id"]),
            "actor": _owner_alias(str(row["wallet"]), campaign_id),
            "grade": str(row["grade"]),
            "points": combat_points(int(row["score"])),
            "created_at": float(row["created_at"]),
        }
        for row in combat
    ]
    events.extend(
        {
            "kind": "forge",
            "id": row["job_id"],
            "actor": row["node_id"],
            "artifact_type": row["artifact_type"],
            "points": row["points"],
            "created_at": row["created_at"],
        }
        for row in forge
    )
    events.sort(key=lambda event: float(event["created_at"]), reverse=True)

    normalized_wallet = wallet.lower() if wallet else None
    personal_combat = [row for row in combat if str(row["wallet"]).lower() == normalized_wallet]
    personal_forge = [row for row in forge if str(row["wallet"]).lower() == normalized_wallet]

    return {
        "schema": "havnai.astra.community-campaign",
        "version": 1,
        "campaign_id": campaign_id,
        "map_id": definition["map_id"],
        "name": definition["name"],
        "operation": definition["operation"],
        "phase": _phase(combat_total, forge_total),
        "secured": combat_total >= COMBAT_TARGET and forge_total >= FORGE_TARGET,
        "starts_at": starts_at,
        "ends_at": ends_at,
        "progress_percent": round((combat_ratio * 0.65 + forge_ratio * 0.35) * 100),
        "combat": {
            "current": combat_total,
            "target": COMBAT_TARGET,
            "percent": round(combat_ratio * 100),
            "accepted_runs": len(combat),
            "contributors": len({str(row["wallet"]).lower() for row in combat}),
        },
        "forge": {
            "current": forge_total,
            "target": FORGE_TARGET,
            "percent": round(forge_ratio * 100),
            "settled_artifacts": len(forge),
            "creator_nodes": len({str(row["node_id"]) for row in forge}),
        },
        "personal": {
            "combat_points": sum(combat_points(int(row["score"])) for row in personal_combat),
            "accepted_runs": len(personal_combat),
            "forge_points": sum(int(row["points"]) for row in personal_forge),
            "settled_artifacts": len(personal_forge),
        } if normalized_wallet else None,
        "recent_events": events[: max(0, min(20, int(event_limit)))],
    }


def get_run_contribution(run_id: str, *, now: Optional[float] = None) -> Dict[str, Any]:
    """Describe one accepted run's effect on the active front."""
    db = get_db()
    row = db.execute(
        "SELECT run_id, wallet, map_id, score, reward, created_at FROM astra_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    campaign = get_campaign(str(row["wallet"]) if row and "wallet" in row.keys() else None, now=now)
    if row is None:
        return {
            "campaign_id": campaign["campaign_id"],
            "eligible": False,
            "combat_points": 0,
            "target_map_id": campaign["map_id"],
        }
    eligible = (
        str(row["map_id"]) == campaign["map_id"]
        and float(row["reward"]) > 0
        and campaign["starts_at"] <= float(row["created_at"]) < campaign["ends_at"]
    )
    return {
        "campaign_id": campaign["campaign_id"],
        "eligible": eligible,
        "combat_points": combat_points(int(row["score"])) if eligible else 0,
        "target_map_id": campaign["map_id"],
        "phase": campaign["phase"],
        "progress_percent": campaign["progress_percent"],
    }

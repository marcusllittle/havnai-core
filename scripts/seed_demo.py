#!/usr/bin/env python3
"""Populate a throwaway coordinator database with a plausible day of network
activity, then keep a handful of simulated GPU nodes heartbeating so the
dashboard stays green while you talk.

This exists for live demos. Nothing here fabricates reward numbers: every
$HAI figure is produced by ``server/rewards.compute_reward`` using the real
manifest weights, so the ledger you show on stage is arithmetic your own
engine performed. What is synthetic is the *history* — the jobs, the node
names, and the wallets.

The script refuses to touch ``db/ledger.db`` unless you say so explicitly,
because a demo should never be one typo away from your real ledger.

Usage::

    # seed a day of history into db/demo.db and hold nodes online
    python scripts/seed_demo.py --serve

    # seed only, no heartbeat loop
    python scripts/seed_demo.py

    # start over
    python scripts/seed_demo.py --reset --serve

Point the coordinator at the seeded database and node registry when you boot it::

    cd server
    HAVNAI_DB_PATH=../db/demo.db HAVNAI_NODES_PATH=../nodes.demo.json \
        SERVER_PORT=5001 python app.py

Both of those paths are gitignored, so a demo never dirties the working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(SERVER_DIR))

import rewards as rewards_module  # noqa: E402  (needs sys.path above)
import astra_rewards  # noqa: E402

DEFAULT_DB = REPO_ROOT / "db" / "demo.db"
LEDGER_DB = REPO_ROOT / "db" / "ledger.db"
MANIFEST = SERVER_DIR / "manifests" / "registry.json"

# Deliberately NOT nodes.json: that file is tracked, and demo node state has no
# business landing in a commit. The coordinator reads HAVNAI_NODES_PATH, so the
# demo gets its own registry and the real one is never touched.
DEFAULT_NODES_FILE = REPO_ROOT / "nodes.demo.json"

# Mirrors REWARD_CONFIG in server/app.py. Kept as env-overridable so a demo
# can be tuned the same way production is.
REWARD_CONFIG: Dict[str, float] = {
    "baseline_runtime": float(os.getenv("REWARD_BASELINE_RUNTIME", "8.0")),
    "sdxl_factor": float(os.getenv("REWARD_SDXL_FACTOR", "1.5")),
    "sd15_factor": float(os.getenv("REWARD_SD15_FACTOR", "1.0")),
    "anime_factor": float(os.getenv("REWARD_ANIME_FACTOR", "0.7")),
    "ltx2_factor": float(os.getenv("REWARD_LTX2_FACTOR", "2.0")),
    "ltx_video_factor": float(os.getenv("REWARD_LTX_VIDEO_FACTOR", "3.0")),
    "base_reward": float(os.getenv("REWARD_BASE_HAI", "0.05")),
}

# Operator rigs. Mixed GPU classes so the node table shows a real spread of
# hardware rather than ten identical 4090s.
DEMO_NODES: List[Dict[str, Any]] = [
    {
        "node_id": "havn-atlas",
        "node_name": "atlas",
        "gpu": "NVIDIA GeForce RTX 4090",
        "vram_mb": 24564,
        "pipelines": ["sdxl", "sd15"],
        "os": "Linux",
    },
    {
        "node_id": "havn-vega",
        "node_name": "vega",
        "gpu": "NVIDIA GeForce RTX 4080 SUPER",
        "vram_mb": 16376,
        "pipelines": ["sdxl", "sd15"],
        "os": "Windows",
    },
    {
        "node_id": "havn-orion",
        "node_name": "orion",
        "gpu": "NVIDIA GeForce RTX 3060",
        "vram_mb": 12288,
        "pipelines": ["sd15", "animatediff"],
        "os": "Linux",
    },
    {
        "node_id": "havn-lyra",
        "node_name": "lyra",
        "gpu": "NVIDIA RTX A5000",
        "vram_mb": 24564,
        "pipelines": ["ltx_video", "ltx23_wangp", "sdxl"],
        "os": "Linux",
    },
]

# Creator wallets. Deterministic so the leaderboard ordering is stable across
# reruns — you do not want the ranking reshuffling between rehearsal and stage.
DEMO_WALLETS = [
    "0x71c7656ec7ab88b098defb751b7401b5f6d8976f",
    "0x2f9a4c1b8d7e6f3a0b5c4d2e1f8a9b7c6d5e4f30",
    "0x8ba1f109551bd432803012645ac136ddd64dba72",
    "0x4e83362442b8d1bec281594cea3050c8eb01311c",
    "0xdc76cd25977e0a5ae17155770273ad58648900d3",
    "0x5aae5c59d642e5fd45b427df6ed478b49d55fefd",
]


def load_manifest_weights() -> Dict[str, Dict[str, Any]]:
    """Read the real registry so routing and rewards use production weights."""
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models", payload)
    catalog: Dict[str, Dict[str, Any]] = {}
    for entry in models:
        name = entry.get("name")
        if not name:
            continue
        weight = entry.get("weight")
        if weight is None:
            weight = entry.get("reward_weight", 10.0)
        catalog[name.lower()] = {
            "name": name,
            "pipeline": (entry.get("pipeline") or "sd15").lower(),
            "task_type": (entry.get("task_type") or "IMAGE_GEN").upper(),
            "weight": float(weight),
        }
    return catalog


def pick_model(catalog: Dict[str, Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    """Weighted routing, same selection rule as ``model="auto"`` in app.py."""
    entries = list(catalog.values())
    weights = [entry["weight"] for entry in entries]
    return rng.choices(entries, weights=weights, k=1)[0]


def runtime_for(pipeline: str, rng: random.Random) -> float:
    """Inference time in ms, scaled to what each pipeline actually costs."""
    envelopes = {
        "sdxl": (6_500, 14_000),
        "sd15": (2_800, 6_500),
        "animatediff": (38_000, 72_000),
        "ltx_video": (55_000, 120_000),
        "ltx23_wangp": (48_000, 96_000),
        "ltx2": (60_000, 130_000),
    }
    low, high = envelopes.get(pipeline, (4_000, 9_000))
    return float(rng.randint(low, high))


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the subset of tables the demo surfaces actually read."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            model TEXT NOT NULL,
            data TEXT,
            task_type TEXT NOT NULL,
            weight REAL NOT NULL,
            status TEXT NOT NULL,
            node_id TEXT,
            timestamp REAL NOT NULL,
            assigned_at REAL,
            completed_at REAL,
            invite_code TEXT,
            attempt_id TEXT,
            lease_expires_at REAL,
            progress REAL NOT NULL DEFAULT 0,
            stage TEXT NOT NULL DEFAULT 'queued',
            resolved_spec TEXT,
            error_code TEXT,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            reward_hai REAL NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS node_wallets (
            node_id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            node_name TEXT,
            updated_at REAL NOT NULL
        );
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
        """
    )
    conn.commit()


def grade_for(score: int) -> str:
    """Letter grade matching the reward tier boundaries in astra_rewards.py."""
    for threshold, letter in ((100_000, "SS"), (50_000, "S"), (25_000, "A"), (10_000, "B+")):
        if score >= threshold:
            return letter
    return "B"


def seed_astra_runs(conn: sqlite3.Connection, rng: random.Random, count: int, window: float, now: float) -> Dict[str, Any]:
    """Fill the in-game leaderboard with completed runs.

    Rewards come from ``astra_rewards._interpolate_reward``, the same curve the
    live endpoint uses, so the credits on the leaderboard are the real payout
    for those scores.
    """
    maps = ["orbital-yard", "helios-belt", "vault-descent", "carrier-run"]
    total = 0.0
    for _ in range(count):
        wallet = rng.choice(DEMO_WALLETS)
        # Skewed toward mid scores so the leaderboard has a believable tail
        # rather than everyone sitting at the SS cap.
        score = int(rng.triangular(5_000, 120_000, 22_000))
        reward = min(astra_rewards._interpolate_reward(score), float(astra_rewards.MAX_CREDITS_PER_RUN))
        created = now - rng.uniform(0, window)
        conn.execute(
            """INSERT OR REPLACE INTO astra_runs
               (run_id, wallet, score, grade, duration_s, map_id, reward, run_hash, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                f"run-{uuid.uuid4().hex[:12]}",
                wallet,
                score,
                grade_for(score),
                rng.uniform(95, 420),
                rng.choice(maps),
                reward,
                uuid.uuid4().hex,
                created,
            ),
        )
        total += reward
    conn.commit()
    return {"runs": count, "credits": round(total, 2)}


def seed(db_path: Path, job_count: int, hours: float, seed_value: int, nodes_file: Path) -> Dict[str, Any]:
    """Write a day of completed work and its rewards into ``db_path``."""
    rng = random.Random(seed_value)
    catalog = load_manifest_weights()

    # Inject the globals rewards.py expects so compute_reward runs for real.
    rewards_module.MODEL_WEIGHTS = {k: v["weight"] for k, v in catalog.items()}
    rewards_module.REWARD_CONFIG = REWARD_CONFIG

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    now = time.time()
    window = hours * 3600.0
    total_hai = 0.0
    succeeded = 0
    failed = 0
    per_model: Dict[str, int] = {}
    per_node: Dict[str, Dict[str, float]] = {}

    for _ in range(job_count):
        entry = pick_model(catalog, rng)
        model_key = entry["name"].lower()
        pipeline = entry["pipeline"]
        wallet = rng.choice(DEMO_WALLETS)
        node = rng.choice([n for n in DEMO_NODES if pipeline in n["pipelines"]] or DEMO_NODES)

        submitted = now - rng.uniform(0, window)
        inference_ms = runtime_for(pipeline, rng)
        # Queue wait plus a little coordinator overhead on top of inference.
        assigned = submitted + rng.uniform(0.2, 4.0)
        completed = assigned + (inference_ms / 1000.0) + rng.uniform(0.3, 1.8)

        # ~4% failure rate keeps success_rate honest instead of a suspicious 100%.
        status = "failed" if rng.random() < 0.04 else "success"
        job_id = f"job-{uuid.uuid4().hex[:12]}"

        reward_hai, _factors = rewards_module.compute_reward(
            model_name=model_key,
            pipeline=pipeline,
            metrics={"inference_time_ms": inference_ms},
            status=status,
        )

        conn.execute(
            """INSERT INTO jobs
               (id, wallet, model, data, task_type, weight, status, node_id,
                timestamp, assigned_at, completed_at, progress, stage, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                wallet,
                entry["name"],
                json.dumps({"prompt": "[demo seed]", "pipeline": pipeline}),
                entry["task_type"],
                entry["weight"],
                # Lowercase on purpose: client.py reports "success"/"failed" and
                # the analytics SQL matches those literals case-sensitively.
                status,
                node["node_id"],
                submitted,
                assigned,
                completed,
                100.0 if status == "success" else 0.0,
                "complete" if status == "success" else "error",
                completed,
            ),
        )

        if status == "success":
            conn.execute(
                "INSERT OR REPLACE INTO rewards (wallet, task_id, reward_hai, timestamp) VALUES (?,?,?,?)",
                (wallet, job_id, reward_hai, completed),
            )
            total_hai += reward_hai
            succeeded += 1
            per_model[entry["name"]] = per_model.get(entry["name"], 0) + 1
            tally = per_node.setdefault(node["node_id"], {"hai": 0.0, "tasks": 0})
            tally["hai"] += reward_hai
            tally["tasks"] += 1
        else:
            failed += 1

    for node in DEMO_NODES:
        conn.execute(
            "INSERT OR REPLACE INTO node_wallets (node_id, wallet, node_name, updated_at) VALUES (?,?,?,?)",
            (node["node_id"], rng.choice(DEMO_WALLETS), node["node_name"], now),
        )

    astra = seed_astra_runs(conn, rng, max(24, job_count // 6), window, now)

    conn.commit()
    conn.close()

    write_nodes_file(per_node, rng, now, nodes_file)

    top_model = max(per_model.items(), key=lambda kv: kv[1])[0] if per_model else None
    return {
        "astra_runs": astra["runs"],
        "astra_credits": astra["credits"],
        "jobs": job_count,
        "succeeded": succeeded,
        "failed": failed,
        "total_hai": round(total_hai, 6),
        "top_model": top_model,
        "success_rate": round(100.0 * succeeded / job_count, 1) if job_count else 0.0,
    }


def write_nodes_file(per_node: Dict[str, Dict[str, float]], rng: random.Random, now: float, nodes_file: Path) -> None:
    """Write the demo node registry so the dashboard's per-node HAI column agrees
    with the ledger.

    The coordinator only credits ``NODES[id]["rewards"]`` when a result arrives
    over ``/results``, and the seeder writes history straight to SQLite. Without
    this the dashboard would show a healthy network total beside a column of
    zeroes. ``/register`` leaves ``rewards`` untouched on nodes it already knows,
    so the values survive every heartbeat.
    """
    from datetime import datetime, timezone

    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")
    payload: Dict[str, Any] = {}
    for node in DEMO_NODES:
        tally = per_node.get(node["node_id"], {"hai": 0.0, "tasks": 0})
        payload[node["node_id"]] = {
            "os": node["os"],
            "gpu": {
                "gpu_name": node["gpu"],
                "memory_total_mb": node["vram_mb"],
                "memory_used_mb": int(node["vram_mb"] * 0.6),
                "utilization": rng.randint(40, 90),
            },
            "role": "creator",
            "node_name": node["node_name"],
            "version": "demo",
            "pipelines": node["pipelines"],
            "models": [],
            "supports": ["image", "video"],
            "rewards": round(tally["hai"], 6),
            "tasks_completed": int(tally["tasks"]),
            "utilization": rng.randint(40, 90),
            "current_task": None,
            "last_result": {},
            "start_time": now - 86_400,
            "last_seen": stamp,
        }
    nodes_file.write_text(json.dumps(payload, indent=2))


def heartbeat(server: str, interval: float) -> None:
    """Keep the demo nodes online. The coordinator drops a node after 120s."""
    import requests

    print(f"[heartbeat] holding {len(DEMO_NODES)} nodes online against {server}")
    print("[heartbeat] ctrl-c to stop\n")
    started = time.time()
    beat = 0
    while True:
        beat += 1
        alive = 0
        for node in DEMO_NODES:
            payload = {
                "node_id": node["node_id"],
                "node_name": node["node_name"],
                "os": node["os"],
                "gpu": node["gpu"],
                "gpu_stats": {
                    "gpu_name": node["gpu"],
                    "memory_total_mb": node["vram_mb"],
                    "memory_used_mb": random.randint(int(node["vram_mb"] * 0.25), int(node["vram_mb"] * 0.85)),
                    "utilization": random.randint(35, 96),
                },
                "role": "creator",
                "version": "demo",
                "pipelines": node["pipelines"],
                "supports": ["image", "video"],
                "start_time": started,
                "uptime": time.time() - started,
            }
            try:
                response = requests.post(f"{server}/register", json=payload, timeout=5)
                if response.ok:
                    alive += 1
            except requests.RequestException as exc:
                print(f"[heartbeat] {node['node_id']}: {exc}")
        print(f"[heartbeat] beat {beat}: {alive}/{len(DEMO_NODES)} online", flush=True)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="database to seed (default: db/demo.db)")
    parser.add_argument(
        "--nodes-file",
        default=str(DEFAULT_NODES_FILE),
        help="node registry to write (default: nodes.demo.json; point the coordinator at it with HAVNAI_NODES_PATH)",
    )
    parser.add_argument("--jobs", type=int, default=340, help="how many jobs of history to write")
    parser.add_argument("--hours", type=float, default=24.0, help="spread history over this many hours")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed; keeps runs reproducible")
    parser.add_argument("--reset", action="store_true", help="delete the database before seeding")
    parser.add_argument("--serve", action="store_true", help="after seeding, hold nodes online")
    parser.add_argument("--server", default="http://127.0.0.1:5001", help="coordinator base URL")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between heartbeats")
    parser.add_argument("--skip-seed", action="store_true", help="heartbeat only, leave the database alone")
    parser.add_argument(
        "--i-know-this-is-the-real-ledger",
        action="store_true",
        help="permit writing to db/ledger.db",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if db_path == LEDGER_DB.resolve() and not args.i_know_this_is_the_real_ledger:
        print(f"refusing to seed the production ledger at {db_path}")
        print("pass --i-know-this-is-the-real-ledger if you truly mean it")
        return 1

    if not args.skip_seed:
        if args.reset and db_path.exists():
            db_path.unlink()
            print(f"removed {db_path}")

        nodes_file = Path(args.nodes_file).resolve()
        summary = seed(db_path, args.jobs, args.hours, args.seed, nodes_file)
        print(f"seeded {db_path}")
        print(f"  nodes       {nodes_file.name}")
        print(f"  jobs        {summary['jobs']} ({summary['succeeded']} ok / {summary['failed']} failed)")
        print(f"  success     {summary['success_rate']}%")
        print(f"  distributed {summary['total_hai']} HAI")
        print(f"  top model   {summary['top_model']}")
        print(f"  astra runs  {summary['astra_runs']} ({summary['astra_credits']} credits paid)")
        print()
        print("boot the coordinator against it:")
        print(f"  cd server && HAVNAI_DB_PATH={db_path} \\")
        print(f"    HAVNAI_NODES_PATH={nodes_file} SERVER_PORT=5001 python app.py")
        print()

    if args.serve:
        try:
            heartbeat(args.server, args.interval)
        except KeyboardInterrupt:
            print("\n[heartbeat] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

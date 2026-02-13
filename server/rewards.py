"""Reward computation for HavnAI node workers."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
LOGGER: logging.Logger
MODEL_WEIGHTS: Dict[str, float]
REWARD_CONFIG: Dict[str, float]


def resolve_weight(model_name: str, default: float = 1.0) -> float:
    return float(MODEL_WEIGHTS.get(model_name, default))


def compute_reward(
    model_name: str,
    pipeline: str,
    metrics: Dict[str, Any],
    status: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute dynamic reward for a completed job.

    Formula:
        reward = base_reward * weight_factor * compute_cost_factor
                  * runtime_factor * success_factor
    """

    try:
        base_reward = float(REWARD_CONFIG.get("base_reward", 0.05))
        # Weight factor based on manifest/model weight
        model_weight = resolve_weight(model_name, 10.0)
        weight_factor = model_weight / 10.0

        # Compute-cost factor based on pipeline family
        pipeline_norm = (pipeline or "sd15").lower()
        if pipeline_norm == "sdxl":
            compute_cost_factor = float(REWARD_CONFIG.get("sdxl_factor", 1.5))
        elif pipeline_norm == "sd15":
            compute_cost_factor = float(REWARD_CONFIG.get("sd15_factor", 1.0))
        elif pipeline_norm == "ltx2":
            compute_cost_factor = float(REWARD_CONFIG.get("ltx2_factor", 2.0))
        elif pipeline_norm in {"anime", "cartoon"}:
            compute_cost_factor = float(REWARD_CONFIG.get("anime_factor", 0.7))
        else:
            compute_cost_factor = 1.0

        # Runtime factor from actual runtime vs baseline
        baseline_runtime = float(REWARD_CONFIG.get("baseline_runtime", 8.0)) or 8.0
        runtime_sec = 0.0
        inf_ms = metrics.get("inference_time_ms")
        if isinstance(inf_ms, (int, float)) and inf_ms > 0:
            runtime_sec = float(inf_ms) / 1000.0
        else:
            dur = metrics.get("duration")
            if isinstance(dur, (int, float)) and dur > 0:
                runtime_sec = float(dur)
        runtime_sec = max(0.0, runtime_sec)
        runtime_factor = max(1.0, runtime_sec / baseline_runtime) if baseline_runtime > 0 else 1.0

        # Success / failure factor
        status_norm = (status or "").lower()
        success_factor = 1.0 if status_norm == "success" else 0.0

        reward = base_reward * weight_factor * compute_cost_factor * runtime_factor * success_factor
        reward = round(float(reward), 6)

        factors = {
            "base_reward": base_reward,
            "weight_factor": weight_factor,
            "compute_cost_factor": compute_cost_factor,
            "runtime_factor": runtime_factor,
            "success_factor": success_factor,
            "runtime_seconds": runtime_sec,
            "model_weight": model_weight,
            "pipeline": pipeline_norm,
            # TODO: future quality verification boost
            # "quality_factor": 1.0,
        }
        return reward, factors
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Reward computation failed for %s: %s", model_name, exc)
        return 0.0, {
            "base_reward": float(REWARD_CONFIG.get("base_reward", 0.05)),
            "weight_factor": 0.0,
            "compute_cost_factor": 1.0,
            "runtime_factor": 1.0,
            "success_factor": 0.0,
            "runtime_seconds": 0.0,
            "error": str(exc),
        }


def record_reward(wallet: Optional[str], task_id: str, reward: float) -> None:
    if not wallet:
        return
    conn = get_db()
    conn.execute(
        """
        INSERT INTO rewards (wallet, task_id, reward_hai, timestamp)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            wallet=excluded.wallet,
            reward_hai=excluded.reward_hai,
            timestamp=excluded.timestamp
        """,
        (wallet, task_id, reward, time.time()),
    )
    conn.commit()

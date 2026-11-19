"""Lock the default HAVN-47 drill prices to the account charging formula."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))

import account_jobs  # type: ignore
import account_ledger  # type: ignore
import credits  # type: ignore
import mixed_model_worker_drill as drill  # type: ignore


def test_default_thirty_job_drill_costs_seventy_credits():
    registry = json.loads((ROOT / "server" / "manifests" / "registry.json").read_text(encoding="utf-8"))
    by_name = {str(model["name"]).lower(): model for model in registry["models"]}
    credits.get_model_config = lambda name: by_name.get(str(name).lower())

    image = credits.resolve_credit_cost("juggernautxl_ragnarokby", "IMAGE_GEN")
    video = credits.resolve_credit_cost("ltx_video_distilled", "LTX_VIDEO_GEN")
    music = credits.resolve_credit_cost("ace_step_1_5_turbo", "MUSIC_GEN")
    assert (image, video, music) == (1.0, 5.0, 1.0)

    plan = drill.build_plan(
        count=30, image_models=["juggernautxl_ragnarokby"], video_models=["ltx_video_distilled"],
        music_models=["ace_step_1_5_turbo"], wallet="w",
    )
    units = 0
    for job in plan:
        cost = {"IMAGE_GEN": image, "LTX_VIDEO_GEN": video, "MUSIC_GEN": music}[job.task_type]
        units += account_jobs.cost_units(cost, int(job.account_payload.get("batch_size", 1)))
    assert account_ledger.SCALE == 1000
    assert units == 70_000
    assert units / account_ledger.SCALE == 70

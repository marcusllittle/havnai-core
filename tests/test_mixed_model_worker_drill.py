from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mixed_model_worker_drill as drill  # type: ignore


def test_build_plan_balances_30_job_mixed_model_sequence() -> None:
    plan = drill.build_plan(
        count=30,
        image_models=["image_a", "image_b"],
        video_models=["video_a"],
        music_models=["music_a"],
        wallet="0x0000000000000000000000000000000000000047",
    )

    assert len(plan) == 30
    assert [job.task_type for job in plan[:6]] == [
        "IMAGE_GEN",
        "LTX_VIDEO_GEN",
        "MUSIC_GEN",
        "IMAGE_GEN",
        "LTX_VIDEO_GEN",
        "MUSIC_GEN",
    ]
    assert sum(1 for job in plan if job.task_type == "IMAGE_GEN") == 10
    assert sum(1 for job in plan if job.task_type == "LTX_VIDEO_GEN") == 10
    assert sum(1 for job in plan if job.task_type == "MUSIC_GEN") == 10
    assert {job.model for job in plan if job.task_type == "IMAGE_GEN"} == {"image_a", "image_b"}
    assert [job.account_type for job in plan[:3]] == ["image", "text_to_video", "text_to_music"]
    assert [job.account_payload["type"] for job in plan[:3]] == ["image", "text_to_video", "text_to_music"]


def test_summary_requires_every_job_to_succeed() -> None:
    plan = drill.build_plan(
        count=3,
        image_models=["image_a"],
        video_models=["video_a"],
        music_models=["music_a"],
        wallet="0x0000000000000000000000000000000000000047",
    )
    summary = drill.summarize_results(
        plan,
        {1: "job-1", 2: "job-2", 3: "job-3"},
        {
            "job-1": {"status": "succeeded"},
            "job-2": {"status": "failed"},
            "job-3": {"status": "completed"},
        },
    )

    assert summary["passed"] is False
    assert summary["by_type"]["IMAGE_GEN"]["succeeded"] == 1
    assert summary["by_type"]["LTX_VIDEO_GEN"]["failed"] == 1
    assert summary["by_type"]["MUSIC_GEN"]["succeeded"] == 1
    assert summary["failures"] == [
        {"index": 2, "job_id": "job-2", "type": "LTX_VIDEO_GEN", "model": "video_a", "status": "failed"}
    ]


def test_artifact_only_result_counts_as_success() -> None:
    plan = drill.build_plan(
        count=1,
        image_models=["image_a"],
        video_models=["video_a"],
        music_models=["music_a"],
        wallet="0x0000000000000000000000000000000000000047",
    )
    summary = drill.summarize_results(plan, {1: "job-1"}, {"job-1": {"image_url": "/static/outputs/job-1.png"}})

    assert summary["passed"] is True
    assert summary["completed_jobs"] == 1
    assert summary["by_type"]["IMAGE_GEN"]["succeeded"] == 1


def test_dry_run_cli_prints_redacted_plan_only() -> None:
    script = ROOT / "scripts" / "mixed_model_worker_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--count", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema"] == "havn-47-mixed-model-worker-drill-plan.v1"
    assert payload["execute"] is False
    assert payload["private_account_jobs"] is False
    assert payload["job_count"] == 3
    assert [item["task_type"] for item in payload["mix"]] == ["IMAGE_GEN", "LTX_VIDEO_GEN", "MUSIC_GEN"]
    assert "TOKEN" not in completed.stdout.upper()


def test_dry_run_accepts_account_token_without_printing_it(monkeypatch) -> None:
    script = ROOT / "scripts" / "mixed_model_worker_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--count", "1", "--account-token", "secret-token"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["private_account_jobs"] is True
    assert "secret-token" not in completed.stdout


def test_execute_requires_visible_production_confirmation() -> None:
    script = ROOT / "scripts" / "mixed_model_worker_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--count", "1", "--execute"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "production-visible jobs" in completed.stderr

from __future__ import annotations

import json
import subprocess
import sys
import pytest
from uuid import UUID
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mixed_model_worker_drill as drill  # type: ignore


def sample_plan(count=30):
    return drill.build_plan(count=count, image_models=["image_a"], video_models=["video_a"],
                            music_models=["music_a"], wallet="test-wallet")


def eligible_catalog():
    return {"models": [
        {"name": "IMAGE_A", "available": True, "task_type": "IMAGE_GEN"},
        {"name": "video_a", "available": True, "task_type": "LTX_VIDEO_GEN", "capabilities": ["text_to_video"]},
        {"name": "music_a", "available": True, "task_type": "MUSIC_GEN", "capabilities": ["text_to_music"]},
    ]}


@pytest.mark.parametrize("change,code", [
    ("missing", "model_not_found"), ("offline", "model_unavailable"),
    ("image_only_video", "model_capability_missing"), ("wrong_task", "model_task_mismatch"),
])
def test_model_catalog_blocks_invalid_plan_independent_of_prices(change, code):
    catalog = eligible_catalog()
    video = catalog["models"][1]
    if change == "missing":
        catalog["models"].remove(video)
    elif change == "offline":
        video["available"] = False
    elif change == "image_only_video":
        video["capabilities"] = ["image_to_video"]
    else:
        video["task_type"] = "IMAGE_GEN"
    result = drill.validate_model_eligibility(sample_plan(), catalog)
    assert result == {"passed": False, "blockers": [{"model": "video_a", "type": "text_to_video", "code": code}]}


def test_catalog_supports_case_insensitive_image_and_explicit_video_music_modes():
    assert drill.validate_model_eligibility(sample_plan(), eligible_catalog())["passed"] is True
    catalog = eligible_catalog()
    catalog["models"][2]["capabilities"] = []
    assert drill.validate_model_eligibility(sample_plan(), catalog)["blockers"][0]["code"] == "model_capability_missing"


def test_default_video_is_explicit_text_to_video_model():
    assert drill.parse_args([]).video_models == "ltx_video_distilled"


@pytest.mark.parametrize("mode", ["--execute", "--preflight"])
def test_invalid_catalog_stops_before_account_prices_or_job_submission(monkeypatch, capsys, mode):
    calls = []

    def fake_json(self, method, path, **kwargs):
        calls.append((method, path))
        assert path == "/models/list"
        return {"models": []}

    monkeypatch.setattr(drill.CoordinatorClient, "_json", fake_json)
    assert drill.main([mode, "--account-token", "secret-token"]) == 2
    assert calls == [("GET", "/models/list")]
    output = capsys.readouterr().out
    assert "model_not_found" in output
    assert "secret-token" not in output


def test_music_plan_pins_one_take_instead_of_inheriting_manifest_batch_default():
    music = [job for job in sample_plan() if job.task_type == "MUSIC_GEN"]
    assert len(music) == 10
    assert all(job.account_payload["batch_size"] == job.payload["batch_size"] == 1 for job in music)


def test_preflight_uses_real_get_contracts_and_keeps_auth_off_public_price_quotes(monkeypatch):
    from io import BytesIO

    seen = []

    def urlopen(req, timeout):
        assert req.get_method() == "GET"
        assert req.data is None
        path = drill.parse.urlsplit(req.full_url).path
        seen.append(path)
        if path == "/v2/account":
            assert req.get_header("Authorization") == "Bearer secret-token"
            data = {"id": "account", "status": "active"}
        elif path == "/v2/account/credits":
            data = {"scale": 1000, "available_units": 70000, "reserved_units": 0}
        else:
            assert path == "/credits/cost"
            assert req.get_header("Authorization") is None
            query = drill.parse.parse_qs(drill.parse.urlsplit(req.full_url).query)
            data = {"cost": {"IMAGE_GEN": 1.0, "LTX_VIDEO_GEN": 5.0, "MUSIC_GEN": 1.0}[query["task_type"][0]],
                    "credits_enabled": False}
        return BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(drill.request, "urlopen", urlopen)
    result = drill.CoordinatorClient("https://coordinator.example", account_token="secret-token").account_preflight(sample_plan())
    # The account route always charges in ledger units, regardless of the legacy
    # credits_enabled flag returned by the public quote endpoint.
    assert result["required_units"] == 70000
    assert result["budget_sufficient"] is True
    assert len(seen) == 5


@pytest.mark.parametrize("success_count,threshold_met", [(29, True), (28, False)])
def test_threshold_reports_95_percent_without_waiving_failed_job_evidence(success_count, threshold_met):
    plan = sample_plan()
    results = {
        str(job.index): {"status": "succeeded" if job.index <= success_count else "failed",
                        "artifacts": [{"kind": {"IMAGE_GEN": "image", "LTX_VIDEO_GEN": "video", "MUSIC_GEN": "audio"}[job.task_type],
                                       "url": "/v2/artifacts/example/content"}]}
        for job in plan
    }
    summary = drill.summarize_results(plan, {job.index: str(job.index) for job in plan}, results)
    assert summary["successful_jobs"] == success_count
    assert summary["success_threshold_met"] is threshold_met
    assert summary["passed"] is False
    assert summary["acceptance_evidence_gaps"]


@pytest.mark.parametrize("result", [
    {"status": "succeeded"},
    {"status": "completed", "artifacts": []},
    {"status": "completed", "artifacts": [{"kind": "audio", "url": "/audio"}]},
    {"status": "completed", "artifacts": [{"kind": "image", "url": " "}]},
    {"status": "completed", "cover_url": "/cover"},
])
def test_terminal_success_requires_primary_artifact(result):
    summary = drill.summarize_results(sample_plan(1), {1: "job"}, {"job": result})
    assert summary["passed"] is False
    assert summary["failures"][0]["status"] == "missing_artifact"
    assert summary["job_evidence"][0]["artifact_present"] is False


def test_private_artifact_evidence_omits_urls_and_account_identity():
    result = {"status": "succeeded", "owner_account_id": "private-owner",
              "created_at": 100, "completed_at": 200,
              "artifacts": [{"kind": "image", "url": "/v2/artifacts/private/content"}]}
    summary = drill.summarize_results(sample_plan(1), {1: "job"}, {"job": result})
    assert summary["passed"] is True
    assert summary["job_evidence"][0]["completed_at"] == 200
    assert "private-owner" not in json.dumps(summary)
    assert "/v2/artifacts/private/content" not in json.dumps(summary)


@pytest.mark.parametrize("available, sufficient", [(65000, True), (64999, False)])
def test_preflight_quotes_entire_plan_in_account_units_without_network_writes(monkeypatch, available, sufficient):
    client = drill.CoordinatorClient("https://coordinator.example", account_token="secret-token")
    calls = []

    def fake_json(method, path, payload=None, headers=None):
        assert method == "GET"
        calls.append(path)
        if path == "/v2/account":
            return {"id": "private-owner", "status": "active"}
        if path == "/v2/account/credits":
            return {"available_units": available, "reserved_units": 1000, "scale": 1000}
        query = drill.parse.parse_qs(drill.parse.urlsplit(path).query)
        return {"cost": {"image_a": "0.5", "video_a": "5", "music_a": "1"}[query["model"][0]]}

    monkeypatch.setattr(client, "_json", fake_json)
    result = client.account_preflight(sample_plan())
    assert result["required_units"] == 65000
    assert result["budget_sufficient"] is sufficient
    assert len(calls) == 5  # account, credits, one quote per task/model pair
    assert "secret-token" not in json.dumps(result)
    assert "private-owner" not in json.dumps(result)


@pytest.mark.parametrize("price", ["NaN", "Infinity", "-1", "0", "0.0001", "not-a-price"])
def test_preflight_rejects_invalid_or_fractional_unit_prices(monkeypatch, price):
    client = drill.CoordinatorClient("https://coordinator.example", account_token="secret-token")

    def fake_json(method, path, **kwargs):
        if path == "/v2/account":
            return {"id": "account"}
        if path == "/v2/account/credits":
            return {"scale": 1000, "available_units": 100000}
        return {"cost": price}

    monkeypatch.setattr(client, "_json", fake_json)
    with pytest.raises(ValueError, match="invalid quoted generation price"):
        client.account_preflight(sample_plan(1))


def test_account_execution_keys_are_unique_per_run_and_recoverable(monkeypatch, capsys):
    keys = []

    class Client:
        def model_eligibility(self, plan):
            return {"passed": True, "blockers": []}

        def __init__(self, *args, **kwargs):
            pass

        def account_preflight(self, plan):
            return {"account_id_present": True, "budget_sufficient": True,
                    "available_units": 70000, "required_units": 70000}

        def submit_account(self, payload, *, idempotency_key):
            keys.append(idempotency_key)
            return idempotency_key

        def account_result(self, job_id):
            return {"status": "succeeded", "image_url": "/image", "video_url": "/video", "audio_url": "/audio"}

        def metrics_text(self):
            return None

    monkeypatch.setattr(drill, "CoordinatorClient", Client)
    args = ["--count", "3", "--execute", "--account-token", "secret-token"]
    reports = []
    for _ in range(2):
        assert drill.main(args) == 0
        output = capsys.readouterr().out
        assert "secret-token" not in output
        decoder = json.JSONDecoder()
        items = []
        while output.strip():
            item, end = decoder.raw_decode(output.lstrip())
            items.append(item)
            output = output.lstrip()[end:]
        reports.append(items)
    first_id = reports[0][0]["run_id"]
    assert str(UUID(first_id)) == first_id
    assert first_id != reports[1][0]["run_id"]
    assert len(set(keys)) == 6
    assert all(len(key) <= 128 for key in keys)
    for items in reports:
        assert items[-1]["run_id"] == items[0]["run_id"]
        assert items[0]["explicit_run_id"] is False
        assert all(item["run_id"] == items[0]["run_id"] for item in items[1:-1])
    assert drill.main(args + ["--run-id", first_id]) == 0
    assert keys[6:] == keys[:3]


def test_invalid_run_id_fails_before_client_creation(monkeypatch):
    import pytest

    def unexpected_client(*args, **kwargs):
        raise AssertionError("invalid run ID reached network client")

    monkeypatch.setattr(drill, "CoordinatorClient", unexpected_client)
    with pytest.raises(SystemExit) as exc:
        drill.main(["--execute", "--run-id", "not-a-uuid"])
    assert exc.value.code == 2


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
            "job-1": {"status": "succeeded", "image_url": "/image"},
            "job-2": {"status": "failed"},
            "job-3": {"status": "completed", "audio_url": "/audio"},
        },
    )

    assert summary["passed"] is False
    assert summary["by_type"]["IMAGE_GEN"]["succeeded"] == 1
    assert summary["by_type"]["LTX_VIDEO_GEN"]["failed"] == 1
    assert summary["by_type"]["MUSIC_GEN"]["succeeded"] == 1
    assert summary["failures"] == [
        {"index": 2, "job_id": "job-2", "type": "LTX_VIDEO_GEN", "model": "video_a", "status": "failed"}
    ]


def test_replayed_job_cannot_count_as_30_successful_worker_executions() -> None:
    plan = drill.build_plan(
        count=30,
        image_models=["image_a"],
        video_models=["video_a"],
        music_models=["music_a"],
        wallet="0x0000000000000000000000000000000000000047",
    )
    summary = drill.summarize_results(
        plan,
        {job.index: "replayed-job" for job in plan},
        {"replayed-job": {"status": "succeeded"}},
    )

    assert summary["passed"] is False
    assert summary["submitted_jobs"] == 30
    assert summary["unique_submitted_jobs"] == 1
    assert summary["completed_jobs"] == 1
    assert summary["duplicate_job_ids"] == {"replayed-job": list(range(1, 31))}
    assert len(summary["failures"]) == 30
    assert all(item["status"] == "duplicate_job_id" for item in summary["failures"])
    assert all(bucket["succeeded"] == 0 for bucket in summary["by_type"].values())


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


def test_duplicate_ids_do_not_invalidate_unrelated_successful_job() -> None:
    plan = drill.build_plan(
        count=3,
        image_models=["image_a"],
        video_models=["video_a"],
        music_models=["music_a"],
        wallet="0x0000000000000000000000000000000000000047",
    )
    summary = drill.summarize_results(
        plan,
        {1: "duplicate", 2: "duplicate", 3: "unique"},
        {"duplicate": {"status": "completed"}, "unique": {"status": "completed", "audio_url": "/audio"}},
    )

    assert summary["passed"] is False
    assert summary["unique_submitted_jobs"] == 2
    assert summary["duplicate_job_ids"] == {"duplicate": [1, 2]}
    assert [item["index"] for item in summary["failures"]] == [1, 2]
    assert summary["by_type"]["MUSIC_GEN"]["succeeded"] == 1


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


def test_preflight_requires_account_token() -> None:
    script = ROOT / "scripts" / "mixed_model_worker_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--count", "1", "--preflight"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    decoder = json.JSONDecoder()
    reports = []
    index = 0
    while index < len(completed.stdout):
        while index < len(completed.stdout) and completed.stdout[index].isspace():
            index += 1
        if index >= len(completed.stdout):
            break
        payload, index = decoder.raw_decode(completed.stdout, index)
        reports.append(payload)
    preflight = reports[-1]
    assert preflight["schema"] == "havn-47-mixed-model-worker-drill-preflight.v1"
    assert preflight["account_token_present"] is False
    assert preflight["passed"] is False
    assert preflight["blockers"] == [
        "No account session source is configured. Pass --token-command, --token-file, "
        "or a short-lived --account-token. /v2 accepts only a Clerk session JWT."
    ]
    assert preflight["session_source"] == "absent"


def test_preflight_reports_funded_account_without_printing_token(monkeypatch, capsys) -> None:
    class Client:
        def model_eligibility(self, plan):
            return {"passed": True, "blockers": []}

        def __init__(self, *args, **kwargs):
            assert kwargs["account_token"] == "secret-token"

        def account_preflight(self, plan):
            return {
                "account_id_present": True,
                "status": "active",
                "available_units": 5000,
                "reserved_units": 0,
                "required_units": 1000,
                "budget_sufficient": True,
            }

    monkeypatch.setattr(drill, "CoordinatorClient", Client)

    code = drill.main(["--count", "1", "--preflight", "--account-token", "secret-token"])
    out = capsys.readouterr().out
    decoder = json.JSONDecoder()
    reports = []
    index = 0
    while index < len(out):
        while index < len(out) and out[index].isspace():
            index += 1
        if index >= len(out):
            break
        payload, index = decoder.raw_decode(out, index)
        reports.append(payload)
    preflight = reports[-1]

    assert code == 0
    assert preflight["schema"] == "havn-47-mixed-model-worker-drill-preflight.v1"
    assert preflight["passed"] is True
    assert preflight["available_units"] == 5000
    assert "secret-token" not in out


def test_preflight_cli_rejects_positive_balance_below_plan_budget(monkeypatch, capsys):
    class Client:
        def model_eligibility(self, plan):
            return {"passed": True, "blockers": []}

        def __init__(self, *args, **kwargs):
            pass

        def account_preflight(self, plan):
            return {"account_id_present": True, "available_units": 1,
                    "required_units": 70000, "budget_sufficient": False}

    monkeypatch.setattr(drill, "CoordinatorClient", Client)
    assert drill.main(["--count", "30", "--preflight", "--account-token", "secret-token"]) == 2
    output = capsys.readouterr().out
    assert "Available credits do not cover the quoted full drill plan." in output
    assert "secret-token" not in output


def test_preflight_cli_returns_redacted_blocker_on_quote_or_auth_failure(monkeypatch, capsys):
    class Client:
        def model_eligibility(self, plan):
            return {"passed": True, "blockers": []}

        def __init__(self, *args, **kwargs):
            pass

        def account_preflight(self, plan):
            raise RuntimeError("upstream response includes secret-token")

    monkeypatch.setattr(drill, "CoordinatorClient", Client)
    assert drill.main(["--preflight", "--account-token", "secret-token"]) == 2
    output = capsys.readouterr().out
    assert "verification failed; no jobs submitted" in output
    assert "secret-token" not in output


def test_prompts_file_overrides_image_and_video_and_leaves_music_pinned(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([
        {"prompt": "image one", "negative_prompt": "blur", "width": 768, "height": 768, "steps": 20, "model": "Image_B"},
        {"index": 2, "type": "text_to_video", "prompt": "video one", "width": 512, "height": 768, "steps": 6},
    ]), encoding="utf-8")
    plan = drill.build_plan(
        count=3, image_models=["image_a"], video_models=["video_a"], music_models=["music_a"],
        wallet="test-wallet", sfw_mode=False, prompt_overrides=drill.load_prompt_overrides(str(path)),
    )
    image, video, music = plan
    assert image.model == "image_b"
    assert image.account_payload["prompt"] == "image one"
    assert image.account_payload["negative_prompt"] == "blur"
    assert image.account_payload["width"] == image.account_payload["height"] == 768
    assert image.account_payload["steps"] == 20
    assert image.account_payload["sfw_mode"] is False
    assert video.account_payload["prompt"] == "video one"
    assert video.account_payload["width"] == 512
    assert video.account_payload["height"] == 768
    assert video.account_payload["steps"] == 6
    assert video.account_payload["sfw_mode"] is False
    assert music.account_payload["batch_size"] == 1
    assert music.account_payload["sfw_mode"] is False
    assert "cinematic synthwave" in music.account_payload["style"]
    assert "launch-quality" in music.prompt


def test_prompts_file_rejects_a_short_list_and_bad_video_dimensions(tmp_path):
    short = tmp_path / "short.json"
    short.write_text(json.dumps([{"prompt": "only image"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="20 image and video"):
        drill.build_plan(count=30, image_models=["image_a"], video_models=["video_a"], music_models=["music_a"],
                         wallet="w", prompt_overrides=drill.load_prompt_overrides(str(short)))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"prompt": "image"},
        {"prompt": "video", "width": 500, "height": 832},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="multiples of 32"):
        drill.build_plan(count=3, image_models=["image_a"], video_models=["video_a"], music_models=["music_a"],
                         wallet="w", prompt_overrides=drill.load_prompt_overrides(str(bad)))


def test_dry_run_shows_prompts_and_does_not_mint_a_session(monkeypatch, tmp_path, capsys):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([
        {"prompt": "kept private"},
        {"prompt": "camera move", "width": 480, "height": 832, "steps": 8},
    ]), encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("token command ran during dry-run")

    monkeypatch.setattr(drill.subprocess, "run", boom)
    code = drill.main([
        "--count", "3", "--dry-run", "--no-sfw-mode", "--prompts-file", str(path),
        "--token-command", "python3 scripts/clerk_session_token.py",
    ])
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["execute"] is False
    assert payload["session_source"] == "command"
    assert payload["private_account_jobs"] is True
    assert payload["sfw_mode"] is False
    assert payload["mix"][0]["prompt"] == "kept private"
    assert payload["mix"][0]["sfw_mode"] is False
    assert payload["mix"][1]["width"] == 480
    assert "TOKEN" not in output.upper()


def test_token_sources_are_exclusive_and_command_failures_are_redacted(monkeypatch):
    assert drill.main(["--account-token", "secret-token", "--token-file", "session.jwt"]) == 2
    script = ROOT / "scripts" / "mixed_model_worker_drill.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], check=True, capture_output=True, text=True,
    )
    assert "--token-command" in completed.stdout
    assert "--prompts-file" in completed.stdout
    assert "--no-sfw-mode" in completed.stdout
    assert "docs/havn-47-mixed-model-drill.md" in completed.stdout

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature\n",
            stderr="failed sk_test_secretvalue eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature",
        )

    monkeypatch.setattr(drill.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="account session command failed") as exc:
        drill.AccountTokenSource(command="python3 scripts/clerk_session_token.py").token()
    assert "eyJ" not in str(exc.value)
    assert "sk_test" not in str(exc.value)


def test_token_file_is_reread_before_each_account_request(monkeypatch, tmp_path):
    path = tmp_path / "session"
    path.write_text("first-session\n", encoding="utf-8")
    seen = []

    def urlopen(req, timeout):
        from io import BytesIO
        header = req.get_header("Authorization")
        route = drill.parse.urlsplit(req.full_url).path
        seen.append((route, header))
        if route == "/v2/account":
            path.write_text("second-session\n", encoding="utf-8")
            data = {"id": "account", "status": "active"}
        elif route == "/v2/account/credits":
            assert header == "Bearer second-session"
            data = {"scale": 1000, "available_units": 70000, "reserved_units": 0}
        else:
            assert header is None
            data = {"cost": 1}
        return BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(drill.request, "urlopen", urlopen)
    client = drill.CoordinatorClient(
        "https://coordinator.example",
        account_token_source=drill.AccountTokenSource(path=str(path)).token,
    )
    result = client.account_preflight(drill.build_plan(
        count=3, image_models=["image_a"], video_models=["video_a"], music_models=["music_a"], wallet="w",
    ))
    assert result["budget_sufficient"] is True
    assert seen[0] == ("/v2/account", "Bearer first-session")
    assert seen[1][0] == "/v2/account/credits"
    assert all(header is None for route, header in seen if route == "/credits/cost")


def test_execute_stops_before_submit_when_credits_do_not_cover_the_plan(monkeypatch, capsys):
    submitted = []

    class Client:
        def __init__(self, *args, **kwargs):
            assert kwargs["account_token"] == "secret-token"

        def model_eligibility(self, plan):
            return {"passed": True, "blockers": []}

        def account_preflight(self, plan):
            return {"account_id_present": True, "available_units": 29000,
                    "required_units": 70000, "budget_sufficient": False}

        def submit_account(self, payload, *, idempotency_key):
            submitted.append(idempotency_key)
            raise AssertionError("short balance submitted a job")

    monkeypatch.setattr(drill, "CoordinatorClient", Client)
    assert drill.main(["--count", "30", "--execute", "--account-token", "secret-token"]) == 2
    output = capsys.readouterr().out
    assert submitted == []
    assert "secret-token" not in output
    assert "70000" in output

"""Run a mixed-model worker stability drill for HAVN-47.

The drill queues a fixed mix of image, video, and music jobs through the same
coordinator routes used by production workers, then polls until each job reaches
a terminal state. By default it prints a dry-run plan and performs no network
writes. Use --execute only from an operator shell with a funded test account
and the required coordinator tokens in the environment.

Each invocation creates a fresh run UUID, recorded before submission and in the
summary. A new run creates new billable jobs. After interruption, recover the
same account-owned requests with --run-id <recorded UUID> and identical plan
arguments. The server rejects changed payloads for an existing key. Recovery
may replay completed jobs: never count it as a fresh worker acceptance run.
Run IDs are not credentials. Keep the account token in the private environment.

The summary's passed field remains a strict all-jobs execution check. The
separate success_threshold_met field reports Jira's 95% numerical criterion;
neither field proves ledger integrity, artifact readability or no manual reboot.
Preflight prices every planned submission using current coordinator quotes and
the account credit scale. Quotes are not a reservation or a price lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib import error, parse, request
from uuid import UUID, uuid4


DEFAULT_JOB_COUNT = 30
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 3
DEFAULT_SUBMIT_RETRY_SECONDS = 65.0
TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled", "expired"}
ARTIFACT_FIELDS = ("image_url", "video_url", "audio_url", "cover_url", "output_url")
PRODUCTION_VISIBLE_WARNING = (
    "live execution creates production-visible jobs; use a private/internal "
    "environment, or pass --allow-production-visible only after confirming "
    "dashboard/gallery surfaces will not show rough QA outputs"
)


@dataclass(frozen=True)
class DrillJob:
    index: int
    task_type: str
    account_type: str
    model: str
    prompt: str
    payload: dict[str, Any]
    account_payload: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def build_plan(
    *,
    count: int,
    image_models: list[str],
    video_models: list[str],
    music_models: list[str],
    wallet: str,
) -> list[DrillJob]:
    if count <= 0:
        raise ValueError("count must be positive")
    pools = [
        ("IMAGE_GEN", image_models),
        ("LTX_VIDEO_GEN", video_models),
        ("MUSIC_GEN", music_models),
    ]
    if not all(models for _kind, models in pools):
        raise ValueError("provide at least one image, video, and music model")

    plan: list[DrillJob] = []
    for index in range(count):
        task_type, models = pools[index % len(pools)]
        model = models[(index // len(pools)) % len(models)]
        prompt = (
            f"HAVN-47 launch-quality stability drill #{index + 1}: cinematic HavnAI operations suite, "
            "premium commercial key art, polished lighting, coherent composition"
        )
        payload: dict[str, Any] = {
            "wallet": wallet,
            "model": model,
            "prompt": prompt,
            "seed": 47000 + index,
            "task_type": task_type,
        }
        account_type = {
            "IMAGE_GEN": "image",
            "LTX_VIDEO_GEN": "text_to_video",
            "MUSIC_GEN": "text_to_music",
        }[task_type]
        account_payload: dict[str, Any] = {
            "type": account_type,
            "model": model,
            "prompt": prompt,
            "seed": 47000 + index,
            "sfw_mode": True,
        }
        if task_type == "IMAGE_GEN":
            payload.update({"steps": 32, "width": 832, "height": 1216, "guidance": 6.5})
            account_payload.update({"steps": 32, "width": 832, "height": 1216, "guidance": 6.5})
        elif task_type == "LTX_VIDEO_GEN":
            payload.update({"steps": 4, "frames": 33, "fps": 16, "duration_seconds": 3})
            account_payload.update({"steps": 4, "frames": 33, "fps": 16, "duration_seconds": 3})
        elif task_type == "MUSIC_GEN":
            # Pin the batch rather than inheriting mutable manifest defaults:
            # account charging multiplies the per-job price by resolved batch_size.
            payload.update({"style": "cinematic synthwave", "instrumental": True, "duration": 30, "batch_size": 1})
            account_payload.update({"style": "cinematic synthwave", "instrumental": True, "duration": 30, "batch_size": 1})
        plan.append(DrillJob(index=index + 1, task_type=task_type, account_type=account_type, model=model, prompt=prompt,
                             payload=payload, account_payload=account_payload))
    return plan


def summarize_results(plan: list[DrillJob], job_ids: dict[int, str], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    job_evidence: list[dict[str, Any]] = []
    submitted_indices: dict[str, list[int]] = {}
    for planned in plan:
        job_id = job_ids.get(planned.index)
        if job_id:
            submitted_indices.setdefault(job_id, []).append(planned.index)
    duplicate_job_ids = {job_id: indices for job_id, indices in submitted_indices.items() if len(indices) > 1}
    for planned in plan:
        bucket = by_type.setdefault(planned.task_type, {"planned": 0, "succeeded": 0, "failed": 0, "other": 0})
        bucket["planned"] += 1
        job_id = job_ids.get(planned.index)
        result = statuses.get(job_id or "") or {}
        status = result_status(result)
        artifact_present = has_primary_artifact(result, planned.task_type)
        if job_id in duplicate_job_ids:
            status = "duplicate_job_id"
        elif status in {"succeeded", "completed"} and not artifact_present:
            status = "missing_artifact"
        job_evidence.append({
            "index": planned.index, "job_id": job_id, "task_type": planned.task_type,
            "model": planned.model, "terminal_status": result_status(result),
            "artifact_present": artifact_present, "validation_status": status,
            "created_at": result.get("created_at"), "completed_at": result.get("completed_at"),
            "error_code": result.get("error_code"),
        })
        if status in {"succeeded", "completed"}:
            bucket["succeeded"] += 1
        elif status in {"failed", "cancelled", "expired", "not_submitted", "duplicate_job_id", "missing_artifact"}:
            bucket["failed"] += 1
            failures.append({"index": planned.index, "job_id": job_id, "type": planned.task_type, "model": planned.model, "status": status})
        else:
            bucket["other"] += 1
            failures.append({"index": planned.index, "job_id": job_id, "type": planned.task_type, "model": planned.model, "status": status})
    successful_jobs = sum(bucket["succeeded"] for bucket in by_type.values())
    return {
        "schema": "havn-47-mixed-model-worker-drill.v1",
        "generated_at": _utc_now(),
        "planned_jobs": len(plan),
        "submitted_jobs": len(job_ids),
        "unique_submitted_jobs": len(submitted_indices),
        "duplicate_job_ids": duplicate_job_ids,
        "successful_jobs": successful_jobs,
        "success_rate": successful_jobs / len(plan) if plan else 0,
        "success_threshold": 0.95,
        "success_threshold_met": bool(plan) and successful_jobs * 100 >= len(plan) * 95,
        "minimum_acceptance_jobs_met": len(plan) >= DEFAULT_JOB_COUNT,
        "job_evidence": job_evidence,
        "acceptance_evidence_gaps": [
            "artifact content retrieval is not verified",
            "account ledger, receipt and reservation audit is not collected",
            "retry counts and automatic recovery evidence are not collected",
            "no-manual-reboot evidence requires operator confirmation",
        ],
        "completed_jobs": sum(1 for data in statuses.values() if result_status(data) in TERMINAL_STATUSES),
        "by_type": by_type,
        "failures": failures,
        "passed": len(job_ids) == len(plan) and not failures,
    }


class CoordinatorClient:
    def __init__(
        self,
        base_url: str,
        node_token: str | None = None,
        submit_retry_seconds: float = DEFAULT_SUBMIT_RETRY_SECONDS,
        account_token: str | None = None,
    ) -> None:
        self.base_url = _clean_base_url(base_url)
        self.node_token = node_token
        self.submit_retry_seconds = submit_retry_seconds
        self.account_token = account_token

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if method == "GET" and exc.code == 404:
                return {"status": "pending", "error": "result_not_found", "detail": detail[:500]}
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail[:500]}") from exc
        return json.loads(data or "{}")

    def submit(self, payload: dict[str, Any]) -> str:
        while True:
            try:
                response = self._json("POST", "/submit-job", payload)
                break
            except RuntimeError as exc:
                if "HTTP 429" not in str(exc):
                    raise
                print(json.dumps({"event": "submit_rate_limited", "retry_seconds": self.submit_retry_seconds}), flush=True)
                time.sleep(self.submit_retry_seconds)
        job_id = str(response.get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError(f"submit response did not include job_id: {response}")
        return job_id

    def submit_account(self, payload: dict[str, Any], *, idempotency_key: str) -> str:
        if not self.account_token:
            raise RuntimeError("account token required")
        headers = {"Authorization": f"Bearer {self.account_token}", "Idempotency-Key": idempotency_key}
        response = self._json("POST", "/v2/jobs", payload, headers=headers)
        job_id = str(response.get("id") or "").strip()
        if not job_id:
            raise RuntimeError(f"account submit response did not include id: {response}")
        return job_id

    def result(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/result/{parse.quote(job_id)}")

    def account_result(self, job_id: str) -> dict[str, Any]:
        if not self.account_token:
            raise RuntimeError("account token required")
        return self._json("GET", f"/v2/jobs/{parse.quote(job_id)}", headers={"Authorization": f"Bearer {self.account_token}"})

    def account_preflight(self, plan: list[DrillJob]) -> dict[str, Any]:
        if not self.account_token:
            raise RuntimeError("account token required")
        headers = {"Authorization": f"Bearer {self.account_token}"}
        account = self._json("GET", "/v2/account", headers=headers)
        credits = self._json("GET", "/v2/account/credits", headers=headers)
        scale = int(credits.get("scale") or 0)
        if scale <= 0:
            raise ValueError("account credit scale must be positive")
        quoted_units: dict[tuple[str, str], int] = {}
        required_units = 0
        for job in plan:
            key = (job.model, job.task_type)
            if key not in quoted_units:
                quote = self._json("GET", "/credits/cost?" + parse.urlencode({"model": job.model, "task_type": job.task_type}))
                try:
                    units = Decimal(str(quote.get("cost"))) * scale
                    if not units.is_finite() or units <= 0 or units != units.to_integral_value():
                        raise ValueError("invalid quoted generation price")
                    quoted_units[key] = int(units)
                except InvalidOperation as exc:
                    raise ValueError("invalid quoted generation price") from exc
            required_units += quoted_units[key] * int(job.account_payload.get("batch_size", 1))
        return {
            "account_id_present": bool(str(account.get("id") or "").strip()),
            "status": account.get("status"),
            "available_units": credits.get("available_units"),
            "reserved_units": credits.get("reserved_units"),
            "required_units": required_units,
            "credit_scale": scale,
            "budget_sufficient": int(credits.get("available_units") or 0) >= required_units,
            "budget_basis": "current /credits/cost quotes for every planned submission; not a price lock",
        }

    def model_eligibility(self, plan: list[DrillJob]) -> dict[str, Any]:
        return validate_model_eligibility(plan, self._json("GET", "/models/list"))

    def metrics_text(self) -> str | None:
        if not self.node_token:
            return None
        req = request.Request(self.base_url + "/metrics", headers={"X-HavnAI-Token": self.node_token})
        try:
            with request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", "replace")
        except Exception:
            return None


def result_status(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").strip().lower()
    if status:
        return status
    if any(result.get(field) for field in ARTIFACT_FIELDS):
        return "succeeded"
    if result.get("error"):
        return "failed"
    return "pending"


def has_primary_artifact(result: dict[str, Any], task_type: str) -> bool:
    kind, field = {"IMAGE_GEN": ("image", "image_url"), "LTX_VIDEO_GEN": ("video", "video_url"),
                   "MUSIC_GEN": ("audio", "audio_url")}[task_type]
    if isinstance(result.get(field), str) and result[field].strip():
        return True
    artifacts = result.get("artifacts")
    return isinstance(artifacts, list) and any(
        isinstance(item, dict) and item.get("kind") == kind
        and isinstance(item.get("url"), str) and item["url"].strip()
        for item in artifacts
    )


def validate_model_eligibility(plan: list[DrillJob], catalog: dict[str, Any]) -> dict[str, Any]:
    models = catalog.get("models")
    if not isinstance(models, list):
        return {"passed": False, "blockers": [{"code": "invalid_model_catalog"}]}
    by_name = {str(item.get("name") or "").lower(): item for item in models if isinstance(item, dict)}
    blockers = []
    checked = set()
    compatible_tasks = {"image": {"IMAGE_GEN"}, "text_to_video": {"LTX_VIDEO_GEN", "VIDEO_GEN", "ANIMATEDIFF"},
                        "text_to_music": {"MUSIC_GEN"}}
    for job in plan:
        key = (job.model.lower(), job.account_type)
        if key in checked:
            continue
        checked.add(key)
        model = by_name.get(key[0])
        code = None
        if model is None:
            code = "model_not_found"
        elif model.get("available") is not True:
            code = "model_unavailable"
        elif str(model.get("task_type") or "").upper() not in compatible_tasks[job.account_type]:
            code = "model_task_mismatch"
        elif job.account_type in {"text_to_video", "text_to_music"} and job.account_type not in (model.get("capabilities") or []):
            code = "model_capability_missing"
        if code:
            blockers.append({"model": job.model, "type": job.account_type, "code": code})
    return {"passed": not blockers, "blockers": blockers}


def _split_models(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--wallet", default=os.environ.get("HAVNAI_DRILL_WALLET", "0x0000000000000000000000000000000000000047"))
    parser.add_argument("--image-models", default=os.environ.get("HAVNAI_DRILL_IMAGE_MODELS", "juggernautxl_ragnarokby"))
    parser.add_argument("--video-models", default=os.environ.get("HAVNAI_DRILL_VIDEO_MODELS", "ltx_video_distilled"))
    parser.add_argument("--music-models", default=os.environ.get("HAVNAI_DRILL_MUSIC_MODELS", "ace_step_1_5_turbo"))
    parser.add_argument("--count", type=int, default=DEFAULT_JOB_COUNT)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--submit-retry-seconds", type=float, default=DEFAULT_SUBMIT_RETRY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--execute", action="store_true", help="Queue jobs and poll live coordinator results")
    parser.add_argument("--run-id", type=UUID, default=None,
                        help="UUID for this drill; reuse only to recover the same run, not for fresh acceptance")
    parser.add_argument("--preflight", action="store_true", help="Validate private account token and credits without submitting jobs")
    parser.add_argument("--account-token", default=os.environ.get("HAVNAI_DRILL_ACCOUNT_TOKEN", ""),
                        help="bearer token for private account-owned drill jobs; preferred for live acceptance")
    parser.add_argument("--allow-production-visible", action="store_true", help="confirm live execution may create visible production jobs")
    parser.add_argument("--node-token", default=os.environ.get("HAVNAI_NODE_TOKEN") or os.environ.get("SERVER_JOIN_TOKEN"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    run_id = str(args.run_id or uuid4())
    plan = build_plan(
        count=args.count,
        image_models=_split_models(args.image_models),
        video_models=_split_models(args.video_models),
        music_models=_split_models(args.music_models),
        wallet=args.wallet,
    )
    print(json.dumps({
        "schema": "havn-47-mixed-model-worker-drill-plan.v1",
        "run_id": run_id,
        "explicit_run_id": args.run_id is not None,
        "generated_at": _utc_now(),
        "base_url": _clean_base_url(args.base_url),
        "execute": bool(args.execute),
        "job_count": len(plan),
        "private_account_jobs": bool(args.account_token.strip()),
        "mix": [{"index": job.index, "task_type": job.task_type, "account_type": job.account_type, "model": job.model} for job in plan],
    }, indent=2, sort_keys=True), flush=True)
    if args.execute and not args.preflight and not args.account_token.strip() and not args.allow_production_visible:
        raise SystemExit(PRODUCTION_VISIBLE_WARNING)
    if args.execute or (args.preflight and args.account_token.strip()):
        eligibility_client = CoordinatorClient(args.base_url, account_token=args.account_token.strip() or None)
        try:
            eligibility = eligibility_client.model_eligibility(plan)
        except (ValueError, RuntimeError, error.URLError):
            eligibility = {"passed": False, "blockers": [{"code": "model_catalog_unavailable"}]}
        print(json.dumps({"schema": "havn-47-model-eligibility.v1", "run_id": run_id, **eligibility}, sort_keys=True), flush=True)
        if not eligibility["passed"]:
            return 2
    if args.preflight:
        account_token = args.account_token.strip()
        if not account_token:
            print(json.dumps({
                "schema": "havn-47-mixed-model-worker-drill-preflight.v1",
                "generated_at": _utc_now(),
                "base_url": _clean_base_url(args.base_url),
                "account_token_present": False,
                "account_id_present": False,
                "available_units": None,
                "reserved_units": None,
                "required_units": None,
                "budget_sufficient": False,
                "passed": False,
                "blockers": ["HAVNAI_DRILL_ACCOUNT_TOKEN is not configured; funded private account token required."],
            }, indent=2, sort_keys=True))
            return 2
        client = CoordinatorClient(args.base_url, node_token=args.node_token, submit_retry_seconds=args.submit_retry_seconds,
                                   account_token=account_token)
        try:
            preflight = client.account_preflight(plan)
        except (ValueError, RuntimeError, error.URLError):
            # Do not print coordinator error bodies or bearer-token details.
            print(json.dumps({
                "schema": "havn-47-mixed-model-worker-drill-preflight.v1",
                "generated_at": _utc_now(),
                "base_url": _clean_base_url(args.base_url),
                "account_token_present": True,
                "required_units": None,
                "budget_sufficient": False,
                "passed": False,
                "blockers": ["Private account or full-plan price verification failed; no jobs submitted."],
            }, indent=2, sort_keys=True))
            return 2
        result = {
            "schema": "havn-47-mixed-model-worker-drill-preflight.v1",
            "generated_at": _utc_now(),
            "base_url": _clean_base_url(args.base_url),
            "account_token_present": True,
            **preflight,
            "passed": bool(preflight.get("account_id_present")) and bool(preflight.get("budget_sufficient")),
            "blockers": ([] if preflight.get("account_id_present") else ["Private account identity is missing."])
                        + ([] if preflight.get("budget_sufficient") else ["Available credits do not cover the quoted full drill plan."]),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if not args.execute:
        return 0
    account_token = args.account_token.strip()
    if not account_token and not args.allow_production_visible:
        raise SystemExit(PRODUCTION_VISIBLE_WARNING)

    client = CoordinatorClient(args.base_url, node_token=args.node_token, submit_retry_seconds=args.submit_retry_seconds,
                               account_token=account_token or None)
    job_ids: dict[int, str] = {}
    for job in plan:
        if account_token:
            job_id = client.submit_account(job.account_payload, idempotency_key=f"havn-47-{run_id}-{job.index}")
        else:
            job_id = client.submit(job.payload)
        job_ids[job.index] = job_id
        print(json.dumps({"event": "submitted", "run_id": run_id, "index": job.index, "task_type": job.task_type, "model": job.model, "job_id": job_id}), flush=True)

    deadline = time.monotonic() + args.timeout_seconds
    statuses: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        pending = []
        for job_id in job_ids.values():
            status = statuses.get(job_id)
            if status and result_status(status) in TERMINAL_STATUSES:
                continue
            result = client.account_result(job_id) if account_token else client.result(job_id)
            statuses[job_id] = result
            if result_status(result) not in TERMINAL_STATUSES:
                pending.append(job_id)
        if not pending:
            break
        print(json.dumps({"event": "poll", "pending": len(pending), "generated_at": _utc_now()}), flush=True)
        time.sleep(args.poll_seconds)

    summary = summarize_results(plan, job_ids, statuses)
    summary["run_id"] = run_id
    summary["explicit_run_id"] = args.run_id is not None
    metrics = client.metrics_text()
    if metrics is not None:
        summary["metrics_observed"] = {
            "worker_model_failures": "havnai_worker_model_failures" in metrics,
            "worker_model_unhealthy": "havnai_worker_model_unhealthy" in metrics,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

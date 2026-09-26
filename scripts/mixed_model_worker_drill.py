"""Run a mixed-model worker stability drill for HAVN-47.

The drill queues a fixed mix of image, video, and music jobs through the same
coordinator routes used by production workers, then polls until each job reaches
a terminal state. By default it prints a dry-run plan and performs no network
writes. Use --execute only from an operator shell with a funded test wallet and
the required coordinator tokens in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, parse, request


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
            payload.update({"style": "cinematic synthwave", "instrumental": True, "duration": 30})
            account_payload.update({"style": "cinematic synthwave", "instrumental": True, "duration": 30})
        plan.append(DrillJob(index=index + 1, task_type=task_type, account_type=account_type, model=model, prompt=prompt,
                             payload=payload, account_payload=account_payload))
    return plan


def summarize_results(plan: list[DrillJob], job_ids: dict[int, str], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for planned in plan:
        bucket = by_type.setdefault(planned.task_type, {"planned": 0, "succeeded": 0, "failed": 0, "other": 0})
        bucket["planned"] += 1
        job_id = job_ids.get(planned.index)
        status = result_status(statuses.get(job_id or "") or {})
        if status in {"succeeded", "completed"}:
            bucket["succeeded"] += 1
        elif status in {"failed", "cancelled", "expired", "not_submitted"}:
            bucket["failed"] += 1
            failures.append({"index": planned.index, "job_id": job_id, "type": planned.task_type, "model": planned.model, "status": status})
        else:
            bucket["other"] += 1
            failures.append({"index": planned.index, "job_id": job_id, "type": planned.task_type, "model": planned.model, "status": status})
    return {
        "schema": "havn-47-mixed-model-worker-drill.v1",
        "generated_at": _utc_now(),
        "planned_jobs": len(plan),
        "submitted_jobs": len(job_ids),
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


def _split_models(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--wallet", default=os.environ.get("HAVNAI_DRILL_WALLET", "0x0000000000000000000000000000000000000047"))
    parser.add_argument("--image-models", default=os.environ.get("HAVNAI_DRILL_IMAGE_MODELS", "juggernautxl_ragnarokby"))
    parser.add_argument("--video-models", default=os.environ.get("HAVNAI_DRILL_VIDEO_MODELS", "ltx23_wangp"))
    parser.add_argument("--music-models", default=os.environ.get("HAVNAI_DRILL_MUSIC_MODELS", "ace_step_1_5_turbo"))
    parser.add_argument("--count", type=int, default=DEFAULT_JOB_COUNT)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--submit-retry-seconds", type=float, default=DEFAULT_SUBMIT_RETRY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--execute", action="store_true", help="Queue jobs and poll live coordinator results")
    parser.add_argument("--account-token", default=os.environ.get("HAVNAI_DRILL_ACCOUNT_TOKEN", ""),
                        help="bearer token for private account-owned drill jobs; preferred for live acceptance")
    parser.add_argument("--allow-production-visible", action="store_true", help="confirm live execution may create visible production jobs")
    parser.add_argument("--node-token", default=os.environ.get("HAVNAI_NODE_TOKEN") or os.environ.get("SERVER_JOIN_TOKEN"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    plan = build_plan(
        count=args.count,
        image_models=_split_models(args.image_models),
        video_models=_split_models(args.video_models),
        music_models=_split_models(args.music_models),
        wallet=args.wallet,
    )
    print(json.dumps({
        "schema": "havn-47-mixed-model-worker-drill-plan.v1",
        "generated_at": _utc_now(),
        "base_url": _clean_base_url(args.base_url),
        "execute": bool(args.execute),
        "job_count": len(plan),
        "private_account_jobs": bool(args.account_token.strip()),
        "mix": [{"index": job.index, "task_type": job.task_type, "account_type": job.account_type, "model": job.model} for job in plan],
    }, indent=2, sort_keys=True))
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
            job_id = client.submit_account(job.account_payload, idempotency_key=f"havn-47-{job.index}-{job.model}")
        else:
            job_id = client.submit(job.payload)
        job_ids[job.index] = job_id
        print(json.dumps({"event": "submitted", "index": job.index, "task_type": job.task_type, "model": job.model, "job_id": job_id}), flush=True)

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

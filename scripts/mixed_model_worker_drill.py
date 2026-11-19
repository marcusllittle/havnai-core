"""Run a mixed-model worker stability drill for HAVN-47.

The drill queues a fixed mix of image, video, and music jobs through the same
coordinator routes used by production workers, then polls until each job reaches
a terminal state. With no mode flag, or with --dry-run, it prints the plan and
does not contact the coordinator. --preflight quotes the plan. --execute submits
private account jobs only after that quote fits the available balance.

/v2 accepts only a Clerk session JWT whose lifetime (exp minus iat) is at most
120 seconds. A static --account-token expires during a multi-hour poll. Pass
--token-command or --token-file instead. The harness reads that source before
every account request and never sends it to /v2 as an admin or join token.
Mint a fresh session JWT with scripts/clerk_session_token.py. The Clerk secret
stays in CLERK_SECRET_KEY on the coordinator host. Full steps, the prompts-file
schema, adult-output policy, and the credit budget are in
docs/havn-47-mixed-model-drill.md.

Operator outline, from the repository root, after Marcus is signed in at
https://joinhavn.io and the account can cover the quoted plan:

  export CLERK_SECRET_KEY=...            # coordinator env; never pass on argv
  export HAVNAI_DRILL_CLERK_SESSION_ID=sess_...
  python3 scripts/mixed_model_worker_drill.py --count 30 --dry-run \\
    --prompts-file /secure/havn-47-prompts.json \\
    --token-command "python3 scripts/clerk_session_token.py"
  python3 scripts/mixed_model_worker_drill.py --count 30 --preflight \\
    --prompts-file /secure/havn-47-prompts.json \\
    --token-command "python3 scripts/clerk_session_token.py"
  python3 scripts/mixed_model_worker_drill.py --count 30 --execute \\
    --prompts-file /secure/havn-47-prompts.json \\
    --token-command "python3 scripts/clerk_session_token.py"

Add --no-sfw-mode only when the account should request output without the
server SFW negative prompt. That flag does not bypass content policy.

Each invocation creates a fresh run UUID, recorded before submission and in the
summary. A new run creates new billable jobs. After interruption, recover the
same account-owned requests with --run-id <recorded UUID> and identical plan
arguments, including the prompts file and sfw flag. The server rejects changed
payloads for an existing key. Recovery may replay completed jobs: never count
it as a fresh worker acceptance run. Run IDs are not credentials.

The summary's passed field remains a strict all-jobs execution check. The
separate success_threshold_met field reports Jira's 95% numerical criterion;
neither field proves ledger integrity, artifact readability or no manual reboot.
Preflight prices every planned submission using current coordinator quotes and
the account credit scale. Quotes are not a reservation or a price lock.
--execute repeats that check and submits nothing when the balance is short.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable
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
MISSING_SESSION_BLOCKER = (
    "No account session source is configured. Pass --token-command, --token-file, "
    "or a short-lived --account-token. /v2 accepts only a Clerk session JWT."
)
PROMPT_FIELDS = {"prompt", "negative_prompt", "width", "height", "steps", "model", "type", "index"}
_SECRET_TEXT = re.compile(
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r"|sk_(?:test|live)_[A-Za-z0-9]+"
    r"|Bearer\s+\S+",
    re.IGNORECASE,
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


def _redact(text: str) -> str:
    return _SECRET_TEXT.sub("[redacted]", text)[:300]


def visual_job_count(count: int) -> int:
    """Image and video jobs in the rotating plan. Music slots are index % 3 == 2."""
    if count <= 0:
        return 0
    return count - len(range(2, count, 3))


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def load_prompt_overrides(path: str) -> list[dict[str, Any]]:
    """Load image and video prompt objects. Music jobs are not listed here."""
    file = Path(path)
    if not file.is_file():
        raise ValueError("prompts file was not found")
    raw = file.read_bytes()
    if len(raw) > 1_048_576:
        raise ValueError("prompts file is too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("prompts file must be a JSON list") from exc
    if not isinstance(data, list):
        raise ValueError("prompts file must be a JSON list")
    overrides: list[dict[str, Any]] = []
    for position, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"prompts entry {position} must be an object")
        unknown = sorted(set(item) - PROMPT_FIELDS)
        if unknown:
            raise ValueError(f"prompts entry {position} has unsupported fields: {', '.join(unknown)}")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompts entry {position} needs a non-empty prompt")
        entry: dict[str, Any] = {"prompt": prompt.strip()}
        if "negative_prompt" in item:
            if not isinstance(item["negative_prompt"], str):
                raise ValueError(f"prompts entry {position} negative_prompt must be a string")
            entry["negative_prompt"] = item["negative_prompt"].strip()
        if "model" in item:
            model = item["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"prompts entry {position} model must be a non-empty string")
            entry["model"] = model.strip().lower()
        if "type" in item:
            if item["type"] not in {"image", "text_to_video"}:
                raise ValueError(f"prompts entry {position} type must be image or text_to_video")
            entry["type"] = item["type"]
        if "index" in item:
            entry["index"] = _strict_int(item["index"], f"prompts entry {position} index")
        for key in ("width", "height", "steps"):
            if key in item:
                entry[key] = _strict_int(item[key], f"prompts entry {position} {key}")
        overrides.append(entry)
    return overrides


def _apply_visual_override(job_index: int, task_type: str, payload: dict[str, Any],
                           account_payload: dict[str, Any], override: dict[str, Any], model: str) -> tuple[str, str]:
    expected = "image" if task_type == "IMAGE_GEN" else "text_to_video"
    if override.get("type", expected) != expected:
        raise ValueError(f"prompts entry for job {job_index} must use type {expected}")
    if "index" in override and override["index"] != job_index:
        raise ValueError(f"prompts entry index {override['index']} does not match planned job {job_index}")
    prompt = override["prompt"]
    payload["prompt"] = prompt
    account_payload["prompt"] = prompt
    if "negative_prompt" in override:
        payload["negative_prompt"] = override["negative_prompt"]
        account_payload["negative_prompt"] = override["negative_prompt"]
    if "model" in override:
        model = override["model"]
        payload["model"] = model
        account_payload["model"] = model
    width = override.get("width")
    height = override.get("height")
    steps = override.get("steps")
    if task_type == "LTX_VIDEO_GEN":
        if (width is None) != (height is None):
            raise ValueError(f"video job {job_index} needs both width and height")
        if width is not None and (not 256 <= width <= 1280 or width % 32 or not 256 <= height <= 1280 or height % 32):
            raise ValueError(f"video job {job_index} width and height must be multiples of 32 from 256 to 1280")
        if width is not None:
            payload["width"] = account_payload["width"] = width
            payload["height"] = account_payload["height"] = height
        if steps is not None and not 1 <= steps <= 150:
            raise ValueError(f"video job {job_index} steps must be from 1 to 150")
    else:
        if width is not None and not 256 <= width <= 1536:
            raise ValueError(f"image job {job_index} width must be from 256 to 1536")
        if height is not None and not 256 <= height <= 1536:
            raise ValueError(f"image job {job_index} height must be from 256 to 1536")
        if steps is not None and not 5 <= steps <= 50:
            raise ValueError(f"image job {job_index} steps must be from 5 to 50")
        if width is not None:
            payload["width"] = account_payload["width"] = width
        if height is not None:
            payload["height"] = account_payload["height"] = height
    if steps is not None:
        payload["steps"] = account_payload["steps"] = steps
    return prompt, model


class AccountTokenSource:
    """Fresh Clerk session material. Read again before every account request."""

    def __init__(self, static: str = "", command: str = "", path: str = "") -> None:
        self.static = static or ""
        self.command = command or ""
        self.path = path or ""

    def ensure_exclusive(self) -> None:
        chosen = [name for name, value in (
            ("--account-token", self.static),
            ("--token-command", self.command),
            ("--token-file", self.path),
        ) if str(value).strip()]
        if len(chosen) > 1:
            raise ValueError("pass only one account session source: " + ", ".join(chosen))

    def kind(self) -> str:
        if self.command.strip():
            return "command"
        if self.path.strip():
            return "file"
        if self.static.strip():
            return "static"
        return "absent"

    def configured(self) -> bool:
        return self.kind() != "absent"

    def token(self) -> str:
        kind = self.kind()
        if kind == "command":
            raw = _run_token_command(self.command)
        elif kind == "file":
            raw = _read_token_file(self.path)
        else:
            raw = self.static
        return _normalize_session(raw)


def _normalize_session(value: str) -> str:
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("account session source must yield exactly one line")
    token = lines[0]
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token or any(character.isspace() for character in token) or len(token) > 16384:
        raise RuntimeError("account session source returned an unusable value")
    return token


def _read_token_file(path: str) -> str:
    file = Path(path)
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise RuntimeError("account session file could not be read") from exc
    if len(data) > 16384:
        raise RuntimeError("account session file is too large")
    try:
        return data.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeError("account session file could not be read") from exc


def _run_token_command(command: str) -> str:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise RuntimeError("account session command could not be parsed") from exc
    if not argv:
        raise RuntimeError("account session command is empty")
    try:
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=30, env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("account session command failed") from exc
    if completed.returncode != 0:
        detail = _redact(completed.stderr).strip()
        raise RuntimeError("account session command failed" + (f": {detail}" if detail else ""))
    return completed.stdout


def build_plan(
    *,
    count: int,
    image_models: list[str],
    video_models: list[str],
    music_models: list[str],
    wallet: str,
    sfw_mode: bool = True,
    prompt_overrides: list[dict[str, Any]] | None = None,
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
    if type(sfw_mode) is not bool:
        raise ValueError("sfw_mode must be a boolean")
    expected_overrides = visual_job_count(count)
    if prompt_overrides is not None and len(prompt_overrides) != expected_overrides:
        raise ValueError(
            f"prompts file must list {expected_overrides} image and video jobs for count {count}, "
            f"found {len(prompt_overrides)}"
        )

    plan: list[DrillJob] = []
    override_at = 0
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
            "sfw_mode": sfw_mode,
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
            "sfw_mode": sfw_mode,
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
        if task_type in {"IMAGE_GEN", "LTX_VIDEO_GEN"} and prompt_overrides is not None:
            prompt, model = _apply_visual_override(
                index + 1, task_type, payload, account_payload, prompt_overrides[override_at], model,
            )
            override_at += 1
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
        account_token_source: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = _clean_base_url(base_url)
        self.node_token = node_token
        self.submit_retry_seconds = submit_retry_seconds
        self.account_token = account_token
        self.account_token_source = account_token_source

    def current_account_token(self) -> str:
        """Read the session immediately before an account request."""
        if self.account_token_source is not None:
            raw = self.account_token_source()
        else:
            raw = self.account_token or ""
        return _normalize_session(str(raw))

    def _account_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.current_account_token()}"}
        if extra:
            headers.update(extra)
        return headers

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
        headers = self._account_headers({"Idempotency-Key": idempotency_key})
        response = self._json("POST", "/v2/jobs", payload, headers=headers)
        job_id = str(response.get("id") or "").strip()
        if not job_id:
            raise RuntimeError(f"account submit response did not include id: {response}")
        return job_id

    def result(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/result/{parse.quote(job_id)}")

    def account_result(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v2/jobs/{parse.quote(job_id)}", headers=self._account_headers())

    def account_preflight(self, plan: list[DrillJob]) -> dict[str, Any]:
        account = self._json("GET", "/v2/account", headers=self._account_headers())
        credits = self._json("GET", "/v2/account/credits", headers=self._account_headers())
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("HAVNAI_COORDINATOR_URL", "https://api.joinhavn.io"))
    parser.add_argument("--wallet", default=os.environ.get("HAVNAI_DRILL_WALLET", "0x0000000000000000000000000000000000000047"))
    parser.add_argument("--image-models", default=os.environ.get("HAVNAI_DRILL_IMAGE_MODELS", "juggernautxl_ragnarokby"))
    parser.add_argument("--video-models", default=os.environ.get("HAVNAI_DRILL_VIDEO_MODELS", "ltx_video_distilled"))
    parser.add_argument("--music-models", default=os.environ.get("HAVNAI_DRILL_MUSIC_MODELS", "ace_step_1_5_turbo"))
    parser.add_argument("--count", type=int, default=DEFAULT_JOB_COUNT)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--submit-retry-seconds", type=float, default=DEFAULT_SUBMIT_RETRY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without contacting the coordinator. This is also the default.")
    parser.add_argument("--execute", action="store_true", help="Queue private account jobs and poll live coordinator results")
    parser.add_argument("--run-id", type=UUID, default=None,
                        help="UUID for this drill; reuse only to recover the same run, not for fresh acceptance")
    parser.add_argument("--preflight", action="store_true", help="Validate the account session and credit quote without submitting jobs")
    parser.add_argument("--account-token", default=os.environ.get("HAVNAI_DRILL_ACCOUNT_TOKEN", ""),
                        help="one Clerk session JWT. It expires within 120 seconds; do not use it for --execute polling")
    parser.add_argument("--token-command", default=os.environ.get("HAVNAI_DRILL_TOKEN_COMMAND", ""),
                        help="command rerun before every account request; stdout is one fresh Clerk session JWT")
    parser.add_argument("--token-file", default=os.environ.get("HAVNAI_DRILL_TOKEN_FILE", ""),
                        help="file reread before every account request; it must contain one fresh Clerk session JWT")
    parser.add_argument("--prompts-file", default="",
                        help="JSON list of image and video jobs: prompt, negative_prompt, width, height, steps, and optional model")
    parser.add_argument("--sfw-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="send sfw_mode to /v2. Default true merges the server SFW negative. --no-sfw-mode does not bypass other policy")
    parser.add_argument("--allow-production-visible", action="store_true", help="confirm live execution may create visible production jobs")
    parser.add_argument("--node-token", default=os.environ.get("HAVNAI_NODE_TOKEN") or os.environ.get("SERVER_JOIN_TOKEN"),
                        help="optional join token for GET /metrics only. It is not accepted by /v2")
    return parser.parse_args(argv)


def _client(args: argparse.Namespace, source: AccountTokenSource) -> CoordinatorClient:
    kwargs: dict[str, Any] = {
        "node_token": args.node_token,
        "submit_retry_seconds": args.submit_retry_seconds,
    }
    if source.kind() == "static":
        kwargs["account_token"] = source.static.strip()
    elif source.configured():
        kwargs["account_token_source"] = source.token
    return CoordinatorClient(args.base_url, **kwargs)


def _preflight_report(args: argparse.Namespace, source: AccountTokenSource, *, passed: bool,
                      blockers: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "havn-47-mixed-model-worker-drill-preflight.v1",
        "generated_at": _utc_now(),
        "base_url": _clean_base_url(args.base_url),
        "account_token_present": source.configured(),
        "session_source": source.kind(),
        "required_units": None,
        "budget_sufficient": False,
        "passed": passed,
        "blockers": blockers,
        **(extra or {}),
    }


def _quote_account(client: CoordinatorClient, plan: list[DrillJob]) -> dict[str, Any]:
    preflight = client.account_preflight(plan)
    passed = bool(preflight.get("account_id_present")) and bool(preflight.get("budget_sufficient"))
    blockers = ([] if preflight.get("account_id_present") else ["Private account identity is missing."])
    blockers += ([] if preflight.get("budget_sufficient") else ["Available credits do not cover the quoted full drill plan."])
    return {**preflight, "passed": passed, "blockers": blockers}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    source = AccountTokenSource(args.account_token, args.token_command, args.token_file)
    try:
        source.ensure_exclusive()
        if args.dry_run and (args.execute or args.preflight):
            raise ValueError("--dry-run only prints the plan; omit --execute and --preflight")
        overrides = load_prompt_overrides(args.prompts_file) if str(args.prompts_file).strip() else None
        plan = build_plan(
            count=args.count,
            image_models=_split_models(args.image_models),
            video_models=_split_models(args.video_models),
            music_models=_split_models(args.music_models),
            wallet=args.wallet,
            sfw_mode=args.sfw_mode,
            prompt_overrides=overrides,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = str(args.run_id or uuid4())
    print(json.dumps({
        "schema": "havn-47-mixed-model-worker-drill-plan.v1",
        "run_id": run_id,
        "explicit_run_id": args.run_id is not None,
        "generated_at": _utc_now(),
        "base_url": _clean_base_url(args.base_url),
        "dry_run": not args.execute and not args.preflight,
        "execute": bool(args.execute),
        "job_count": len(plan),
        "private_account_jobs": source.configured(),
        "session_source": source.kind(),
        "sfw_mode": args.sfw_mode,
        "prompts_file": args.prompts_file or None,
        "mix": [{
            "index": job.index,
            "task_type": job.task_type,
            "account_type": job.account_type,
            "model": job.model,
            "prompt": job.prompt,
            "negative_prompt": job.account_payload.get("negative_prompt", ""),
            "width": job.account_payload.get("width"),
            "height": job.account_payload.get("height"),
            "steps": job.account_payload.get("steps"),
            "sfw_mode": job.account_payload["sfw_mode"],
        } for job in plan],
    }, indent=2, sort_keys=True), flush=True)
    if args.execute and not args.preflight and not source.configured() and not args.allow_production_visible:
        raise SystemExit(PRODUCTION_VISIBLE_WARNING)
    if args.execute or (args.preflight and source.configured()):
        try:
            eligibility = _client(args, source).model_eligibility(plan)
        except (ValueError, RuntimeError, error.URLError):
            eligibility = {"passed": False, "blockers": [{"code": "model_catalog_unavailable"}]}
        print(json.dumps({"schema": "havn-47-model-eligibility.v1", "run_id": run_id, **eligibility}, sort_keys=True), flush=True)
        if not eligibility["passed"]:
            return 2
    if args.preflight:
        if not source.configured():
            print(json.dumps(_preflight_report(args, source, passed=False, blockers=[MISSING_SESSION_BLOCKER], extra={
                "account_id_present": False, "available_units": None, "reserved_units": None,
            }), indent=2, sort_keys=True))
            return 2
        client = _client(args, source)
        try:
            quoted = _quote_account(client, plan)
        except (ValueError, RuntimeError, error.URLError):
            # Do not print coordinator error bodies or bearer session details.
            print(json.dumps(_preflight_report(
                args, source, passed=False,
                blockers=["Private account or full-plan price verification failed; no jobs submitted."],
            ), indent=2, sort_keys=True))
            return 2
        result = _preflight_report(args, source, passed=quoted["passed"], blockers=quoted["blockers"], extra=quoted)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if not args.execute:
        return 0

    client = _client(args, source)
    use_account = source.configured()
    if use_account:
        if source.kind() == "static":
            print(json.dumps({
                "event": "static_session_expires",
                "run_id": run_id,
                "detail": "This session value is reused for every request and expires within 120 seconds. "
                          "Use --token-command or --token-file for a drill that polls.",
            }, sort_keys=True), flush=True)
        try:
            quoted = _quote_account(client, plan)
        except (ValueError, RuntimeError, error.URLError):
            print(json.dumps({
                "event": "budget_checked",
                "run_id": run_id,
                "passed": False,
                "detail": "Private account or full-plan price verification failed; no jobs submitted.",
            }, sort_keys=True), flush=True)
            return 2
        print(json.dumps({
            "event": "budget_checked",
            "run_id": run_id,
            "required_units": quoted.get("required_units"),
            "available_units": quoted.get("available_units"),
            "budget_sufficient": quoted.get("budget_sufficient"),
            "blockers": quoted["blockers"],
            "passed": quoted["passed"],
        }, sort_keys=True), flush=True)
        if not quoted["passed"]:
            return 2
    job_ids: dict[int, str] = {}
    try:
        for job in plan:
            if use_account:
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
                result = client.account_result(job_id) if use_account else client.result(job_id)
                statuses[job_id] = result
                if result_status(result) not in TERMINAL_STATUSES:
                    pending.append(job_id)
            if not pending:
                break
            print(json.dumps({"event": "poll", "pending": len(pending), "generated_at": _utc_now()}), flush=True)
            time.sleep(args.poll_seconds)
    except (RuntimeError, error.URLError, ValueError) as exc:
        print(json.dumps({
            "event": "account_request_failed",
            "run_id": run_id,
            "detail": _redact(str(exc)),
            "submitted_jobs": len(job_ids),
        }, sort_keys=True), flush=True)
        return 2

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

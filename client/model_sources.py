"""Model artifact acquisition for HavnAI nodes.

The coordinator manifest describes *which* models the network wants a creator
node to serve. This module is responsible for turning that description into
actual files on local disk.

Three delivery routes are supported, selected per model by ``source.kind``:

``hf``
    Open-license weights pulled from the Hugging Face CDN. Bandwidth is free,
    transfers resume, and the CDN is closer to the operator than we are.
``coordinator``
    Weights we cannot redistribute publicly. Streamed from
    ``/models/download/<name>`` with the node's join token, resumable via HTTP
    Range requests.
``operator``
    The operator supplies the file. We never download it; we only report the
    exact filename that must appear in the creator models directory.

Every download lands in a ``.part`` file and is only moved into place after the
digest check passes, so an interrupted run can never leave a truncated
checkpoint that later loads as corrupt weights.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

CHUNK_SIZE = 8 * 1024 * 1024
MODEL_EXTENSION = ".safetensors"

# Progress callbacks receive (model_name, downloaded_bytes, total_bytes).
# ``total_bytes`` is 0 when the server declines to report a length.
ProgressCallback = Callable[[str, int, int], None]


class ModelDownloadError(RuntimeError):
    """Raised when a model artifact could not be obtained or verified."""


@dataclass
class ModelRequirement:
    """A single model the node needs on disk, resolved from the manifest."""

    name: str
    kind: str
    destination: Path
    repo_id: str = ""
    filename: str = ""
    revision: str = "main"
    sha256: str = ""
    size_bytes: int = 0
    license: str = ""
    notes: str = ""

    @property
    def present(self) -> bool:
        return self.destination.exists() and self.destination.stat().st_size > 0

    @property
    def downloadable(self) -> bool:
        """Whether this node can fetch the artifact without operator action."""

        return self.kind in {"hf", "coordinator"}


@dataclass
class DownloadResult:
    """Outcome of a single model acquisition attempt."""

    name: str
    status: str  # "present" | "downloaded" | "skipped" | "failed"
    destination: Optional[Path] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"present", "downloaded"}


def models_dir(havnai_home: Path) -> Path:
    """Return the creator model directory, honouring the usual overrides."""

    override = (
        os.environ.get("HAVNAI_MODEL_DIR")
        or os.environ.get("HAI_MODEL_DIR")
        or os.environ.get("MODEL_DIR")
        or ""
    ).strip()
    if override:
        return Path(override).expanduser()
    return havnai_home / "models" / "creator"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def requirements_from_manifest(
    models: Iterable[Dict[str, Any]],
    havnai_home: Path,
    *,
    pipelines: Optional[Iterable[str]] = None,
) -> List[ModelRequirement]:
    """Build the download plan from ``/models/list`` style entries.

    ``pipelines`` optionally restricts the plan to the pipelines this node can
    actually run, so a node without a video engine does not pull video weights.
    """

    wanted = {p.strip().lower() for p in pipelines or [] if p and p.strip()}
    target_dir = models_dir(havnai_home)
    plan: List[ModelRequirement] = []
    seen: set[str] = set()

    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model_key") or "").strip()
        if not name or name.lower() in seen:
            continue
        pipeline = str(entry.get("pipeline") or "").strip().lower()
        if wanted and pipeline and pipeline not in wanted:
            continue
        seen.add(name.lower())

        source = entry.get("source")
        source = source if isinstance(source, dict) else {}
        kind = str(source.get("kind") or "operator").strip().lower()
        filename = str(source.get("filename") or "").strip() or f"{name}{MODEL_EXTENSION}"

        plan.append(
            ModelRequirement(
                name=name,
                kind=kind,
                destination=target_dir / filename,
                repo_id=str(source.get("repo_id") or "").strip(),
                filename=filename,
                revision=str(source.get("revision") or "main").strip() or "main",
                sha256=str(source.get("sha256") or "").strip().lower(),
                size_bytes=_as_int(source.get("size_bytes")),
                license=str(source.get("license") or "").strip(),
                notes=str(source.get("notes") or "").strip(),
            )
        )
    return plan


def _digest(path: Path, *, progress: Optional[Callable[[int], None]] = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return digest.hexdigest()


def verify(requirement: ModelRequirement) -> bool:
    """Return whether the on-disk artifact matches the manifest digest.

    A requirement with no published digest is accepted on presence alone; we
    would rather serve a model we cannot checksum than refuse to start.
    """

    if not requirement.present:
        return False
    if not requirement.sha256:
        return True
    return _digest(requirement.destination) == requirement.sha256


def _finalize(part: Path, requirement: ModelRequirement) -> None:
    """Verify a completed ``.part`` file and move it into place."""

    if requirement.sha256:
        actual = _digest(part)
        if actual != requirement.sha256:
            part.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"{requirement.name}: checksum mismatch "
                f"(expected {requirement.sha256[:12]}…, got {actual[:12]}…)"
            )
    part.replace(requirement.destination)


def _download_coordinator(
    requirement: ModelRequirement,
    *,
    server_url: str,
    token: str,
    session: requests.Session,
    on_progress: Optional[ProgressCallback],
    timeout: int,
) -> None:
    """Stream a self-hosted artifact, resuming a partial transfer if present."""

    url = f"{server_url.rstrip('/')}/models/download/{requirement.name}"
    part = requirement.destination.with_suffix(requirement.destination.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    resume_from = part.stat().st_size if part.exists() else 0
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Havnai-Join-Token"] = token
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    response = session.get(url, headers=headers, stream=True, timeout=timeout)
    if response.status_code == 416:
        # The partial file is already the full length; let verification decide.
        response.close()
        _finalize(part, requirement)
        return
    if response.status_code in (401, 403):
        response.close()
        raise ModelDownloadError(
            f"{requirement.name}: coordinator rejected the join token "
            f"(HTTP {response.status_code}). Check JOIN_TOKEN in your node .env."
        )
    if response.status_code == 404:
        response.close()
        raise ModelDownloadError(
            f"{requirement.name}: coordinator has no artifact for this model."
        )
    response.raise_for_status()

    # A server that ignores our Range header restarts the file from zero.
    if resume_from and response.status_code != 206:
        resume_from = 0

    total = _as_int(response.headers.get("Content-Length")) + resume_from
    if not total:
        total = requirement.size_bytes

    mode = "ab" if resume_from else "wb"
    downloaded = resume_from
    with response, part.open(mode) as handle:
        if on_progress is not None:
            on_progress(requirement.name, downloaded, total)
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if on_progress is not None:
                on_progress(requirement.name, downloaded, total)

    _finalize(part, requirement)


def _download_hf(
    requirement: ModelRequirement,
    *,
    on_progress: Optional[ProgressCallback],
) -> None:
    """Pull an open-license artifact from the Hugging Face CDN."""

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelDownloadError(
            f"{requirement.name}: huggingface-hub is not installed; "
            "run the node installer again to repair dependencies."
        ) from exc

    if not requirement.repo_id or not requirement.filename:
        raise ModelDownloadError(
            f"{requirement.name}: manifest is missing repo_id/filename for a Hugging Face source."
        )

    requirement.destination.parent.mkdir(parents=True, exist_ok=True)
    if on_progress is not None:
        on_progress(requirement.name, 0, requirement.size_bytes)

    try:
        cached = hf_hub_download(
            repo_id=requirement.repo_id,
            filename=requirement.filename,
            revision=requirement.revision,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise ModelDownloadError(f"{requirement.name}: Hugging Face download failed: {exc}") from exc

    # hf_hub_download returns a path inside its own cache (usually a symlink).
    # Copy it into the creator directory so the node has a stable, real file
    # even if the operator later clears the HF cache.
    part = requirement.destination.with_suffix(requirement.destination.suffix + ".part")
    shutil.copyfile(cached, part)
    if on_progress is not None:
        size = part.stat().st_size
        on_progress(requirement.name, size, size)
    _finalize(part, requirement)


def ensure_model(
    requirement: ModelRequirement,
    *,
    server_url: str = "",
    token: str = "",
    session: Optional[requests.Session] = None,
    on_progress: Optional[ProgressCallback] = None,
    timeout: int = 60,
    retries: int = 3,
) -> DownloadResult:
    """Make a single model available locally, downloading it if required."""

    if requirement.present and verify(requirement):
        return DownloadResult(requirement.name, "present", requirement.destination)

    if requirement.present:
        # Present but failed verification: the file is corrupt, not usable.
        requirement.destination.unlink(missing_ok=True)

    if not requirement.downloadable:
        detail = requirement.notes or (
            f"Place {requirement.filename} in {requirement.destination.parent}"
        )
        return DownloadResult(requirement.name, "skipped", requirement.destination, detail)

    owned_session = session is None
    session = session or requests.Session()
    last_error = ""
    try:
        for attempt in range(1, max(1, retries) + 1):
            try:
                if requirement.kind == "hf":
                    _download_hf(requirement, on_progress=on_progress)
                else:
                    _download_coordinator(
                        requirement,
                        server_url=server_url,
                        token=token,
                        session=session,
                        on_progress=on_progress,
                        timeout=timeout,
                    )
                return DownloadResult(requirement.name, "downloaded", requirement.destination)
            except ModelDownloadError as exc:
                # Auth, missing artifact and checksum failures do not get better
                # by trying again.
                return DownloadResult(requirement.name, "failed", requirement.destination, str(exc))
            except Exception as exc:  # network hiccup, partial read, etc.
                last_error = str(exc)
                if attempt < max(1, retries):
                    time.sleep(min(2 ** attempt, 16))
    finally:
        if owned_session:
            session.close()

    return DownloadResult(
        requirement.name,
        "failed",
        requirement.destination,
        f"download failed after {retries} attempts: {last_error}",
    )


def ensure_models(
    requirements: Iterable[ModelRequirement],
    *,
    server_url: str = "",
    token: str = "",
    on_progress: Optional[ProgressCallback] = None,
    timeout: int = 60,
    retries: int = 3,
) -> List[DownloadResult]:
    """Resolve a whole download plan, continuing past individual failures."""

    results: List[DownloadResult] = []
    with requests.Session() as session:
        for requirement in requirements:
            results.append(
                ensure_model(
                    requirement,
                    server_url=server_url,
                    token=token,
                    session=session,
                    on_progress=on_progress,
                    timeout=timeout,
                    retries=retries,
                )
            )
    return results


__all__ = [
    "DownloadResult",
    "ModelDownloadError",
    "ModelRequirement",
    "ensure_model",
    "ensure_models",
    "models_dir",
    "requirements_from_manifest",
    "verify",
]

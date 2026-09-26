"""Worker-local cleanup, authorized only by completed coordinator purges."""
from pathlib import Path
import re


def valid_job(value):
    return isinstance(value, str) and re.fullmatch(r"job-[A-Za-z0-9-]{1,160}", value) is not None


def candidates(home):
    home = Path(home).resolve()
    jobs = set()
    for directory in ("outputs", "outputs/originals", "outputs/manifests", "outputs/music", "assets"):
        root = home / directory
        if (not root.is_dir() or not root.resolve().is_relative_to(home)
                or any(parent.is_symlink() for parent in (root, *root.parents) if parent != home)):
            continue
        for path in root.iterdir():
            if path.is_symlink():
                continue
            name = path.name if path.is_dir() else path.stem
            for prefix in ("video_", "animatediff_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            if valid_job(name):
                jobs.add(name)
    return sorted(jobs)


def cleanup_job(home, job):
    if not valid_job(job):
        raise ValueError("invalid_cleanup_job")
    home = Path(home).resolve()
    outputs = home / "outputs"
    paths = [outputs / f"{job}.png", outputs / "originals" / f"{job}.png",
             outputs / "manifests" / f"{job}.json"]
    paths.extend(outputs / f"{prefix}{job}.mp4" for prefix in ("", "video_", "animatediff_"))
    directories = [outputs / "music" / job, home / "assets" / job]
    for directory in directories:
        paths.append(directory)
        if directory.is_dir() and not directory.is_symlink():
            paths.extend(directory.rglob("*"))
    # Validate the whole batch before unlinking. Never follow symlink ancestors
    # into models, another job, or a path outside the worker home.
    for path in paths:
        if not path.resolve().is_relative_to(home):
            raise ValueError("unsafe_cleanup_path")
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != home):
            raise ValueError("unsafe_cleanup_path")
        if path.exists() and not (path.is_file() or path.is_dir()):
            raise ValueError("unsafe_cleanup_path")
    removed = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            removed += 1
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and (path in directories or any(parent in directories for parent in path.parents)):
            path.rmdir()
    return removed


def run_batch(session, base_url, headers, node_id, home, *, after=""):
    jobs = candidates(home)
    batch = [job for job in jobs if job > after][:25]
    if not batch:
        batch = jobs[:25]
    if not batch:
        return "", 0
    response = session.post(base_url.rstrip("/") + "/v1/node/artifact-purges",
                            headers=headers, json={"node_id": node_id, "job_ids": batch}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    approved = payload.get("job_ids") if isinstance(payload, dict) else None
    if not isinstance(approved, list) or any(not valid_job(job) or job not in batch for job in approved):
        raise ValueError("invalid_cleanup_authorization")
    failures = 0
    for job in set(approved):
        try:
            cleanup_job(home, job)
        except (OSError, ValueError):
            failures += 1
    # Advance even after a filesystem failure so older failures cannot starve
    # later jobs. Surviving files are rediscovered on the next complete sweep.
    return batch[-1], failures

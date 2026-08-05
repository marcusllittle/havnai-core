"""Download every model artifact this node needs to serve its capabilities.

Run after installation, or any time the coordinator manifest gains a model:

    python -m client.fetch_models              # everything in the manifest
    python -m client.fetch_models --face-assets  # + InstantID / antelopev2
    python -m client.fetch_models --json       # progress as JSON lines

The ``--json`` mode emits one JSON object per line so a supervising process
(the desktop app) can render progress without parsing human text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from client.model_sources import (
        DownloadResult,
        ModelRequirement,
        ensure_model,
        models_dir,
        requirements_from_manifest,
    )
except ImportError:  # direct execution from ~/.havnai
    from model_sources import (  # type: ignore
        DownloadResult,
        ModelRequirement,
        ensure_model,
        models_dir,
        requirements_from_manifest,
    )

# InstantID's face analysis pack. insightface's own auto-download for this pack
# is unreliable, so we fetch it explicitly from the InstantID release mirror.
ANTELOPEV2_URL = "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/antelopev2.zip"


def _havnai_home() -> Path:
    return Path(os.environ.get("HAVNAI_HOME") or (Path.home() / ".havnai")).expanduser()


def _server_url() -> str:
    return (
        os.environ.get("HAVNAI_SERVER_URL")
        or os.environ.get("SERVER_URL")
        or os.environ.get("COORDINATOR_URL")
        or "https://api.joinhavn.io"
    ).rstrip("/")


def _join_token() -> str:
    return (
        os.environ.get("JOIN_TOKEN")
        or os.environ.get("HAVNAI_NODE_TOKEN")
        or os.environ.get("SERVER_JOIN_TOKEN")
        or ""
    ).strip()


def _load_env_file() -> None:
    env_path = _havnai_home() / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        pass


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class ProgressPrinter:
    """Renders download progress as a terminal bar or as JSON lines."""

    def __init__(self, *, as_json: bool, stream: Any = None) -> None:
        self.as_json = as_json
        self.stream = stream or sys.stdout
        self._last_emit = 0.0
        self._current = ""

    def emit(self, payload: Dict[str, Any]) -> None:
        if self.as_json:
            self.stream.write(json.dumps(payload) + "\n")
            self.stream.flush()

    def on_progress(self, name: str, done: int, total: int) -> None:
        now = time.monotonic()
        finished = total > 0 and done >= total
        # Throttle so a fast local transfer does not flood the console.
        if not finished and now - self._last_emit < 0.25 and name == self._current:
            return
        self._last_emit = now
        self._current = name

        if self.as_json:
            self.emit(
                {
                    "event": "progress",
                    "model": name,
                    "downloaded": done,
                    "total": total,
                    "percent": round(done / total * 100, 1) if total else None,
                }
            )
            return

        if total:
            fraction = min(done / total, 1.0)
            filled = int(fraction * 30)
            bar = "█" * filled + "░" * (30 - filled)
            line = f"\r  {name[:28]:<28} [{bar}] {fraction * 100:5.1f}% {_human(done)}"
        else:
            line = f"\r  {name[:28]:<28} {_human(done)}"
        self.stream.write(line)
        self.stream.flush()

    def finish_line(self) -> None:
        if not self.as_json:
            self.stream.write("\n")
            self.stream.flush()


def fetch_manifest(server_url: str, timeout: int = 30) -> List[Dict[str, Any]]:
    response = requests.get(f"{server_url}/models/list", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    return models if isinstance(models, list) else []


def fetch_face_assets(printer: ProgressPrinter, *, force: bool = False) -> bool:
    """Download the antelopev2 pack used by InstantID face analysis."""

    cache = _havnai_home() / "instantid"
    target = cache / "models" / "antelopev2"
    if target.exists() and any(target.glob("*.onnx")) and not force:
        printer.emit({"event": "face_assets", "status": "present", "path": str(target)})
        if not printer.as_json:
            print(f"  antelopev2 already present at {target}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    archive = cache / "antelopev2.zip"
    printer.emit({"event": "face_assets", "status": "downloading", "url": ANTELOPEV2_URL})
    if not printer.as_json:
        print("  downloading antelopev2 face analysis pack…")

    try:
        with urllib.request.urlopen(ANTELOPEV2_URL, timeout=120) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with archive.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    printer.on_progress("antelopev2", downloaded, total)
        printer.finish_line()

        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target.parent)
        archive.unlink(missing_ok=True)

        # Some mirrors nest the models one directory deeper.
        nested = target / "antelopev2"
        if nested.is_dir() and not any(target.glob("*.onnx")):
            for item in nested.iterdir():
                shutil.move(str(item), str(target / item.name))
            nested.rmdir()

        ok = any(target.glob("*.onnx"))
        printer.emit(
            {"event": "face_assets", "status": "ready" if ok else "failed", "path": str(target)}
        )
        if not printer.as_json:
            print(f"  antelopev2 {'ready' if ok else 'extraction produced no models'}")
        return ok
    except Exception as exc:
        archive.unlink(missing_ok=True)
        printer.emit({"event": "face_assets", "status": "failed", "error": str(exc)})
        if not printer.as_json:
            print(f"  antelopev2 download failed: {exc}")
        return False


def run(
    *,
    only: Optional[List[str]] = None,
    pipelines: Optional[List[str]] = None,
    face_assets: bool = False,
    as_json: bool = False,
    dry_run: bool = False,
) -> int:
    _load_env_file()
    printer = ProgressPrinter(as_json=as_json)
    server_url = _server_url()
    home = _havnai_home()
    target_dir = models_dir(home)

    printer.emit({"event": "start", "server": server_url, "models_dir": str(target_dir)})
    if not as_json:
        print(f"\nCoordinator: {server_url}")
        print(f"Model directory: {target_dir}\n")

    try:
        manifest = fetch_manifest(server_url)
    except Exception as exc:
        printer.emit({"event": "error", "error": f"manifest fetch failed: {exc}"})
        if not as_json:
            print(f"Could not fetch the model manifest: {exc}")
        return 1

    plan: List[ModelRequirement] = requirements_from_manifest(
        manifest, home, pipelines=pipelines
    )
    if only:
        wanted = {name.strip().lower() for name in only}
        plan = [item for item in plan if item.name.lower() in wanted]

    if not plan:
        printer.emit({"event": "error", "error": "nothing to download"})
        if not as_json:
            print("No models matched the requested filters.")
        return 1

    if dry_run:
        for item in plan:
            state = "present" if item.present else ("fetch" if item.downloadable else "manual")
            printer.emit(
                {
                    "event": "plan",
                    "model": item.name,
                    "kind": item.kind,
                    "state": state,
                    "destination": str(item.destination),
                    "size_bytes": item.size_bytes,
                }
            )
            if not as_json:
                print(f"  [{state:>7}] {item.name}  ({item.kind}) → {item.destination.name}")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    results: List[DownloadResult] = []
    token = _join_token()

    with requests.Session() as session:
        for index, item in enumerate(plan, start=1):
            printer.emit(
                {
                    "event": "model_start",
                    "model": item.name,
                    "index": index,
                    "count": len(plan),
                    "kind": item.kind,
                    "size_bytes": item.size_bytes,
                }
            )
            if not as_json:
                label = "present" if item.present else item.kind
                print(f"[{index}/{len(plan)}] {item.name} ({label})")

            result = ensure_model(
                item,
                server_url=server_url,
                token=token,
                session=session,
                on_progress=printer.on_progress,
            )
            printer.finish_line()
            results.append(result)
            printer.emit(
                {
                    "event": "model_done",
                    "model": result.name,
                    "status": result.status,
                    "detail": result.detail,
                }
            )
            if not as_json and result.detail:
                print(f"  {result.status}: {result.detail}")

    if face_assets:
        if not as_json:
            print("\nFace swap assets")
        fetch_face_assets(printer)

    downloaded = [item for item in results if item.status == "downloaded"]
    present = [item for item in results if item.status == "present"]
    skipped = [item for item in results if item.status == "skipped"]
    failed = [item for item in results if item.status == "failed"]

    printer.emit(
        {
            "event": "summary",
            "downloaded": len(downloaded),
            "present": len(present),
            "skipped": len(skipped),
            "failed": len(failed),
        }
    )

    if not as_json:
        print(
            f"\n{len(downloaded)} downloaded, {len(present)} already present, "
            f"{len(skipped)} need manual copy, {len(failed)} failed"
        )
        for item in skipped:
            print(f"  manual: {item.name} — {item.detail}")
        for item in failed:
            print(f"  failed: {item.name} — {item.detail}")

    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="havnai-fetch-models",
        description="Download the model weights this node needs.",
    )
    parser.add_argument("--only", nargs="*", help="download just these model names")
    parser.add_argument(
        "--pipelines", nargs="*", help="restrict to these pipelines (sdxl, ltx2, animatediff…)"
    )
    parser.add_argument(
        "--face-assets", action="store_true", help="also fetch InstantID / antelopev2 assets"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON lines instead of a bar")
    parser.add_argument(
        "--dry-run", action="store_true", help="show the plan without downloading anything"
    )
    args = parser.parse_args(argv)

    return run(
        only=args.only,
        pipelines=args.pipelines,
        face_assets=args.face_assets,
        as_json=args.json,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

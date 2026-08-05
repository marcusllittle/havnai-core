"""Packages the complete node runtime for distribution to operators.

The installer used to fetch ``client.py`` and ``registry.py`` and nothing else,
which left every node without the InstantID pipelines (face swap) or the
``engines`` package (video). Those nodes registered as healthy creators and
then failed at import time on the first real job.

This module builds one tarball containing everything ``client.py`` imports at
runtime, laid out exactly like the repository so ``python -m client.client``
resolves the same way it does in development.

The archive is built once and cached in memory, keyed by the digest of its
contents, so repeated installs are cheap. Set ``HAVNAI_BUNDLE_CACHE=0`` to
rebuild on every request while iterating locally.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import tarfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Everything the node runtime imports, relative to the repository root.
# Directories are included recursively (Python sources only).
BUNDLE_FILES: List[str] = [
    "client/client.py",
    "client/registry.py",
    "client/task_executor.py",
    "client/model_sources.py",
    "client/fetch_models.py",
    "client/doctor.py",
    "client/requirements-node.txt",
    "client/pipeline_stable_diffusion_xl_instantid.py",
    "client/pipeline_stable_diffusion_xl_instantid_inpaint.py",
]

BUNDLE_DIRS: List[str] = [
    "client/ip_adapter",
    "engines",
    "shared",
    "common",
]

# Packages that must be importable once extracted; missing __init__ files are
# synthesised so a directory that was never a package still imports cleanly.
REQUIRED_PACKAGES: List[str] = [
    "client",
    "client/ip_adapter",
    "engines",
    "shared",
    "common",
]

EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_DIRS = {"__pycache__", ".git", ".idea", "tests"}


class BundleError(RuntimeError):
    """Raised when the runtime bundle cannot be assembled."""


def _is_excluded(path: Path) -> bool:
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_DIRS for part in path.parts)


def collect_files(base_dir: Path) -> List[Tuple[Path, str]]:
    """Return ``(absolute_path, archive_name)`` pairs for the bundle."""

    collected: List[Tuple[Path, str]] = []
    seen: set[str] = set()

    for relative in BUNDLE_FILES:
        source = base_dir / relative
        if not source.is_file():
            # A missing optional pipeline should not break the whole bundle,
            # but the core client absolutely must be there.
            if relative == "client/client.py":
                raise BundleError(f"node bundle is missing its entry point: {relative}")
            continue
        if relative not in seen:
            collected.append((source, relative))
            seen.add(relative)

    for directory in BUNDLE_DIRS:
        root = base_dir / directory
        if not root.is_dir():
            continue
        for source in sorted(root.rglob("*")):
            if not source.is_file() or _is_excluded(source):
                continue
            if source.suffix not in {".py", ".json", ".txt", ".yaml", ".yml"}:
                continue
            relative = source.relative_to(base_dir).as_posix()
            if relative not in seen:
                collected.append((source, relative))
                seen.add(relative)

    return collected


def bundle_manifest(base_dir: Path) -> Dict[str, object]:
    """Per-file digests, so a node can verify or repair its own install."""

    files = []
    for source, relative in collect_files(base_dir):
        data = source.read_bytes()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    combined = hashlib.sha256()
    for entry in files:
        combined.update(entry["path"].encode("utf-8"))  # type: ignore[union-attr]
        combined.update(entry["sha256"].encode("utf-8"))  # type: ignore[union-attr]
    return {
        "bundle_sha256": combined.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def build_bundle(base_dir: Path) -> Tuple[bytes, str]:
    """Build the gzipped tarball. Returns ``(payload, sha256)``."""

    files = collect_files(base_dir)
    if not files:
        raise BundleError("no runtime files were collected for the node bundle")

    # Build the tar uncompressed first, then gzip it with a pinned mtime.
    # tarfile's "w:gz" mode stamps the current time into the gzip header, which
    # would change the digest on every build and make every install look like
    # an upgrade. Zeroing the member mtimes alone is not enough.
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        packaged: set[str] = set()

        for source, relative in files:
            info = archive.gettarinfo(str(source), arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = "havnai"
            with source.open("rb") as handle:
                archive.addfile(info, handle)
            packaged.add(relative)

        for package in REQUIRED_PACKAGES:
            init_name = f"{package}/__init__.py"
            if init_name in packaged:
                continue
            if not (base_dir / package).is_dir():
                continue
            payload = b'"""HavnAI node runtime package."""\n'
            info = tarfile.TarInfo(name=init_name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uname = info.gname = "havnai"
            archive.addfile(info, io.BytesIO(payload))
            packaged.add(init_name)

    gz_buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=gz_buffer, mode="wb", compresslevel=6, mtime=0
    ) as compressor:
        compressor.write(tar_buffer.getvalue())

    payload = gz_buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


class BundleCache:
    """Caches the built archive until the underlying sources change."""

    def __init__(self, base_dir: Path, *, enabled: bool = True) -> None:
        self.base_dir = base_dir
        self.enabled = enabled
        self._lock = threading.Lock()
        self._payload: Optional[bytes] = None
        self._digest: str = ""
        self._signature: str = ""
        self._built_at: float = 0.0

    def _source_signature(self) -> str:
        """Cheap fingerprint of the source tree: paths, sizes and mtimes."""

        digest = hashlib.sha256()
        for source, relative in collect_files(self.base_dir):
            try:
                stat = source.stat()
            except OSError:
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime)).encode("utf-8"))
        return digest.hexdigest()

    def get(self) -> Tuple[bytes, str]:
        if not self.enabled:
            return build_bundle(self.base_dir)

        signature = self._source_signature()
        with self._lock:
            if self._payload is not None and signature == self._signature:
                return self._payload, self._digest
            payload, digest = build_bundle(self.base_dir)
            self._payload = payload
            self._digest = digest
            self._signature = signature
            self._built_at = time.time()
            return payload, digest

    @property
    def built_at(self) -> float:
        return self._built_at


def cache_enabled() -> bool:
    return os.getenv("HAVNAI_BUNDLE_CACHE", "1").strip().lower() not in {"0", "false", "no"}


__all__ = [
    "BUNDLE_DIRS",
    "BUNDLE_FILES",
    "BundleCache",
    "BundleError",
    "build_bundle",
    "bundle_manifest",
    "cache_enabled",
    "collect_files",
]

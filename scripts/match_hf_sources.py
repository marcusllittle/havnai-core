"""Find Hugging Face mirrors of manifest checkpoints, verified by checksum.

Serving multi-gigabyte checkpoints from the coordinator costs us bandwidth and
is slow for operators on the far side of a home uplink. Where the identical
file already exists on the Hugging Face CDN we would rather point nodes there.

The danger in doing that by name is subtle and expensive: the community
re-uploads checkpoints constantly, often under a name that matches ours while
the bytes are a different version, a different quantisation, or an inpainting
variant. A node given the wrong weights does not fail - it quietly produces
different output for the same prompt, which is close to undebuggable from the
outside.

So nothing here is matched by name. For each model we hash the local artifact,
find candidate files on Hugging Face whose published size matches, and compare
SHA-256. A model is only rewritten to an ``hf`` source when the bytes are
identical, and the digest is recorded in the manifest so nodes verify what they
downloaded. Anything unmatched is left exactly as it was.

    python scripts/match_hf_sources.py                    # report only
    python scripts/match_hf_sources.py --pipelines sdxl   # skip pipelines you do not serve
    python scripts/match_hf_sources.py --apply            # rewrite the manifest
    python scripts/match_hf_sources.py --only NAME ...    # restrict to some models

Hashing dominates the runtime, so filter before you run: --pipelines skips
non-matching models without reading them. Hashes are cached against size and
mtime, so interrupting and re-running never repeats work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(os.getenv("HAVNAI_MANIFEST_FILE", REPO_ROOT / "server" / "manifests" / "registry.json"))
CACHE = Path(os.getenv("HAVNAI_HASH_CACHE", REPO_ROOT / ".hf-hash-cache.json"))
API = "https://huggingface.co/api"
CHUNK = 8 * 1024 * 1024
MAX_REPOS_PER_MODEL = 8


# ---------------------------------------------------------------------------
# Hugging Face queries
# ---------------------------------------------------------------------------


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "havnai-source-matcher"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None




def search_terms(name: str) -> List[str]:
    """Plausible repo-name searches derived from a checkpoint filename."""

    base = name.split("_")[0]
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base)
    candidates = [name, name.replace("_", " "), spaced, spaced.split()[0] if spaced else base]
    seen: set[str] = set()
    ordered: List[str] = []
    for term in candidates:
        term = term.strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            ordered.append(term)
    return ordered


def candidate_repos(name: str) -> List[str]:
    repos: List[str] = []
    seen: set[str] = set()
    for term in search_terms(name):
        results = _get_json(f"{API}/models?search={urllib.parse.quote(term)}&limit=5&sort=downloads&direction=-1")
        for entry in results or []:
            repo = entry.get("modelId") or entry.get("id")
            if repo and repo not in seen:
                seen.add(repo)
                repos.append(repo)
        if len(repos) >= MAX_REPOS_PER_MODEL:
            break
        time.sleep(0.15)
    return repos[:MAX_REPOS_PER_MODEL]


def files_matching_size(repo: str, size: int) -> List[Tuple[str, str]]:
    """Return ``(path, sha256)`` for safetensors files of exactly ``size``.

    The digest comes from the tree API's LFS ``oid``, which is the file's real
    SHA-256. Do not be tempted to read it from a HEAD of the resolve URL: that
    redirects to the CDN, and the CDN's ETag is a storage-layer checksum, not
    the content hash. Comparing against it silently matches nothing.

    Size is a free pre-filter that eliminates almost every wrong version.
    """

    tree = _get_json(f"{API}/models/{repo}/tree/main?expand=true&recursive=true")
    matches: List[Tuple[str, str]] = []
    for entry in tree or []:
        path = str(entry.get("path", ""))
        if not path.endswith(".safetensors"):
            continue
        lfs = entry.get("lfs") or {}
        entry_size = lfs.get("size") or entry.get("size") or 0
        oid = str(lfs.get("oid") or "").lower()
        if oid and int(entry_size) == size:
            matches.append((path, oid))
    return matches


def repo_commit(repo: str) -> str:
    """Current commit of a repo, so a match can be pinned to exact content."""

    detail = _get_json(f"{API}/models/{repo}")
    return str((detail or {}).get("sha") or "").strip()


# ---------------------------------------------------------------------------
# Local hashing
# ---------------------------------------------------------------------------


def load_cache() -> Dict[str, Any]:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    try:
        CACHE.write_text(json.dumps(cache, indent=2))
    except OSError:
        pass


def local_digest(path: Path, cache: Dict[str, Any], *, quiet: bool = False) -> Optional[Tuple[str, int]]:
    """SHA-256 of a local checkpoint, cached against its size and mtime."""

    try:
        stat = path.stat()
    except OSError:
        return None

    key = str(path)
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime") == int(stat.st_mtime):
        return entry["sha256"], stat.st_size

    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if not quiet and stat.st_size:
                pct = done / stat.st_size * 100
                sys.stderr.write(f"\r    hashing {path.name}: {pct:5.1f}%")
                sys.stderr.flush()
    if not quiet:
        sys.stderr.write("\r" + " " * 60 + "\r")

    value = digest.hexdigest()
    cache[key] = {"sha256": value, "size": stat.st_size, "mtime": int(stat.st_mtime)}
    save_cache(cache)
    return value, stat.st_size


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def find_mirror(name: str, sha256: str, size: int) -> Optional[Dict[str, Any]]:
    """Locate a Hugging Face file with identical bytes, or return None."""

    target = sha256.lower()
    for repo in candidate_repos(name):
        for filename, remote_sha in files_matching_size(repo, size):
            if remote_sha == target:
                # Pin the revision to the current commit so a later force-push
                # to a third-party repo cannot swap the bytes underneath us.
                return {
                    "repo_id": repo,
                    "filename": filename,
                    "revision": repo_commit(repo) or "main",
                    "sha256": target,
                    "size_bytes": size,
                }
        time.sleep(0.1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the manifest (default: report only)")
    parser.add_argument("--only", nargs="*", help="restrict to these model names")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        help="restrict to these pipelines (e.g. sdxl), skipping the rest entirely",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress hashing progress")
    args = parser.parse_args()

    data = json.loads(MANIFEST.read_text())
    models = data.get("models", [])
    cache = load_cache()

    wanted = {n.lower() for n in (args.only or [])}
    pipelines = {p.lower() for p in (args.pipelines or [])}
    matched: List[Tuple[str, Dict[str, Any]]] = []
    unmatched: List[Tuple[str, str]] = []

    for entry in models:
        name = str(entry.get("name", ""))
        source = entry.get("source") or {}
        if source.get("kind") != "coordinator":
            continue
        if wanted and name.lower() not in wanted:
            continue
        # Filter before hashing: reading a 7 GB checkpoint we do not serve is
        # the most expensive thing this script can do for no reason.
        if pipelines and str(entry.get("pipeline", "")).lower() not in pipelines:
            continue

        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            unmatched.append((name, f"local file not found: {path}"))
            print(f"[skip ] {name}: local file not found")
            continue

        result = local_digest(path, cache, quiet=args.quiet)
        if not result:
            unmatched.append((name, "could not hash local file"))
            continue
        sha256, size = result

        mirror = find_mirror(name, sha256, size)
        if mirror:
            matched.append((name, mirror))
            print(f"[MATCH] {name}")
            print(f"         {mirror['repo_id']}/{mirror['filename']}")
            print(f"         sha256 {sha256[:16]}… ({size:,} bytes)")
            entry["source"] = {"kind": "hf", **mirror, "license": source.get("license", "")}
        else:
            unmatched.append((name, "no byte-identical file on Hugging Face"))
            print(f"[keep ] {name}: no byte-identical mirror; leaving on coordinator")

    print()
    print(f"{len(matched)} model(s) can be served from Hugging Face; {len(unmatched)} stay as they are.")

    if matched and args.apply:
        MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
        print(f"manifest updated: {MANIFEST}")
    elif matched:
        print("re-run with --apply to write these changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

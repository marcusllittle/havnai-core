"""Tests for the node runtime bundle served to operators.

The bug these guard against: the installer used to deliver only client.py and
registry.py, so face swap and video died on import on every fresh node. If a
module the client imports at runtime stops being packaged, these fail.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

import node_bundle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every module client.py imports at runtime for a capability the network
# advertises. Missing any one of these is a broken node, not a degraded one.
CAPABILITY_MODULES = [
    "client/client.py",
    "client/registry.py",
    "client/pipeline_stable_diffusion_xl_instantid.py",
    "client/pipeline_stable_diffusion_xl_instantid_inpaint.py",
    "client/ip_adapter/__init__.py",
    "client/ip_adapter/resampler.py",
    "client/ip_adapter/attention_processor.py",
    "client/ip_adapter/utils.py",
    "engines/ltx_video/runner.py",
    "engines/ltx_video/config.py",
    "engines/ltx2/ltx2_runner.py",
    "engines/animatediff/animatediff_runner.py",
    "engines/wangp/runner.py",
]


class BundleContentsTests(unittest.TestCase):
    def test_bundle_carries_every_capability_module(self) -> None:
        payload, _ = node_bundle.build_bundle(REPO_ROOT)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            names = set(archive.getnames())

        missing = [module for module in CAPABILITY_MODULES if module not in names]
        self.assertEqual(missing, [], f"runtime bundle is missing: {missing}")

    def test_bundle_includes_the_operator_tooling(self) -> None:
        payload, _ = node_bundle.build_bundle(REPO_ROOT)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            names = set(archive.getnames())

        for module in (
            "client/doctor.py",
            "client/fetch_models.py",
            "client/model_sources.py",
            "client/requirements-node.txt",
        ):
            self.assertIn(module, names)

    def test_bundle_excludes_bytecode_and_caches(self) -> None:
        payload, _ = node_bundle.build_bundle(REPO_ROOT)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            names = archive.getnames()

        self.assertFalse([name for name in names if "__pycache__" in name])
        self.assertFalse([name for name in names if name.endswith(".pyc")])

    def test_extracted_bundle_forms_importable_packages(self) -> None:
        payload, _ = node_bundle.build_bundle(REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                archive.extractall(root)

            # Synthesised where the repo has none, so `-m client.client` works.
            for package in ("client", "client/ip_adapter", "engines", "shared"):
                self.assertTrue(
                    (root / package / "__init__.py").exists(),
                    f"{package} is not an importable package after extraction",
                )

    def test_build_is_reproducible(self) -> None:
        first, first_digest = node_bundle.build_bundle(REPO_ROOT)
        second, second_digest = node_bundle.build_bundle(REPO_ROOT)
        # Identical sources must produce an identical digest, otherwise every
        # install would look like an upgrade.
        self.assertEqual(first_digest, second_digest)
        self.assertEqual(len(first), len(second))

    def test_missing_entry_point_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(node_bundle.BundleError):
                node_bundle.build_bundle(Path(tmp))


class BundleManifestTests(unittest.TestCase):
    def test_manifest_digests_match_the_files(self) -> None:
        import hashlib

        manifest = node_bundle.bundle_manifest(REPO_ROOT)
        self.assertGreater(manifest["file_count"], 20)

        entry = next(
            item for item in manifest["files"] if item["path"] == "client/client.py"
        )
        actual = hashlib.sha256((REPO_ROOT / "client" / "client.py").read_bytes()).hexdigest()
        self.assertEqual(entry["sha256"], actual)

    def test_manifest_digest_is_stable(self) -> None:
        first = node_bundle.bundle_manifest(REPO_ROOT)["bundle_sha256"]
        second = node_bundle.bundle_manifest(REPO_ROOT)["bundle_sha256"]
        self.assertEqual(first, second)


class BundleCacheTests(unittest.TestCase):
    def test_cache_returns_the_same_payload(self) -> None:
        cache = node_bundle.BundleCache(REPO_ROOT, enabled=True)
        first, first_digest = cache.get()
        second, second_digest = cache.get()
        self.assertEqual(first_digest, second_digest)
        self.assertIs(first, second)  # served from cache, not rebuilt

    def test_disabled_cache_rebuilds(self) -> None:
        cache = node_bundle.BundleCache(REPO_ROOT, enabled=False)
        first, _ = cache.get()
        second, _ = cache.get()
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()

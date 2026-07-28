"""Tests for non-blocking model provenance lookups."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ModelHashCacheTests(unittest.TestCase):
    def test_lookup_uses_current_cache_without_hashing_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.safetensors"
            cache_path = root / "model_hashes.json"
            model_path.write_bytes(b"model")
            stat = model_path.stat()
            expected = "a" * 64
            cache_path.write_text(
                json.dumps(
                    {
                        str(model_path.resolve()): {
                            "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
                            "sha256": expected,
                        }
                    }
                ),
                encoding="utf-8",
            )
            script = """
import sys
from pathlib import Path
from client import client

model_path = Path(sys.argv[1])
client._MODEL_HASH_CACHE_PATH = Path(sys.argv[2])
client._file_sha256 = lambda _path: (_ for _ in ()).throw(AssertionError("model file hashed"))
assert client._get_cached_model_sha256(model_path) == sys.argv[3]
client._MODEL_HASH_CACHE_PATH = Path(sys.argv[4])
assert client._get_cached_model_sha256(model_path) is None
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(model_path),
                    str(cache_path),
                    expected,
                    str(root / "missing.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

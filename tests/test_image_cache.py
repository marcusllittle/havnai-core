"""Tests for image pipeline cache behavior in the node client."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT))

import client as client_module  # type: ignore


class _FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGenerator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        return self


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeCuda()
    float16 = "float16"
    float32 = "float32"
    Generator = _FakeGenerator

    @staticmethod
    def inference_mode() -> _FakeInferenceMode:
        return _FakeInferenceMode()


class _FakePipe:
    def __call__(self, *args, **kwargs):
        image = client_module.Image.new("RGB", (64, 64), color=(10, 20, 30))
        return SimpleNamespace(images=[image])


class ImagePipelineCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_cache_size = client_module.IMAGE_PIPELINE_CACHE_SIZE
        self._orig_fast_preview = client_module.FAST_PREVIEW
        self._orig_torch = client_module.torch
        self._orig_diffusers = client_module.diffusers
        client_module.IMAGE_PIPELINE_CACHE_SIZE = 1
        client_module.FAST_PREVIEW = False
        client_module.torch = _FakeTorch()  # type: ignore[assignment]
        client_module.diffusers = object()  # type: ignore[assignment]
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()

    def tearDown(self) -> None:
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()
        client_module.IMAGE_PIPELINE_CACHE_SIZE = self._orig_cache_size
        client_module.FAST_PREVIEW = self._orig_fast_preview
        client_module.torch = self._orig_torch
        client_module.diffusers = self._orig_diffusers

    def test_acquire_pipeline_hits_cache_on_second_lookup(self) -> None:
        entry = SimpleNamespace(name="m1")
        model_a = Path("/tmp/model-a.safetensors")
        build_calls = {"count": 0}

        def _build(*args, **kwargs):
            build_calls["count"] += 1
            return _FakePipe(), 42

        with patch.object(client_module, "_construct_base_image_pipeline", side_effect=_build):
            pipe1, hit1, load1 = client_module._acquire_base_image_pipeline(
                entry, model_a, "sdxl", "float16", True, "cpu"
            )
            pipe2, hit2, load2 = client_module._acquire_base_image_pipeline(
                entry, model_a, "sdxl", "float16", True, "cpu"
            )

        self.assertIs(pipe1, pipe2)
        self.assertFalse(hit1)
        self.assertEqual(load1, 42)
        self.assertTrue(hit2)
        self.assertEqual(load2, 0)
        self.assertEqual(build_calls["count"], 1)

    def test_lora_run_uses_transient_pipeline_and_does_not_use_cache_path(self) -> None:
        entry = SimpleNamespace(name="m2", pipeline="sd15")
        model_path = Path("/tmp/model-b.safetensors")
        fake_pipe = _FakePipe()
        lora_entries = [(Path("/tmp/test.safetensors"), 0.55, "lora_style_test")]

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", False, "sd15")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=lora_entries
        ), patch.object(
            client_module, "_construct_base_image_pipeline", return_value=(fake_pipe, 15)
        ) as construct_mock, patch.object(
            client_module, "_acquire_base_image_pipeline", side_effect=RuntimeError("cache path should not be used")
        ) as acquire_mock, patch.object(
            client_module, "_apply_explicit_loras", return_value=(["test.safetensors:0.55"], 7)
        ), patch.object(
            client_module, "_release_image_pipeline"
        ) as release_mock:
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-cache-lora",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="portrait",
                negative_prompt="",
                job_settings={"loras": [{"name": "test", "weight": 0.55}]},
            )

        self.assertEqual(metrics["status"], "success")
        self.assertFalse(metrics["pipeline_cache_hit"])
        self.assertEqual(metrics["pipeline_load_ms"], 15)
        self.assertEqual(metrics["lora_load_ms"], 7)
        self.assertGreaterEqual(metrics["generation_ms"], 0)
        construct_mock.assert_called_once()
        acquire_mock.assert_not_called()
        release_mock.assert_called_once_with(fake_pipe)


if __name__ == "__main__":
    unittest.main()

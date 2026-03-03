"""Tests for image pipeline cache behavior in the node client."""

from __future__ import annotations

import sys
import types
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
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(images=["fake-image"])


class _FakeFaceAnalysisApp:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return list(self.faces)


class _FakeNumPy:
    uint8 = "uint8"

    @staticmethod
    def array(value):
        return value


class _FakeCv2:
    COLOR_RGB2BGR = 1

    @staticmethod
    def cvtColor(image, code):
        return image


class ImagePipelineCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_cache_size = client_module.IMAGE_PIPELINE_CACHE_SIZE
        self._orig_fast_preview = client_module.FAST_PREVIEW
        self._orig_torch = client_module.torch
        self._orig_diffusers = client_module.diffusers
        self._orig_face_analysis = client_module.FaceAnalysis
        self._orig_save_output_image = client_module._save_output_image
        self._orig_np = client_module.np
        self._orig_cv2 = client_module.cv2
        self._orig_image = client_module.Image
        client_module.IMAGE_PIPELINE_CACHE_SIZE = 1
        client_module.FAST_PREVIEW = False
        client_module.torch = _FakeTorch()  # type: ignore[assignment]
        client_module.diffusers = object()  # type: ignore[assignment]
        client_module.FaceAnalysis = object()  # type: ignore[assignment]
        if client_module.Image is None:
            client_module.Image = object()  # type: ignore[assignment]
        if client_module.np is None:
            client_module.np = _FakeNumPy()  # type: ignore[assignment]
        if client_module.cv2 is None:
            client_module.cv2 = _FakeCv2()  # type: ignore[assignment]
        client_module._save_output_image = (
            lambda img, path, task_id=None: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"fake")
        )
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()
        client_module._REFERENCE_FACE_PIPE = None
        client_module._REFERENCE_FACE_PIPE_MODEL = ""

    def tearDown(self) -> None:
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()
        client_module.IMAGE_PIPELINE_CACHE_SIZE = self._orig_cache_size
        client_module.FAST_PREVIEW = self._orig_fast_preview
        client_module.torch = self._orig_torch
        client_module.diffusers = self._orig_diffusers
        client_module.FaceAnalysis = self._orig_face_analysis
        client_module._save_output_image = self._orig_save_output_image
        client_module.np = self._orig_np
        client_module.cv2 = self._orig_cv2
        client_module.Image = self._orig_image
        client_module._REFERENCE_FACE_PIPE = None
        client_module._REFERENCE_FACE_PIPE_MODEL = ""

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
        self.assertFalse(metrics["reference_face_used"])
        self.assertNotIn("reference_face_pipeline", metrics)
        construct_mock.assert_called_once()
        acquire_mock.assert_not_called()
        release_mock.assert_called_once_with(fake_pipe)

    def test_reference_face_run_uses_instantid_pipeline(self) -> None:
        entry = SimpleNamespace(name="m3", pipeline="sdxl")
        model_path = Path("/tmp/model-ref.safetensors")
        fake_pipe = _FakePipe()
        face_image = [[[0, 0, 0]]]
        fake_module = types.ModuleType("pipeline_stable_diffusion_xl_instantid")
        fake_module.draw_kps = lambda image, kps: image

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=[]
        ), patch.object(
            client_module, "_acquire_reference_face_pipeline", return_value=(fake_pipe, True, 0)
        ) as acquire_mock, patch.object(
            client_module, "get_face_analysis",
            return_value=_FakeFaceAnalysisApp(
                [
                    {
                        "bbox": [0, 0, 10, 10],
                        "kps": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ]
            ),
        ), patch.object(
            client_module, "load_image_source_with_error", return_value=(face_image, None)
        ), patch.dict(
            sys.modules, {"pipeline_stable_diffusion_xl_instantid": fake_module}
        ):
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-reference-face",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="portrait",
                negative_prompt="",
                job_settings={"reference_face_url": "https://example.com/face.png"},
            )

        self.assertEqual(metrics["status"], "success")
        self.assertTrue(metrics["reference_face_used"])
        self.assertEqual(metrics["reference_face_pipeline"], "instantid")
        acquire_mock.assert_called_once_with(entry, model_path, "float32", "cpu")
        self.assertEqual(fake_pipe.calls[0][1]["image_embeds"], [0.1, 0.2, 0.3])
        self.assertEqual(fake_pipe.calls[0][1]["controlnet_conditioning_scale"], 0.8)
        self.assertEqual(fake_pipe.calls[0][1]["ip_adapter_scale"], 0.8)

    def test_reference_face_run_fails_when_no_face_detected(self) -> None:
        entry = SimpleNamespace(name="m4", pipeline="sdxl")
        model_path = Path("/tmp/model-ref-fail.safetensors")
        fake_pipe = _FakePipe()
        face_image = [[[0, 0, 0]]]
        fake_module = types.ModuleType("pipeline_stable_diffusion_xl_instantid")
        fake_module.draw_kps = lambda image, kps: image

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=[]
        ), patch.object(
            client_module, "_acquire_reference_face_pipeline", return_value=(fake_pipe, True, 0)
        ), patch.object(
            client_module, "get_face_analysis", return_value=_FakeFaceAnalysisApp([])
        ), patch.object(
            client_module, "load_image_source_with_error", return_value=(face_image, None)
        ), patch.dict(
            sys.modules, {"pipeline_stable_diffusion_xl_instantid": fake_module}
        ):
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-reference-face-fail",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="portrait",
                negative_prompt="",
                job_settings={"reference_face_url": "https://example.com/face.png"},
            )

        self.assertEqual(metrics["status"], "failed")
        self.assertTrue(metrics["reference_face_used"])
        self.assertEqual(metrics["reference_face_pipeline"], "instantid")
        self.assertIn("No face detected in reference face image", metrics["error"])


if __name__ == "__main__":
    unittest.main()

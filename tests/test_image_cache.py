"""Tests for image pipeline cache behavior in the node client."""

from __future__ import annotations

import base64
import io
import sys
import threading
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

    def __call__(self, *args, callback_on_step_end=None, **kwargs):
        if callback_on_step_end is not None:
            total_steps = int(kwargs.get("num_inference_steps") or 1)
            if kwargs.get("image") is not None:
                total_steps = max(1, int(total_steps * float(kwargs.get("strength") or 1.0)))
            for step_index in range(total_steps):
                callback_on_step_end(self, step_index, 0, {"latents": "state"})
        self.calls.append((args, kwargs))
        source = kwargs.get("image")
        image_size = source.size if source is not None else (64, 64)
        image = client_module.Image.new("RGB", image_size, color=(10, 20, 30))
        return SimpleNamespace(images=[image])


class _ModernCallbackPipe:
    def __call__(self, prompt=None, callback_on_step_end=None):
        return None


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

    def test_diffusers_callback_reports_throttled_step_progress(self) -> None:
        reports = []
        callback_kwargs = client_module._pipeline_cancel_kwargs(
            _ModernCallbackPipe(),
            threading.Event(),
            lambda progress, stage: reports.append((progress, stage)),
            total_steps=10,
            stage="generating_image",
        )
        callback = callback_kwargs["callback_on_step_end"]
        pipeline_state = {"latents": "state"}

        self.assertIs(callback(_ModernCallbackPipe(), 4, 0, pipeline_state), pipeline_state)
        callback(_ModernCallbackPipe(), 4, 0, pipeline_state)
        callback(_ModernCallbackPipe(), 9, 0, pipeline_state)

        self.assertEqual(reports, [(50, "generating_image"), (90, "generating_image")])

    def test_diffusers_callback_keeps_cancellation_behavior(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        callback = client_module._pipeline_cancel_kwargs(
            _ModernCallbackPipe(),
            cancel_event,
            total_steps=10,
        )["callback_on_step_end"]

        with self.assertRaises(client_module.TaskCancelled):
            callback(_ModernCallbackPipe(), 0, 0, {"latents": "state"})

    def test_relative_image_source_uses_coordinator_base_url(self) -> None:
        source = client_module.Image.new("RGB", (32, 24), color=(15, 25, 35))
        encoded = io.BytesIO()
        source.save(encoded, format="PNG")
        response = SimpleNamespace(
            content=encoded.getvalue(),
            raise_for_status=lambda: None,
        )

        with patch.object(client_module.requests, "get", return_value=response) as get_mock:
            loaded, error = client_module.load_image_source_with_error(
                "/static/outputs/job-refine.png",
                base_url="http://192.168.4.105:5001",
            )

        self.assertIsNone(error)
        self.assertEqual(loaded.size, (32, 24))
        get_mock.assert_called_once_with(
            "http://192.168.4.105:5001/static/outputs/job-refine.png",
            timeout=30,
            headers={"User-Agent": "HavnAI/1.0"},
        )

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

    def test_img2img_passes_reference_image_and_preservation_strength(self) -> None:
        entry = SimpleNamespace(name="m3", pipeline="sdxl")
        model_path = Path("/tmp/model-c.safetensors")
        fake_pipe = _FakePipe()
        reported_progress = []
        source = client_module.Image.new("RGB", (80, 48), color=(90, 40, 20))
        encoded = io.BytesIO()
        source.save(encoded, format="PNG")
        source_data = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_acquire_base_image_pipeline", return_value=(fake_pipe, True, 0)
        ) as acquire_mock:
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-img2img",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="preserve the composition",
                negative_prompt="",
                job_settings={
                    "init_image": source_data,
                    "img2img_strength": 0.2,
                    "width": 64,
                    "height": 64,
                    "_return_b64": False,
                    "_progress_fn": lambda progress, stage: reported_progress.append((progress, stage)),
                },
            )

        self.assertEqual(metrics["status"], "success")
        self.assertTrue(metrics["image_to_image_used"])
        self.assertEqual(metrics["img2img_strength"], 0.2)
        self.assertTrue(metrics["preserve_reference_aspect"])
        self.assertEqual(metrics["reference_preparation"], "source_aspect")
        self.assertEqual(metrics["reference_source_width"], 80)
        self.assertEqual(metrics["reference_source_height"], 48)
        self.assertEqual(metrics["width"], 448)
        self.assertEqual(metrics["height"], 256)
        self.assertEqual(reported_progress[0], (5, "loading_image_model"))
        self.assertEqual(reported_progress[-1], (90, "generating_image"))
        acquire_mock.assert_called_once_with(
            entry, model_path, "sdxl", "float32", True, "cpu", "img2img"
        )
        _, kwargs = fake_pipe.calls[0]
        self.assertEqual(kwargs["strength"], 0.2)
        self.assertEqual(kwargs["image"].size, (448, 256))
        self.assertNotIn("height", kwargs)
        self.assertNotIn("width", kwargs)

    def test_img2img_explicit_size_center_crops_without_stretching(self) -> None:
        source = client_module.Image.new("RGB", (1200, 600), color=(30, 60, 90))

        prepared, mode = client_module._prepare_img2img_reference(
            source,
            (768, 768),
            preserve_source_aspect=False,
        )

        self.assertEqual(prepared.size, (768, 768))
        self.assertEqual(mode, "center_crop")

    def test_inpainting_regenerates_only_selected_pixels(self) -> None:
        entry = SimpleNamespace(name="m-inpaint", pipeline="sdxl")
        model_path = Path("/tmp/model-inpaint.safetensors")
        fake_pipe = _FakePipe()
        source = client_module.Image.new("RGB", (64, 64), color=(80, 90, 100))
        mask = client_module.Image.new("RGB", (64, 64), color="black")
        client_module.ImageDraw.Draw(mask).rectangle((0, 0, 31, 63), fill="white")
        source_encoded = io.BytesIO()
        mask_encoded = io.BytesIO()
        source.save(source_encoded, format="PNG")
        mask.save(mask_encoded, format="PNG")
        source_data = "data:image/png;base64," + base64.b64encode(source_encoded.getvalue()).decode("ascii")
        mask_data = "data:image/png;base64," + base64.b64encode(mask_encoded.getvalue()).decode("ascii")
        saved = {}

        def _capture_output(image, *_args, **_kwargs):
            saved["image"] = image.copy()

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_acquire_base_image_pipeline", return_value=(fake_pipe, True, 0)
        ) as acquire_mock, patch.object(client_module, "_save_output_image", side_effect=_capture_output):
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-inpaint",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="localized change",
                negative_prompt="",
                job_settings={
                    "init_image": source_data,
                    "inpaint_mask": mask_data,
                    "img2img_strength": 0.35,
                    "width": 256,
                    "height": 256,
                    "_return_b64": False,
                },
            )

        self.assertEqual(metrics["status"], "success")
        self.assertTrue(metrics["image_to_image_used"])
        self.assertTrue(metrics["inpainting_used"])
        acquire_mock.assert_called_once_with(
            entry, model_path, "sdxl", "float32", True, "cpu", "inpaint"
        )
        _, kwargs = fake_pipe.calls[0]
        self.assertEqual(kwargs["mask_image"].mode, "L")
        self.assertEqual(kwargs["mask_image"].size, (256, 256))
        self.assertEqual(saved["image"].getpixel((32, 128)), (10, 20, 30))
        self.assertEqual(saved["image"].getpixel((224, 128)), (80, 90, 100))

    def test_reference_output_size_tracks_portrait_source_ratio(self) -> None:
        width, height = client_module._reference_output_size((768, 1344), (768, 768))

        self.assertEqual((width, height), (576, 1024))

    def test_txt2img_does_not_receive_img2img_arguments(self) -> None:
        entry = SimpleNamespace(name="m4", pipeline="sdxl")
        model_path = Path("/tmp/model-d.safetensors")
        fake_pipe = _FakePipe()

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_acquire_base_image_pipeline", return_value=(fake_pipe, True, 0)
        ) as acquire_mock:
            metrics, _, _ = client_module.run_image_generation(
                task_id="job-txt2img",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="new composition",
                negative_prompt="",
                job_settings={"_return_b64": False},
            )

        self.assertEqual(metrics["status"], "success")
        self.assertFalse(metrics["image_to_image_used"])
        acquire_mock.assert_called_once_with(
            entry, model_path, "sdxl", "float32", True, "cpu", "txt2img"
        )
        _, kwargs = fake_pipe.calls[0]
        self.assertNotIn("image", kwargs)
        self.assertNotIn("strength", kwargs)
        self.assertEqual(kwargs["height"], client_module.IMAGE_HEIGHT)
        self.assertEqual(kwargs["width"], client_module.IMAGE_WIDTH)


if __name__ == "__main__":
    unittest.main()

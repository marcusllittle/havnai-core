"""Tests for image pipeline cache behavior in the node client."""

from __future__ import annotations

import sys
import io
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


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

    def test_refinement_and_masks_reach_the_correct_pipeline_without_cache_mutation(self) -> None:
        for masked in (False, True):
            with self.subTest(masked=masked), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.png"
                client_module.Image.new("RGB", (80, 40), (20, 40, 60)).save(source)
                mask_bytes = io.BytesIO()
                client_module.Image.new("L", (40, 20), 255).save(mask_bytes, format="PNG")
                mask = "data:image/png;base64," + base64.b64encode(mask_bytes.getvalue()).decode()
                base = Mock()
                converted = Mock(side_effect=_FakePipe())
                converter = Mock()
                converter.from_pipe.return_value = converted
                runtime = SimpleNamespace(**{
                    "AutoPipelineForInpainting" if masked else "AutoPipelineForImage2Image": converter,
                })
                with patch.object(client_module, "diffusers", runtime), patch.object(
                    client_module, "read_gpu_stats", return_value={"utilization": 0}
                ), patch.object(client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")), patch.object(
                    client_module, "_collect_explicit_loras", return_value=[(Path("style"), 0.5, "adapter")]
                ), patch.object(client_module, "_construct_base_image_pipeline", return_value=(base, 12)), patch.object(
                    client_module, "_acquire_base_image_pipeline"
                ) as cache, patch.object(client_module, "_apply_explicit_loras", return_value=(["style"], 1)) as loras, patch.object(
                    client_module, "_truncate_image_prompts", return_value=("edit", "noise")
                ), patch.object(client_module, "_apply_image_sampler"), patch.object(
                    client_module, "_pipeline_cancel_kwargs", return_value={}
                ), patch.object(client_module, "_save_output_image"), patch.object(client_module, "_release_image_pipeline") as release:
                    metrics, _, _ = client_module.run_image_generation(
                        task_id="refine-test", entry=SimpleNamespace(name="sdxl", pipeline="sdxl"),
                        model_path=Path("model.safetensors"), reward_weight=1, prompt="edit", negative_prompt="noise",
                        job_settings={"init_image": str(source), "inpaint_mask": mask if masked else None,
                                      "img2img_strength": 0.4, "steps": 20, "width": 512, "height": 512,
                                      "preserve_reference_aspect": True, "_return_b64": False},
                    )
                self.assertEqual(metrics["status"], "success", metrics)
                self.assertEqual((metrics["width"], metrics["height"]), (512, 256))
                converter.from_pipe.assert_called_once_with(base)
                base.assert_not_called()
                cache.assert_not_called()
                self.assertIs(loras.call_args.args[0], converted)
                release.assert_called_once_with(converted)
                arguments = converted.call_args.kwargs
                self.assertEqual(arguments["image"].size, (512, 256))
                self.assertEqual(arguments["image"].getpixel((0, 0)), (20, 40, 60))
                self.assertEqual(arguments["strength"], 0.4)
                if masked:
                    self.assertEqual(arguments["mask_image"].mode, "L")
                    self.assertEqual(arguments["mask_image"].size, (512, 256))
                else:
                    self.assertNotIn("mask_image", arguments)

    def test_invalid_refinement_never_falls_back_to_text_only(self) -> None:
        cases = [
            {"init_image": "invalid source"},
            {"inpaint_mask": "mask without source"},
            {"init_image": "source", "img2img_strength": 0},
            {"init_image": "source", "img2img_strength": float("nan")},
        ]
        for settings in cases:
            with self.subTest(settings=settings), patch.object(client_module, "read_gpu_stats", return_value={}), patch.object(
                client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
            ), patch.object(client_module, "_collect_explicit_loras", return_value=[]), patch.object(
                client_module, "_construct_base_image_pipeline"
            ) as construct, patch.object(client_module, "_acquire_base_image_pipeline") as cache:
                metrics, _, output = client_module.run_image_generation(
                    task_id="invalid-refine", entry=SimpleNamespace(name="sdxl"), model_path=Path("model"),
                    reward_weight=1, prompt="edit", negative_prompt="", job_settings=settings,
                )
            self.assertEqual(metrics["status"], "failed")
            self.assertIsNone(output)
            construct.assert_not_called()
            cache.assert_not_called()

    def test_downloaded_owned_images_reach_generation_and_prompt_json_is_literal(self) -> None:
        prompt = '{"init_image":"/private/file","steps":1,"prompt":"injected"}'
        task = {"task_id": "owned-image", "type": "IMAGE_GEN", "model_name": "sdxl",
                "prompt": prompt, "resolved_spec": {"schema_version": 1},
                "source_asset_id": "source-owned", "mask_asset_id": "mask-owned",
                "img2img_strength": 0.4, "preserve_reference_aspect": True, "steps": 20}
        with patch.object(client_module, "ROLE", "creator"), patch.object(client_module, "_is_model_allowed", return_value=True), patch.object(
            client_module, "_download_task_asset", side_effect=lambda asset, job, kind: Path("/tmp") / f"{asset}.png"
        ) as download, patch.object(client_module, "ensure_model_entry", return_value=SimpleNamespace(name="sdxl")), patch.object(
            client_module, "ensure_model_path", return_value=Path("model")
        ), patch.object(client_module, "run_image_generation", return_value=({"status": "success"}, 0, None)) as generate, patch.object(
            client_module, "_task_output_path", return_value=None
        ), patch.object(client_module.SESSION, "post") as post:
            post.return_value.json.return_value = {"reward": 0}
            client_module.execute_task(task)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(generate.call_args.args[4], prompt)
        settings = generate.call_args.args[6]
        self.assertEqual(settings["init_image"], "/tmp/source-owned.png")
        self.assertEqual(settings["inpaint_mask"], "/tmp/mask-owned.png")
        self.assertEqual(settings["steps"], 20)
        self.assertEqual(settings["img2img_strength"], 0.4)
        self.assertTrue(settings["preserve_reference_aspect"])


if __name__ == "__main__":
    unittest.main()

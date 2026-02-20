"""Tests for image pipeline cache behavior in the node client."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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
    class Tensor:  # pragma: no cover - test shim
        pass

    @staticmethod
    def from_numpy(_value):
        return _value

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
        lora_entries = [(Path("/tmp/test.safetensors"), 0.55, "lora_style_test", 0.55, "test")]

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", False, "sd15")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=lora_entries
        ), patch.object(
            client_module, "_construct_base_image_pipeline", return_value=(fake_pipe, 15)
        ) as construct_mock, patch.object(
            client_module, "_acquire_base_image_pipeline", side_effect=RuntimeError("cache path should not be used")
        ) as acquire_mock, patch.object(
            client_module,
            "_apply_explicit_loras",
            return_value=(["test.safetensors:0.55"], [{"name": "test", "applied_weight": 0.55}], 7),
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


class LoraWeightPolicyTests(unittest.TestCase):
    def test_collect_explicit_loras_preserves_exact_user_weight(self) -> None:
        entry = SimpleNamespace(name="m1", pipeline="sdxl")
        with patch.object(client_module, "resolve_lora_path", return_value=Path("/tmp/incase_style.safetensors")):
            collected = client_module._collect_explicit_loras(
                [{"name": "incase_style", "weight": 0.74}],
                entry,
                "sdxl",
            )
        self.assertEqual(len(collected), 1)
        self.assertAlmostEqual(collected[0][1], 0.74, places=6)
        self.assertAlmostEqual(collected[0][3], 0.74, places=6)

    def test_collect_explicit_loras_accepts_strength_fallback(self) -> None:
        entry = SimpleNamespace(name="m1", pipeline="sdxl")
        with patch.object(client_module, "resolve_lora_path", return_value=Path("/tmp/incase_style.safetensors")):
            collected = client_module._collect_explicit_loras(
                [{"name": "incase_style", "strength": 0.66}],
                entry,
                "sdxl",
            )
        self.assertEqual(len(collected), 1)
        self.assertAlmostEqual(collected[0][1], 0.66, places=6)

    def test_collect_explicit_loras_defaults_to_one_when_unspecified(self) -> None:
        entry = SimpleNamespace(name="m1", pipeline="sdxl")
        with patch.object(client_module, "resolve_lora_path", return_value=Path("/tmp/incase_style.safetensors")):
            collected = client_module._collect_explicit_loras(
                [{"name": "incase_style"}],
                entry,
                "sdxl",
            )
        self.assertEqual(len(collected), 1)
        self.assertAlmostEqual(collected[0][1], 1.0, places=6)

    def test_collect_explicit_loras_defaults_to_one_for_known_roles(self) -> None:
        entry = SimpleNamespace(name="m1", pipeline="sd15")
        with patch.object(client_module, "resolve_lora_path", return_value=Path("/tmp/POVDoggy.safetensors")):
            collected = client_module._collect_explicit_loras(
                [{"name": "POVDoggy"}],
                entry,
                "sd15",
            )
        self.assertEqual(len(collected), 1)
        self.assertAlmostEqual(collected[0][1], 1.0, places=6)


class LoraLoggingRegressionTests(unittest.TestCase):
    def test_apply_explicit_loras_does_not_collide_with_logrecord_fields(self) -> None:
        class _LogSafePipe:
            def __init__(self) -> None:
                self.loaded = []
                self.adapters = None

            def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
                self.loaded.append((path, adapter_name))

            def set_adapters(self, names, weights=None) -> None:
                self.adapters = (names, weights)

        pipe = _LogSafePipe()
        entry = SimpleNamespace(name="cyberrealisticPony_v160")
        lora_path = Path("/tmp/incase_style.safetensors")
        lora_entries = [(lora_path, 0.25, "lora_style_incase_style", 0.25, "incase_style")]

        loaded, applied, elapsed_ms = client_module._apply_explicit_loras(pipe, entry, lora_entries)

        self.assertEqual(loaded, ["incase_style.safetensors:0.25"])
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["filename"], "incase_style.safetensors")
        self.assertAlmostEqual(applied[0]["requested_weight"], 0.25, places=6)
        self.assertAlmostEqual(applied[0]["applied_weight"], 0.25, places=6)
        self.assertIsNotNone(pipe.adapters)
        self.assertGreaterEqual(elapsed_ms, 0)


class ImageTimeoutAndProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_cache_size = client_module.IMAGE_PIPELINE_CACHE_SIZE
        self._orig_fast_preview = client_module.FAST_PREVIEW
        self._orig_torch = client_module.torch
        self._orig_diffusers = client_module.diffusers
        self._orig_timeout = client_module.IMAGE_JOB_TIMEOUT_SECONDS
        client_module.IMAGE_PIPELINE_CACHE_SIZE = 1
        client_module.FAST_PREVIEW = False
        client_module.torch = _FakeTorch()  # type: ignore[assignment]
        client_module.diffusers = object()  # type: ignore[assignment]
        client_module.IMAGE_JOB_TIMEOUT_SECONDS = 480
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()

    def tearDown(self) -> None:
        with client_module._IMAGE_PIPELINE_CACHE_LOCK:
            client_module._IMAGE_PIPELINE_CACHE.clear()
        client_module.IMAGE_PIPELINE_CACHE_SIZE = self._orig_cache_size
        client_module.FAST_PREVIEW = self._orig_fast_preview
        client_module.torch = self._orig_torch
        client_module.diffusers = self._orig_diffusers
        client_module.IMAGE_JOB_TIMEOUT_SECONDS = self._orig_timeout

    def test_timeout_callback_marks_failed_with_image_timeout_reason(self) -> None:
        class _TimeoutPipe:
            def __call__(self, *args, **kwargs):
                callback = kwargs.get("callback_on_step_end")
                if callback is not None:
                    for step_idx in range(20):
                        callback(self, step_idx, None, {})
                image = client_module.Image.new("RGB", (64, 64), color=(1, 2, 3))
                return SimpleNamespace(images=[image])

        entry = SimpleNamespace(name="timeout-model", pipeline="sdxl")
        model_path = Path("/tmp/timeout-model.safetensors")
        fake_pipe = _TimeoutPipe()
        ticks = {"value": 0.0}

        def _fake_time() -> float:
            ticks["value"] += 2.0
            return ticks["value"]

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=[]
        ), patch.object(
            client_module, "_acquire_base_image_pipeline", return_value=(fake_pipe, False, 10)
        ), patch.object(
            client_module, "IMAGE_JOB_TIMEOUT_SECONDS", 8
        ), patch.object(
            client_module.time, "time", side_effect=_fake_time
        ):
            metrics, _, image_b64 = client_module.run_image_generation(
                task_id="job-timeout",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="portrait",
                negative_prompt="",
                job_settings={"steps": 30, "guidance": 7.0},
            )

        self.assertEqual(metrics["status"], "failed")
        self.assertEqual(metrics["status_reason"], "image_timeout")
        self.assertEqual(metrics["timeout_seconds"], 8)
        self.assertGreater(metrics["elapsed_ms"], 0)
        self.assertIn("exceeded timeout", metrics.get("error", ""))
        self.assertIsNone(image_b64)

    def test_progress_callback_logs_start_progress_and_complete(self) -> None:
        class _ProgressPipe:
            def __call__(self, *args, **kwargs):
                callback = kwargs.get("callback_on_step_end")
                if callback is not None:
                    for step_idx in range(10):
                        callback(self, step_idx, None, {})
                image = client_module.Image.new("RGB", (64, 64), color=(10, 20, 30))
                return SimpleNamespace(images=[image])

        entry = SimpleNamespace(name="progress-model", pipeline="sdxl")
        model_path = Path("/tmp/progress-model.safetensors")
        fake_pipe = _ProgressPipe()

        with patch.object(client_module, "read_gpu_stats", return_value={"utilization": 0}), patch.object(
            client_module, "_resolve_image_runtime", return_value=("cpu", "float32", True, "sdxl")
        ), patch.object(
            client_module, "_collect_explicit_loras", return_value=[]
        ), patch.object(
            client_module, "_acquire_base_image_pipeline", return_value=(fake_pipe, False, 12)
        ), patch.object(
            client_module, "log"
        ) as log_mock:
            metrics, _, image_b64 = client_module.run_image_generation(
                task_id="job-progress",
                entry=entry,
                model_path=model_path,
                reward_weight=1.0,
                prompt="portrait",
                negative_prompt="",
                job_settings={"steps": 10, "guidance": 7.0},
            )

        messages = [str(call.args[0]) for call in log_mock.call_args_list if call.args]
        self.assertEqual(metrics["status"], "success")
        self.assertTrue(any(msg == "Starting denoise" for msg in messages))
        self.assertTrue(any(msg == "Denoise progress" for msg in messages))
        self.assertTrue(any(msg == "Denoise complete" for msg in messages))
        self.assertIsNotNone(image_b64)


class FaceSwapGuidancePassThroughTests(unittest.TestCase):
    def test_faceswap_guidance_from_task_reaches_pipeline(self) -> None:
        class _FakeImageProj:
            def to(self, **kwargs):
                return self

        class _FakeFaceSwapPipe:
            def __init__(self) -> None:
                self.kwargs = None
                self.image_proj_model = _FakeImageProj()

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                image = client_module.Image.new("RGB", (64, 64), color=(20, 30, 40))
                return SimpleNamespace(images=[image])

        class _FakeFaceApp:
            def get(self, _img):
                return [{"embedding": [0.1, 0.2, 0.3]}]

        class _FakeCv2:
            COLOR_RGB2BGR = 0

            @staticmethod
            def cvtColor(arr, _code):
                return arr

        class _FakeNumpy:
            ndarray = tuple

            @staticmethod
            def array(_obj):
                return [[0]]

        fake_pipe = _FakeFaceSwapPipe()
        entry = SimpleNamespace(name="epicrealismXL_vxviiCrystalclear", pipeline="sdxl")
        model_path = Path("/tmp/epicrealismXL_vxviiCrystalclear.safetensors")
        task = {
            "prompt": "faceswap portrait",
            "base_image_url": "https://example.com/base.png",
            "face_source_url": "https://example.com/face.png",
            "num_steps": 14,
            "guidance": 4.6,
            "strength": 0.72,
            "seed": 1234,
        }
        base_img = client_module.Image.new("RGB", (64, 64), color=(1, 2, 3))
        face_img = client_module.Image.new("RGB", (64, 64), color=(5, 6, 7))

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_key = f"{model_path}|controlnet-ref|/tmp/adapter.bin"
            with patch.object(client_module, "torch", _FakeTorch()), patch.object(
                client_module, "diffusers", object()
            ), patch.object(
                client_module, "cv2", _FakeCv2()
            ), patch.object(
                client_module, "np", _FakeNumpy()
            ), patch.object(
                client_module, "_FACE_SWAP_PIPE", fake_pipe
            ), patch.object(
                client_module, "_FACE_SWAP_PIPE_MODEL", cache_key
            ), patch.object(
                client_module, "_resolve_instantid_adapter_path", return_value=Path("/tmp/adapter.bin")
            ), patch.object(
                client_module, "_resolve_instantid_controlnet_ref", return_value="controlnet-ref"
            ), patch.object(
                client_module, "load_image_source_with_error", side_effect=[(base_img, None), (face_img, None)]
            ), patch.object(
                client_module, "get_face_analysis", return_value=_FakeFaceApp()
            ), patch.object(
                client_module, "pick_primary_face", side_effect=lambda faces: faces[0] if faces else None
            ), patch.object(
                client_module,
                "prepare_mask_and_pose_control",
                return_value=((base_img, base_img, base_img), (0, 0, 64, 64)),
            ), patch.object(
                client_module, "read_gpu_stats", return_value={"utilization": 0}
            ), patch.object(
                client_module, "OUTPUTS_DIR", Path(tmpdir)
            ):
                metrics, _util, image_b64 = client_module._run_faceswap_task(
                    "job-face-guidance",
                    entry,
                    model_path,
                    1.0,
                    task,
                )

        self.assertEqual(metrics["status"], "success")
        self.assertAlmostEqual(float(metrics.get("guidance", 0.0)), 4.6, places=6)
        self.assertIsNotNone(image_b64)
        self.assertIsNotNone(fake_pipe.kwargs)
        self.assertAlmostEqual(float(fake_pipe.kwargs.get("guidance_scale", 0.0)), 4.6, places=6)


class AnimateDiffCapabilityTests(unittest.TestCase):
    def test_resolve_animatediff_base_model_prefers_fuzzy_task_hint(self) -> None:
        entry_a = SimpleNamespace(name="realisticVisionV60B1_v51HyperVAE", pipeline="sd15", task_type="IMAGE_GEN")
        entry_b = SimpleNamespace(name="lyriel_v16", pipeline="sd15", task_type="IMAGE_GEN")
        path_a = Path("/tmp/realisticVisionV60B1_v51HyperVAE.safetensors")
        path_b = Path("/tmp/lyriel_v16.safetensors")

        def _ensure(entry):
            if entry is entry_a:
                return path_a
            if entry is entry_b:
                return path_b
            raise FileNotFoundError("unexpected model")

        with patch.object(client_module.REGISTRY, "list_entries", return_value=[entry_b, entry_a]), patch.object(
            client_module, "ensure_model_path", side_effect=_ensure
        ):
            resolved_entry, resolved_path, source = client_module.resolve_animatediff_base_model("realisticVision")

        self.assertIs(resolved_entry, entry_a)
        self.assertEqual(resolved_path, path_a)
        self.assertEqual(source, "task.base_model")

    def test_discover_supports_adds_animatediff_when_runner_and_model_ready(self) -> None:
        with patch.object(client_module, "_has_image_generation_model", return_value=False), patch.object(
            client_module, "_has_sdxl_base_model", return_value=False
        ), patch.object(
            client_module, "_instantid_assets_ready", return_value=False
        ), patch.object(
            client_module, "resolve_animatediff_base_model", return_value=(SimpleNamespace(name="rv"), Path("/tmp/rv.safetensors"), "default")
        ), patch.object(
            client_module,
            "_has_runner",
            side_effect=lambda module_name, function_name: function_name == "run_animatediff",
        ):
            supports = client_module.discover_supports({"pipelines": [], "models": []})

        self.assertIn("animatediff", supports)

    def test_discover_capabilities_includes_animatediff_model_when_runtime_ready(self) -> None:
        entry = SimpleNamespace(name="juggernautXL_ragnarokBy", pipeline="sdxl", task_type="IMAGE_GEN")
        with patch.object(client_module.REGISTRY, "list_entries", return_value=[entry]), patch.object(
            client_module, "ensure_model_path", return_value=Path("/tmp/juggernaut.safetensors")
        ), patch.object(
            client_module, "_has_runner", return_value=True
        ), patch.object(
            client_module, "resolve_animatediff_base_model", return_value=(SimpleNamespace(name="rv"), Path("/tmp/rv.safetensors"), "default")
        ), patch.object(
            client_module, "_manifest_animatediff_model_names", return_value=["animatediff"]
        ):
            caps = client_module.discover_capabilities()

        self.assertIn("animatediff", [p.lower() for p in caps["pipelines"]])
        self.assertIn("animatediff", [m.lower() for m in caps["models"]])

    def test_execute_task_animatediff_uses_resolved_base_model_path(self) -> None:
        task = {
            "task_id": "job-ad-test",
            "type": "ANIMATEDIFF",
            "model_name": "animatediff",
            "reward_weight": 1.0,
            "prompt": "test",
            "seed": 123,
            "wallet": "0xabc",
        }
        ad_entry = SimpleNamespace(name="animatediff", pipeline="animatediff", task_type="ANIMATEDIFF")
        resolved_entry = SimpleNamespace(name="realisticVisionV60B1_v51HyperVAE")
        resolved_path = Path("/tmp/rv.safetensors")

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"reward": 0.1}

        with patch.object(client_module, "ensure_model_entry", return_value=ad_entry), patch.object(
            client_module, "ensure_model_path", side_effect=RuntimeError("should not be called for animatediff")
        ), patch.object(
            client_module, "resolve_animatediff_base_model", return_value=(resolved_entry, resolved_path, "task.base_model")
        ), patch.object(
            client_module, "_run_animatediff_task", return_value=({"status": "success"}, 70, "dGVzdA==")
        ) as run_ad_mock, patch.object(
            client_module.SESSION, "post", return_value=fake_response
        ):
            client_module.execute_task(task)

        args, kwargs = run_ad_mock.call_args
        self.assertEqual(args[2], resolved_path)
        self.assertEqual(kwargs["resolved_base_model"], resolved_entry.name)
        self.assertEqual(kwargs["resolution_source"], "task.base_model")


if __name__ == "__main__":
    unittest.main()

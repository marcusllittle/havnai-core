"""Wan 2.2 I2V registry, config overrides, and the diffusers load plan.

Torch and diffusers are fakes. Nothing here downloads weights or needs a GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines.wan22_i2v import config as wan22_config  # noqa: E402
from engines.wan22_i2v import generator as wan22_generator  # noqa: E402
from engines.wan22_i2v import runner as wan22_runner  # noqa: E402


REGISTRY_PATH = ROOT / "server" / "manifests" / "registry.json"
APP_PATH = ROOT / "server" / "app.py"
ENV_KEYS = (
    "HAI_WAN22_I2V",
    "HAI_WAN22_ROOT",
    "HAI_WAN22_CONFIG",
    "HAI_WAN22_STEPS",
    "HAI_WAN22_GUIDANCE",
    "HAI_WAN22_TEXT_ENCODER",
    "HAI_WAN22_ALLOW_720P",
    "HAI_WAN22_OFFLOAD",
    "HAI_WAN22_NO_OFFLOAD",
    "HAI_WAN22_BACKEND",
)


def _silence_env() -> dict:
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    wan22_config.reset_config_cache()
    wan22_generator.unload_pipelines()
    return previous


def _restore_env(previous: dict) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    wan22_config.reset_config_cache()
    wan22_generator.unload_pipelines()


class _Recording:
    def __init__(self) -> None:
        self.calls = []


class _Expert:
    def __init__(self, name: str, recording: _Recording) -> None:
        self.name = name
        self.recording = recording

    def load_lora_adapter(self, path, adapter_name=None, **kwargs) -> None:
        self.recording.calls.append(("lora", self.name, path, adapter_name, kwargs))

    def set_adapters(self, names, weights=None, **kwargs) -> None:
        self.recording.calls.append(("adapters", self.name, list(names), weights))

    def fuse_lora(self, *args, **kwargs) -> None:
        self.recording.calls.append(("fuse", self.name))

    def enable_tiling(self) -> None:
        self.recording.calls.append(("tiling", self.name))


class _Scheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.sigmas = None

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, **kwargs) -> None:
        self.sigmas = list(sigmas) if sigmas is not None else None
        self.steps = num_inference_steps


class _Pipe:
    def __init__(self, kwargs: dict, recording: _Recording) -> None:
        self.transformer = kwargs["transformer"]
        self.transformer_2 = kwargs["transformer_2"]
        self.vae = kwargs["vae"]
        self.text_encoder = kwargs["text_encoder"]
        self.scheduler = _Scheduler()
        self.config = SimpleNamespace()
        self.recording = recording
        self.call = None

    def register_to_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def enable_model_cpu_offload(self) -> None:
        self.recording.calls.append(("offload", "model"))

    def enable_sequential_cpu_offload(self) -> None:
        self.recording.calls.append(("offload", "sequential"))

    def enable_vae_slicing(self) -> None:
        self.recording.calls.append(("vae_slicing",))

    def set_progress_bar_config(self, disable: bool = False) -> None:
        self.recording.calls.append(("progress", disable))

    def to(self, device) -> None:
        self.recording.calls.append(("to", device))

    def __call__(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(frames=["frame-batch"])


class _Torch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    class Generator:
        def __init__(self, device=None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> "_Torch.Generator":
            self.seed = seed
            return self


def _diffusers(recording: _Recording, *, encoder_fails: bool) -> SimpleNamespace:
    class GGUFQuantizationConfig:
        def __init__(self, compute_dtype=None) -> None:
            self.compute_dtype = compute_dtype

    class WanTransformer3DModel:
        @staticmethod
        def from_single_file(path, **kwargs):
            recording.calls.append(("expert", path, dict(kwargs)))
            return _Expert(str(path), recording)

    class AutoencoderKLWan:
        @staticmethod
        def from_single_file(path, **kwargs):
            recording.calls.append(("vae", path, dict(kwargs)))
            return _Expert("vae", recording)

        @staticmethod
        def from_pretrained(repo, **kwargs):
            recording.calls.append(("vae_repo", repo, dict(kwargs)))
            return _Expert("vae", recording)

    class UMT5EncoderModel:
        @staticmethod
        def from_single_file(path, **kwargs):
            recording.calls.append(("encoder_file", path, dict(kwargs)))
            if encoder_fails:
                raise RuntimeError("comfy fp8 umt5 is not a diffusers checkpoint")
            return _Expert("encoder", recording)

        @staticmethod
        def from_pretrained(repo, **kwargs):
            recording.calls.append(("encoder_repo", repo, dict(kwargs)))
            return _Expert("encoder", recording)

    class T5TokenizerFast:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            recording.calls.append(("tokenizer", repo, dict(kwargs)))
            return "tokenizer"

    class WanImageToVideoPipeline:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            recording.calls.append(("pipe", repo, sorted(kwargs)))
            return _Pipe(kwargs, recording)

    return SimpleNamespace(
        GGUFQuantizationConfig=GGUFQuantizationConfig,
        WanTransformer3DModel=WanTransformer3DModel,
        AutoencoderKLWan=AutoencoderKLWan,
        UMT5EncoderModel=UMT5EncoderModel,
        T5TokenizerFast=T5TokenizerFast,
        WanImageToVideoPipeline=WanImageToVideoPipeline,
    )


def _write_weights(root: Path) -> None:
    config = wan22_config.load_config()
    for key in ("transformer_high", "transformer_low", "text_encoder", "vae", "lora_high", "lora_low"):
        path = root / config.files[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")


class RegistryTests(unittest.TestCase):
    def test_registry_entry_is_opt_in_and_does_not_move_defaults(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        entry = next(item for item in payload["models"] if item["name"] == "wan22-i2v-a14b")
        self.assertEqual(entry["pipeline"], "wan22-i2v")
        self.assertEqual(entry["task_type"], "LTX_VIDEO_GEN")
        self.assertEqual(entry["capabilities"], ["image_to_video"])
        self.assertEqual(entry["video_defaults"]["steps"], 4)
        self.assertEqual(entry["video_defaults"]["guidance"], 1.0)
        self.assertEqual(entry["video_defaults"]["width"], 480)
        self.assertEqual(entry["video_defaults"]["height"], 832)
        self.assertEqual(entry["video_defaults"]["frames"], 129)
        self.assertEqual(entry["video_defaults"]["flow_shift"], 5.0)
        self.assertEqual(entry["timeout_seconds"], 3600)
        self.assertEqual(entry["credit_cost"], 12.0)
        self.assertEqual(entry["video_defaults"]["fps"], 16)
        self.assertEqual(entry["reward_weight"], 1)
        self.assertEqual(entry["license_status"], "apache-2.0")
        juggernaut = next(item for item in payload["models"] if item["name"] == "juggernautXL_ragnarokBy")
        ltx = next(item for item in payload["models"] if item["name"] == "ltx23_wangp_distilled")
        zimage = next(item for item in payload["models"] if item["name"] == "zimage-turbo")
        self.assertEqual(juggernaut["reward_weight"], 20)
        self.assertEqual(ltx["reward_weight"], 35)
        self.assertEqual(zimage["pipeline"], "zimage-turbo")
        self.assertGreater(juggernaut["reward_weight"], entry["reward_weight"])
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('"ltx23_wangp_distilled" if job_type == "image_to_video"', app_source)
        self.assertIn('else "juggernautXL_ragnarokBy"', app_source)
        config = wan22_config.load_config()
        self.assertEqual(
            sorted(item["id"] for item in entry["motion_presets"]),
            config.preset_names(),
        )

    def test_sources_do_not_import_a_wrapper(self) -> None:
        for path in (ROOT / "engines" / "wan22_i2v").glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("import wan2gp", text)
            self.assertNotIn("from wan2gp", text)


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = _silence_env()

    def tearDown(self) -> None:
        _restore_env(self._env)

    def test_flag_defaults_off_and_blocks_admission(self) -> None:
        self.assertFalse(wan22_config.enabled())
        blocked = wan22_config.admission(False, [])
        self.assertFalse(blocked["advertise"])
        self.assertIn("HAI_WAN22_I2V", blocked["error"])
        os.environ["HAI_WAN22_I2V"] = "1"
        self.assertTrue(wan22_config.enabled())
        self.assertTrue(wan22_config.admission(True, [])["advertise"])
        self.assertFalse(wan22_config.admission(True, ["vae/wan_2.1_vae.safetensors"])["advertise"])

    def test_root_defaults_to_the_wsl_path_and_env_overrides_it(self) -> None:
        config = wan22_config.load_config()
        self.assertEqual(str(config.weights_root()), wan22_config.DEFAULT_ROOT)
        os.environ["HAI_WAN22_ROOT"] = "/tmp/wan22-root"
        self.assertEqual(str(config.weights_root()), "/tmp/wan22-root")

    def test_recommended_split_and_job_override(self) -> None:
        ratio = wan22_config.boundary_ratio_for_split([1000, 750, 500, 250], 2)
        self.assertAlmostEqual(ratio, 0.625)
        settings = wan22_config.resolve_settings({"prompt": "a calm portrait", "seed": 1})
        self.assertEqual(settings["steps"], 4)
        self.assertEqual(settings["guidance"], 1.0)
        self.assertEqual(settings["guidance_2"], 1.0)
        self.assertEqual((settings["width"], settings["height"]), (480, 832))
        self.assertEqual(settings["frames"], 129)
        self.assertEqual(settings["fps"], 16)
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertEqual(settings["scheduler"], "euler")
        self.assertEqual(settings["backend"], "comfy")
        self.assertAlmostEqual(settings["boundary_ratio"], 0.625)
        self.assertEqual(settings["high_noise_steps"], 2)
        self.assertEqual(settings["lora_scale_high"], 1.0)
        self.assertEqual(settings["lora_scale_low"], 1.0)
        self.assertEqual(settings["strength"], 1.0)
        self.assertEqual(settings["offload"], "model")
        os.environ["HAI_WAN22_STEPS"] = "6"
        from_env = wan22_config.resolve_settings({"prompt": "a calm portrait"})
        self.assertEqual(from_env["steps"], 6)
        self.assertEqual(from_env["sources"]["steps"], "env")
        from_job = wan22_config.resolve_settings({"prompt": "a calm portrait", "steps": 3, "guidance": 1.5})
        self.assertEqual(from_job["steps"], 3)
        self.assertEqual(from_job["guidance"], 1.5)
        self.assertEqual(from_job["sources"]["steps"], "job")

    def test_one_high_noise_step_moves_the_boundary(self) -> None:
        settings = wan22_config.resolve_settings({"prompt": "pose", "high_noise_steps": 1})
        self.assertAlmostEqual(settings["boundary_ratio"], 0.875)

    def test_preset_appends_motion_and_supplies_the_negative(self) -> None:
        settings = wan22_config.resolve_settings({
            "prompt": "portrait of a person by a window",
            "motion_preset": "blink_smile",
        })
        self.assertIn("slow blink", settings["prompt"])
        self.assertIn("portrait of a person", settings["prompt"])
        self.assertIn("jitter", settings["negative_prompt"])
        self.assertEqual(settings["motion_preset"], "blink_smile")
        custom = wan22_config.resolve_settings({
            "prompt": "portrait",
            "motion_preset": "hair_breeze",
            "negative_prompt": "blurry hands",
        })
        self.assertEqual(custom["negative_prompt"], "blurry hands")
        self.assertIn("breeze", custom["prompt"])
        with self.assertRaises(ValueError):
            wan22_config.resolve_settings({"prompt": "portrait", "motion_preset": "not-a-preset"})

    def test_strength_stays_full_and_720p_is_clamped_until_allowed(self) -> None:
        partial = wan22_config.resolve_settings({"prompt": "portrait", "strength": 0.4})
        self.assertEqual(partial["strength"], 1.0)
        self.assertEqual(partial["requested_strength"], 0.4)
        landscape = wan22_config.resolve_settings({"prompt": "portrait", "width": 832, "height": 480})
        self.assertEqual((landscape["width"], landscape["height"]), (832, 480))
        clamped = wan22_config.resolve_settings({"prompt": "portrait", "width": 720, "height": 1280, "frames": 81})
        self.assertEqual((clamped["width"], clamped["height"]), (480, 832))
        wider = wan22_config.resolve_settings({
            "prompt": "portrait",
            "source_width": 480,
            "source_height": 704,
        })
        self.assertEqual((wider["width"], wider["height"]), (480, 704))
        kept = wan22_config.resolve_settings({"prompt": "portrait", "frames": 81})
        self.assertEqual(kept["frames"], 81)
        os.environ["HAI_WAN22_ALLOW_720P"] = "1"
        allowed = wan22_config.resolve_settings({"prompt": "portrait", "width": 720, "height": 1280, "frames": 81})
        self.assertEqual((allowed["width"], allowed["height"]), (720, 1280))
        self.assertEqual(allowed["frames"], 49)

    def test_odd_frame_count_snaps_to_4n_plus_1(self) -> None:
        settings = wan22_config.resolve_settings({"prompt": "portrait", "frames": 80})
        self.assertEqual(settings["frames"], 81)


class LoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = _silence_env()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["HAI_WAN22_ROOT"] = str(self.root)
        _write_weights(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _restore_env(self._env)

    def _load(self, *, encoder_fails: bool) -> tuple:
        recording = _Recording()
        diffusers = _diffusers(recording, encoder_fails=encoder_fails)
        config = wan22_config.load_config()
        settings = wan22_config.resolve_settings({"prompt": "portrait", "seed": 7})
        pipe = wan22_generator._load_pipeline(
            config,
            settings,
            torch_module=_Torch(),
            diffusers_module=diffusers,
        )
        return recording, pipe, settings

    def test_loads_both_experts_and_does_not_fuse_loras(self) -> None:
        recording, pipe, settings = self._load(encoder_fails=True)
        expert_calls = [call for call in recording.calls if call[0] == "expert"]
        self.assertEqual(len(expert_calls), 2)
        subfolders = {call[2]["subfolder"] for call in expert_calls}
        self.assertEqual(subfolders, {"transformer", "transformer_2"})
        for call in expert_calls:
            self.assertIn("quantization_config", call[2])
            self.assertEqual(call[2]["config"], "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            self.assertTrue(str(call[1]).endswith(".gguf"))
        lora_calls = [call for call in recording.calls if call[0] == "lora"]
        self.assertEqual(len(lora_calls), 2)
        self.assertIn("high_noise", lora_calls[0][2])
        self.assertIn("low_noise", lora_calls[1][2])
        self.assertNotIn("fuse", [call[0] for call in recording.calls])
        self.assertEqual(pipe.config.boundary_ratio, settings["boundary_ratio"])
        self.assertIn(("offload", "model"), recording.calls)
        self.assertIn(("vae_slicing",), recording.calls)
        self.assertIn(("tiling", "vae"), recording.calls)
        self.assertIn(("encoder_repo", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"), [call[:2] for call in recording.calls])
        pipe.scheduler.set_timesteps(4, device="cpu")
        self.assertEqual(pipe.scheduler.sigmas, [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_fp8_encoder_is_kept_when_it_loads(self) -> None:
        recording, pipe, _settings = self._load(encoder_fails=False)
        kinds = [call[0] for call in recording.calls]
        self.assertIn("encoder_file", kinds)
        self.assertNotIn("encoder_repo", kinds)
        self.assertEqual(pipe._wan22_text_encoder_source, "local-fp8")

    def test_generate_maps_the_resolved_job(self) -> None:
        recording, _pipe, _settings = self._load(encoder_fails=False)
        config = wan22_config.load_config()
        settings = wan22_config.resolve_settings({
            "prompt": "portrait by a window",
            "motion_preset": "camera_push",
            "seed": 3,
        })
        frames = wan22_generator.generate_frames(
            config,
            prompt=settings["prompt"],
            negative_prompt=settings["negative_prompt"],
            image="source-image",
            settings=settings,
            seed=3,
            torch_module=_Torch(),
            diffusers_module=_diffusers(recording, encoder_fails=False),
        )
        self.assertEqual(frames, "frame-batch")
        pipe = wan22_generator._PIPES[next(iter(wan22_generator._PIPES))]
        self.assertEqual(pipe.call["image"], "source-image")
        self.assertEqual(pipe.call["num_frames"], 129)
        self.assertEqual(pipe.call["num_inference_steps"], 4)
        self.assertEqual(pipe.call["guidance_scale"], 1.0)
        self.assertEqual(pipe.call["guidance_scale_2"], 1.0)
        self.assertIn("camera push-in", pipe.call["prompt"])
        self.assertIn("jitter", pipe.call["negative_prompt"])


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = _silence_env()
        os.environ["HAI_WAN22_BACKEND"] = "diffusers"

    def tearDown(self) -> None:
        _restore_env(self._env)

    def test_runner_encodes_mp4_and_records_the_preset(self) -> None:
        captured = {}

        def fake_generate(config, **kwargs):
            captured.update(kwargs)
            return [[[0, 0, 0]]]

        def fake_save(frames, path: Path, fps: int) -> None:
            path.write_bytes(b"mp4")

        original_generate = wan22_runner.generate_frames
        original_save = wan22_runner._save_video
        original_load = wan22_runner._load_init_image
        wan22_runner.generate_frames = fake_generate
        wan22_runner._save_video = fake_save
        wan22_runner._load_init_image = lambda source: "image"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                metrics, util, path = wan22_runner.run_wan22_i2v(
                    {
                        "task_id": "job-wan",
                        "prompt": "portrait",
                        "seed": 4,
                        "motion_preset": "fabric_movement",
                        "init_image": "source.png",
                        "strength": 0.2,
                    },
                    outputs_dir=Path(tmp),
                    read_gpu_stats=lambda: {"utilization": 3},
                )
        finally:
            wan22_runner.generate_frames = original_generate
            wan22_runner._save_video = original_save
            wan22_runner._load_init_image = original_load
        self.assertEqual(metrics["status"], "success")
        self.assertEqual(metrics["model_family"], "wan22_i2v")
        self.assertEqual(metrics["steps"], 4)
        self.assertEqual(metrics["motion_preset"], "fabric_movement")
        self.assertEqual(metrics["strength"], 1.0)
        self.assertEqual(metrics["requested_strength"], 0.2)
        self.assertEqual(util, 3)
        self.assertIsNotNone(path)
        self.assertTrue(str(path).endswith("video_job-wan.mp4"))
        self.assertIn("fabric", captured["prompt"])

    def test_runner_rejects_a_missing_image_and_an_unknown_preset(self) -> None:
        missing, _util, path = wan22_runner.run_wan22_i2v(
            {"prompt": "portrait", "seed": 1},
        )
        self.assertEqual(missing["status"], "failed")
        self.assertIn("source image", missing["error"])
        self.assertIsNone(path)
        unknown, _util, _path = wan22_runner.run_wan22_i2v(
            {"prompt": "portrait", "seed": 1, "init_image": object(), "motion_preset": "nope"},
        )
        self.assertEqual(unknown["status"], "failed")
        self.assertIn("unknown motion preset", unknown["error"])


if __name__ == "__main__":
    unittest.main()

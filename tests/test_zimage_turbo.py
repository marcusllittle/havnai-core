"""Z-Image-Turbo registry, routing, and the resident Q5 runtime.

Torch and diffusers are fakes. Nothing here downloads weights.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT / "server"))

import zimage_turbo  # noqa: E402


REGISTRY_PATH = ROOT / "server" / "manifests" / "registry.json"


class FakeTensor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def to(self, device: object) -> "FakeTensor":
        self.device = str(device)
        return self

    def detach(self) -> "FakeTensor":
        return self


class FakeModule:
    def __init__(self, name: str, events: list) -> None:
        self.name = name
        self.device = "cpu"
        self.events = events

    def to(self, device: object) -> "FakeModule":
        self.device = str(device)
        self.events.append((self.name, "to", self.device))
        return self


class FakeGenerator:
    def __init__(self, device: object) -> None:
        self.device = device

    def manual_seed(self, seed: int) -> "FakeGenerator":
        self.seed = seed
        return self


class FakePipe:
    def __init__(self, events: list) -> None:
        self.events = events
        self.text_encoder = FakeModule("text_encoder", events)
        self.transformer = FakeModule("transformer", events)
        self.vae = FakeModule("vae", events)
        self.hooks_removed = False

    def set_progress_bar_config(self, disable: bool = False) -> None:
        self.events.append(("progress", disable))

    def remove_all_hooks(self) -> None:
        self.hooks_removed = True
        self.events.append(("remove_all_hooks",))

    def encode_prompt(self, prompt: str, device: object = None, do_classifier_free_guidance: bool = True):
        self.events.append(
            (
                "encode",
                prompt,
                str(device),
                do_classifier_free_guidance,
                self.text_encoder.device,
            )
        )
        return [FakeTensor(str(device))], []

    def __call__(self, **kwargs: object):
        self.events.append(
            (
                "denoise",
                kwargs.get("guidance_scale"),
                kwargs.get("width"),
                kwargs.get("height"),
                kwargs.get("num_inference_steps"),
                "negative_prompt" in kwargs,
                kwargs.get("prompt"),
                self.text_encoder.device,
                self.transformer.device,
            )
        )
        return type("Result", (), {"images": ["image"]})()


class FakeCuda:
    def __init__(self, events: list) -> None:
        self.events = events

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.events.append(("empty_cache",))


class FakeTorch:
    bfloat16 = "bf16"

    def __init__(self, events: list) -> None:
        self.cuda = FakeCuda(events)
        self.events = events

    def device(self, kind: str, index: int = 0) -> str:
        return f"{kind}:{index}"

    def Generator(self, device: object) -> FakeGenerator:
        return FakeGenerator(device)


class FakeQuant:
    def __init__(self, compute_dtype: object) -> None:
        self.compute_dtype = compute_dtype


class FakeDiffusers:
    calls = []

    class GGUFQuantizationConfig(FakeQuant):
        pass

    class ZImageTransformer2DModel:
        @staticmethod
        def from_single_file(path: str, **kwargs: object) -> str:
            FakeDiffusers.calls.append(("from_single_file", path, kwargs))
            return "transformer"

    class ZImagePipeline:
        @staticmethod
        def from_pretrained(snapshot: str, **kwargs: object) -> FakePipe:
            FakeDiffusers.calls.append(("from_pretrained", snapshot, kwargs))
            return FakeDiffusers.calls[-1][2]["pipe"]  # type: ignore[index]


class FakeHub:
    def snapshot_download(self, **kwargs: object) -> str:
        self.snapshot_kwargs = kwargs
        return "/snap/z-image-turbo"

    def hf_hub_download(self, **kwargs: object) -> str:
        self.gguf_kwargs = kwargs
        return "/weights/z-image-turbo-Q5_K_M.gguf"


class RegistryAndCapTests(unittest.TestCase):
    def test_registry_entry_is_q5_turbo(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        entry = next(item for item in payload["models"] if item["name"] == "zimage-turbo")
        self.assertEqual(entry["pipeline"], "zimage-turbo")
        self.assertEqual(entry["steps"], 9)
        self.assertEqual(entry["guidance"], 0.0)
        self.assertEqual(entry["width"], 1024)
        self.assertEqual(entry["height"], 1024)
        self.assertEqual(entry["reward_weight"], 1)
        self.assertEqual(entry["source"]["repo_id"], "unsloth/Z-Image-Turbo-GGUF")
        self.assertEqual(entry["source"]["filename"], "z-image-turbo-Q5_K_M.gguf")
        juggernaut = next(item for item in payload["models"] if item["name"] == "juggernautXL_ragnarokBy")
        self.assertGreater(juggernaut["reward_weight"], entry["reward_weight"])

    def test_default_cap_is_1024_and_ignores_sdxl_env(self) -> None:
        previous = {key: os.environ.get(key) for key in (
            "HAI_SDXL_CAP_WIDTH",
            "HAI_SDXL_CAP_HEIGHT",
            "HAI_SDXL_MAX_PIXELS",
            "HAI_SDXL_MAX_STEPS",
            "HAI_ZIMAGE_MAX_WIDTH",
            "HAI_ZIMAGE_MAX_HEIGHT",
            "HAI_ZIMAGE_MAX_PIXELS",
        )}
        try:
            os.environ["HAI_SDXL_CAP_WIDTH"] = "768"
            os.environ["HAI_SDXL_CAP_HEIGHT"] = "1024"
            os.environ["HAI_SDXL_MAX_PIXELS"] = "589824"
            os.environ["HAI_SDXL_MAX_STEPS"] = "28"
            for key in ("HAI_ZIMAGE_MAX_WIDTH", "HAI_ZIMAGE_MAX_HEIGHT", "HAI_ZIMAGE_MAX_PIXELS"):
                os.environ.pop(key, None)
            self.assertEqual(zimage_turbo.cap_resolution(1024, 1024), (1024, 1024))
            self.assertEqual(zimage_turbo.cap_resolution(1600, 1600), (1024, 1024))
            os.environ["HAI_ZIMAGE_MAX_WIDTH"] = "768"
            os.environ["HAI_ZIMAGE_MAX_HEIGHT"] = "768"
            os.environ["HAI_ZIMAGE_MAX_PIXELS"] = str(768 * 768)
            self.assertEqual(zimage_turbo.cap_resolution(1024, 1024), (768, 768))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_switch_evicts_the_other_family(self) -> None:
        self.assertFalse(zimage_turbo.should_release_on_switch("zimage"))
        self.assertTrue(zimage_turbo.should_release_on_switch("base"))
        self.assertTrue(zimage_turbo.should_release_on_switch(""))

    def test_flag_defaults_off(self) -> None:
        previous = os.environ.pop("HAI_ZIMAGE_TURBO", None)
        try:
            self.assertFalse(zimage_turbo.enabled())
            os.environ["HAI_ZIMAGE_TURBO"] = "1"
            self.assertTrue(zimage_turbo.enabled())
        finally:
            if previous is None:
                os.environ.pop("HAI_ZIMAGE_TURBO", None)
            else:
                os.environ["HAI_ZIMAGE_TURBO"] = previous


class RuntimeTests(unittest.TestCase):
    def test_load_uses_snapshot_config_and_keeps_encoder_on_cpu(self) -> None:
        events: list = []
        pipe = FakePipe(events)
        FakeDiffusers.calls = []

        def from_pretrained(snapshot: str, **kwargs: object) -> FakePipe:
            FakeDiffusers.calls.append(("from_pretrained", snapshot, dict(kwargs)))
            return pipe

        FakeDiffusers.ZImagePipeline.from_pretrained = staticmethod(from_pretrained)
        hub = FakeHub()
        runtime = zimage_turbo.Runtime()
        loaded = runtime.load(
            torch_module=FakeTorch(events),
            diffusers_module=FakeDiffusers,
            hub_module=hub,
            gpu_index=0,
        )
        self.assertIs(loaded, pipe)
        single = FakeDiffusers.calls[0]
        self.assertEqual(single[0], "from_single_file")
        self.assertEqual(single[1], "/weights/z-image-turbo-Q5_K_M.gguf")
        self.assertEqual(single[2]["config"], "/snap/z-image-turbo")
        self.assertEqual(single[2]["subfolder"], "transformer")
        self.assertTrue(single[2]["local_files_only"])
        self.assertEqual(single[2]["quantization_config"].compute_dtype, "bf16")
        self.assertEqual(hub.gguf_kwargs["repo_id"], "unsloth/Z-Image-Turbo-GGUF")
        self.assertEqual(hub.snapshot_kwargs["repo_id"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(pipe.transformer.device, "cuda:0")
        self.assertEqual(pipe.vae.device, "cuda:0")
        self.assertEqual(pipe.text_encoder.device, "cpu")

    def test_generate_encodes_then_frees_encoder_before_denoise(self) -> None:
        events: list = []
        pipe = FakePipe(events)
        runtime = zimage_turbo.Runtime()
        runtime.pipe = pipe
        torch = FakeTorch(events)
        image = runtime.generate(
            "a hangar sign that reads ASTRA",
            seed=7,
            width=1280,
            height=1280,
            steps=30,
            torch_module=torch,
        )
        self.assertEqual(image, "image")
        encode = next(item for item in events if item[0] == "encode")
        self.assertEqual(encode[3], False)
        self.assertTrue(encode[4].startswith("cuda"))
        empty_at = events.index(("empty_cache",))
        denoise_at = next(index for index, item in enumerate(events) if item[0] == "denoise")
        self.assertLess(empty_at, denoise_at)
        denoise = events[denoise_at]
        self.assertEqual(denoise[1], 0.0)
        self.assertEqual(denoise[2], 1024)
        self.assertEqual(denoise[3], 1024)
        self.assertEqual(denoise[4], 9)
        self.assertFalse(denoise[5])
        self.assertIsNone(denoise[6])
        self.assertEqual(denoise[7], "cpu")
        self.assertTrue(str(denoise[8]).startswith("cuda"))

    def test_unload_drops_hooks_and_moves_modules_to_cpu(self) -> None:
        events: list = []
        pipe = FakePipe(events)
        pipe.transformer.device = "cuda:0"
        runtime = zimage_turbo.Runtime()
        runtime.pipe = pipe
        runtime.unload(torch_module=FakeTorch(events))
        self.assertIsNone(runtime.pipe)
        self.assertTrue(pipe.hooks_removed)
        self.assertEqual(pipe.transformer.device, "cpu")
        self.assertEqual(pipe.text_encoder.device, "cpu")
        self.assertEqual(pipe.vae.device, "cpu")
        self.assertIn(("empty_cache",), events)


class RoutingTests(unittest.TestCase):
    def test_turbo_guidance_stays_zero_and_sdxl_does_not(self) -> None:
        import image_routing

        self.assertEqual(image_routing.clamp_guidance("zimage-turbo", 0.0), 0.0)
        self.assertEqual(image_routing.clamp_guidance("sdxl", 0.0), 1.0)
        self.assertEqual(image_routing.negative_prompt_override("zimage-turbo"), "")
        self.assertIsNone(image_routing.negative_prompt_override("sdxl"))
        self.assertEqual(image_routing.positive_suffix_override("zimage-turbo"), "")
        self.assertIsNone(image_routing.positive_suffix_override("sdxl"))

    def test_node_must_advertise_the_pipeline(self) -> None:
        import image_routing

        advertised = {
            "models": ["zimage-turbo"],
            "pipelines": ["zimage-turbo"],
        }
        self.assertTrue(image_routing.node_advertises_model(advertised, "zimage-turbo", "zimage-turbo"))
        sdxl_only = {
            "models": ["juggernautXL_ragnarokBy"],
            "pipelines": ["sdxl"],
        }
        self.assertFalse(image_routing.node_advertises_model(sdxl_only, "zimage-turbo", "zimage-turbo"))
        self.assertTrue(
            image_routing.node_advertises_model(sdxl_only, "juggernautXL_ragnarokBy", "sdxl")
        )


if __name__ == "__main__":
    unittest.main()

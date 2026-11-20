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
    def __init__(self, device: str = "cpu", dtype: object = None) -> None:
        self.device = device
        self.dtype = dtype

    def to(self, device: object = None, dtype: object = None, **kwargs: object) -> "FakeTensor":
        if device is None:
            device = kwargs.get("device")
        if dtype is None:
            dtype = kwargs.get("dtype")
        if device is not None:
            self.device = str(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def detach(self) -> "FakeTensor":
        return self


class FakeIds:
    def __init__(self, length: int, device: str = "cpu") -> None:
        self.shape = (1, length)
        self.device = device

    def to(self, device: object = None, dtype: object = None, **kwargs: object) -> "FakeIds":
        if device is None:
            device = kwargs.get("device")
        moved = FakeIds(self.shape[1], str(device) if device is not None else self.device)
        return moved

    def bool(self) -> "FakeIds":
        return self

    def __getitem__(self, index: object) -> "FakeIds":
        return self


class FakeTokenizer:
    def __init__(self, events: list) -> None:
        self.events = events

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.events.append(("template", tokenize, add_generation_prompt, enable_thinking, messages[0]["role"]))
        return "chat:" + messages[0]["content"]

    def __call__(self, text, padding=None, truncation=None, max_length=None, return_tensors=None):
        self.events.append(("tokenize", padding, truncation, max_length, return_tensors, len(text)))
        length = min(len(text), max_length or len(text))
        return type("Batch", (), {"input_ids": FakeIds(length), "attention_mask": FakeIds(length)})()


class FakeModule:
    def __init__(self, name: str, events: list) -> None:
        self.name = name
        self.device = "cpu"
        self.dtype = None
        self.events = events

    def to(self, device: object = None, dtype: object = None, **kwargs: object) -> "FakeModule":
        if device is None:
            device = kwargs.get("device")
        if dtype is None:
            dtype = kwargs.get("dtype")
        if device is not None:
            self.device = str(device)
        if dtype is not None:
            self.dtype = dtype
        self.events.append((self.name, "to", self.device, dtype))
        return self

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        self.events.append(("encoder_forward", input_ids.shape, str(input_ids.device), self.device))

        class _Row:
            def __getitem__(self, mask: object) -> FakeTensor:
                return FakeTensor(device="cpu", dtype="fp32")

        class _Batch:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: object) -> _Row:
                return _Row()

        hidden = _Batch()
        return type("Out", (), {"hidden_states": (hidden, hidden)})()


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
        self.tokenizer = FakeTokenizer(events)
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
                self.transformer.device,
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
                getattr((kwargs.get("prompt_embeds") or [None])[0], "dtype", None),
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
    float32 = "fp32"

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

    def _resident_pipe(self, events: list) -> FakePipe:
        pipe = FakePipe(events)
        pipe.transformer.device = "cuda:0"
        pipe.vae.device = "cuda:0"
        pipe.text_encoder.device = "cpu"
        return pipe

    def test_cpu_mode_encodes_on_cpu_while_transformer_stays(self) -> None:
        previous = os.environ.pop("HAI_ZIMAGE_ENCODER_MODE", None)
        try:
            events: list = []
            pipe = self._resident_pipe(events)
            runtime = zimage_turbo.Runtime()
            runtime.pipe = pipe
            image = runtime.generate(
                "a hangar sign that reads ASTRA",
                seed=7,
                width=1280,
                height=1280,
                steps=30,
                torch_module=FakeTorch(events),
            )
            self.assertEqual(image, "image")
            tokenize = next(item for item in events if item[0] == "tokenize")
            self.assertFalse(tokenize[1])
            self.assertNotEqual(tokenize[1], "max_length")
            self.assertEqual(tokenize[3], 512)
            forward = next(item for item in events if item[0] == "encoder_forward")
            self.assertNotEqual(forward[1][1], 512)
            self.assertFalse(str(forward[2]).startswith("cuda"))
            self.assertEqual(pipe.text_encoder.dtype, "fp32")
            self.assertFalse(str(pipe.text_encoder.device).startswith("cuda"))
            self.assertTrue(str(pipe.transformer.device).startswith("cuda"))
            self.assertFalse(_encoder_overlapped_transformer(events))
            denoise = next(item for item in events if item[0] == "denoise")
            self.assertEqual(denoise[1], 0.0)
            self.assertEqual(denoise[2], 1024)
            self.assertEqual(denoise[4], 9)
            self.assertFalse(denoise[5])
            self.assertEqual(denoise[-1], "bf16")
            self.assertEqual(runtime.last_stats["encoder_mode"], "cpu")
        finally:
            if previous is None:
                os.environ.pop("HAI_ZIMAGE_ENCODER_MODE", None)
            else:
                os.environ["HAI_ZIMAGE_ENCODER_MODE"] = previous

    def test_swap_mode_moves_transformer_off_before_the_encoder(self) -> None:
        previous = os.environ.get("HAI_ZIMAGE_ENCODER_MODE")
        os.environ["HAI_ZIMAGE_ENCODER_MODE"] = "swap"
        zimage_turbo._SWAP_WARNED = False
        try:
            events: list = []
            pipe = self._resident_pipe(events)
            runtime = zimage_turbo.Runtime()
            runtime.pipe = pipe
            with self.assertLogs("zimage_turbo", level="WARNING") as caught:
                runtime.generate("second prompt", seed=3, torch_module=FakeTorch(events))
            self.assertIn("9 GB", caught.output[0])
            encode_at = next(index for index, item in enumerate(events) if item[0] == "encoder_forward")
            forward = events[encode_at]
            self.assertTrue(str(forward[2]).startswith("cuda"))
            self.assertTrue(str(forward[3]).startswith("cuda"))
            transformer_off = next(
                index
                for index, item in enumerate(events)
                if item[:3] == ("transformer", "to", "cpu")
            )
            encoder_on = next(
                index
                for index, item in enumerate(events)
                if item[0] == "text_encoder" and item[1] == "to" and str(item[2]).startswith("cuda")
            )
            encoder_off = next(
                index
                for index, item in enumerate(events)
                if item[:3] == ("text_encoder", "to", "cpu") and index > encoder_on
            )
            transformer_back = next(
                index
                for index, item in enumerate(events)
                if item[0] == "transformer" and item[1] == "to" and str(item[2]).startswith("cuda")
            )
            self.assertLess(transformer_off, encoder_on)
            self.assertLess(encoder_on, encode_at)
            self.assertLess(encode_at, encoder_off)
            self.assertLess(encoder_off, transformer_back)
            self.assertFalse(_encoder_overlapped_transformer(events))
            self.assertEqual(pipe.text_encoder.device, "cpu")
            self.assertTrue(pipe.transformer.device.startswith("cuda"))
            self.assertEqual(runtime.last_stats["encoder_mode"], "swap")
        finally:
            if previous is None:
                os.environ.pop("HAI_ZIMAGE_ENCODER_MODE", None)
            else:
                os.environ["HAI_ZIMAGE_ENCODER_MODE"] = previous

    def test_vram_guard_names_the_mode(self) -> None:
        events: list = []
        pipe = self._resident_pipe(events)

        total = 12 * 1024 * 1024 * 1024
        peak = int(total * 0.96)

        class HotCuda(FakeCuda):
            def max_memory_allocated(self) -> int:
                return peak

            def get_device_properties(self, index: int) -> object:
                return type("Props", (), {"total_memory": total})()

        class HotTorch(FakeTorch):
            def __init__(self) -> None:
                super().__init__(events)
                self.cuda = HotCuda(events)

        previous = os.environ.pop("HAI_ZIMAGE_ENCODER_MODE", None)
        try:
            runtime = zimage_turbo.Runtime()
            runtime.pipe = pipe
            with self.assertLogs("zimage_turbo", level="WARNING") as caught:
                runtime.generate("prompt", torch_module=HotTorch())
            self.assertTrue(runtime.last_stats["vram_warning"])
            self.assertEqual(runtime.last_stats["encoder_mode"], "cpu")
            self.assertIn("encoder mode cpu", caught.output[0])
            self.assertAlmostEqual(runtime.last_stats["peak_allocated_mib"], round(peak / (1024.0 * 1024.0), 1))
        finally:
            if previous is None:
                os.environ.pop("HAI_ZIMAGE_ENCODER_MODE", None)
            else:
                os.environ["HAI_ZIMAGE_ENCODER_MODE"] = previous

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


def _encoder_overlapped_transformer(events: list) -> bool:
    """Replay moves from the post-load state. True if both were on CUDA together."""
    encoder = "cpu"
    transformer = "cuda:0"
    overlapped = False
    for item in events:
        if item[0] == "text_encoder" and item[1] == "to":
            encoder = str(item[2])
        elif item[0] == "transformer" and item[1] == "to":
            transformer = str(item[2])
        elif item[0] == "encoder_forward":
            encoder = str(item[3])
        if encoder.startswith("cuda") and transformer.startswith("cuda"):
            overlapped = True
    return overlapped


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

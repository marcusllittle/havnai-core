"""The user-facing Wan 2.2 and LTX-2.3 recipes, without a GPU."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from engines.gpu_handoff import (  # noqa: E402
    CommandResult,
    GpuBusy,
    finish_music,
    prepare_for_music,
    release_for_video,
    restore_after_video,
)
import engines.gpu_handoff as gpu_handoff  # noqa: E402
from engines.wan22_i2v import config as wan22_config  # noqa: E402
from engines.wan22_i2v.comfy import build_workflow, run_graph  # noqa: E402
from engines.wangp.runner import ltx23_sample  # noqa: E402
import platform_v1  # noqa: E402


class _Response:
    def __init__(self, payload=None, status=200, content=b""):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class _Session:
    def __init__(self, history, video=b"mp4-bytes"):
        self.history = history
        self.video = video
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/upload/image"):
            return _Response({"name": "havnai_job.png"})
        if url.endswith("/prompt"):
            return _Response({"prompt_id": "prompt-1"})
        if url.endswith("/free"):
            return _Response({"ok": True})
        if url.endswith("/v1/unload_model"):
            return _Response({"ok": True})
        return _Response({}, status=404)

    def get(self, url, **kwargs):
        self.calls.append(("get", url))
        if "/history/" in url:
            return _Response(self.history)
        if url.endswith("/view"):
            return _Response(content=self.video)
        if url.endswith("/health"):
            return _Response(self.history)
        return _Response({}, status=404)


class ComfyWorkflowTests(unittest.TestCase):
    def test_graph_matches_the_measured_recipe(self) -> None:
        config = wan22_config.load_config()
        settings = wan22_config.resolve_settings({"prompt": "a calm portrait", "seed": 4})
        graph = build_workflow(settings, config=config, image_name="source.png", seed=4)
        self.assertEqual(graph["5"]["inputs"]["shift"], 5.0)
        self.assertEqual(graph["6"]["inputs"]["shift"], 5.0)
        self.assertEqual(graph["12"]["inputs"]["length"], 129)
        self.assertEqual((graph["12"]["inputs"]["width"], graph["12"]["inputs"]["height"]), (480, 832))
        high = graph["13"]["inputs"]
        low = graph["14"]["inputs"]
        self.assertEqual(high["sampler_name"], "euler")
        self.assertEqual(low["sampler_name"], "euler")
        self.assertEqual(high["cfg"], 1.0)
        self.assertEqual((high["start_at_step"], high["end_at_step"]), (0, 2))
        self.assertEqual((low["start_at_step"], low["end_at_step"]), (2, 4))
        self.assertEqual(high["add_noise"], "enable")
        self.assertEqual(low["add_noise"], "disable")
        self.assertIn("HighNoise", graph["1"]["inputs"]["unet_name"])
        self.assertIn("LowNoise", graph["2"]["inputs"]["unet_name"])
        self.assertIn("umt5", graph["7"]["inputs"]["clip_name"])
        self.assertIn("lightx2v", graph["3"]["inputs"]["lora_name"])

    def test_client_polls_then_frees_the_card(self) -> None:
        history = {
            "prompt-1": {
                "outputs": {
                    "17": {"videos": [{"filename": "havnai/wan22_00001.mp4", "subfolder": "", "type": "output"}]}
                }
            }
        }
        session = _Session(history)
        data = run_graph(
            {"11": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}},
            b"png",
            "old.png",
            base_url="http://comfy.test",
            session=session,
            timeout=5,
            sleeper=lambda _seconds: None,
            clock=lambda: 0.0,
        )
        self.assertEqual(data, b"mp4-bytes")
        posts = [call[1] for call in session.calls if call[0] == "post"]
        self.assertTrue(any(url.endswith("/free") for url in posts))
        self.assertEqual(session.calls[0][2]["files"]["image"][0], "old.png")


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        gpu_handoff._cooldown_until = 0.0
        gpu_handoff._gpu_lock = threading.Lock()
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("HAVNAI_ACESTEP_STOP_CMD", "HAVNAI_ACESTEP_START_CMD", "ACESTEP_API_URL")
        }
        for key in self._saved_env:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gpu_handoff._cooldown_until = 0.0
        lock = gpu_handoff._gpu_lock
        if lock.locked():
            try:
                lock.release()
            except RuntimeError:
                pass

    def test_zimage_and_ace_leave_before_the_job_and_zimage_returns(self) -> None:
        events = []
        session = _Session({"loaded_model": "acestep-v15-turbo", "models_initialized": True})
        # First health says loaded. After unload, health says empty.
        health = [
            _Response({"loaded_model": "acestep-v15-turbo", "models_initialized": True}),
            _Response({"loaded_model": "", "models_initialized": False}),
        ]

        def get(url, **kwargs):
            session.calls.append(("get", url))
            if url.endswith("/health"):
                return health.pop(0)
            return _Response({})

        session.get = get
        state = release_for_video(
            session=session,
            zimage_loaded=lambda: True,
            unload_zimage=lambda: events.append("unload-zimage"),
            load_zimage=lambda: events.append("load-zimage"),
            run_command=lambda command: events.append(command) or 0,
        )
        self.assertIn("unload-zimage", events)
        self.assertTrue(state.ace_unloaded)
        self.assertTrue(state.zimage_was_loaded)
        restore_after_video(state, load_zimage=lambda: events.append("load-zimage"))
        self.assertEqual(events.count("load-zimage"), 1)

    def test_busy_ace_blocks_the_video_and_restores_zimage(self) -> None:
        events = []
        session = _Session({})

        def get(url, **kwargs):
            return _Response({"loaded_model": "acestep-v15-turbo", "models_initialized": True})

        def post(url, **kwargs):
            return _Response({}, status=404)

        session.get = get
        session.post = post
        with self.assertRaises(GpuBusy):
            release_for_video(
                session=session,
                zimage_loaded=lambda: True,
                unload_zimage=lambda: events.append("unload-zimage"),
                load_zimage=lambda: events.append("load-zimage"),
                run_command=lambda command: 1,
            )
        self.assertEqual(events, ["unload-zimage", "load-zimage"])

    def test_video_stops_ace_with_systemctl_and_waits_for_vram(self) -> None:
        commands = []
        reads = {"n": 0}
        stopped = {"value": False}

        def run(command: str) -> CommandResult:
            commands.append(command)
            if command.startswith("nvidia-smi"):
                reads["n"] += 1
                # The first samples are still ACE-Step. Later samples are the idle card.
                used = 7800 if reads["n"] < 3 else 400
                return CommandResult(0, f"{used}\n")
            if command == "systemctl --user stop acestep-api":
                stopped["value"] = True
                return CommandResult(0, "")
            return CommandResult(0, "")

        def get(url, **kwargs):
            if url.endswith("/health"):
                if stopped["value"] and reads["n"] >= 3:
                    raise RuntimeError("connection refused")
                return _Response({"loaded_model": "acestep-v15-turbo", "models_initialized": True})
            return _Response({}, status=404)

        session = _Session({})
        session.get = get
        session.post = lambda url, **kwargs: _Response({}, status=404)
        now = {"t": 0.0}

        state = release_for_video(
            session=session,
            zimage_loaded=lambda: False,
            unload_zimage=lambda: None,
            run_command=run,
            sleeper=lambda _seconds: now.__setitem__("t", now["t"] + 1.0),
            clock=lambda: now["t"],
            vram_wait=10,
        )
        self.assertTrue(state.ace_stopped)
        self.assertIn("systemctl --user stop acestep-api", commands)
        self.assertGreaterEqual(reads["n"], 3)
        self.assertNotIn("systemctl --user start acestep-api", commands)
        restore_after_video(state, run_command=run)
        self.assertEqual(commands[-1], "systemctl --user start acestep-api")

    def test_stop_and_start_commands_honor_the_env_override(self) -> None:
        os.environ["HAVNAI_ACESTEP_STOP_CMD"] = "custom-stop"
        os.environ["HAVNAI_ACESTEP_START_CMD"] = "custom-start"
        commands = []
        stopped = {"value": False}

        def run(command: str) -> CommandResult:
            commands.append(command)
            if command.startswith("nvidia-smi"):
                return CommandResult(0, "200\n" if stopped["value"] else "8000\n")
            if command == "custom-stop":
                stopped["value"] = True
                return CommandResult(0, "")
            return CommandResult(0, "")

        def get(url, **kwargs):
            if url.endswith("/health") and not stopped["value"]:
                return _Response({"loaded_model": "acestep-v15-turbo", "models_initialized": True})
            raise RuntimeError("connection refused")

        session = _Session({})
        session.get = get
        session.post = lambda url, **kwargs: _Response({}, status=404)
        state = release_for_video(
            session=session,
            zimage_loaded=lambda: True,
            unload_zimage=lambda: None,
            run_command=run,
            sleeper=lambda _seconds: None,
            clock=lambda: 0.0,
        )
        self.assertIn("custom-stop", commands)
        self.assertNotIn("systemctl --user stop acestep-api", commands)
        restore_after_video(state, run_command=run, load_zimage=lambda: commands.append("load-zimage"))
        self.assertIn("custom-start", commands)
        self.assertNotIn("load-zimage", commands)

    def test_vram_that_stays_taken_restarts_ace_once_and_does_not_loop(self) -> None:
        commands = []
        events = []
        stopped = {"value": False}

        def run(command: str) -> CommandResult:
            commands.append(command)
            if command.startswith("nvidia-smi"):
                return CommandResult(0, "9000\n")
            if "stop" in command:
                stopped["value"] = True
            return CommandResult(0, "")

        def get(url, **kwargs):
            if stopped["value"]:
                raise RuntimeError("connection refused")
            return _Response({"loaded_model": "acestep-v15-turbo", "models_initialized": True})

        session = _Session({})
        session.get = get
        session.post = lambda url, **kwargs: _Response({}, status=404)
        now = {"t": 0.0}

        def sleeper(_seconds: float) -> None:
            now["t"] += 1.0

        with self.assertRaises(GpuBusy):
            release_for_video(
                session=session,
                zimage_loaded=lambda: True,
                unload_zimage=lambda: events.append("unload-zimage"),
                load_zimage=lambda: events.append("load-zimage"),
                run_command=run,
                sleeper=sleeper,
                clock=lambda: now["t"],
                vram_wait=3,
            )
        self.assertEqual(events, ["unload-zimage"])
        self.assertIn("systemctl --user stop acestep-api", commands)
        self.assertIn("systemctl --user start acestep-api", commands)
        commands.clear()
        events.clear()
        with self.assertRaises(GpuBusy) as caught:
            release_for_video(
                session=session,
                zimage_loaded=lambda: True,
                unload_zimage=lambda: events.append("unload-zimage"),
                load_zimage=lambda: events.append("load-zimage"),
                run_command=run,
                sleeper=sleeper,
                clock=lambda: now["t"],
            )
        self.assertIn("not retrying", str(caught.exception))
        self.assertEqual(commands, [])
        self.assertEqual(events, [])

    def test_music_waits_for_the_video_job_then_starts_ace(self) -> None:
        commands = []
        video = release_for_video(
            session=_down_session(),
            zimage_loaded=lambda: False,
            unload_zimage=lambda: None,
            run_command=lambda command: commands.append(command) or 0,
        )
        started = threading.Event()
        finished = threading.Event()
        seen = []

        def music() -> None:
            started.set()
            state = prepare_for_music(
                session=_down_session(),
                zimage_loaded=lambda: True,
                unload_zimage=lambda: seen.append("unload-zimage"),
                run_command=lambda command: seen.append(command) or CommandResult(0, "200\n"),
                wait_timeout=2,
                sleeper=lambda _seconds: None,
                clock=lambda: 0.0,
            )
            finish_music(state)
            finished.set()

        thread = threading.Thread(target=music)
        thread.start()
        self.assertTrue(started.wait(1))
        blocked_until = time.monotonic() + 0.3
        while time.monotonic() < blocked_until:
            self.assertFalse(finished.is_set())
            time.sleep(0.05)
        restore_after_video(video, run_command=lambda command: commands.append(command) or 0)
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertIn("unload-zimage", seen)
        self.assertIn("systemctl --user start acestep-api", seen)

    def test_music_leaves_a_running_ace_alone(self) -> None:
        commands = []
        session = _Session({"loaded_model": "", "models_initialized": False})
        state = prepare_for_music(
            session=session,
            zimage_loaded=lambda: False,
            unload_zimage=lambda: commands.append("unload-zimage"),
            run_command=lambda command: commands.append(command) or 0,
            wait_timeout=1,
        )
        finish_music(state)
        self.assertEqual(commands, [])
        self.assertFalse(gpu_handoff._gpu_lock.locked())


def _down_session():
    session = _Session({})

    def get(url, **kwargs):
        raise RuntimeError("connection refused")

    session.get = get
    return session


class LtxRecipeTests(unittest.TestCase):
    def test_default_sample_matches_the_local_run(self) -> None:
        sample = ltx23_sample({})
        self.assertEqual(sample["width"], 448)
        self.assertEqual(sample["height"], 832)
        self.assertEqual(sample["frames"], 217)
        self.assertEqual(sample["fps"], 24)
        self.assertEqual(sample["steps"], 8)
        self.assertEqual(sample["guidance"], 1.0)
        self.assertTrue(sample["audio"])
        self.assertEqual(sample["resolution"], "448x832")
        self.assertEqual(sample["duration_seconds"], 9.0)

    def test_studio_profile_uses_the_same_clip_when_the_client_omits_size(self) -> None:
        registry = json.loads((ROOT / "server" / "manifests" / "registry.json").read_text(encoding="utf-8"))
        ltx = next(item for item in registry["models"] if item["name"] == "ltx23_wangp_distilled")
        wan = next(item for item in registry["models"] if item["name"] == "wan22-i2v-a14b")
        ltx_spec = platform_v1.resolve_video_spec(
            {"prompt": "leaves move"},
            model=ltx["name"],
            profile=ltx["video_defaults"],
        )
        self.assertEqual(ltx_spec["parameters"]["width"], 448)
        self.assertEqual(ltx_spec["parameters"]["height"], 832)
        self.assertEqual(ltx_spec["parameters"]["frames"], 217)
        self.assertEqual(ltx_spec["parameters"]["fps"], 24)
        self.assertEqual(ltx_spec["parameters"]["steps"], 8)
        wan_spec = platform_v1.resolve_video_spec(
            {"prompt": "a slow blink", "aspect_ratio": "9:16"},
            model=wan["name"],
            profile=wan["video_defaults"],
        )
        self.assertEqual(wan_spec["parameters"]["frames"], 129)
        self.assertEqual(wan_spec["parameters"]["fps"], 16)
        self.assertEqual((wan_spec["parameters"]["width"], wan_spec["parameters"]["height"]), (480, 832))
        self.assertEqual(wan_spec["timeout_seconds"], 3600)


if __name__ == "__main__":
    unittest.main()

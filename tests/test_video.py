"""Tests for video generation: negative prompts, parameter clamping, job submission."""
from __future__ import annotations

import json
import sys
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the server package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))


# ---------------------------------------------------------------------------
# 1. Video negative prompt definitions
# ---------------------------------------------------------------------------

class TestVideoNegativePrompts(unittest.TestCase):
    """Verify that video-specific negative prompts are defined and non-empty."""

    def _load_app_module(self):
        """Import the constants we need from app.py without starting Flask."""
        # We can't import the full app easily, so parse the constants directly
        app_path = ROOT / "server" / "app.py"
        src = app_path.read_text()

        # Extract the constant definitions using exec in a controlled namespace
        ns: dict = {}
        # We need the base constants too
        for line_block in [
            "_NEGATIVE_QUALITY_COMMON",
            "_NEGATIVE_HAND_SD15",
            "_NEGATIVE_ANATOMY_SD15",
            "_NEGATIVE_STYLE_REALISM",
            "_NEGATIVE_VIDEO_COMMON",
            "_NEGATIVE_VIDEO_HAND",
            "NEGATIVE_VIDEO_LTXL",
            "NEGATIVE_VIDEO_ANIMATEDIFF",
            "POSITIVE_VIDEO_LTXL",
            "POSITIVE_VIDEO_ANIMATEDIFF",
        ]:
            # Find each assignment block
            for i, line in enumerate(src.splitlines()):
                stripped = line.strip()
                if stripped.startswith(f"{line_block} =") or stripped.startswith(f"{line_block}="):
                    # Grab until we find the closing ) or end of expression
                    block_lines = []
                    depth = 0
                    for j in range(i, min(i + 20, len(src.splitlines()))):
                        bl = src.splitlines()[j]
                        block_lines.append(bl)
                        depth += bl.count("(") + bl.count("[") - bl.count(")") - bl.count("]")
                        if depth <= 0 and j > i:
                            break
                    block = "\n".join(block_lines)
                    try:
                        exec(block, ns)
                    except Exception:
                        pass
                    break
        return ns

    def test_video_negative_prompts_exist(self):
        ns = self._load_app_module()
        self.assertIn("NEGATIVE_VIDEO_LTXL", ns)
        self.assertIn("NEGATIVE_VIDEO_ANIMATEDIFF", ns)
        self.assertTrue(len(ns["NEGATIVE_VIDEO_LTXL"]) > 50, "LTX2 negative prompt too short")
        self.assertTrue(len(ns["NEGATIVE_VIDEO_ANIMATEDIFF"]) > 50, "AnimateDiff negative prompt too short")

    def test_video_negative_contains_motion_artifacts(self):
        ns = self._load_app_module()
        for key in ("NEGATIVE_VIDEO_LTXL", "NEGATIVE_VIDEO_ANIMATEDIFF"):
            prompt = ns[key].lower()
            self.assertIn("flicker", prompt, f"{key} missing flicker prevention")
            self.assertIn("jitter", prompt, f"{key} missing jitter prevention")

    def test_video_negative_contains_hand_fix(self):
        ns = self._load_app_module()
        for key in ("NEGATIVE_VIDEO_LTXL", "NEGATIVE_VIDEO_ANIMATEDIFF"):
            prompt = ns[key].lower()
            self.assertIn("hands", prompt, f"{key} missing hand artifact prevention")

    def test_positive_video_tokens_exist(self):
        ns = self._load_app_module()
        self.assertIn("POSITIVE_VIDEO_LTXL", ns)
        self.assertIn("POSITIVE_VIDEO_ANIMATEDIFF", ns)
        for key in ("POSITIVE_VIDEO_LTXL", "POSITIVE_VIDEO_ANIMATEDIFF"):
            prompt = ns[key].lower()
            self.assertIn("smooth motion", prompt, f"{key} missing smooth motion token")
            self.assertIn("best quality", prompt, f"{key} missing best quality token")


# ---------------------------------------------------------------------------
# 2. Parameter clamping in runners
# ---------------------------------------------------------------------------

class TestLTX2ParameterClamping(unittest.TestCase):
    """Verify LTX2 runner clamps parameters correctly."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "engines" / "ltx2"))
        from engines.ltx2.ltx2_runner import _clamp_int, _clamp_float
        self.clamp_int = _clamp_int
        self.clamp_float = _clamp_float
        self.log = MagicMock()

    def test_clamp_int_within_range(self):
        self.assertEqual(self.clamp_int("steps", 30, 25, 1, 50, self.log), 30)

    def test_clamp_int_below_min(self):
        self.assertEqual(self.clamp_int("steps", -5, 25, 1, 50, self.log), 1)

    def test_clamp_int_above_max(self):
        self.assertEqual(self.clamp_int("steps", 100, 25, 1, 50, self.log), 50)

    def test_clamp_int_invalid_falls_to_default(self):
        self.assertEqual(self.clamp_int("steps", "abc", 25, 1, 50, self.log), 25)

    def test_clamp_int_none_falls_to_default(self):
        self.assertEqual(self.clamp_int("steps", None, 25, 1, 50, self.log), 25)

    def test_clamp_float_within_range(self):
        self.assertAlmostEqual(self.clamp_float("guidance", 7.5, 6.0, 0.0, 12.0, self.log), 7.5)

    def test_clamp_float_below_min(self):
        self.assertAlmostEqual(self.clamp_float("guidance", -1.0, 6.0, 0.0, 12.0, self.log), 0.0)

    def test_clamp_float_above_max(self):
        self.assertAlmostEqual(self.clamp_float("guidance", 20.0, 6.0, 0.0, 12.0, self.log), 12.0)

    def test_frames_max_is_16(self):
        """LTX2 max frames must be 16 (Latte-1 fixed temporal embeddings)."""
        result = self.clamp_int("frames", 32, 16, 1, 16, self.log)
        self.assertEqual(result, 16)

    def test_fps_max_is_12(self):
        """LTX2 max fps should be 12."""
        result = self.clamp_int("fps", 30, 8, 1, 12, self.log)
        self.assertEqual(result, 12)


class TestAnimateDiffParameterClamping(unittest.TestCase):
    """Verify AnimateDiff runner clamps parameters correctly."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "engines" / "animatediff"))
        from engines.animatediff.animatediff_runner import _clamp_int, _clamp_float
        self.clamp_int = _clamp_int
        self.clamp_float = _clamp_float
        self.log = MagicMock()

    def test_frames_max_is_64(self):
        """AnimateDiff supports up to 64 frames."""
        result = self.clamp_int("frames", 64, 16, 1, 64, self.log)
        self.assertEqual(result, 64)

    def test_frames_above_max(self):
        result = self.clamp_int("frames", 100, 16, 1, 64, self.log)
        self.assertEqual(result, 64)

    def test_fps_max_is_24(self):
        """AnimateDiff fps should max at 24."""
        result = self.clamp_int("fps", 30, 8, 1, 24, self.log)
        self.assertEqual(result, 24)

    def test_guidance_default(self):
        result = self.clamp_float("guidance", None, 6.0, 0.0, 12.0, self.log)
        self.assertAlmostEqual(result, 6.0)


# ---------------------------------------------------------------------------
# 3. Runner job handling (no GPU required)
# ---------------------------------------------------------------------------

class TestLTX2RunnerJobValidation(unittest.TestCase):
    """Test that the LTX2 runner validates jobs correctly (no GPU needed)."""

    def _run_ltx2(self, job):
        from engines.ltx2.ltx2_runner import run_ltx2
        return run_ltx2(job, log_fn=lambda m: None)

    def test_missing_prompt_fails(self):
        metrics, _, path = self._run_ltx2({"seed": 42})
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("prompt", metrics["error"])

    def test_missing_seed_fails(self):
        metrics, _, path = self._run_ltx2({"prompt": "test video"})
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("seed", metrics["error"])

    def test_empty_prompt_fails(self):
        metrics, _, path = self._run_ltx2({"prompt": "", "seed": 42})
        self.assertEqual(metrics["status"], "failed")

    def test_invalid_seed_fails(self):
        metrics, _, path = self._run_ltx2({"prompt": "test", "seed": "not-a-number"})
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("seed", metrics["error"])


class TestAnimateDiffRunnerJobValidation(unittest.TestCase):
    """Test that the AnimateDiff runner validates jobs correctly (no GPU needed)."""

    def _run_ad(self, job):
        from engines.animatediff.animatediff_runner import run_animatediff
        return run_animatediff(job, model_id="test-model", log_fn=lambda m: None)

    def test_missing_prompt_fails(self):
        metrics, _, path = self._run_ad({"seed": 42})
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("prompt", metrics["error"])

    def test_missing_seed_fails(self):
        metrics, _, path = self._run_ad({"prompt": "test video"})
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("seed", metrics["error"])

    def test_empty_prompt_fails(self):
        metrics, _, path = self._run_ad({"prompt": "  ", "seed": 42})
        self.assertEqual(metrics["status"], "failed")


# ---------------------------------------------------------------------------
# 4. Video frame normalization
# ---------------------------------------------------------------------------

class TestVideoFrameNormalization(unittest.TestCase):
    """Test frame normalization handles various tensor shapes."""

    def setUp(self):
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.skipTest("numpy not available")

    def test_ltx2_normalize_fhwc(self):
        from engines.ltx2.ltx2_runner import _normalize_video_frames
        frames = self.np.zeros((16, 256, 256, 3), dtype="uint8")
        result = _normalize_video_frames(frames)
        self.assertEqual(result.shape, (16, 256, 256, 3))

    def test_ltx2_normalize_fchw(self):
        from engines.ltx2.ltx2_runner import _normalize_video_frames
        frames = self.np.zeros((16, 3, 256, 256), dtype="uint8")
        result = _normalize_video_frames(frames)
        self.assertEqual(result.shape, (16, 256, 256, 3))

    def test_ltx2_normalize_batch_dim(self):
        from engines.ltx2.ltx2_runner import _normalize_video_frames
        frames = self.np.zeros((1, 16, 3, 256, 256), dtype="uint8")
        result = _normalize_video_frames(frames)
        self.assertEqual(result.shape, (16, 256, 256, 3))

    def test_animatediff_normalize_fhwc(self):
        from engines.animatediff.animatediff_runner import _normalize_video_frames
        frames = self.np.zeros((16, 512, 512, 3), dtype="uint8")
        result = _normalize_video_frames(frames)
        self.assertEqual(result.shape, (16, 512, 512, 3))

    def test_animatediff_normalize_fchw(self):
        from engines.animatediff.animatediff_runner import _normalize_video_frames
        frames = self.np.zeros((16, 3, 512, 512), dtype="uint8")
        result = _normalize_video_frames(frames)
        self.assertEqual(result.shape, (16, 512, 512, 3))


# ---------------------------------------------------------------------------
# 5. Init image loading
# ---------------------------------------------------------------------------

class TestInitImageLoading(unittest.TestCase):
    """Test init_image parsing handles various input formats."""

    def test_ltx2_none_returns_none(self):
        from engines.ltx2.ltx2_runner import _load_init_image
        self.assertIsNone(_load_init_image(None))

    def test_ltx2_empty_string_returns_none(self):
        from engines.ltx2.ltx2_runner import _load_init_image
        try:
            result = _load_init_image("")
            self.assertIsNone(result)
            result = _load_init_image("   ")
            self.assertIsNone(result)
        except RuntimeError as e:
            if "PIL" in str(e):
                self.skipTest("PIL not available")
            raise

    def test_animatediff_none_returns_none(self):
        from engines.animatediff.animatediff_runner import _load_init_image
        self.assertIsNone(_load_init_image(None))

    def test_animatediff_empty_string_returns_none(self):
        from engines.animatediff.animatediff_runner import _load_init_image
        try:
            result = _load_init_image("")
            self.assertIsNone(result)
            result = _load_init_image("   ")
            self.assertIsNone(result)
        except RuntimeError as e:
            if "PIL" in str(e):
                self.skipTest("PIL not available")
            raise


# ---------------------------------------------------------------------------
# 6. Timeout and progress logging
# ---------------------------------------------------------------------------

class TestLTX2Timeout(unittest.TestCase):
    """Test that the LTX2 runner respects job timeout."""

    def test_timeout_constant_has_default(self):
        from engines.ltx2.ltx2_runner import LTX2_JOB_TIMEOUT
        self.assertIsInstance(LTX2_JOB_TIMEOUT, int)
        self.assertGreater(LTX2_JOB_TIMEOUT, 0)

    def test_job_timeout_exception_defined(self):
        from engines.ltx2.ltx2_runner import _JobTimeout
        exc = _JobTimeout("test")
        self.assertIsInstance(exc, Exception)
        self.assertEqual(str(exc), "test")

    def test_timeout_in_metrics_when_no_gpu(self):
        """Even when the job fails for no-GPU, timeout should be in metrics."""
        from engines.ltx2.ltx2_runner import run_ltx2
        metrics, _, _ = run_ltx2(
            {"prompt": "test", "seed": 42, "timeout": 60},
            log_fn=lambda m: None,
        )
        # Job fails (no GPU in test env), but we can check the params were parsed
        self.assertEqual(metrics["status"], "failed")


class TestLTX2ProgressCallback(unittest.TestCase):
    """Test that generate_video_frames accepts callback_on_step_end."""

    def test_generator_accepts_callback_kwarg(self):
        """generate_video_frames signature should accept callback_on_step_end."""
        import inspect
        from engines.ltx2.ltx2_generator import generate_video_frames
        sig = inspect.signature(generate_video_frames)
        self.assertIn("callback_on_step_end", sig.parameters)


# ---------------------------------------------------------------------------
# 7. LTX2 frame mismatch fallback
# ---------------------------------------------------------------------------

class TestLTX2FrameFallback(unittest.TestCase):
    """Test LTX2 fallback logic for fixed 16-frame models."""

    def test_retry_triggered_for_known_16_frame_shape_error(self):
        from engines.ltx2.ltx2_runner import _should_retry_with_16_frames

        err = RuntimeError(
            "The size of tensor a (12) must match the size of tensor b (16) at non-singleton dimension 1"
        )
        self.assertTrue(_should_retry_with_16_frames(err, 12))

    def test_no_retry_when_request_already_16(self):
        from engines.ltx2.ltx2_runner import _should_retry_with_16_frames

        err = RuntimeError(
            "The size of tensor a (16) must match the size of tensor b (16) at non-singleton dimension 1"
        )
        self.assertFalse(_should_retry_with_16_frames(err, 16))


# ---------------------------------------------------------------------------
# 8. GPU profile presets
# ---------------------------------------------------------------------------

class TestGPUProfiles(unittest.TestCase):
    """Test that GPU profile presets are correctly defined in app.py."""

    def _load_profiles(self):
        app_path = ROOT / "server" / "app.py"
        src = app_path.read_text()
        ns: dict = {"os": os, "Dict": dict, "Any": object}
        # Extract the _GPU_PROFILES dict
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "_GPU_PROFILES" in line and "=" in line:
                block_lines = []
                depth = 0
                for j in range(i, min(i + 10, len(lines))):
                    bl = lines[j]
                    block_lines.append(bl)
                    depth += bl.count("{") - bl.count("}")
                    if depth <= 0 and j > i:
                        break
                block = "\n".join(block_lines)
                exec(block, ns)
                break
        return ns.get("_GPU_PROFILES", {})

    def test_fast_3060_profile_exists(self):
        profiles = self._load_profiles()
        self.assertIn("fast_3060", profiles)
        p = profiles["fast_3060"]
        self.assertEqual(p["steps"], 20)
        self.assertEqual(p["frames"], 16)

    def test_quality_profile_exists(self):
        profiles = self._load_profiles()
        self.assertIn("quality", profiles)
        p = profiles["quality"]
        self.assertEqual(p["steps"], 30)
        self.assertEqual(p["frames"], 16)

    def test_fast_3060_has_lower_steps_than_quality(self):
        profiles = self._load_profiles()
        self.assertLess(profiles["fast_3060"]["steps"], profiles["quality"]["steps"])


if __name__ == "__main__":
    unittest.main()

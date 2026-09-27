"""Tests for node reward runtime accounting."""

from __future__ import annotations

import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
REWARDS_PATH = ROOT / "server" / "rewards.py"
REWARDS_SPEC = importlib.util.spec_from_file_location("havnai_server_rewards", REWARDS_PATH)
assert REWARDS_SPEC is not None and REWARDS_SPEC.loader is not None
rewards_module = importlib.util.module_from_spec(REWARDS_SPEC)
sys.modules[REWARDS_SPEC.name] = rewards_module
REWARDS_SPEC.loader.exec_module(rewards_module)


class RewardRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        patches = (
            mock.patch.object(
                rewards_module,
                "MODEL_WEIGHTS",
                {"image-model": 10.0},
                create=True,
            ),
            mock.patch.object(
                rewards_module,
                "REWARD_CONFIG",
                {
                    "base_reward": 0.05,
                    "baseline_runtime": 8.0,
                    "sdxl_factor": 1.5,
                },
                create=True,
            ),
            mock.patch.object(
                rewards_module,
                "LOGGER",
                logging.getLogger("test-rewards"),
                create=True,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_generation_time_excludes_cold_model_load(self) -> None:
        reward, factors = rewards_module.compute_reward(
            "image-model",
            "sdxl",
            {
                "generation_ms": 40_000,
                "pipeline_load_ms": 200_000,
                "inference_time_ms": 240_000,
            },
            "success",
        )

        self.assertEqual(reward, 0.375)
        self.assertEqual(factors["runtime_seconds"], 40.0)
        self.assertEqual(factors["runtime_factor"], 5.0)
        self.assertEqual(factors["runtime_source"], "generation_ms")

    def test_legacy_nodes_fall_back_to_inference_time(self) -> None:
        reward, factors = rewards_module.compute_reward(
            "image-model",
            "sdxl",
            {"inference_time_ms": 16_000},
            "success",
        )

        self.assertEqual(reward, 0.15)
        self.assertEqual(factors["runtime_seconds"], 16.0)
        self.assertEqual(factors["runtime_source"], "inference_time_ms")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for credit cost resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import credits as credits_module


class CreditCostResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_get_model_config = getattr(credits_module, "get_model_config", None)
        credits_module.get_model_config = lambda model_name: None

    def tearDown(self) -> None:
        if self._orig_get_model_config is None:
            delattr(credits_module, "get_model_config")
        else:
            credits_module.get_model_config = self._orig_get_model_config

    def test_face_swap_uses_face_swap_rate_even_for_sdxl_model(self) -> None:
        credits_module.get_model_config = lambda model_name: {
            "name": model_name,
            "pipeline": "sdxl",
        }
        cost = credits_module.resolve_credit_cost("epicrealismXL_purefix", "FACE_SWAP")
        self.assertEqual(cost, credits_module.DEFAULT_CREDIT_COSTS["face_swap"])

    def test_image_generation_uses_pipeline_default_when_no_explicit_override(self) -> None:
        credits_module.get_model_config = lambda model_name: {
            "name": model_name,
            "pipeline": "sd15",
        }
        cost = credits_module.resolve_credit_cost("perfectdeliberate_v5SD15", "IMAGE_GEN")
        self.assertEqual(cost, credits_module.DEFAULT_CREDIT_COSTS["sd15"])


if __name__ == "__main__":
    unittest.main()

"""Tests for output watermark application."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT))

try:
    import client as client_module  # type: ignore
    CLIENT_IMPORT_ERROR = None
except SystemExit as exc:  # pragma: no cover - depends on local runtime deps
    client_module = None  # type: ignore
    CLIENT_IMPORT_ERROR = str(exc)


class OutputWatermarkTests(unittest.TestCase):
    def setUp(self) -> None:
        if client_module is None:
            self.skipTest(CLIENT_IMPORT_ERROR or "client module unavailable")
        if client_module.Image is None:
            self.skipTest("Pillow is not available in this environment")
        self._orig_enabled = client_module.WATERMARK_ENABLED

    def tearDown(self) -> None:
        client_module.WATERMARK_ENABLED = self._orig_enabled

    def test_apply_output_watermark_preserves_size_and_changes_pixels(self) -> None:
        base = client_module.Image.new("RGB", (128, 128), color=(10, 20, 30))
        mark = client_module.Image.new("RGBA", (40, 20), color=(255, 255, 255, 255))
        client_module.WATERMARK_ENABLED = True

        with patch.object(client_module, "_load_watermark_image", return_value=mark):
            result = client_module._apply_output_watermark(base, task_id="wm-test")

        self.assertEqual(result.size, base.size)
        self.assertNotEqual(list(result.getdata()), list(base.getdata()))


if __name__ == "__main__":
    unittest.main()

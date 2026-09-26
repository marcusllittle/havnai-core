"""The isolated executor must be stoppable on every platform the node runs on."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class StopExecutorTests(unittest.TestCase):
    def test_stops_a_running_executor_on_this_platform(self) -> None:
        # Runs in a subprocess because importing the client has side effects.
        # On Windows this used to raise AttributeError: os.killpg does not
        # exist there, so a cancelled or timed-out video job crashed its thread.
        script = """
import subprocess, sys
from client import client
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    start_new_session=True,
)
client._stop_executor(child)
assert child.returncode is not None, "executor still running"
print("stopped", child.returncode)
"""
        with tempfile.TemporaryDirectory() as home:
            # A wallet in .env keeps the client import from prompting for one.
            Path(home, ".env").write_text("WALLET=0x" + "1" * 40 + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env={**os.environ, "HAVNAI_HOME": home},
                capture_output=True,
                text=True,
                # Cold imports of torch and friends can take a while on first run.
                timeout=180,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("stopped", result.stdout)


if __name__ == "__main__":
    unittest.main()

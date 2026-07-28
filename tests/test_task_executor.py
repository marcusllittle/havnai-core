"""Tests for the isolated GPU task process entry point."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from client.task_executor import main


class TaskExecutorTests(unittest.TestCase):
    def test_refreshes_manifest_before_execution(self) -> None:
        events: list[str] = []
        fake_client = types.ModuleType("client.client")

        def refresh_manifest_with_backoff(**_kwargs: object) -> bool:
            events.append("refresh")
            return True

        def execute_task(task: dict[str, object]) -> None:
            self.assertEqual(task["task_id"], "job-test")
            events.append("execute")

        fake_client.refresh_manifest_with_backoff = refresh_manifest_with_backoff  # type: ignore[attr-defined]
        fake_client.execute_task = execute_task  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory) / "task.json"
            task_path.write_text(json.dumps({"task_id": "job-test"}), encoding="utf-8")
            with patch.dict(sys.modules, {"client.client": fake_client}), patch.object(
                sys, "argv", ["client.task_executor", str(task_path)]
            ):
                self.assertEqual(main(), 0)

        self.assertEqual(events, ["refresh", "execute"])


if __name__ == "__main__":
    unittest.main()

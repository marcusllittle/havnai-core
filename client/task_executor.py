"""Isolated GPU task entry point supervised by the lightweight node agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m client.task_executor TASK.json")
    task_path = Path(sys.argv[1])
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise RuntimeError("task payload must be an object")
    from client.client import execute_task, refresh_manifest_with_backoff

    if not refresh_manifest_with_backoff(reason="isolated_executor", max_attempts=3):
        raise RuntimeError("isolated executor could not refresh the model manifest")
    execute_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

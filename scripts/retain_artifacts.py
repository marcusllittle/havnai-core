#!/usr/bin/env python3
"""Legacy timer entry point: age alone never authorizes artifact deletion."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    required = ("HAVNAI_DB_PATH", "HAVNAI_OUTPUTS_DIR", "HAVNAI_ASSETS_DIR")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        print("Retention requires explicit configuration: " + ", ".join(missing), file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    from artifact_lifecycle import main as lifecycle_main
    return lifecycle_main([
        "--database", os.environ["HAVNAI_DB_PATH"],
        "--outputs-dir", os.environ["HAVNAI_OUTPUTS_DIR"],
        "--assets-dir", os.environ["HAVNAI_ASSETS_DIR"],
        "--limit", "25",
    ])


if __name__ == "__main__":
    raise SystemExit(main())

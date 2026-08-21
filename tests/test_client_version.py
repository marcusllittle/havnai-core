from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT))

import client as client_module  # type: ignore  # noqa: E402


def test_source_root_git_revision_wins_over_installed_version(tmp_path: Path) -> None:
    installed_version = tmp_path / "installed-version"
    installed_version.write_text("old-release\n")

    with (
        patch.dict(
            os.environ,
            {"HAVNAI_SOURCE_ROOT": str(tmp_path / "source")},
            clear=False,
        ),
        patch.object(client_module, "VERSION_SEARCH_PATHS", [installed_version]),
        patch.object(client_module.subprocess, "check_output", return_value=b"abc1234\n"),
    ):
        assert client_module.load_version() == "abc1234"


def test_source_root_version_file_is_used_when_git_is_unavailable(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "VERSION").write_text("source-release\n")

    with (
        patch.dict(os.environ, {"HAVNAI_SOURCE_ROOT": str(source_root)}, clear=False),
        patch.object(client_module.subprocess, "check_output", side_effect=OSError),
    ):
        assert client_module.load_version() == "source-release"


def test_installed_version_remains_default_without_source_override(tmp_path: Path) -> None:
    installed_version = tmp_path / "VERSION"
    installed_version.write_text("installed-release\n")

    with (
        patch.dict(os.environ, {}, clear=False),
        patch.object(client_module, "VERSION_SEARCH_PATHS", [installed_version]),
    ):
        os.environ.pop("HAVNAI_SOURCE_ROOT", None)
        os.environ.pop("HAVNAI_CLIENT_VERSION", None)
        assert client_module.load_version() == "installed-release"


def test_explicit_client_version_has_highest_priority() -> None:
    with patch.dict(
        os.environ,
        {"HAVNAI_CLIENT_VERSION": "operator-build", "HAVNAI_SOURCE_ROOT": "/missing"},
        clear=False,
    ):
        assert client_module.load_version() == "operator-build"

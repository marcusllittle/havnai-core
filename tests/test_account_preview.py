"""The preview must never reuse production storage or payment credentials."""
import base64
from pathlib import Path

import pytest

from scripts.account_preview import preview_environment


def development_env(tmp_path):
    source = tmp_path / "web.env"
    host = base64.b64encode(b"preview.clerk.accounts.dev$").decode()
    source.write_text(f"NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_{host}\nCLERK_SECRET_KEY=sk_test_fixture\n")
    return source


def test_preview_isolates_storage_and_disables_funding(tmp_path):
    source = development_env(tmp_path)
    directory = tmp_path / "preview"
    result = preview_environment(source, directory, 5101, "http://localhost:3100")
    assert result["HAVNAI_CLERK_ISSUER"] == "https://preview.clerk.accounts.dev"
    assert result["CLERK_SECRET_KEY"] == "sk_test_fixture"
    assert result["CLERK_JWT_KEY"] == ""
    assert result["CLERK_WEBHOOK_SIGNING_SECRET"] == result["HAVNAI_CLERK_INSTANCE_ID"] == ""
    assert result["CORS_ORIGINS"] == result["HAVNAI_ACCOUNT_ORIGINS"] == "http://localhost:3100"
    assert Path(result["HAVNAI_DB_PATH"]).parent == directory
    assert result["SERVER_BIND"] == "127.0.0.1"
    assert result["STRIPE_SECRET_KEY"] == result["STRIPE_ACCOUNT_WEBHOOK_SECRET"] == ""
    assert result["STRIPE_ENABLED"] == result["HAVNAI_ACCOUNT_CHECKOUT_ENABLED"] == "false"
    assert result["HAVNAI_HAI_FUNDING_ENABLED"] == "0"
    assert result["PYTHON_DOTENV_DISABLED"] == "1"
    assert list(directory.iterdir()) == [directory / ".havnai-account-preview"]
    # Reusing an explicitly marked preview is allowed; credentials rotate.
    again = preview_environment(source, directory, 5101, "http://localhost:3100")
    assert again["SERVER_JOIN_TOKEN"] != result["SERVER_JOIN_TOKEN"]


def test_preview_rejects_existing_unmarked_storage(tmp_path):
    source = development_env(tmp_path)
    directory = tmp_path / "existing"
    directory.mkdir()
    existing = directory / "coordinator.db"
    existing.write_bytes(b"existing data")
    with pytest.raises(ValueError, match="empty directory"):
        preview_environment(source, directory, 5101, "http://localhost:3100")
    assert existing.read_bytes() == b"existing data"
    assert not (directory / ".havnai-account-preview").exists()


@pytest.mark.parametrize("origin,port", [("https://joinhavn.io", 5101), ("http://localhost:3100/", 5101), ("http://localhost:3100", 80)])
def test_preview_rejects_nonlocal_or_privileged_binding(tmp_path, origin, port):
    with pytest.raises(ValueError):
        preview_environment(development_env(tmp_path), tmp_path / "preview", port, origin)
    assert not (tmp_path / "preview").exists()


def test_preview_rejects_live_keys_without_echoing_them(tmp_path):
    source = development_env(tmp_path)
    source.write_text(source.read_text().replace("sk_test_fixture", "sk_live_do_not_echo"))
    with pytest.raises(ValueError) as error:
        preview_environment(source, tmp_path / "preview", 5101, "http://localhost:3100")
    assert "sk_live" not in str(error.value)
    assert not (tmp_path / "preview").exists()

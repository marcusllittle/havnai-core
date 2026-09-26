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
    assert result["HAVNAI_ACCOUNT_IMPORT_ENABLED"] == "0"
    assert result["PYTHON_DOTENV_DISABLED"] == "1"
    assert list(directory.iterdir()) == [directory / ".havnai-account-preview"]
    # Reusing an explicitly marked preview is allowed; credentials rotate.
    again = preview_environment(source, directory, 5101, "http://localhost:3100")
    assert again["SERVER_JOIN_TOKEN"] != result["SERVER_JOIN_TOKEN"]
    enabled = preview_environment(source, directory, 5101, "http://localhost:3100", enable_imports=True)
    assert enabled["HAVNAI_ACCOUNT_IMPORT_ENABLED"] == "1"
    assert enabled["STRIPE_ENABLED"] == "false"


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


def sandbox_env(tmp_path, **overrides):
    values = {
        "STRIPE_SECRET_KEY": "sk_test_fixture_key",
        "STRIPE_ACCOUNT_WEBHOOK_SECRET": "whsec_fixture_secret",
        "HAVNAI_CREDIT_TERMS_VERSION": "sandbox-v1",
        "HAVNAI_CREDIT_TERMS_URL": "http://localhost:3100/test-terms",
        "HAVNAI_CREDIT_REFUND_URL": "http://localhost:3100/test-refunds",
        # Even a supplied file cannot redirect storage, bind publicly, or enable legacy funding.
        "HAVNAI_DB_PATH": "/production/coordinator.db", "SERVER_BIND": "0.0.0.0",
        "STRIPE_ENABLED": "true", "HAVNAI_HAI_FUNDING_ENABLED": "1",
        "HAVNAI_CHECKOUT_ORIGIN": "https://joinhavn.io",
    }
    values.update(overrides)
    source = tmp_path / "sandbox.env"
    source.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    return source


@pytest.mark.parametrize("key", ["sk_test_fixture_key", "rk_test_fixture_key"])
def test_explicit_sandbox_enables_only_account_test_checkout(tmp_path, key):
    directory = tmp_path / "preview"
    result = preview_environment(development_env(tmp_path), directory, 5101, "http://localhost:3100",
                                 sandbox_payments_env=sandbox_env(tmp_path, STRIPE_SECRET_KEY=key))
    assert result["STRIPE_SECRET_KEY"] == key
    assert result["HAVNAI_ACCOUNT_CHECKOUT_ENABLED"] == "true"
    assert result["HAVNAI_CHECKOUT_ORIGIN"] == "http://localhost:3100"
    assert result["HAVNAI_CREDIT_TERMS_VERSION"] == "sandbox-v1"
    assert Path(result["HAVNAI_DB_PATH"]).parent == directory
    assert result["SERVER_BIND"] == "127.0.0.1"
    assert result["STRIPE_ENABLED"] == "false"
    assert result["HAVNAI_HAI_FUNDING_ENABLED"] == "0"
    assert result["HAVNAI_ACCOUNT_IMPORT_ENABLED"] == "0"


@pytest.mark.parametrize("overrides", [
    {"STRIPE_SECRET_KEY": "sk_live_do_not_echo"},
    {"STRIPE_SECRET_KEY": "rk_live_do_not_echo"},
    {"STRIPE_SECRET_KEY": ""},
    {"STRIPE_ACCOUNT_WEBHOOK_SECRET": ""},
    {"HAVNAI_CREDIT_TERMS_VERSION": ""},
    {"HAVNAI_CREDIT_TERMS_URL": "https://joinhavn.io/terms"},
    {"HAVNAI_CREDIT_REFUND_URL": "http://localhost:3101/refunds"},
    {"HAVNAI_CREDIT_REFUND_URL": "http://user@localhost:3100/refunds"},
])
def test_invalid_sandbox_config_fails_before_storage_changes(tmp_path, overrides):
    with pytest.raises(ValueError) as error:
        preview_environment(development_env(tmp_path), tmp_path / "preview", 5101, "http://localhost:3100",
                            sandbox_payments_env=sandbox_env(tmp_path, **overrides))
    assert "do_not_echo" not in str(error.value)
    assert not (tmp_path / "preview").exists()


def test_sandbox_requires_explicit_existing_file(tmp_path):
    with pytest.raises(ValueError, match="file is missing"):
        preview_environment(development_env(tmp_path), tmp_path / "preview", 5101, "http://localhost:3100",
                            sandbox_payments_env=tmp_path / "missing.env")
    assert not (tmp_path / "preview").exists()

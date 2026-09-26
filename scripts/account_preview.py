"""Run the account branch locally with private Clerk development keys.

Uses a dedicated database and storage directory, binds only to loopback, and
disables payments/blockchain funding by default. An explicit private sandbox
environment file can enable test Checkout. Never prints provider keys.
"""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import re
import runpy
import secrets
import sys
from urllib.parse import urlsplit

from dotenv import dotenv_values


def sandbox_payment_environment(source: Path, web_origin: str) -> dict[str, str]:
    """Allow only test payments; never load arbitrary environment overrides."""
    if not source.is_file():
        raise ValueError("The private sandbox payment environment file is missing.")
    values = dotenv_values(source, interpolate=False)
    key = (values.get("STRIPE_SECRET_KEY") or "").strip()
    webhook = (values.get("STRIPE_ACCOUNT_WEBHOOK_SECRET") or "").strip()
    if not key.startswith(("sk_test_", "rk_test_")) or len(key.split("_", 2)[-1]) < 8:
        raise ValueError("The preview requires a Stripe sandbox server key.")
    if not webhook.startswith("whsec_") or len(webhook) < 14:
        raise ValueError("The preview requires a sandbox listener signing secret.")
    result = {"STRIPE_SECRET_KEY": key, "STRIPE_ACCOUNT_WEBHOOK_SECRET": webhook,
              "HAVNAI_ACCOUNT_CHECKOUT_ENABLED": "true", "HAVNAI_CHECKOUT_ORIGIN": web_origin}
    version = (values.get("HAVNAI_CREDIT_TERMS_VERSION") or "").strip()
    if not version:
        raise ValueError("Set an explicit sandbox credit policy revision.")
    result["HAVNAI_CREDIT_TERMS_VERSION"] = version
    for name in ("HAVNAI_CREDIT_TERMS_URL", "HAVNAI_CREDIT_REFUND_URL"):
        value = (values.get(name) or "").strip()
        try:
            parts = urlsplit(value)
            valid = (parts.scheme == "http" and f"http://{parts.netloc}" == web_origin
                     and not parts.username and not parts.password and bool(parts.path)
                     and not parts.query and not parts.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Sandbox policy URLs must be pages on the local web origin.")
        result[name] = value
    return result


def preview_environment(web_env: Path, data_dir: Path, port: int, web_origin: str, *, enable_imports: bool = False,
                        sandbox_payments_env: Path | None = None) -> dict[str, str]:
    values = dotenv_values(web_env)
    publishable = values.get("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY") or ""
    secret = values.get("CLERK_SECRET_KEY") or ""
    if not publishable.startswith("pk_test_") or not secret.startswith("sk_test_"):
        raise ValueError("Save both Clerk development keys in the web environment file first.")
    try:
        encoded = publishable[len("pk_test_"):]
        host = base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True).decode().removesuffix("$")
    except (ValueError, UnicodeError):
        raise ValueError("The Clerk publishable key is not valid.") from None
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.clerk\.accounts\.dev", host):
        raise ValueError("The key must belong to a Clerk development instance.")
    if not re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):[0-9]{2,5}", web_origin):
        raise ValueError("Use an exact local web origin such as http://localhost:3100.")
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an unprivileged local port.")
    payment_environment = sandbox_payment_environment(sandbox_payments_env, web_origin) if sandbox_payments_env else {}
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / ".havnai-account-preview"
    if not marker.exists():
        if any(data_dir.iterdir()):
            raise ValueError("Choose an empty directory for the isolated account preview.")
        marker.write_text("HAVN-11 local account preview\n", encoding="utf-8")
    environment = {
        "CLERK_SECRET_KEY": secret, "CLERK_JWT_KEY": "",
        "CLERK_WEBHOOK_SIGNING_SECRET": "", "HAVNAI_CLERK_INSTANCE_ID": "",
        "HAVNAI_CLERK_ISSUER": "https://" + host, "HAVNAI_CLERK_AUDIENCE": "havnai-api",
        "HAVNAI_ACCOUNT_ORIGINS": web_origin, "CORS_ORIGINS": web_origin, "PYTHON_DOTENV_DISABLED": "1",
        "HAVNAI_DB_PATH": str(data_dir / "account-preview.sqlite3"),
        "HAVNAI_STATIC_DIR": str(data_dir / "static"), "HAVNAI_INSTALLER_DIR": str(data_dir / "installers"),
        "HAVNAI_LOG_DIR": str(data_dir / "logs"), "HAVNAI_OUTPUTS_DIR": str(data_dir / "static" / "outputs"),
        "HAVNAI_ASSETS_DIR": str(data_dir / "static" / "assets"), "HAVNAI_NODES_PATH": str(data_dir / "nodes.json"),
        "HAVNAI_OWNER_TOKEN": secrets.token_urlsafe(32), "HAVNAI_ADMIN_TOKEN": secrets.token_urlsafe(32),
        "SERVER_JOIN_TOKEN": secrets.token_urlsafe(32),
        "STRIPE_ENABLED": "false", "HAVNAI_ACCOUNT_CHECKOUT_ENABLED": "false",
        "HAVNAI_ACCOUNT_IMPORT_ENABLED": "1" if enable_imports else "0",
        "STRIPE_SECRET_KEY": "", "STRIPE_WEBHOOK_SECRET": "", "STRIPE_ACCOUNT_WEBHOOK_SECRET": "",
        "HAVNAI_HAI_FUNDING_ENABLED": "0",
        "SERVER_BIND": "127.0.0.1", "SERVER_PORT": str(port),
    }
    environment.update(payment_environment)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-env", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".local/state/havnai/account-preview")
    parser.add_argument("--port", type=int, default=5101)
    parser.add_argument("--web-origin", default="http://localhost:3100")
    parser.add_argument("--enable-imports", action="store_true", help="Enable signed imports only in this isolated local preview")
    parser.add_argument("--sandbox-payments-env", type=Path,
                        help="Private environment file with test-only Stripe credentials and local policy URLs")
    args = parser.parse_args()
    try:
        environment = preview_environment(args.web_env, args.data_dir.resolve(), args.port, args.web_origin,
                                          enable_imports=args.enable_imports, sandbox_payments_env=args.sandbox_payments_env)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    os.environ.update(environment)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    sys.path.insert(0, str(root))
    payment_status = "sandbox payments enabled" if args.sandbox_payments_env else "payments disabled"
    print(f"Account preview: http://127.0.0.1:{args.port}; {payment_status}.", flush=True)
    runpy.run_path(str(root / "server/app.py"), run_name="__main__")


if __name__ == "__main__":
    main()

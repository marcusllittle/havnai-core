"""Mint one short-lived Clerk session JWT for an existing signed-in session.

The mixed-model drill calls this before every /v2 request. /v2 accepts only a
Clerk session token, and server/account_auth.py rejects any token whose exp
minus iat is greater than 120 seconds. This helper asks Clerk for a session
token in that window. It does not create a session, mint a custom JWT template
(those can omit sid), or produce an admin, owner, or join token.

The Clerk secret key is read from CLERK_SECRET_KEY. Never pass it on the
command line. The session must already exist because Marcus signed in at
https://joinhavn.io. Copy its sess_ id from the Clerk dashboard, or point
--browser-token-file at the signed-in __session cookie. The cookie is decoded
only to read sid. It is not forwarded to /v2.

Stdout is the JWT and nothing else. The drill's --token-command consumes it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from urllib import error, parse, request
from urllib.parse import urlsplit


MIN_SESSION_SECONDS = 30
MAX_SESSION_SECONDS = 120
DEFAULT_SESSION_SECONDS = 60
SESSION_ID_RE = re.compile(r"^sess_[A-Za-z0-9]+$")
JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


class SessionTokenError(RuntimeError):
    pass


class _RejectRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        raise SessionTokenError(f"clerk session token endpoint returned HTTP {code}")


def _api_base(value: str) -> str:
    raw = value.strip().rstrip("/")
    parts = urlsplit(raw)
    loopback = parts.hostname in {"127.0.0.1", "localhost"}
    if parts.scheme == "https" and parts.hostname and not parts.username and not parts.password:
        return raw
    if parts.scheme == "http" and loopback and not parts.username and not parts.password:
        return raw
    raise SessionTokenError("CLERK_API_URL must be https://api.clerk.com")


def session_id_from_browser_token(value: str) -> str:
    """Read sid from a Clerk session JWT without trusting or reusing it."""
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise SessionTokenError("browser session file is not a Clerk session JWT")
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionTokenError("browser session file has no session id") from exc
    session_id = payload.get("sid") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise SessionTokenError("browser session file has no session id")
    return session_id


def _session_id_from_file(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read(16385)
    except OSError as exc:
        raise SessionTokenError("browser session file could not be read") from exc
    if len(data) >= 16384:
        raise SessionTokenError("browser session file is too large")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise SessionTokenError("browser session file could not be read") from exc
    return session_id_from_browser_token(text)


def resolve_session_id(session_id: str, browser_token_file: str) -> str:
    explicit = session_id.strip()
    path = browser_token_file.strip()
    if explicit and path:
        raise SessionTokenError("pass only one of --session-id or --browser-token-file")
    if path:
        explicit = _session_id_from_file(path)
    if explicit.startswith("sk_"):
        raise SessionTokenError("session id must be a Clerk session (sess_...), not a secret key")
    if not SESSION_ID_RE.fullmatch(explicit):
        raise SessionTokenError("session id must look like sess_...")
    return explicit


def resolve_lifetime(seconds: int) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise SessionTokenError("expires_in_seconds must be an integer")
    if not MIN_SESSION_SECONDS <= seconds <= MAX_SESSION_SECONDS:
        raise SessionTokenError(
            f"expires_in_seconds must be {MIN_SESSION_SECONDS}..{MAX_SESSION_SECONDS} "
            "so /v2 can accept the session"
        )
    return seconds


def mint_session_token(*, session_id: str, secret_key: str, expires_in_seconds: int, api_base: str) -> str:
    secret = secret_key.strip()
    if not secret.startswith(("sk_test_", "sk_live_")):
        raise SessionTokenError("CLERK_SECRET_KEY must be a Clerk secret key in the environment")
    session_id = resolve_session_id(session_id, "")
    expires_in_seconds = resolve_lifetime(expires_in_seconds)
    base = _api_base(api_base)
    body = json.dumps({"expires_in_seconds": expires_in_seconds}).encode("utf-8")
    req = request.Request(
        f"{base}/v1/sessions/{parse.quote(session_id)}/tokens",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    opener = request.build_opener(_RejectRedirect)
    try:
        with opener.open(req, timeout=20) as response:
            raw = response.read(32768)
            status = getattr(response, "status", 200)
    except error.HTTPError as exc:
        detail = exc.read(2048)
        code = ""
        try:
            parsed = json.loads(detail.decode("utf-8", "replace") or "{}")
            errors = parsed.get("errors") if isinstance(parsed, dict) else None
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                code = str(errors[0].get("code") or "")
        except (UnicodeError, json.JSONDecodeError):
            code = ""
        suffix = f" {code}" if code else ""
        raise SessionTokenError(f"clerk session token request failed: HTTP {exc.code}{suffix}") from exc
    except error.URLError as exc:
        raise SessionTokenError("clerk session token request failed") from exc
    if status != 200:
        raise SessionTokenError(f"clerk session token request failed: HTTP {status}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SessionTokenError("clerk session token response was not a session JWT") from exc
    jwt = payload.get("jwt") if isinstance(payload, dict) else None
    if not isinstance(jwt, str) or not JWT_RE.fullmatch(jwt):
        raise SessionTokenError("clerk session token response was not a session JWT")
    return jwt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", default=os.environ.get("HAVNAI_DRILL_CLERK_SESSION_ID", ""),
                        help="existing Clerk session id (sess_...). Defaults to HAVNAI_DRILL_CLERK_SESSION_ID")
    parser.add_argument("--browser-token-file", default=os.environ.get("HAVNAI_DRILL_BROWSER_TOKEN_FILE", ""),
                        help="signed-in __session cookie file. Only its sid claim is read")
    parser.add_argument("--expires-in-seconds", type=int,
                        default=int(os.environ.get("HAVNAI_DRILL_SESSION_SECONDS", DEFAULT_SESSION_SECONDS)),
                        help=f"Clerk session lifetime, {MIN_SESSION_SECONDS}..{MAX_SESSION_SECONDS} (default {DEFAULT_SESSION_SECONDS})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        session_id = resolve_session_id(args.session_id, args.browser_token_file)
        token = mint_session_token(
            session_id=session_id,
            secret_key=os.environ.get("CLERK_SECRET_KEY", ""),
            expires_in_seconds=args.expires_in_seconds,
            api_base=os.environ.get("CLERK_API_URL", "https://api.clerk.com"),
        )
    except SessionTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

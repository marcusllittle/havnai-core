from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import clerk_session_token as clerk  # type: ignore


def _jwt(payload: dict) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'RS256'})}.{encode(payload)}.sig"


def test_lifetime_above_the_account_limit_never_calls_clerk(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("clerk was contacted for a rejected lifetime")

    monkeypatch.setattr(clerk.request, "urlopen", boom)
    with pytest.raises(clerk.SessionTokenError, match="30..120"):
        clerk.mint_session_token(
            session_id="sess_abc", secret_key="sk_test_fixture",
            expires_in_seconds=121, api_base="https://api.clerk.com",
        )


def test_secret_must_come_from_the_environment_shape_not_a_session():
    with pytest.raises(clerk.SessionTokenError, match="CLERK_SECRET_KEY"):
        clerk.mint_session_token(session_id="sess_abc", secret_key="", expires_in_seconds=60, api_base="https://api.clerk.com")
    with pytest.raises(clerk.SessionTokenError, match="not a secret key"):
        clerk.resolve_session_id("sk_test_fixture", "")


def test_browser_token_file_yields_sid_and_is_not_the_minted_jwt(tmp_path, monkeypatch):
    browser = _jwt({"sid": "sess_frombrowser", "sub": "user_marcus"})
    path = tmp_path / "session.jwt"
    path.write_text(browser, encoding="utf-8")
    minted = _jwt({"sid": "sess_frombrowser", "exp": 60})

    def urlopen(req, timeout):
        assert req.get_method() == "POST"
        assert req.full_url == "https://api.clerk.com/v1/sessions/sess_frombrowser/tokens"
        assert req.get_header("Authorization") == "Bearer sk_test_fixture"
        assert json.loads(req.data.decode()) == {"expires_in_seconds": 60}
        assert browser.encode() not in req.data

        class Response:
            status = 200

            def read(self, limit):
                return json.dumps({"object": "token", "jwt": minted}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    monkeypatch.setattr(clerk.request, "build_opener", lambda *handlers: type("Opener", (), {"open": staticmethod(urlopen)})())
    token = clerk.mint_session_token(
        session_id=clerk.resolve_session_id("", str(path)),
        secret_key="sk_test_fixture",
        expires_in_seconds=60,
        api_base="https://api.clerk.com",
    )
    assert token == minted
    assert token != browser


def test_helper_script_prints_only_the_jwt(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            assert self.path == "/v1/sessions/sess_live/tokens"
            assert self.headers["Authorization"] == "Bearer sk_test_fixture"
            assert json.loads(body)["expires_in_seconds"] == 60
            payload = json.dumps({"jwt": _jwt({"sid": "sess_live"})}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "clerk_session_token.py"), "--session-id", "sess_live"],
            check=False, capture_output=True, text=True,
            env={**os.environ, "CLERK_SECRET_KEY": "sk_test_fixture", "CLERK_API_URL": f"http://{host}:{port}"},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 0
    assert completed.stdout.strip() == _jwt({"sid": "sess_live"})
    assert "sk_test_fixture" not in completed.stdout
    assert "sk_test_fixture" not in completed.stderr


def test_helper_script_hides_clerk_error_bodies(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            payload = json.dumps({"errors": [{"code": "resource_not_found", "message": "sk_test_fixture leaked"}]}).encode()
            self.send_response(404)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "clerk_session_token.py"), "--session-id", "sess_missing", "--expires-in-seconds", "60"],
            check=False, capture_output=True, text=True,
            env={**os.environ, "CLERK_SECRET_KEY": "sk_test_fixture", "CLERK_API_URL": f"http://{host}:{port}"},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 2
    assert "resource_not_found" in completed.stderr
    assert "sk_test_fixture" not in completed.stderr
    assert completed.stdout == ""

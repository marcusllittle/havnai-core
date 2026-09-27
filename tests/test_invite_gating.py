from __future__ import annotations

from types import SimpleNamespace

import pytest

import server.invite as invite


@pytest.fixture(autouse=True)
def restore_invite_state():
    original_gating = invite.INVITE_GATING
    original_request = getattr(invite, "request", None)
    original_jsonify = getattr(invite, "jsonify", None)
    try:
        yield
    finally:
        invite.INVITE_GATING = original_gating
        if original_request is not None:
            invite.request = original_request
        if original_jsonify is not None:
            invite.jsonify = original_jsonify


def test_invite_config_file_does_not_enable_launch_gating(monkeypatch) -> None:
    invite.INVITE_GATING = False
    monkeypatch.setattr(invite, "load_invite_config", lambda: {"alpha": {"enabled": True}})

    assert invite.invite_gating_enabled({"alpha": {"enabled": True}}) is False
    invite_code, error = invite.enforce_invite_limits({"prompt": "launch generation"})

    assert invite_code is None
    assert error is None


def test_explicit_invite_gating_still_rejects_missing_code(monkeypatch) -> None:
    invite.INVITE_GATING = True
    invite.request = SimpleNamespace(headers={})
    invite.jsonify = lambda payload: payload
    monkeypatch.setattr(invite, "load_invite_config", lambda: {"alpha": {"enabled": True}})

    invite_code, error = invite.enforce_invite_limits({"prompt": "gated generation"})

    assert invite_code is None
    assert error == ({"error": "invite_required", "message": "Invite code required."}, 403)

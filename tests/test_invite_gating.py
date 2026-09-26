from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

import invite


def test_invite_config_does_not_enable_gating_by_default(monkeypatch):
    monkeypatch.setattr(invite, "INVITE_GATING", False)

    assert invite.invite_gating_enabled({"launch": {"enabled": True}}) is False


def test_invite_gating_remains_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(invite, "INVITE_GATING", True)

    assert invite.invite_gating_enabled({}) is True

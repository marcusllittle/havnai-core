"""Verified Clerk lifecycle events revoke access without deleting owned data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import sqlite3
import time
from urllib.parse import urlsplit

from svix.webhooks import Webhook, WebhookVerificationError


EVENT_TYPES = frozenset({"user.created", "user.updated", "user.deleted",
                         "session.ended", "session.removed", "session.revoked"})


class LifecycleError(ValueError):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.status = status


@dataclass(frozen=True)
class Config:
    issuer: str
    instance_id: str
    signing_secret: str

    @classmethod
    def from_environment(cls):
        return cls(os.getenv("HAVNAI_CLERK_ISSUER", "").strip().rstrip("/"),
                   os.getenv("HAVNAI_CLERK_INSTANCE_ID", "").strip(),
                   os.getenv("CLERK_WEBHOOK_SIGNING_SECRET", "").strip())


def initialize(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_provider_users (
            issuer TEXT NOT NULL, subject TEXT NOT NULL,
            banned INTEGER NOT NULL DEFAULT 0, deleted INTEGER NOT NULL DEFAULT 0,
            version_ms INTEGER NOT NULL, PRIMARY KEY(issuer,subject)
        );
        CREATE TABLE IF NOT EXISTS account_revoked_sessions (
            issuer TEXT NOT NULL, session_id TEXT NOT NULL,
            revoked_at_ms INTEGER NOT NULL, PRIMARY KEY(issuer,session_id)
        );
        CREATE TABLE IF NOT EXISTS account_lifecycle_events (
            issuer TEXT NOT NULL, event_id TEXT NOT NULL, event_type TEXT NOT NULL,
            payload_hash TEXT NOT NULL, processed_at REAL NOT NULL,
            PRIMARY KEY(issuer,event_id)
        );
        CREATE TRIGGER IF NOT EXISTS account_lifecycle_no_update BEFORE UPDATE ON account_lifecycle_events
            BEGIN SELECT RAISE(ABORT,'account lifecycle audit is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_lifecycle_no_delete BEFORE DELETE ON account_lifecycle_events
            BEGIN SELECT RAISE(ABORT,'account lifecycle audit is append-only'); END;
    """)


def blocked_reason(conn, principal):
    user = conn.execute("SELECT banned,deleted FROM account_provider_users WHERE issuer=? AND subject=?",
                        (principal.issuer, principal.subject)).fetchone()
    if user and (user[0] or user[1]):
        return "account_suspended"
    if conn.execute("SELECT 1 FROM account_revoked_sessions WHERE issuer=? AND session_id=?",
                    (principal.issuer, principal.session_id)).fetchone():
        return "account_session_revoked"
    return None


def webhook(conn, payload: bytes, headers, *, config: Config):
    if len(payload) > 1024 * 1024:
        raise LifecycleError("payload_too_large", 413)
    try:
        issuer = urlsplit(config.issuer)
        if (issuer.scheme != "https" or not issuer.hostname or issuer.username or issuer.password
                or issuer.query or issuer.fragment or not config.instance_id.startswith("ins_")
                or not config.signing_secret.startswith("whsec_")):
            raise ValueError()
        verifier = Webhook(config.signing_secret)
    except ValueError:
        raise LifecycleError("account_webhook_not_configured", 503) from None
    normalized = {key.lower(): value for key, value in headers.items()}
    try:
        verifier.verify(payload, normalized)
        # Svix 2.x verifies the bytes and returns None; parse only after verification.
        event = json.loads(payload)
    except (WebhookVerificationError, ValueError, UnicodeError):
        raise LifecycleError("invalid_account_webhook") from None
    if not isinstance(event, dict) or event.get("instance_id") != config.instance_id:
        raise LifecycleError("wrong_account_provider_instance")
    event_id = normalized.get("svix-id", "")
    kind = event.get("type")
    if not isinstance(kind, str) or not event_id or len(event_id) > 256:
        raise LifecycleError("invalid_account_webhook")
    if kind not in EVENT_TYPES:
        return {"status": "ignored"}
    data = event.get("data")
    timestamp = event.get("timestamp")
    if (event.get("object") != "event" or not isinstance(data, dict)
            or type(timestamp) is not int or not 0 <= timestamp < 2**63):
        raise LifecycleError("invalid_account_webhook")
    identifier = data.get("id")
    prefix = "user_" if kind.startswith("user.") else "sess_"
    if not isinstance(identifier, str) or not identifier.startswith(prefix) or len(identifier) > 256:
        raise LifecycleError("invalid_account_webhook")
    version = timestamp
    if kind in {"user.created", "user.updated"}:
        version = data.get("updated_at")
        if type(data.get("banned")) is not bool or type(version) is not int or not 0 <= version < 2**63:
            raise LifecycleError("invalid_account_webhook")
    digest = hashlib.sha256(payload).hexdigest()
    if conn.in_transaction:
        raise RuntimeError("lifecycle operation requires an idle connection")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        previous = conn.execute("SELECT payload_hash FROM account_lifecycle_events WHERE issuer=? AND event_id=?",
                                (config.issuer, event_id)).fetchone()
        if previous:
            if previous[0] != digest:
                raise LifecycleError("account_webhook_replay_mismatch", 409)
            return {"status": "duplicate"}
        if kind.startswith("session."):
            # Session identifiers are never revived by created/updated events.
            conn.execute("INSERT OR IGNORE INTO account_revoked_sessions VALUES (?,?,?)",
                         (config.issuer, identifier, timestamp))
        else:
            deleted = int(kind == "user.deleted")
            banned = int(bool(data.get("banned"))) if not deleted else 1
            conn.execute("""INSERT INTO account_provider_users VALUES (?,?,?,?,?)
                ON CONFLICT(issuer,subject) DO UPDATE SET
                  deleted=MAX(deleted,excluded.deleted),
                  banned=CASE WHEN excluded.deleted=1 THEN 1
                    WHEN excluded.version_ms>version_ms THEN excluded.banned
                    WHEN excluded.version_ms=version_ms THEN MAX(banned,excluded.banned)
                    ELSE banned END,
                  version_ms=MAX(version_ms,excluded.version_ms)""",
                         (config.issuer, identifier, banned, deleted, version))
        # No profile payloads, emails, tokens, balances, or content are copied.
        conn.execute("INSERT INTO account_lifecycle_events VALUES (?,?,?,?,?)",
                     (config.issuer, event_id, kind, digest, time.time()))
    return {"status": "processed"}

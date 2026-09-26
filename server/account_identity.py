"""Account identity storage used by the commercial account and studio APIs.

Callers MUST supply a provider-verified principal, never request body identity.
Each operation owns its transaction and requires an otherwise idle connection.
No function in this module grants access to or migrates legacy wallet assets.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

import account_lifecycle


class IdentityError(ValueError):
    """Stable API error code; contains no private identity."""


@dataclass(frozen=True)
class VerifiedPrincipal:
    issuer: str
    subject: str
    session_id: str


def initialize(conn: sqlite3.Connection) -> None:
    """Add only identity tables. Existing application tables remain untouched."""
    if conn.in_transaction:
        raise RuntimeError("identity operation requires an idle connection")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','suspended')),
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account_identities (
            issuer TEXT NOT NULL, subject TEXT NOT NULL,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            PRIMARY KEY(issuer, subject)
        );
        CREATE TABLE IF NOT EXISTS wallet_links (
            id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
            wallet TEXT NOT NULL, namespace TEXT NOT NULL DEFAULT 'eip155',
            verified_at REAL NOT NULL, linked_at REAL NOT NULL, unlinked_at REAL,
            UNIQUE(id, account_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS wallet_links_active_address
            ON wallet_links(namespace, wallet) WHERE unlinked_at IS NULL;
        CREATE TABLE IF NOT EXISTS account_wallet_challenges (
            id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
            session_hash TEXT NOT NULL, wallet TEXT NOT NULL,
            purpose TEXT NOT NULL CHECK(purpose IN ('wallet_link','wallet_unlink')),
            link_id TEXT, message TEXT NOT NULL, expires_at REAL NOT NULL,
            used_at REAL
        );
        CREATE TABLE IF NOT EXISTS account_audit_events (
            id TEXT PRIMARY KEY, actor_account_id TEXT NOT NULL REFERENCES accounts(id),
            session_hash TEXT NOT NULL, operation TEXT NOT NULL,
            target_id TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS account_audit_no_update
            BEFORE UPDATE ON account_audit_events BEGIN
                SELECT RAISE(ABORT, 'account audit is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS account_audit_no_delete
            BEFORE DELETE ON account_audit_events BEGIN
                SELECT RAISE(ABORT, 'account audit is append-only'); END;
    """)
    account_lifecycle.initialize(conn)


def _begin(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        raise RuntimeError("identity operation requires an idle connection")
    conn.execute("BEGIN IMMEDIATE")


def _session_hash(principal: VerifiedPrincipal) -> str:
    if not all(isinstance(v, str) and v.strip() for v in
               (principal.issuer, principal.subject, principal.session_id)):
        raise IdentityError("invalid_principal")
    # Length-delimited fields avoid ambiguous bindings; never persist bearer tokens.
    value = "".join(f"{len(v)}:{v}" for v in
                    (principal.issuer, principal.subject, principal.session_id))
    return hashlib.sha256(value.encode()).hexdigest()


def _account(conn: sqlite3.Connection, principal: VerifiedPrincipal) -> str:
    _session_hash(principal)
    _require_provider_access(conn, principal)
    row = conn.execute("""SELECT a.id, a.status FROM accounts a
        JOIN account_identities i ON i.account_id=a.id
        WHERE i.issuer=? AND i.subject=?""", (principal.issuer, principal.subject)).fetchone()
    if row is None:
        raise IdentityError("account_required")
    if row[1] != "active":
        raise IdentityError("account_suspended")
    return str(row[0])


def ensure_account(conn: sqlite3.Connection, principal: VerifiedPrincipal) -> str:
    """Map a verified issuer/subject to an immutable account, including races."""
    _session_hash(principal)
    _begin(conn)
    with conn:
        _require_provider_access(conn, principal)
        existing = conn.execute("SELECT account_id FROM account_identities WHERE issuer=? AND subject=?",
                                (principal.issuer, principal.subject)).fetchone()
        if existing:
            return _account(conn, principal)
        account_id = "acct_" + uuid.uuid4().hex
        conn.execute("INSERT INTO accounts(id, created_at) VALUES (?,?)", (account_id, time.time()))
        conn.execute("INSERT INTO account_identities VALUES (?,?,?)",
                     (principal.issuer, principal.subject, account_id))
        return account_id


def _require_provider_access(conn, principal):
    reason = account_lifecycle.blocked_reason(conn, principal)
    if reason:
        raise IdentityError(reason)


def issue_wallet_challenge(
    conn: sqlite3.Connection, principal: VerifiedPrincipal, *, wallet: str,
    origin: str, chain_id: int, purpose: str = "wallet_link", link_id: str | None = None,
) -> dict:
    """Origin/chain and recent reauthentication MUST be enforced by the adapter."""
    wallet = wallet.lower()
    if not re.fullmatch(r"0x[0-9a-f]{40}", wallet) or int(wallet[2:], 16) == 0:
        raise IdentityError("invalid_wallet")
    parsed = urlsplit(origin)
    secure_origin = parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
    if (not secure_origin or not parsed.netloc or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment or any(c.isspace() for c in origin)):
        raise IdentityError("invalid_origin")
    if type(chain_id) is not int or chain_id <= 0 or chain_id >= 2**63:
        raise IdentityError("invalid_chain")
    if purpose not in {"wallet_link", "wallet_unlink"}:
        raise IdentityError("invalid_purpose")
    _begin(conn)
    with conn:
        account_id = _account(conn, principal)
        if purpose == "wallet_unlink":
            if not conn.execute("""SELECT 1 FROM wallet_links
                WHERE id=? AND account_id=? AND wallet=? AND unlinked_at IS NULL""",
                                (link_id, account_id, wallet)).fetchone():
                raise IdentityError("wallet_link_not_found")
        elif link_id is not None:
            raise IdentityError("invalid_link_context")
        now = time.time()
        expires = now + 300
        challenge_id = secrets.token_hex(32)
        message = "\n".join([
            "HavnAI optional wallet authorization", f"origin: {origin}",
            f"account_id: {account_id}", f"wallet: {wallet}", f"chain_id: {chain_id}",
            f"purpose: {purpose}", f"link_id: {link_id or ''}",
            f"session_binding: {_session_hash(principal)}", f"nonce: {challenge_id}",
            f"issued_at: {now:.6f}", f"expires_at: {expires:.6f}",
            "This action does not move content, credits, tokens, or rewards.",
        ])
        conn.execute("""INSERT INTO account_wallet_challenges
            (id,account_id,session_hash,wallet,purpose,link_id,message,expires_at)
            VALUES (?,?,?,?,?,?,?,?)""", (challenge_id, account_id, _session_hash(principal),
                                         wallet, purpose, link_id, message, expires))
        return {"challenge_id": challenge_id, "message": message, "expires_at": expires}


def complete_wallet_challenge(
    conn: sqlite3.Connection, principal: VerifiedPrincipal, *, challenge_id: str,
    signature: str, purpose: str,
) -> str:
    """Recover an EOA signer and atomically consume proof, mutate link, and audit.

    purpose is selected by the route, never copied from client request fields.
    Returns a link ID. It is not a content ownership or rewards authorization.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    # Cheap validation before signature recovery; repeat state checks under the lock.
    account_id = _account(conn, principal)
    row = conn.execute("""SELECT account_id,session_hash,wallet,purpose,link_id,message,expires_at,used_at
        FROM account_wallet_challenges WHERE id=?""", (challenge_id,)).fetchone()
    if (not row or row[0] != account_id or row[1] != _session_hash(principal)
            or row[3] != purpose or row[7] is not None or row[6] <= time.time()):
        raise IdentityError("invalid_challenge")
    try:
        signer = Account.recover_message(encode_defunct(text=row[5]), signature=signature)
    except Exception as exc:
        raise IdentityError("invalid_signature") from exc
    if signer.lower() != row[2]:
        raise IdentityError("invalid_signature")
    _begin(conn)
    try:
        with conn:
            _account(conn, principal)
            now = time.time()
            consumed = conn.execute("""UPDATE account_wallet_challenges SET used_at=?
                WHERE id=? AND used_at IS NULL AND expires_at>?""", (now, challenge_id, now))
            if consumed.rowcount != 1:
                raise IdentityError("invalid_challenge")
            link_id = row[4]
            if purpose == "wallet_link":
                link_id = "link_" + uuid.uuid4().hex
                conn.execute("""INSERT INTO wallet_links
                    (id,account_id,wallet,verified_at,linked_at) VALUES (?,?,?,?,?)""",
                             (link_id, account_id, row[2], now, now))
            else:
                changed = conn.execute("""UPDATE wallet_links SET unlinked_at=?
                    WHERE id=? AND account_id=? AND wallet=? AND unlinked_at IS NULL""",
                                       (now, link_id, account_id, row[2]))
                if changed.rowcount != 1:
                    raise IdentityError("wallet_link_not_found")
                conn.execute("""UPDATE account_wallet_challenges SET used_at=?
                    WHERE account_id=? AND wallet=? AND used_at IS NULL""", (now, account_id, row[2]))
            conn.execute("INSERT INTO account_audit_events VALUES (?,?,?,?,?,?)",
                         (uuid.uuid4().hex, account_id, _session_hash(principal), purpose, link_id, now))
            return str(link_id)
    except sqlite3.IntegrityError as exc:
        if "wallet_links.namespace, wallet_links.wallet" in str(exc):
            raise IdentityError("wallet_already_linked") from exc
        raise

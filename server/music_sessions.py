"""Short-lived, read-only music sessions. Tokens cannot authorize mutations."""

import hashlib
import secrets
import time

TTL_SECONDS = 8 * 60 * 60


def init_tables(db):
    db.execute("""CREATE TABLE IF NOT EXISTS music_read_sessions (
        token_hash TEXT PRIMARY KEY, wallet TEXT NOT NULL, expires_at REAL NOT NULL
    )""")


def create(db, wallet):
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TTL_SECONDS
    db.execute("DELETE FROM music_read_sessions WHERE expires_at <= ?", (time.time(),))
    db.execute("INSERT INTO music_read_sessions VALUES (?, ?, ?)",
               (hashlib.sha256(token.encode()).hexdigest(), wallet, expires_at))
    db.commit()
    return {"token": token, "wallet": wallet, "expires_at": expires_at}


def resolve(db, token):
    if not isinstance(token, str) or not token or len(token) > 256:
        return None
    row = db.execute(
        "SELECT wallet FROM music_read_sessions WHERE token_hash=? AND expires_at>?",
        (hashlib.sha256(token.encode()).hexdigest(), time.time()),
    ).fetchone()
    return row["wallet"] if row else None

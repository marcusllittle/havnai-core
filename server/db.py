"""Database connection factory supporting SQLite (dev) and PostgreSQL (prod).

Usage
-----
* Development / single-node:  leave DATABASE_URL unset.  SQLite is used at
  the path configured by HAVNAI_DB_PATH (default: <repo-root>/db/ledger.db).
* Production:  set DATABASE_URL to a postgresql:// connection string.  The
  psycopg2 package must be installed (see requirements-prod.txt).

This module is intentionally kept small.  It provides connection primitives
only; all schema definitions and SQL live in app.py and the module files.
Wiring app.py to use get_connection() instead of sqlite3.connect() directly
is tracked as a follow-up task.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "").strip() or None

_BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_PATH = Path(os.getenv("HAVNAI_DB_PATH", str(_BASE_DIR / "db" / "ledger.db")))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_postgres() -> bool:
    """Return True when the runtime is configured to use PostgreSQL."""
    return bool(DATABASE_URL)


def placeholder() -> str:
    """Return the parameter placeholder for the active driver ('?' or '%s')."""
    return "%s" if is_postgres() else "?"


def placeholders(count: int) -> str:
    """Return a comma-separated list of *count* parameter placeholders."""
    p = placeholder()
    return ", ".join([p] * count)


def get_connection() -> Any:
    """Return a ready-to-use database connection for the current environment.

    SQLite connections have WAL mode and foreign keys enabled.
    PostgreSQL connections use RealDictCursor so rows behave like dicts.
    """
    if is_postgres():
        return _get_postgres_connection()
    return _get_sqlite_connection()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_sqlite_connection() -> sqlite3.Connection:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _get_postgres_connection() -> Any:
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL. "
            "Install it with: pip install psycopg2-binary"
        ) from exc

    conn = psycopg2.connect(
        DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    conn.autocommit = False
    return conn

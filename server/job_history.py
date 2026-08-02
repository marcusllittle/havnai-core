"""Per-wallet job history for the Library / Collection page.

The web Library was backed only by browser localStorage. That had three
consequences nobody could see from the server side:

  - clearing site data, using a private window, or opening the site on a
    second device showed an empty Collection even though every job was
    still in the database;
  - the local list was capped at 200 entries, so generation 201 silently
    evicted the oldest one;
  - nothing was keyed to the wallet, so reconnecting restored nothing.

The ``jobs`` table has carried a non-null ``wallet`` on every row from the
start, so the history was always there — it just was not reachable. This
module exposes it.

Follows the module pattern used by ``astra_rewards.py``: ``get_db`` is
injected by ``app.py`` at import time so the logic stays unit-testable
against an in-memory sqlite connection.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Injected by app.py. Declared here so the module imports standalone in tests.
get_db: Optional[Callable[[], Any]] = None

# Task types that produce a video artifact rather than a still.
VIDEO_TASK_TYPES = {"VIDEO_GEN", "ANIMATEDIFF"}

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def _media_type(task_type: Any) -> str:
    """Map a job's task type to what the client needs to render.

    The client only needs to choose between <video> and <img>; it resolves
    the real URL from /result separately.
    """
    return "video" if str(task_type or "").upper() in VIDEO_TASK_TYPES else "image"


def get_wallet_jobs(
    wallet: str, limit: int = DEFAULT_LIMIT, offset: int = 0
) -> Dict[str, Any]:
    """Return ``wallet``'s own jobs, newest first, with a total count.

    The total is returned unclamped so the client can tell the difference
    between "you have 40 generations" and "you have 4,000 and this is page
    one".
    """
    assert get_db is not None, "job_history.get_db was never injected"
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))

    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, model, task_type, status, timestamp, completed_at
        FROM jobs WHERE wallet = ?
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """,
        (wallet, limit, offset),
    ).fetchall()
    total_row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE wallet = ?", (wallet,)
    ).fetchone()

    jobs: List[Dict[str, Any]] = [
        {
            "job_id": row["id"],
            "model": row["model"],
            "task_type": row["task_type"],
            "type": _media_type(row["task_type"]),
            "status": row["status"],
            "timestamp": row["timestamp"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]

    return {
        "wallet": wallet,
        "jobs": jobs,
        "total": int(total_row["n"]) if total_row else 0,
        "limit": limit,
        "offset": offset,
    }

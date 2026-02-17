"""Workflow CRUD and marketplace for HavnAI.

Allows users to create, manage, and publish reusable generation workflows
that can be shared on the marketplace.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Will be injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
WALLET_REGEX: Any  # re.Pattern


def init_workflow_tables(conn: sqlite3.Connection) -> None:
    """Create workflow tables if they don't exist.

    Note: The workflow_registry table is also created by blockchain.py
    for on-chain tracking.  This function ensures it exists and adds
    any additional columns needed for the CRUD / marketplace features.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            creator_wallet TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            config TEXT DEFAULT '{}',
            published INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    # Add optional columns for marketplace features
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(workflow_registry)").fetchall()}
    if "category" not in columns:
        conn.execute("ALTER TABLE workflow_registry ADD COLUMN category TEXT DEFAULT ''")
    if "tags" not in columns:
        conn.execute("ALTER TABLE workflow_registry ADD COLUMN tags TEXT DEFAULT '[]'")
    if "usage_count" not in columns:
        conn.execute("ALTER TABLE workflow_registry ADD COLUMN usage_count INTEGER DEFAULT 0")
    conn.commit()


def create_workflow(
    creator_wallet: str,
    name: str,
    description: str = "",
    config: Optional[Dict[str, Any]] = None,
    category: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a new workflow."""
    conn = get_db()
    now = time.time()
    config_json = json.dumps(config or {})
    tags_json = json.dumps(tags or [])

    cursor = conn.execute(
        """
        INSERT INTO workflow_registry (creator_wallet, name, description, config, published, created_at, updated_at, category, tags, usage_count)
        VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, 0)
        """,
        (creator_wallet, name, description, config_json, now, now, category, tags_json),
    )
    conn.commit()
    workflow_id = cursor.lastrowid
    log_event("Workflow created", workflow_id=workflow_id, wallet=creator_wallet, name=name)
    return get_workflow(workflow_id) or {"id": workflow_id}


def get_workflow(workflow_id: int) -> Optional[Dict[str, Any]]:
    """Get a workflow by ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM workflow_registry WHERE id = ?", (workflow_id,)
    ).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def update_workflow(
    workflow_id: int,
    creator_wallet: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Update a workflow.  Only the creator can update their workflow."""
    conn = get_db()
    existing = conn.execute(
        "SELECT creator_wallet FROM workflow_registry WHERE id = ?", (workflow_id,)
    ).fetchone()
    if not existing:
        return None
    if existing["creator_wallet"] != creator_wallet:
        return None  # Unauthorized

    updates: List[str] = []
    params: List[Any] = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if config is not None:
        updates.append("config = ?")
        params.append(json.dumps(config))
    if category is not None:
        updates.append("category = ?")
        params.append(category)
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))

    if not updates:
        return get_workflow(workflow_id)

    updates.append("updated_at = ?")
    params.append(time.time())
    params.append(workflow_id)

    conn.execute(
        f"UPDATE workflow_registry SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    conn.commit()
    log_event("Workflow updated", workflow_id=workflow_id)
    return get_workflow(workflow_id)


def list_workflows(
    search: Optional[str] = None,
    category: Optional[str] = None,
    creator_wallet: Optional[str] = None,
    published_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List workflows with optional filters and pagination."""
    conn = get_db()
    conditions: List[str] = []
    params: List[Any] = []

    if published_only:
        conditions.append("published = 1")
    if creator_wallet:
        conditions.append("creator_wallet = ?")
        params.append(creator_wallet)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if search:
        conditions.append("(name LIKE ? OR description LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Count total matches
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM workflow_registry {where}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    # Fetch page
    limit_int = max(1, min(limit, 200))
    offset_int = max(0, offset)
    rows = conn.execute(
        f"""
        SELECT * FROM workflow_registry {where}
        ORDER BY updated_at DESC
        LIMIT {limit_int} OFFSET {offset_int}
        """,
        params,
    ).fetchall()

    workflows = [_row_to_dict(row) for row in rows]
    return {
        "workflows": workflows,
        "total": total,
        "limit": limit_int,
        "offset": offset_int,
    }


def publish_workflow(workflow_id: int, creator_wallet: str) -> Optional[Dict[str, Any]]:
    """Publish a workflow to the marketplace."""
    conn = get_db()
    existing = conn.execute(
        "SELECT creator_wallet FROM workflow_registry WHERE id = ?", (workflow_id,)
    ).fetchone()
    if not existing:
        return None
    if existing["creator_wallet"] != creator_wallet:
        return None  # Unauthorized

    conn.execute(
        "UPDATE workflow_registry SET published = 1, updated_at = ? WHERE id = ?",
        (time.time(), workflow_id),
    )
    conn.commit()
    log_event("Workflow published", workflow_id=workflow_id, wallet=creator_wallet)
    return get_workflow(workflow_id)


def browse_marketplace(
    search: Optional[str] = None,
    category: Optional[str] = None,
    sort: str = "popular",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Browse published workflows in the marketplace."""
    conn = get_db()
    conditions = ["published = 1"]
    params: List[Any] = []

    if search:
        conditions.append("(name LIKE ? OR description LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])
    if category:
        conditions.append("category = ?")
        params.append(category)

    where = f"WHERE {' AND '.join(conditions)}"

    # Count
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM workflow_registry {where}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    # Sort
    if sort == "popular":
        order = "usage_count DESC, updated_at DESC"
    elif sort == "newest":
        order = "created_at DESC"
    elif sort == "updated":
        order = "updated_at DESC"
    else:
        order = "usage_count DESC, updated_at DESC"

    limit_int = max(1, min(limit, 200))
    offset_int = max(0, offset)
    rows = conn.execute(
        f"""
        SELECT * FROM workflow_registry {where}
        ORDER BY {order}
        LIMIT {limit_int} OFFSET {offset_int}
        """,
        params,
    ).fetchall()

    workflows = [_row_to_dict(row) for row in rows]
    return {
        "workflows": workflows,
        "total": total,
        "limit": limit_int,
        "offset": offset_int,
        "sort": sort,
    }


def increment_usage(workflow_id: int) -> None:
    """Increment the usage counter for a workflow."""
    conn = get_db()
    conn.execute(
        "UPDATE workflow_registry SET usage_count = usage_count + 1 WHERE id = ?",
        (workflow_id,),
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a workflow DB row to a serializable dict."""
    d = dict(row)
    # Parse JSON fields
    for key in ("config", "tags"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    # Ensure boolean-like published field
    d["published"] = bool(d.get("published", 0))
    return d

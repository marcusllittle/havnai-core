"""Gallery marketplace for HavnAI — browse & purchase generated outputs.

Allows users to list completed job outputs for sale. Other users can
browse the gallery and purchase items using credits. The seller receives
credits when a sale completes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
WALLET_REGEX: Any  # re.Pattern


def init_gallery_tables(conn: sqlite3.Connection) -> None:
    """Create gallery tables if they don't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            seller_wallet TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            description TEXT DEFAULT '',
            price_credits REAL NOT NULL CHECK (price_credits > 0),
            category TEXT DEFAULT '',
            asset_type TEXT DEFAULT 'image',
            model TEXT DEFAULT '',
            prompt TEXT DEFAULT '',
            listed INTEGER NOT NULL DEFAULT 1,
            sold INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL REFERENCES gallery_listings(id),
            buyer_wallet TEXT NOT NULL,
            seller_wallet TEXT NOT NULL,
            price_paid REAL NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def create_listing(
    job_id: str,
    seller_wallet: str,
    title: str,
    price_credits: float,
    description: str = "",
    category: str = "",
    asset_type: str = "image",
    model: str = "",
    prompt: str = "",
) -> Dict[str, Any]:
    """Create a new gallery listing from a completed job."""
    conn = get_db()
    now = time.time()

    # Prevent duplicate listings for the same job
    existing = conn.execute(
        "SELECT id FROM gallery_listings WHERE job_id = ? AND seller_wallet = ? AND listed = 1",
        (job_id, seller_wallet),
    ).fetchone()
    if existing:
        return get_listing(existing["id"]) or {}

    cursor = conn.execute(
        """
        INSERT INTO gallery_listings
            (job_id, seller_wallet, title, description, price_credits,
             category, asset_type, model, prompt, listed, sold, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
        """,
        (job_id, seller_wallet, title, description, price_credits,
         category, asset_type, model, prompt, now, now),
    )
    conn.commit()
    listing_id = cursor.lastrowid
    log_event("Gallery listing created", listing_id=listing_id, job_id=job_id, wallet=seller_wallet)
    return get_listing(listing_id) or {"id": listing_id}


def get_listing(listing_id: int) -> Optional[Dict[str, Any]]:
    """Get a listing by ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM gallery_listings WHERE id = ?", (listing_id,)
    ).fetchone()
    if not row:
        return None
    return _listing_to_dict(row)


def delist(listing_id: int, seller_wallet: str) -> bool:
    """Remove a listing (seller only).  Returns True if delisted."""
    conn = get_db()
    cur = conn.execute(
        "UPDATE gallery_listings SET listed = 0, updated_at = ? WHERE id = ? AND seller_wallet = ? AND listed = 1",
        (time.time(), listing_id, seller_wallet),
    )
    conn.commit()
    if cur.rowcount > 0:
        log_event("Gallery listing removed", listing_id=listing_id, wallet=seller_wallet)
        return True
    return False


def browse_gallery(
    search: Optional[str] = None,
    category: Optional[str] = None,
    asset_type: Optional[str] = None,
    sort: str = "newest",
    limit: int = 24,
    offset: int = 0,
) -> Dict[str, Any]:
    """Browse listed (unsold) gallery items."""
    conn = get_db()
    conditions = ["listed = 1", "sold = 0"]
    params: List[Any] = []

    if search:
        conditions.append("(title LIKE ? OR description LIKE ? OR prompt LIKE ? OR model LIKE ?)")
        params.extend([f"%{search}%"] * 4)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if asset_type:
        conditions.append("asset_type = ?")
        params.append(asset_type)

    where = f"WHERE {' AND '.join(conditions)}"

    count_row = conn.execute(
        f"SELECT COUNT(*) FROM gallery_listings {where}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    if sort == "price_low":
        order = "price_credits ASC, created_at DESC"
    elif sort == "price_high":
        order = "price_credits DESC, created_at DESC"
    elif sort == "oldest":
        order = "created_at ASC"
    else:  # newest
        order = "created_at DESC"

    limit_int = max(1, min(limit, 200))
    offset_int = max(0, offset)
    rows = conn.execute(
        f"""
        SELECT * FROM gallery_listings {where}
        ORDER BY {order}
        LIMIT {limit_int} OFFSET {offset_int}
        """,
        params,
    ).fetchall()

    listings = [_listing_to_dict(row) for row in rows]
    return {
        "listings": listings,
        "total": total,
        "limit": limit_int,
        "offset": offset_int,
        "sort": sort,
    }


def purchase_listing(listing_id: int, buyer_wallet: str) -> Dict[str, Any]:
    """Purchase a gallery listing.

    Returns a dict with ``ok``, ``sale``, and optionally ``error``.
    Credit deduction and seller credit are handled by the caller in app.py
    so that we don't create a circular dependency on credits.py.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM gallery_listings WHERE id = ? AND listed = 1 AND sold = 0",
        (listing_id,),
    ).fetchone()
    if not row:
        return {"ok": False, "error": "listing_not_found"}

    listing = _listing_to_dict(row)

    if listing["seller_wallet"].lower() == buyer_wallet.lower():
        return {"ok": False, "error": "cannot_buy_own_listing"}

    now = time.time()

    # Mark as sold
    conn.execute(
        "UPDATE gallery_listings SET sold = 1, updated_at = ? WHERE id = ?",
        (now, listing_id),
    )

    # Record sale
    conn.execute(
        """
        INSERT INTO gallery_sales (listing_id, buyer_wallet, seller_wallet, price_paid, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (listing_id, buyer_wallet, listing["seller_wallet"], listing["price_credits"], now),
    )
    conn.commit()

    sale = {
        "listing_id": listing_id,
        "buyer_wallet": buyer_wallet,
        "seller_wallet": listing["seller_wallet"],
        "price_paid": listing["price_credits"],
        "job_id": listing["job_id"],
    }
    log_event(
        "Gallery sale completed",
        listing_id=listing_id,
        buyer=buyer_wallet,
        seller=listing["seller_wallet"],
        price=listing["price_credits"],
    )
    return {"ok": True, "sale": sale, "listing": listing}


def seller_listings(wallet: str, include_sold: bool = False) -> List[Dict[str, Any]]:
    """Get all listings for a seller."""
    conn = get_db()
    if include_sold:
        rows = conn.execute(
            "SELECT * FROM gallery_listings WHERE seller_wallet = ? ORDER BY created_at DESC",
            (wallet,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM gallery_listings WHERE seller_wallet = ? AND listed = 1 ORDER BY created_at DESC",
            (wallet,),
        ).fetchall()
    return [_listing_to_dict(row) for row in rows]


def buyer_purchases(wallet: str) -> List[Dict[str, Any]]:
    """Get purchase history for a buyer."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT s.*, l.job_id, l.title, l.asset_type, l.model, l.prompt
        FROM gallery_sales s
        JOIN gallery_listings l ON l.id = s.listing_id
        WHERE s.buyer_wallet = ?
        ORDER BY s.created_at DESC
        """,
        (wallet,),
    ).fetchall()
    return [dict(row) for row in rows]


def _listing_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a gallery listing row to a dict."""
    d = dict(row)
    d["listed"] = bool(d.get("listed", 0))
    d["sold"] = bool(d.get("sold", 0))
    return d

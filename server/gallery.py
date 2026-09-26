"""Gallery marketplace for HavnAI — browse & purchase generated outputs.

Allows users to list completed job outputs for sale. Other users can
browse the gallery and purchase items using credits. The seller receives
credits when a sale completes.

Ownership model: each listing tracks an ``owner_wallet`` that represents
the current exclusive owner.  On purchase the owner transfers to the
buyer who may then re-list, delist, or hold the asset.  A provenance
log (``gallery_ownership_log``) records every transfer.
"""

from __future__ import annotations

import math
import credits
import account_marketplace

import json
import os
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
WALLET_REGEX: Any  # re.Pattern
build_result_payload: Callable[[str], Optional[Dict[str, Any]]]
resolve_job_metadata: Callable[[str], Optional[Dict[str, Any]]]


def _account_owned_job(conn: sqlite3.Connection, job_id: str) -> bool:
    """Account ownership overrides all historical wallet listing records."""
    return bool(conn.execute(
        "SELECT 1 FROM jobs WHERE id=? AND owner_account_id IS NOT NULL", (job_id,)
    ).fetchone())


def legacy_public_gallery_enabled() -> bool:
    """Return whether pre-account wallet gallery routes are publicly browseable."""
    return os.getenv("HAVNAI_LEGACY_GALLERY_PUBLIC_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def init_gallery_tables(conn: sqlite3.Connection) -> None:
    """Create gallery tables if they don't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            seller_wallet TEXT NOT NULL,
            owner_wallet TEXT NOT NULL DEFAULT '',
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery_ownership_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            listing_id INTEGER,
            from_wallet TEXT NOT NULL,
            to_wallet TEXT NOT NULL,
            event_type TEXT NOT NULL,
            price_credits REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ownership_log_job ON gallery_ownership_log(job_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ownership_log_to ON gallery_ownership_log(to_wallet)"
    )

    # Migration: add owner_wallet column if missing (table may predate this column)
    try:
        conn.execute("SELECT owner_wallet FROM gallery_listings LIMIT 0")
    except sqlite3.OperationalError:
        try:
            conn.execute(
                "ALTER TABLE gallery_listings ADD COLUMN owner_wallet TEXT NOT NULL DEFAULT ''"
            )
        except Exception:
            pass

    # Migration: backfill owner_wallet for existing rows
    try:
        conn.execute(
            "UPDATE gallery_listings SET owner_wallet = seller_wallet WHERE owner_wallet = '' OR owner_wallet IS NULL"
        )
    except Exception:
        pass

    conn.commit()
    account_marketplace.initialize(conn)


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

    if _account_owned_job(conn, job_id):
        raise ValueError("account_owned_job")

    # Prevent duplicate active listings for the same job. Sold rows stay in the
    # table for history and should not block a fresh relist.
    existing = conn.execute(
        "SELECT id FROM gallery_listings WHERE job_id = ? AND listed = 1 AND sold = 0",
        (job_id,),
    ).fetchone()
    if existing:
        listing = get_listing(existing["id"]) or {"id": existing["id"]}
        listing["already_listed"] = True
        return listing

    cursor = conn.execute(
        """
        INSERT INTO gallery_listings
            (job_id, seller_wallet, owner_wallet, title, description, price_credits,
             category, asset_type, model, prompt, listed, sold, created_at, updated_at)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE id=? AND owner_account_id IS NOT NULL)
        """,
        (job_id, seller_wallet, seller_wallet, title, description, price_credits,
         category, asset_type, model, prompt, now, now, job_id),
    )

    if cursor.rowcount != 1:
        conn.rollback()
        raise ValueError("account_owned_job")
    listing_id = cursor.lastrowid

    # Record the mint event in ownership log
    conn.execute(
        """
        INSERT INTO gallery_ownership_log
            (job_id, listing_id, from_wallet, to_wallet, event_type, price_credits, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, listing_id, "", seller_wallet, "mint", 0, now),
    )

    conn.commit()
    log_event("Gallery listing created", listing_id=listing_id, job_id=job_id, wallet=seller_wallet)
    return get_listing(listing_id) or {"id": listing_id}


def get_listing(listing_id: int) -> Optional[Dict[str, Any]]:
    """Get a listing by ID."""
    if not legacy_public_gallery_enabled():
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM gallery_listings WHERE id = ?", (listing_id,)
    ).fetchone()
    if not row:
        return None
    if _account_owned_job(conn, row["job_id"]):
        return None
    return _listing_to_dict(row)


def delist(listing_id: int, owner_wallet: str) -> bool:
    """Remove a listing (current owner only).  Returns True if delisted."""
    if not legacy_public_gallery_enabled():
        return False
    conn = get_db()
    cur = conn.execute(
        """UPDATE gallery_listings SET listed = 0, updated_at = ?
        WHERE id = ? AND owner_wallet = ? AND listed = 1
        AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.id=gallery_listings.job_id
                        AND jobs.owner_account_id IS NOT NULL)""",
        (time.time(), listing_id, owner_wallet),
    )
    conn.commit()
    if cur.rowcount > 0:
        log_event("Gallery listing removed", listing_id=listing_id, wallet=owner_wallet)
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
    if not legacy_public_gallery_enabled():
        limit_int = max(1, min(limit, 200))
        return {"listings": [], "total": 0, "limit": limit_int, "offset": max(0, offset), "sort": sort}
    conn = get_db()
    conditions = ["listed = 1", "sold = 0", """NOT EXISTS
        (SELECT 1 FROM jobs WHERE jobs.id=gallery_listings.job_id
         AND jobs.owner_account_id IS NOT NULL)"""]
    params: List[Any] = []

    if search:
        conditions.append("(title LIKE ? OR description LIKE ? OR model LIKE ?)")
        params.extend([f"%{search}%"] * 3)
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

    listings = [_listing_to_dict(row, strip_prompt=True) for row in rows]
    return {
        "listings": listings,
        "total": total,
        "limit": limit_int,
        "offset": offset_int,
        "sort": sort,
    }


def purchase_listing(listing_id: int, buyer_wallet: str, *, settle_credits: bool = False,
                     expected_price: Optional[float] = None) -> Dict[str, Any]:
    """Transfer ownership, sale history and (for API purchases) credits atomically."""
    if not legacy_public_gallery_enabled():
        return {"ok": False, "error": "listing_not_found"}
    conn = get_db()
    buyer_wallet = buyer_wallet.strip().lower()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM gallery_listings WHERE id=? AND listed=1 AND sold=0", (listing_id,)).fetchone()
        if not row:
            conn.rollback()
            return {"ok": False, "error": "listing_not_found"}
        listing = dict(row)
        if _account_owned_job(conn, listing["job_id"]):
            conn.rollback()
            return {"ok": False, "error": "listing_not_found"}
        previous_owner = str(listing["owner_wallet"] or listing["seller_wallet"]).lower()
        price = float(listing["price_credits"])
        if not math.isfinite(price) or price <= 0 or (expected_price is not None and price != expected_price):
            conn.rollback()
            return {"ok": False, "error": "listing_price_changed"}
        if previous_owner == buyer_wallet:
            conn.rollback()
            return {"ok": False, "error": "cannot_buy_own_listing"}
        now = time.time()
        remaining = None
        if settle_credits:
            debit = conn.execute("""UPDATE credits SET balance=balance-?, total_spent=total_spent+?, updated_at=?
                WHERE wallet=? AND balance>=?""", (price, price, now, buyer_wallet, price))
            balance = conn.execute("SELECT balance FROM credits WHERE wallet=?", (buyer_wallet,)).fetchone()
            remaining = float(balance[0]) if balance else 0.0
            if debit.rowcount != 1:
                conn.rollback()
                return {"ok": False, "error": "insufficient_credits", "balance": remaining, "cost": price}
            credits.deposit_in_transaction(conn, previous_owner, price)
        conn.execute("UPDATE gallery_listings SET sold=1, listed=0, owner_wallet=?, updated_at=? WHERE id=?", (buyer_wallet, now, listing_id))
        conn.execute("INSERT INTO gallery_sales (listing_id,buyer_wallet,seller_wallet,price_paid,created_at) VALUES (?,?,?,?,?)",
                     (listing_id, buyer_wallet, previous_owner, price, now))
        conn.execute("""INSERT INTO gallery_ownership_log (job_id,listing_id,from_wallet,to_wallet,event_type,price_credits,created_at)
                     VALUES (?,?,?,?,?,?,?)""", (listing["job_id"], listing_id, previous_owner, buyer_wallet, "sale", price, now))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    sale = {"listing_id": listing_id, "buyer_wallet": buyer_wallet, "seller_wallet": previous_owner,
            "price_paid": price, "job_id": listing["job_id"]}
    log_event("Gallery sale completed", listing_id=listing_id, buyer=buyer_wallet, seller=previous_owner, price=price)
    return {"ok": True, "sale": sale, "remaining_credits": remaining}


def relist_owned_asset(
    job_id: str,
    owner_wallet: str,
    title: str,
    price_credits: float,
    description: str = "",
    category: str = "",
) -> Dict[str, Any]:
    """Re-list an asset that the caller currently owns (purchased previously).

    This creates a new listing row.  The caller must be the current owner
    (verified via the most recent sold listing or ownership log).
    """
    if not legacy_public_gallery_enabled():
        return {"ok": False, "error": "not_owner"}
    conn = get_db()

    if _account_owned_job(conn, job_id):
        return {"ok": False, "error": "not_owner"}

    # Verify ownership: find the most recent listing for this job where owner_wallet matches
    ownership_row = conn.execute(
        """
        SELECT id, asset_type, model, prompt, owner_wallet FROM gallery_listings
        WHERE job_id = ?
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    ).fetchone()

    if not ownership_row or str(ownership_row["owner_wallet"]).lower() != owner_wallet.lower():
        return {"ok": False, "error": "not_owner"}

    # Prevent duplicate active listings
    active = conn.execute(
        "SELECT id FROM gallery_listings WHERE job_id = ? AND listed = 1 AND sold = 0",
        (job_id,),
    ).fetchone()
    if active:
        return {"ok": False, "error": "already_listed", "listing_id": active["id"]}

    now = time.time()
    asset_type = ownership_row["asset_type"] or "image"
    model = ownership_row["model"] or ""
    prompt = ownership_row["prompt"] or ""

    cursor = conn.execute(
        """
        INSERT INTO gallery_listings
            (job_id, seller_wallet, owner_wallet, title, description, price_credits,
             category, asset_type, model, prompt, listed, sold, created_at, updated_at)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE id=? AND owner_account_id IS NOT NULL)
        """,
        (job_id, owner_wallet, owner_wallet, title, description, price_credits,
         category, asset_type, model, prompt, now, now, job_id),
    )

    if cursor.rowcount != 1:
        conn.rollback()
        raise ValueError("account_owned_job")
    listing_id = cursor.lastrowid

    # Log re-list event
    conn.execute(
        """
        INSERT INTO gallery_ownership_log
            (job_id, listing_id, from_wallet, to_wallet, event_type, price_credits, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, listing_id, owner_wallet, owner_wallet, "relist", price_credits, now),
    )

    conn.commit()
    log_event("Gallery asset re-listed", listing_id=listing_id, job_id=job_id, wallet=owner_wallet)
    return {"ok": True, "listing": get_listing(listing_id)}


def get_owned_assets(wallet: str) -> List[Dict[str, Any]]:
    """Get all assets currently owned by a wallet (most recent listing per job_id)."""
    if not legacy_public_gallery_enabled():
        return []
    conn = get_db()
    rows = conn.execute(
        """SELECT gl.* FROM gallery_listings gl WHERE LOWER(gl.owner_wallet)=?
           AND gl.id=(SELECT latest.id FROM gallery_listings latest WHERE latest.job_id=gl.job_id
                      ORDER BY latest.updated_at DESC, latest.id DESC LIMIT 1)
           ORDER BY gl.updated_at DESC, gl.id DESC""", (wallet.strip().lower(),)
    ).fetchall()
    return [_listing_to_dict(row) for row in rows if not _account_owned_job(conn, row["job_id"])]


def get_ownership_history(job_id: str) -> List[Dict[str, Any]]:
    """Get the full ownership provenance chain for an asset."""
    if not legacy_public_gallery_enabled():
        return []
    conn = get_db()
    if _account_owned_job(conn, job_id):
        return []
    rows = conn.execute(
        """
        SELECT * FROM gallery_ownership_log
        WHERE job_id = ?
        ORDER BY created_at ASC
        """,
        (job_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_asset_owner(job_id: str) -> Optional[str]:
    """Return the current owner wallet for a given job_id, or None if never listed."""
    if not legacy_public_gallery_enabled():
        return None
    conn = get_db()
    if _account_owned_job(conn, job_id):
        return None
    row = conn.execute(
        """
        SELECT owner_wallet FROM gallery_listings
        WHERE job_id = ?
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    ).fetchone()
    if row:
        return row["owner_wallet"]
    return None


def seller_listings(wallet: str, include_sold: bool = False) -> List[Dict[str, Any]]:
    """Get all listings for a seller."""
    if not legacy_public_gallery_enabled():
        return []
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
    return [_listing_to_dict(row) for row in rows if not _account_owned_job(conn, row["job_id"])]


def buyer_purchases(wallet: str) -> List[Dict[str, Any]]:
    """Get purchase history for a buyer."""
    if not legacy_public_gallery_enabled():
        return []
    conn = get_db()
    rows = conn.execute(
        """
        SELECT s.*, l.job_id, l.title, l.asset_type, l.model, l.prompt, l.owner_wallet
        FROM gallery_sales s
        JOIN gallery_listings l ON l.id = s.listing_id
        WHERE s.buyer_wallet = ?
        ORDER BY s.created_at DESC
        """,
        (wallet,),
    ).fetchall()
    purchases: List[Dict[str, Any]] = []
    for row in rows:
        if _account_owned_job(conn, row["job_id"]):
            continue
        purchase = dict(row)
        purchases.append(_attach_result_urls(purchase))
    return purchases


def _listing_to_dict(row: sqlite3.Row, strip_prompt: bool = False) -> Dict[str, Any]:
    """Convert a gallery listing row to a dict."""
    d = dict(row)
    d["listed"] = bool(d.get("listed", 0))
    d["sold"] = bool(d.get("sold", 0))
    # Derive a clear status string
    if d["sold"]:
        d["status"] = "sold"
    elif d["listed"]:
        d["status"] = "active"
    else:
        d["status"] = "delisted"
    # Ensure owner_wallet is present (migration compat)
    if not d.get("owner_wallet"):
        d["owner_wallet"] = d.get("seller_wallet", "")
    if strip_prompt:
        d.pop("prompt", None)
    return _attach_result_urls(d)


def _attach_result_urls(record: Dict[str, Any]) -> Dict[str, Any]:
    """Attach image/video/preview URLs using the job's output artifacts when available."""
    job_id = str(record.get("job_id") or "").strip()
    if not job_id:
        return record

    metadata_resolver = globals().get("resolve_job_metadata")
    if callable(metadata_resolver):
        try:
            metadata = metadata_resolver(job_id)  # type: ignore[misc]
        except Exception:
            metadata = None
        if isinstance(metadata, dict):
            canonical_model = str(metadata.get("model_name") or "").strip()
            canonical_prompt = str(metadata.get("prompt") or "").strip()
            if canonical_model:
                record["model"] = canonical_model
                record["model_key"] = metadata.get("model_key")
                record["model_tier"] = metadata.get("tier")
                record["model_reward_weight"] = metadata.get("reward_weight")
                record["model_credit_cost"] = metadata.get("credit_cost")
                record["model_pipeline"] = metadata.get("pipeline")
                record["model_task_type"] = metadata.get("task_type")
            if canonical_prompt and "prompt" in record:
                record["prompt"] = canonical_prompt

    resolver = globals().get("build_result_payload")
    if not callable(resolver):
        return record

    try:
        payload = resolver(job_id)  # type: ignore[misc]
    except Exception:
        return record

    if not payload:
        return record

    image_url = payload.get("image_url")
    video_url = payload.get("video_url")
    if image_url:
        record["image_url"] = image_url
    if video_url:
        record["video_url"] = video_url
    if image_url or video_url:
        record["preview_url"] = video_url or image_url
    return record

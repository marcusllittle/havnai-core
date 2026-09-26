"""Account marketplace: ownership, settlement and retry receipts share a transaction."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import warnings

import account_ledger
import adult_content
import artifact_lifecycle
import platform_v1


class MarketplaceError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


def initialize(conn):
    for table, fields in {
        "gallery_listings": {"owner_account_id": "TEXT REFERENCES accounts(id)", "seller_account_id": "TEXT REFERENCES accounts(id)",
                             "creator_account_id": "TEXT REFERENCES accounts(id)", "price_units": "INTEGER", "artifact_id": "TEXT REFERENCES artifacts(id)",
                             "adult_content": "INTEGER NOT NULL DEFAULT 0", "adult_policy_reason": "TEXT DEFAULT ''"},
        "gallery_sales": {"buyer_account_id": "TEXT REFERENCES accounts(id)", "seller_account_id": "TEXT REFERENCES accounts(id)",
                          "price_units": "INTEGER", "account_sale_id": "TEXT REFERENCES account_credit_sales(sale_id)"},
        "gallery_ownership_log": {"from_account_id": "TEXT REFERENCES accounts(id)", "to_account_id": "TEXT REFERENCES accounts(id)", "price_units": "INTEGER"},
    }.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in fields.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_marketplace_requests (
            account_id TEXT NOT NULL REFERENCES accounts(id), operation TEXT NOT NULL,
            request_key TEXT NOT NULL, payload_hash TEXT NOT NULL, result TEXT NOT NULL,
            created_at REAL NOT NULL, PRIMARY KEY(account_id,operation,request_key));
        CREATE UNIQUE INDEX IF NOT EXISTS account_gallery_active_job ON gallery_listings(job_id)
            WHERE owner_account_id IS NOT NULL AND listed=1 AND sold=0;
    """)
    conn.commit()


def _intent(key, body):
    if not isinstance(key, str) or not key.strip() or len(key) > 128:
        raise MarketplaceError("idempotency_key_required")
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _previous(conn, account, operation, key, digest):
    row = conn.execute("SELECT payload_hash,result FROM account_marketplace_requests WHERE account_id=? AND operation=? AND request_key=?",
                       (account, operation, key)).fetchone()
    if row:
        if row[0] != digest:
            raise MarketplaceError("idempotency_conflict", 409)
        return json.loads(row[1])


def _remember(conn, account, operation, key, digest, result):
    conn.execute("INSERT INTO account_marketplace_requests VALUES (?,?,?,?,?,?)",
                 (account, operation, key, digest, json.dumps(result), time.time()))
    return result


def _price(value):
    if type(value) is not int or not 0 < value <= account_ledger.MAX_UNITS:
        raise MarketplaceError("invalid_price_units")
    return value


def _active_account(conn, account):
    if not conn.execute("SELECT 1 FROM accounts WHERE id=? AND status='active'", (account,)).fetchone():
        raise MarketplaceError("account_unavailable", 403)


def _artifact(conn, job_id, artifact_id):
    if artifact_lifecycle.deleted(conn, job_id):
        raise MarketplaceError("marketplace_artifact_unavailable", 409)
    row = conn.execute("SELECT id,kind,content_type,path FROM artifacts WHERE id=? AND job_id=?", (artifact_id, job_id)).fetchone()
    if (not row or row["kind"] != "image" or row["content_type"] not in {"image/png", "image/jpeg", "image/webp"}
            or not Path(row["path"]).is_file()):
        raise MarketplaceError("marketplace_artifact_unavailable", 409)


def create(conn, account, key, body):
    if not isinstance(body, dict) or set(body) - {"job_id", "artifact_id", "title", "description", "category", "price_units"}:
        raise MarketplaceError("invalid_listing")
    units = _price(body.get("price_units"))
    for field, limit in (("job_id", 128), ("artifact_id", 128), ("title", 200), ("description", 2000), ("category", 100)):
        value = body.get(field, "")
        if not isinstance(value, str) or len(value) > limit or (field in {"job_id", "artifact_id", "title"} and not value.strip()):
            raise MarketplaceError("invalid_listing")
    digest = _intent(key, body)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _active_account(conn, account)
        old = _previous(conn, account, "list", key, digest)
        if old:
            return old
        job = conn.execute("SELECT * FROM jobs WHERE id=? AND owner_account_id=?", (body["job_id"], account)).fetchone()
        if not job:
            raise MarketplaceError("job_not_found", 404)
        # Preserve the existing marketplace's image-generation eligibility. Face
        # swaps and other task types are not silently made tradable by account auth.
        if job["task_type"] != "IMAGE_GEN" or platform_v1.canonical_job_state(job["status"]) != "succeeded":
            raise MarketplaceError("marketplace_ineligible", 409)
        reservation = conn.execute("SELECT state FROM account_credit_reservations WHERE job_id=?", (job["id"],)).fetchone()
        imported = conn.execute("""SELECT 1 FROM account_import_job_transfers t
            JOIN account_import_receipts r ON r.snapshot_id=t.snapshot_id AND r.account_id=t.account_id
            WHERE t.job_id=? LIMIT 1""", (job["id"],)).fetchone()
        # An imported legacy creation has no account-generation charge. Its
        # signed import receipt establishes provenance across subsequent resales.
        # An existing uncaptured reservation still blocks publication.
        if (reservation and reservation[0] != "captured") or (not reservation and not imported):
            raise MarketplaceError("marketplace_unsettled", 409)
        _artifact(conn, job["id"], body["artifact_id"])
        adult_reason = adult_content.from_job(job, body.get("title"), body.get("description"), body.get("category"))
        if adult_reason:
            raise MarketplaceError("adult_content_restricted", 409)
        if conn.execute("SELECT 1 FROM gallery_listings WHERE job_id=? AND listed=1 AND sold=0 AND owner_account_id IS NOT NULL", (job["id"],)).fetchone():
            raise MarketplaceError("already_listed", 409)
        now = time.time()
        listing_id = conn.execute("""INSERT INTO gallery_listings
            (job_id,seller_wallet,owner_wallet,title,description,price_credits,category,asset_type,model,prompt,
             listed,sold,created_at,updated_at,owner_account_id,seller_account_id,creator_account_id,price_units,artifact_id,
             adult_content,adult_policy_reason)
            VALUES (?,'','',?,?,?,?,'image',?,'',1,0,?,?,?,?,?,?,?,0,'')""",
            (job["id"], body["title"].strip(), body.get("description", "").strip(), units / account_ledger.SCALE,
             body.get("category", "").strip(), job["model"], now, now, account, account,
             job["creator_account_id"], units, body["artifact_id"])).lastrowid
        conn.execute("""INSERT INTO gallery_ownership_log
            (job_id,listing_id,from_wallet,to_wallet,event_type,price_credits,created_at,from_account_id,to_account_id,price_units)
            VALUES (?,?,'','','account_list',0,?,?,?,0)""", (job["id"], listing_id, now, account, account))
        return _remember(conn, account, "list", key, digest, {"listing_id": listing_id, "job_id": job["id"], "price_units": units})


def purchase(conn, buyer, listing_id, key, body):
    if not isinstance(body, dict) or set(body) != {"expected_price_units"}:
        raise MarketplaceError("invalid_purchase")
    expected = _price(body["expected_price_units"])
    digest = _intent(key, {"listing_id": listing_id, **body})
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _active_account(conn, buyer)
        old = _previous(conn, buyer, "purchase", key, digest)
        if old:
            return old
        listing = conn.execute("""SELECT l.* FROM gallery_listings l JOIN jobs j ON j.id=l.job_id
            WHERE l.id=? AND l.listed=1 AND l.sold=0 AND l.owner_account_id IS NOT NULL
            AND j.owner_account_id=l.owner_account_id""", (listing_id,)).fetchone()
        if not listing:
            raise MarketplaceError("listing_not_found", 404)
        if listing["price_units"] != expected:
            raise MarketplaceError("listing_price_changed", 409)
        seller = listing["owner_account_id"]
        _artifact(conn, listing["job_id"], listing["artifact_id"])
        sale_id = "gallery-sale-" + uuid.uuid4().hex
        receipt = account_ledger.settle_sale_in_transaction(conn, buyer, seller, expected, sale_id=sale_id)
        now = time.time()
        conn.execute("UPDATE jobs SET owner_account_id=?,updated_at=? WHERE id=? AND owner_account_id=?",
                     (buyer, now, listing["job_id"], seller))
        # A repurchased creation must reappear even if this buyer hid it during
        # an earlier period of ownership.
        conn.execute("DELETE FROM account_collection_hidden WHERE account_id=? AND job_id=?", (buyer, listing["job_id"]))
        conn.execute("UPDATE gallery_listings SET owner_account_id=?,listed=0,sold=1,updated_at=? WHERE id=?",
                     (buyer, now, listing_id))
        conn.execute("""INSERT INTO gallery_sales
            (listing_id,buyer_wallet,seller_wallet,price_paid,created_at,buyer_account_id,seller_account_id,price_units,account_sale_id)
            VALUES (?,'','',?,?,?,?,?,?)""", (listing_id, expected / account_ledger.SCALE, now, buyer, seller, expected, sale_id))
        conn.execute("""INSERT INTO gallery_ownership_log
            (job_id,listing_id,from_wallet,to_wallet,event_type,price_credits,created_at,from_account_id,to_account_id,price_units)
            VALUES (?,?,'','','account_sale',?,?,?,?,?)""",
            (listing["job_id"], listing_id, expected / account_ledger.SCALE, now, seller, buyer, expected))
        return _remember(conn, buyer, "purchase", key, digest,
                         {**receipt, "listing_id": listing_id, "job_id": listing["job_id"], "price_units": expected})


def delist(conn, account, listing_id):
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        _active_account(conn, account)
        row = conn.execute("""SELECT l.id FROM gallery_listings l JOIN jobs j ON j.id=l.job_id
            WHERE l.id=? AND l.owner_account_id=? AND j.owner_account_id=?""", (listing_id, account, account)).fetchone()
        if not row:
            raise MarketplaceError("listing_not_found", 404)
        conn.execute("UPDATE gallery_listings SET listed=0,updated_at=? WHERE id=?", (time.time(), listing_id))


_PUBLIC_FROM = """FROM gallery_listings l JOIN jobs j ON j.id=l.job_id
    JOIN accounts owner ON owner.id=j.owner_account_id
    WHERE l.owner_account_id=j.owner_account_id AND owner.status='active'
    AND l.listed=1 AND l.sold=0 AND l.price_units IS NOT NULL
    AND COALESCE(l.adult_content,0)=0"""


def _public_listing(row):
    # An explicit allowlist prevents prompt, source assets, paths and internal
    # account identifiers from appearing in public catalog responses.
    return {**{key: row[key] for key in ("id", "title", "description", "category", "asset_type", "model", "price_units", "created_at")},
            "scale": account_ledger.SCALE, "status": "active",
            "preview_url": f"/v2/marketplace/listings/{row['id']}/preview"}


def browse(conn, *, search="", category="", sort="newest", limit=24, offset=0):
    if not isinstance(search, str) or len(search) > 200 or not isinstance(category, str) or len(category) > 100:
        raise MarketplaceError("invalid_search")
    orders = {"newest": "l.created_at DESC,l.id DESC", "oldest": "l.created_at,l.id",
              "price_low": "l.price_units,l.id DESC", "price_high": "l.price_units DESC,l.id DESC"}
    if sort not in orders or type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000:
        raise MarketplaceError("invalid_pagination")
    where, params = _PUBLIC_FROM, []
    if search.strip():
        escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where += " AND (l.title LIKE ? ESCAPE '\\' OR l.description LIKE ? ESCAPE '\\' OR l.model LIKE ? ESCAPE '\\')"
        params.extend([f"%{escaped}%"] * 3)
    if category:
        where += " AND l.category=?"
        params.append(category)
    total = conn.execute("SELECT COUNT(*) " + where, params).fetchone()[0]
    rows = conn.execute("SELECT l.* " + where + " ORDER BY " + orders[sort] + " LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
    return {"listings": [_public_listing(row) for row in rows], "total": total, "limit": limit, "offset": offset, "sort": sort}


def detail(conn, listing_id):
    row = conn.execute("SELECT l.* " + _PUBLIC_FROM + " AND l.id=?", (listing_id,)).fetchone()
    if not row:
        raise MarketplaceError("listing_not_found", 404)
    return _public_listing(row)


def account_listings(conn, account, *, limit=24, offset=0):
    _pagination(limit, offset)
    where = """FROM gallery_listings l JOIN jobs j ON j.id=l.job_id
        WHERE j.owner_account_id=? AND l.owner_account_id=?
        AND NOT EXISTS (SELECT 1 FROM artifact_lifecycle d WHERE d.job_id=j.id AND d.restored_at IS NULL)
        AND l.id=(SELECT MAX(latest.id) FROM gallery_listings latest
                  WHERE latest.job_id=l.job_id AND latest.owner_account_id IS NOT NULL)"""
    params = (account, account)
    total = conn.execute("SELECT COUNT(*) " + where, params).fetchone()[0]
    rows = conn.execute("SELECT l.* " + where + " ORDER BY l.updated_at DESC,l.id DESC LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
    listings = []
    for row in rows:
        item = _public_listing(row)
        item.update({"status": "active" if row["listed"] and not row["sold"] else "sold" if row["sold"] else "delisted",
                     "job_id": row["job_id"], "artifact_id": row["artifact_id"],
                     "owner_account_id": account, "creator_account_id": row["creator_account_id"],
                     "original_url": f"/v2/artifacts/{row['artifact_id']}/content"})
        # The public preview endpoint closes when a listing is sold/delisted.
        if item["status"] != "active":
            item["preview_url"] = None
        listings.append(item)
    return {"listings": listings, "total": total, "limit": limit, "offset": offset}


def _pagination(limit, offset):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000:
        raise MarketplaceError("invalid_pagination")


def receipts(conn, account, *, limit=24, offset=0):
    _pagination(limit, offset)
    where = """FROM account_credit_sales s JOIN gallery_sales g ON g.account_sale_id=s.sale_id
        JOIN gallery_listings l ON l.id=g.listing_id
        WHERE s.buyer_account_id=? OR s.seller_account_id=?"""
    total = conn.execute("SELECT COUNT(*) " + where, (account, account)).fetchone()[0]
    rows = conn.execute("SELECT s.*,g.listing_id,l.job_id,l.title " + where + " ORDER BY s.created_at DESC,s.sale_id DESC LIMIT ? OFFSET ?",
                        (account, account, limit, offset)).fetchall()
    return {"receipts": [{"id": row["sale_id"], "listing_id": row["listing_id"], "job_id": row["job_id"],
                          "title": row["title"], "price_units": row["units"], "scale": account_ledger.SCALE,
                          "direction": "purchase" if row["buyer_account_id"] == account else "sale",
                          "ledger_entry_id": row["debit_entry_id"] if row["buyer_account_id"] == account else row["credit_entry_id"],
                          "created_at": row["created_at"]} for row in rows],
            "total": total, "limit": limit, "offset": offset}


def preview(conn, listing_id, *, outputs_dir):
    row = conn.execute("SELECT l.* " + _PUBLIC_FROM + " AND l.id=?", (listing_id,)).fetchone()
    if not row:
        raise MarketplaceError("listing_not_found", 404)
    source = conn.execute("SELECT path,content_type FROM artifacts WHERE id=? AND job_id=? AND kind='image'",
                          (row["artifact_id"], row["job_id"])).fetchone()
    if not source or source["content_type"] not in {"image/png", "image/jpeg", "image/webp"}:
        raise MarketplaceError("preview_unavailable", 409)
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise MarketplaceError("preview_unavailable", 503) from exc
    try:
        path = Path(source["path"]).resolve()
        if not path.is_relative_to(Path(outputs_dir).resolve()) or path.stat().st_size > 32 * 1024 * 1024:
            raise MarketplaceError("preview_unavailable", 409)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as original:
                if original.width * original.height > 20_000_000:
                    raise MarketplaceError("preview_unavailable", 409)
                image = ImageOps.exif_transpose(original)
                image.thumbnail((640, 640))
                # A fresh RGB canvas deliberately drops EXIF, text and profiles.
                clean = Image.new("RGB", image.size, "#101820")
                if "A" in image.getbands():
                    clean.paste(image.convert("RGB"), mask=image.getchannel("A"))
                else:
                    clean.paste(image.convert("RGB"))
                draw = ImageDraw.Draw(clean)
                draw.rectangle((0, max(0, clean.height - 24), clean.width, clean.height), fill="#101820")
                draw.text((8, max(0, clean.height - 19)), "HavnAI preview", fill="white")
                output = io.BytesIO()
                clean.save(output, format="JPEG", quality=80)
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MarketplaceError("preview_unavailable", 409) from exc
    # Encoding happens without a write lock. Recheck publication/ownership before
    # releasing bytes, including delists and purchases during image processing.
    current = conn.execute("SELECT l.artifact_id,l.owner_account_id " + _PUBLIC_FROM + " AND l.id=?", (listing_id,)).fetchone()
    if not current or tuple(current) != (row["artifact_id"], row["owner_account_id"]):
        raise MarketplaceError("listing_not_found", 404)
    return output.getvalue()

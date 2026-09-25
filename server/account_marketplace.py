"""Account marketplace: ownership, settlement and retry receipts share a transaction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

import account_ledger
import platform_v1


class MarketplaceError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


def initialize(conn):
    for table, fields in {
        "gallery_listings": {"owner_account_id": "TEXT REFERENCES accounts(id)", "seller_account_id": "TEXT REFERENCES accounts(id)",
                             "creator_account_id": "TEXT REFERENCES accounts(id)", "price_units": "INTEGER", "artifact_id": "TEXT REFERENCES artifacts(id)"},
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
        captured = conn.execute("SELECT 1 FROM account_credit_reservations WHERE job_id=? AND state='captured'", (job["id"],)).fetchone()
        if not captured:
            raise MarketplaceError("marketplace_unsettled", 409)
        _artifact(conn, job["id"], body["artifact_id"])
        if conn.execute("SELECT 1 FROM gallery_listings WHERE job_id=? AND listed=1 AND sold=0 AND owner_account_id IS NOT NULL", (job["id"],)).fetchone():
            raise MarketplaceError("already_listed", 409)
        now = time.time()
        listing_id = conn.execute("""INSERT INTO gallery_listings
            (job_id,seller_wallet,owner_wallet,title,description,price_credits,category,asset_type,model,prompt,
             listed,sold,created_at,updated_at,owner_account_id,seller_account_id,creator_account_id,price_units,artifact_id)
            VALUES (?,'','',?,?,?,?,'image',?,'',1,0,?,?,?,?,?,?,?)""",
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

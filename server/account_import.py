"""Inventory and immutable selection snapshots. Neither operation transfers assets."""
import hashlib
import json
import secrets
import time
from decimal import Decimal, InvalidOperation

import account_identity
import account_ledger


class MigrationError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


def initialize(conn):
    if conn.in_transaction:
        raise RuntimeError("import initialization requires an idle connection")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_import_snapshots (
            id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
            link_id TEXT NOT NULL REFERENCES wallet_links(id), session_hash TEXT NOT NULL,
            request_key TEXT NOT NULL, selection_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL, digest TEXT NOT NULL,
            created_at REAL NOT NULL, expires_at REAL NOT NULL,
            UNIQUE(account_id, request_key)
        );
        CREATE TRIGGER IF NOT EXISTS account_import_snapshot_no_update
            BEFORE UPDATE ON account_import_snapshots BEGIN
                SELECT RAISE(ABORT, 'import snapshot is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS account_import_snapshot_no_delete
            BEFORE DELETE ON account_import_snapshots BEGIN
                SELECT RAISE(ABORT, 'import snapshot is immutable'); END;
    """)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _rows(conn, query, args):
    cursor = conn.execute(query, args)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def prepare(conn, principal, link_id, request_key, selection):
    """Persist exactly the selected inventory, without changing ownership/balances.

    This is not a wallet challenge or transfer authorization. The eventual
    signed execution must recompute dependencies inside its own write lock.
    """
    if (not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 128
            or not isinstance(selection, dict) or set(selection) != {"job_ids", "include_credits"}):
        raise MigrationError("invalid_import_selection")
    ids, include_credits = selection["job_ids"], selection["include_credits"]
    if (not isinstance(ids, list) or len(ids) > 100 or type(include_credits) is not bool
            or any(not isinstance(value, str) or not value.strip() or len(value) > 200 for value in ids)
            or len(set(ids)) != len(ids) or (not ids and not include_credits)):
        raise MigrationError("invalid_import_selection")
    ids = sorted(ids)
    selection_hash = _digest({"link_id": link_id, "job_ids": ids, "include_credits": include_credits})
    if conn.in_transaction:
        raise RuntimeError("import preparation requires an idle connection")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        account = account_identity._account(conn, principal)
        session = account_identity._session_hash(principal)
        previous = conn.execute("""SELECT id,selection_hash,session_hash FROM account_import_snapshots
            WHERE account_id=? AND request_key=?""", (account, request_key)).fetchone()
        if previous:
            if previous[2] != session:
                raise MigrationError("import_snapshot_not_found", 404)
            if previous[1] != selection_hash:
                raise MigrationError("idempotency_conflict", 409)
            return _load(conn, account, session, previous[0])
        inventory = _preview(conn, account, link_id, limit=100, selected=ids)
        if inventory["total"] != len(ids) or inventory["eligible_count"] != len(ids):
            raise MigrationError("import_selection_unavailable", 409)
        if include_credits and inventory["credits"]["exclusion"] is not None:
            raise MigrationError("import_credits_require_review", 409)
        # Persist hashes, not private prompts or filesystem paths. Full rows
        # bind state even when an edit leaves the public summary unchanged.
        jobs = []
        for job in inventory["jobs"]:
            job_id = job["id"]
            state = {
                "job": _rows(conn, "SELECT * FROM jobs WHERE id=?", (job_id,)),
                "listings": _rows(conn, "SELECT * FROM gallery_listings WHERE job_id=? ORDER BY id", (job_id,)),
                "artifacts": _rows(conn, "SELECT * FROM artifacts WHERE job_id=? ORDER BY id", (job_id,)),
            }
            jobs.append({**job, "state_digest": _digest(state)})
        credits = None
        if include_credits:
            state = _rows(conn, "SELECT * FROM credits WHERE LOWER(wallet)=? ORDER BY wallet", (inventory["wallet"],))
            credits = {**inventory["credits"], "state_digest": _digest(state)}
        now = time.time()
        snapshot = {"version": 1, "id": "import_" + secrets.token_hex(24),
                    "account_id": account, "link_id": link_id, "wallet": inventory["wallet"],
                    "session_binding": session, "created_at": now, "expires_at": now + 300,
                    "scope": (["generation_history"] if ids else []) + (["available_credits"] if include_credits else []),
                    "jobs": jobs, "credits": credits}
        digest = _digest(snapshot)
        conn.execute("""INSERT INTO account_import_snapshots
            (id,account_id,link_id,session_hash,request_key,selection_hash,snapshot_json,digest,created_at,expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (snapshot["id"], account, link_id, session, request_key,
                selection_hash, _json(snapshot), digest, now, snapshot["expires_at"]))
        return {**snapshot, "digest": digest, "transfer_authorized": False}


def _load(conn, account, session, snapshot_id):
    row = conn.execute("""SELECT s.snapshot_json,s.digest,s.expires_at FROM account_import_snapshots s
        JOIN wallet_links w ON w.id=s.link_id AND w.account_id=s.account_id
        WHERE s.id=? AND s.account_id=? AND s.session_hash=? AND w.unlinked_at IS NULL""",
        (snapshot_id, account, session)).fetchone()
    if not row:
        raise MigrationError("import_snapshot_not_found", 404)
    if row[2] <= time.time():
        raise MigrationError("import_snapshot_expired", 409)
    return {**json.loads(row[0]), "digest": row[1], "transfer_authorized": False}


def load(conn, principal, snapshot_id):
    if conn.in_transaction:
        raise RuntimeError("import snapshot read requires an idle connection")
    conn.execute("BEGIN")
    with conn:
        account = account_identity._account(conn, principal)
        return _load(conn, account, account_identity._session_hash(principal), snapshot_id)


def preview(conn, account, link_id, *, limit=50, offset=0):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000:
        raise MigrationError("invalid_pagination")
    if conn.in_transaction:
        raise RuntimeError("import preview requires an idle connection")
    # One read snapshot keeps ownership, exclusion counts and balances consistent.
    conn.execute("BEGIN")
    with conn:
        return _preview(conn, account, link_id, limit=limit, offset=offset)


def _preview(conn, account, link_id, *, limit=50, offset=0, selected=None):
    link = conn.execute("""SELECT w.wallet FROM wallet_links w JOIN accounts a ON a.id=w.account_id
        WHERE w.id=? AND w.account_id=? AND w.namespace='eip155' AND w.unlinked_at IS NULL AND a.status='active'""",
        (link_id, account)).fetchone()
    if not link:
        raise MigrationError("wallet_link_not_found", 404)
    wallet = link[0].lower()
    # Gallery transfers override the original job wallet. Never present a
    # sold-away job as importable merely because the caller created it.
    inventory = """WITH candidates AS (
        SELECT j.id,j.status,j.task_type,j.timestamp,j.owner_account_id,
               LOWER(j.wallet) AS creator_wallet,g.id AS listing_id,
               LOWER(COALESCE(NULLIF(g.owner_wallet,''),g.seller_wallet)) AS gallery_wallet,
               g.owner_account_id AS gallery_account,
               EXISTS(SELECT 1 FROM gallery_listings active WHERE active.job_id=j.id
                      AND active.listed=1 AND active.sold=0) AS active_listing
        FROM jobs j LEFT JOIN gallery_listings g ON g.id=(
            SELECT latest.id FROM gallery_listings latest WHERE latest.job_id=j.id
            ORDER BY latest.updated_at DESC,latest.id DESC LIMIT 1)
        WHERE LOWER(j.wallet)=? OR LOWER(COALESCE(NULLIF(g.owner_wallet,''),g.seller_wallet))=?
    ), classified AS (
        SELECT *,CASE
            WHEN owner_account_id IS NOT NULL OR gallery_account IS NOT NULL THEN 'already_account_owned'
            WHEN listing_id IS NOT NULL AND COALESCE(gallery_wallet,'')<>? THEN 'not_current_owner'
            WHEN active_listing=1 THEN 'active_listing'
            WHEN LOWER(TRIM(COALESCE(status,''))) NOT IN
                ('success','completed','done','succeeded','failed','error','cancelled','canceled','expired') THEN 'job_not_final'
            ELSE NULL END AS exclusion
        FROM candidates)
    """
    params = (wallet, wallet, wallet)
    if selected is not None:
        inventory += ", selected AS (SELECT * FROM classified WHERE id IN (" + ",".join("?" for _ in selected) + ")) "
        source = "selected"
        params = (*params, *selected)
    else:
        source = "classified"
    groups = conn.execute(inventory + f"SELECT exclusion,COUNT(*) FROM {source} GROUP BY exclusion", params).fetchall()
    exclusions = {row[0]: row[1] for row in groups if row[0] is not None}
    eligible = sum(row[1] for row in groups if row[0] is None)
    total = sum(row[1] for row in groups)
    rows = conn.execute(inventory + f"SELECT id,status,task_type,timestamp,exclusion FROM {source} ORDER BY id LIMIT ? OFFSET ?",
                        (*params, limit, offset)).fetchall()
    jobs = [{"id": row[0], "status": row[1], "type": row[2], "created_at": row[3],
             "eligible": row[4] is None, "exclusion": row[4]} for row in rows]
    balances = conn.execute("SELECT balance FROM credits WHERE LOWER(wallet)=?", (wallet,)).fetchall()
    units, credit_exclusion = 0, None
    if len(balances) > 1:
        credit_exclusion = "ambiguous_wallet_balance"
    else:
        try:
            amount = Decimal(str(balances[0][0])) if balances else Decimal(0)
            scaled = amount * account_ledger.SCALE
            if not scaled.is_finite() or scaled < 0 or scaled > account_ledger.MAX_UNITS:
                credit_exclusion = "invalid_wallet_balance"
            elif scaled != scaled.to_integral_value():
                credit_exclusion = "balance_precision_review"
            else:
                units = int(scaled)
        except (InvalidOperation, ValueError):
            credit_exclusion = "invalid_wallet_balance"
    return {"link_id": link_id, "wallet": wallet, "read_only": True,
            "scope": ["generation_history", "available_credits"],
            "jobs": jobs, "total": total, "eligible_count": eligible, "exclusions": exclusions,
            "limit": limit, "offset": offset,
            "credits": {"available_units": units if credit_exclusion is None else None,
                        "scale": account_ledger.SCALE, "exclusion": credit_exclusion},
            "confirmation_required": True,
            "notice": "This preview does not transfer content or credits. Import requires a separate signed confirmation of an unchanged snapshot."}

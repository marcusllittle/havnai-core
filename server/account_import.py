"""Explicit legacy import inventory, confirmation and internal atomic execution."""
import hashlib
import json
import secrets
import time
from urllib.parse import urlsplit
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
        CREATE TABLE IF NOT EXISTS account_import_challenges (
            id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL UNIQUE REFERENCES account_import_snapshots(id),
            origin TEXT NOT NULL, chain_id INTEGER NOT NULL, message TEXT NOT NULL,
            expires_at REAL NOT NULL, used_at REAL
        );
        CREATE TABLE IF NOT EXISTS account_import_receipts (
            snapshot_id TEXT PRIMARY KEY REFERENCES account_import_snapshots(id),
            account_id TEXT NOT NULL REFERENCES accounts(id),
            challenge_id TEXT NOT NULL UNIQUE REFERENCES account_import_challenges(id),
            receipt_json TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS account_import_receipt_immutable_update
            BEFORE UPDATE ON account_import_receipts BEGIN
                SELECT RAISE(ABORT, 'import receipt is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS account_import_receipt_immutable_delete
            BEFORE DELETE ON account_import_receipts BEGIN
                SELECT RAISE(ABORT, 'import receipt is immutable'); END;
        CREATE TABLE IF NOT EXISTS account_import_job_transfers (
            snapshot_id TEXT NOT NULL REFERENCES account_import_receipts(snapshot_id),
            job_id TEXT NOT NULL REFERENCES jobs(id),
            account_id TEXT NOT NULL REFERENCES accounts(id),
            previous_owner_wallet TEXT NOT NULL, creator_wallet TEXT,
            PRIMARY KEY(snapshot_id,job_id)
        );
        CREATE INDEX IF NOT EXISTS account_import_job_lookup ON account_import_job_transfers(job_id);
        CREATE TRIGGER IF NOT EXISTS account_import_job_no_update
            BEFORE UPDATE ON account_import_job_transfers BEGIN
                SELECT RAISE(ABORT, 'import job provenance is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS account_import_job_no_delete
            BEFORE DELETE ON account_import_job_transfers BEGIN
                SELECT RAISE(ABORT, 'import job provenance is immutable'); END;
    """)
    # Receipts predate the indexed per-job provenance table. Backfill only the
    # server-authored transfer records, never infer imports from wallet links.
    with conn:
        conn.execute("""INSERT OR IGNORE INTO account_import_job_transfers
            (snapshot_id,job_id,account_id,previous_owner_wallet,creator_wallet)
            SELECT r.snapshot_id,json_extract(j.value,'$.id'),r.account_id,
                   json_extract(j.value,'$.previous_owner_wallet'),json_extract(j.value,'$.creator_wallet')
            FROM account_import_receipts r,json_each(r.receipt_json,'$.jobs') j
            JOIN jobs existing ON existing.id=json_extract(j.value,'$.id')""")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _rows(conn, query, args):
    cursor = conn.execute(query, args)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _job_digest(conn, job_id):
    return _digest({
        "job": _rows(conn, "SELECT * FROM jobs WHERE id=?", (job_id,)),
        "listings": _rows(conn, "SELECT * FROM gallery_listings WHERE job_id=? ORDER BY id", (job_id,)),
        "artifacts": _rows(conn, "SELECT * FROM artifacts WHERE job_id=? ORDER BY id", (job_id,)),
    })


def _credit_digest(conn, wallet):
    return _digest(_rows(conn, "SELECT * FROM credits WHERE LOWER(wallet)=? ORDER BY wallet", (wallet,)))


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
            jobs.append({**job, "state_digest": _job_digest(conn, job["id"])})
        credits = None
        if include_credits:
            credits = {**inventory["credits"], "state_digest": _credit_digest(conn, inventory["wallet"])}
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


def revalidate_in_transaction(conn, principal, snapshot_id):
    """Caller must hold BEGIN IMMEDIATE through eventual proof consumption/transfer.

    Only the selected inventory is checked. This does not expand selection or
    certify the as-yet unsupported publication/reference migration dependencies.
    """
    if not conn.in_transaction:
        raise RuntimeError("import revalidation requires a write transaction")
    account = account_identity._account(conn, principal)
    snapshot = _load(conn, account, account_identity._session_hash(principal), snapshot_id)
    ids = [job["id"] for job in snapshot["jobs"]]
    inventory = _preview(conn, account, snapshot["link_id"], limit=100, selected=ids)
    if (inventory["wallet"] != snapshot["wallet"] or inventory["total"] != len(ids)
            or inventory["eligible_count"] != len(ids)):
        raise MigrationError("import_snapshot_changed", 409)
    try:
        if any(_job_digest(conn, job["id"]) != job["state_digest"] for job in snapshot["jobs"]):
            raise MigrationError("import_snapshot_changed", 409)
        if snapshot["credits"] is not None:
            if (inventory["credits"]["exclusion"] is not None
                    or inventory["credits"]["available_units"] != snapshot["credits"]["available_units"]
                    or _credit_digest(conn, snapshot["wallet"]) != snapshot["credits"]["state_digest"]):
                raise MigrationError("import_snapshot_changed", 409)
    except MigrationError:
        raise
    except (TypeError, ValueError) as exc:
        # Non-finite legacy numbers or malformed state must never be treated as
        # an unchanged selection.
        raise MigrationError("import_snapshot_changed", 409) from exc
    return snapshot


def issue_challenge(conn, principal, snapshot_id, *, origin, chain_id):
    """Issue a separate EIP-191 message; a link signature cannot authorize import.

    The HTTP adapter must enforce recent authentication and its Origin allowlist.
    No execution endpoint is enabled until all migration dependencies are covered.
    """
    if not isinstance(origin, str):
        raise MigrationError("invalid_origin", 403)
    try:
        parsed = urlsplit(origin)
    except ValueError as exc:
        raise MigrationError("invalid_origin", 403) from exc
    secure = parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
    if (not secure or not parsed.netloc or parsed.username or parsed.password or parsed.path
            or parsed.query or parsed.fragment or any(c.isspace() for c in origin)):
        raise MigrationError("invalid_origin", 403)
    if type(chain_id) is not int or chain_id not in {1, 11155111}:
        raise MigrationError("unsupported_chain")
    if conn.in_transaction:
        raise RuntimeError("import challenge requires an idle connection")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        snapshot = revalidate_in_transaction(conn, principal, snapshot_id)
        existing = conn.execute("""SELECT id,origin,chain_id,message,expires_at,used_at
            FROM account_import_challenges WHERE snapshot_id=?""", (snapshot_id,)).fetchone()
        if existing:
            if existing[1] != origin or existing[2] != chain_id:
                raise MigrationError("import_challenge_context_changed", 409)
            if existing[5] is not None or existing[4] <= time.time():
                raise MigrationError("invalid_import_challenge", 409)
            return {"challenge_id": existing[0], "message": existing[3], "expires_at": existing[4]}
        nonce = secrets.token_hex(32)
        expires = snapshot["expires_at"]
        credits = snapshot["credits"]["available_units"] if snapshot["credits"] else 0
        message = "\n".join([
            "HavnAI legacy content import authorization", f"origin: {origin}",
            f"account_id: {snapshot['account_id']}", f"wallet: {snapshot['wallet']}",
            f"chain_id: {chain_id}", "purpose: legacy_import", f"link_id: {snapshot['link_id']}",
            f"session_binding: {snapshot['session_binding']}", f"snapshot_id: {snapshot_id}",
            f"snapshot_digest: {snapshot['digest']}", f"nonce: {nonce}",
            f"job_ids: {_json([job['id'] for job in snapshot['jobs']])}",
            f"credit_units: {credits}", f"credit_scale: {account_ledger.SCALE}",
            f"expires_at: {expires:.6f}",
            "Authorize only this unchanged selection to move from the linked wallet to this account.",
            "No on-chain tokens, rewards, or unselected content are transferred.",
        ])
        conn.execute("""INSERT INTO account_import_challenges
            (id,snapshot_id,origin,chain_id,message,expires_at) VALUES (?,?,?,?,?,?)""",
            (nonce, snapshot_id, origin, chain_id, message, expires))
        return {"challenge_id": nonce, "message": message, "expires_at": expires}


def execute(conn, principal, snapshot_id, *, challenge_id, signature, origin, chain_id):
    """Internal signed transfer; not exposed by an HTTP route during rollout.

    The adapter must enforce recent authentication and an allowlisted Origin.
    Jobs with publication dependencies and balances with legacy Stripe history
    are refused until their explicit migration/provenance flows are implemented.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    if (not isinstance(challenge_id, str) or not isinstance(signature, str)
            or len(signature) > 1024 or type(chain_id) is not int):
        raise MigrationError("invalid_import_proof")
    if conn.in_transaction:
        raise RuntimeError("import execution requires an idle connection")
    # Verify the actual stored message, never a message/digest supplied by a client.
    row = conn.execute("""SELECT c.message,s.account_id,s.session_hash,s.snapshot_json,c.origin,c.chain_id
        FROM account_import_challenges c JOIN account_import_snapshots s ON s.id=c.snapshot_id
        WHERE c.id=? AND s.id=?""", (challenge_id, snapshot_id)).fetchone()
    account = account_identity._account(conn, principal)
    session = account_identity._session_hash(principal)
    if not row or row[1] != account or row[2] != session or row[4] != origin or row[5] != chain_id:
        raise MigrationError("invalid_import_proof")
    try:
        signer = Account.recover_message(encode_defunct(text=row[0]), signature=signature)
    except Exception as exc:
        raise MigrationError("invalid_import_signature") from exc
    if signer.lower() != json.loads(row[3])["wallet"]:
        raise MigrationError("invalid_import_signature")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        # Revocation/suspension and persisted proof context can change while the
        # ECDSA recovery runs. Recheck under the same lock as ownership and money.
        account_identity._account(conn, principal)
        current = conn.execute("""SELECT message,origin,chain_id,expires_at,used_at FROM account_import_challenges
            WHERE id=? AND snapshot_id=?""", (challenge_id, snapshot_id)).fetchone()
        if not current or tuple(current[:3]) != (row[0], origin, chain_id):
            raise MigrationError("invalid_import_proof")
        prior = conn.execute("SELECT receipt_json FROM account_import_receipts WHERE snapshot_id=? AND account_id=? AND challenge_id=?",
                             (snapshot_id, account, challenge_id)).fetchone()
        if prior:
            return json.loads(prior[0])
        if current[4] is not None or current[3] <= time.time():
            raise MigrationError("invalid_import_challenge", 409)
        snapshot = revalidate_in_transaction(conn, principal, snapshot_id)
        ids = [job["id"] for job in snapshot["jobs"]]
        for job_id in ids:
            if conn.execute("SELECT 1 FROM music_publications WHERE job_id=? LIMIT 1", (job_id,)).fetchone():
                raise MigrationError("import_publication_migration_required", 409)
        if snapshot["credits"] is not None and conn.execute(
                "SELECT 1 FROM stripe_payments WHERE LOWER(wallet)=? LIMIT 1", (snapshot["wallet"],)).fetchone():
            raise MigrationError("import_payment_provenance_required", 409)
        now = time.time()
        consumed = conn.execute("UPDATE account_import_challenges SET used_at=? WHERE id=? AND used_at IS NULL AND expires_at>?",
                                (now, challenge_id, now))
        if consumed.rowcount != 1:
            raise MigrationError("invalid_import_challenge", 409)
        transferred = []
        for job_id in ids:
            before = conn.execute("SELECT wallet,creator_account_id FROM jobs WHERE id=?", (job_id,)).fetchone()
            # Keep the original wallet attribution. A purchaser must not become
            # the creator simply by importing a previously purchased creation.
            creator = before[1] or (account if str(before[0] or "").lower() == snapshot["wallet"] else None)
            changed = conn.execute("UPDATE jobs SET owner_account_id=?,creator_account_id=? WHERE id=? AND owner_account_id IS NULL",
                                   (account, creator, job_id))
            if changed.rowcount != 1:
                raise MigrationError("import_snapshot_changed", 409)
            transferred.append({"id": job_id, "previous_owner_wallet": snapshot["wallet"],
                                "creator_wallet": before[0], "owner_account_id": account})
        credit_receipt = None
        units = snapshot["credits"]["available_units"] if snapshot["credits"] else 0
        if units:
            credit_receipt = account_ledger.import_legacy_in_transaction(conn, account, snapshot["wallet"],
                                                                       units, migration_id=snapshot_id)
        receipt = {"id": snapshot_id, "account_id": account, "wallet": snapshot["wallet"],
                   "digest": snapshot["digest"], "jobs": transferred, "credit_units": units,
                   "credit_receipt": credit_receipt, "created_at": now}
        conn.execute("INSERT INTO account_import_receipts VALUES (?,?,?,?,?)",
                     (snapshot_id, account, challenge_id, _json(receipt), now))
        conn.executemany("INSERT INTO account_import_job_transfers VALUES (?,?,?,?,?)",
            [(snapshot_id, job["id"], account, job["previous_owner_wallet"], job["creator_wallet"]) for job in transferred])
        conn.execute("INSERT INTO account_audit_events VALUES (?,?,?,?,?,?)",
                     (secrets.token_hex(16), account, session, "legacy_import", snapshot_id, now))
        return receipt


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

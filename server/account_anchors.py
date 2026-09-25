"""Account-owned reusable face references; wallet-era anchors stay separate."""
import re
import time


SLUG = re.compile(r"[a-z0-9_-]{1,64}")
TAG = re.compile(r"\[\s*identity\s+anchor\s*:\s*([A-Za-z0-9_-]{1,64})\s*\]", re.I)
OPEN = re.compile(r"\[\s*identity\s+anchor\b", re.I)


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS account_identity_anchors (
        account_id TEXT NOT NULL REFERENCES accounts(id),
        slug TEXT NOT NULL, display_name TEXT NOT NULL,
        asset_id TEXT NOT NULL REFERENCES assets(id), created_at REAL NOT NULL,
        PRIMARY KEY(account_id, slug))""")
    conn.commit()


def save(conn, account_id, slug, body):
    if (not SLUG.fullmatch(slug) or not isinstance(body, dict)
        or set(body) != {"asset_id", "display_name"}
        or not isinstance(body["asset_id"], str) or not 1 <= len(body["asset_id"]) <= 128
        or not isinstance(body["display_name"], str) or not 1 <= len(body["display_name"].strip()) <= 100):
        raise ValueError("invalid_identity_anchor")
    name = body["display_name"].strip()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        if not conn.execute("SELECT 1 FROM assets WHERE id=? AND owner_account_id=? AND kind='image'",
                            (body["asset_id"], account_id)).fetchone():
            raise ValueError("asset_not_found")
        old = conn.execute("SELECT * FROM account_identity_anchors WHERE account_id=? AND slug=?", (account_id, slug)).fetchone()
        if old:
            if old["asset_id"] != body["asset_id"] or old["display_name"] != name:
                raise ValueError("identity_anchor_slug_exists")
        else:
            conn.execute("INSERT INTO account_identity_anchors VALUES (?,?,?,?,?)",
                         (account_id, slug, name, body["asset_id"], time.time()))
        return dict(conn.execute("SELECT slug,display_name,asset_id,created_at FROM account_identity_anchors WHERE account_id=? AND slug=?",
                                 (account_id, slug)).fetchone())


def resolve(conn, account_id, prompt, job_type, explicit_face):
    """Return a cleaned prompt and stable private asset reference for this job."""
    if not OPEN.search(prompt):
        return prompt, None, None
    matches = list(TAG.finditer(prompt))
    if len(matches) != 1 or len(OPEN.findall(prompt)) != 1:
        raise ValueError("invalid_identity_anchor_tag")
    if job_type != "image" or explicit_face:
        raise ValueError("invalid_face_conditioning")
    cleaned = TAG.sub("", prompt).strip()
    if not cleaned:
        raise ValueError("missing_prompt")
    slug = matches[0][1].lower()
    row = conn.execute("SELECT asset_id FROM account_identity_anchors WHERE account_id=? AND slug=?", (account_id, slug)).fetchone()
    if not row:
        raise ValueError("identity_anchor_not_found")
    return cleaned, row["asset_id"], slug

"""Account-owned reusable workflows; no wallet claims grant access."""
import json
import time

import account_identity


class WorkflowError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.status = status


def payload(data, *, partial=False):
    if not isinstance(data, dict) or not data or set(data) - {"name", "description", "config", "category", "tags", "published"}:
        raise WorkflowError("invalid_workflow")
    result = {} if partial else {"description": "", "config": {}, "category": "", "tags": [], "published": False}
    result.update(data)
    if not partial and "name" not in result:
        raise WorkflowError("workflow_name_required")
    for field, limit in (("name", 160), ("description", 4000), ("category", 80)):
        if field in result:
            value = result[field]
            if not isinstance(value, str) or len(value) > limit or (field == "name" and not value.strip()):
                raise WorkflowError("invalid_workflow_" + field)
            result[field] = value.strip()
    if "config" in result and not isinstance(result["config"], dict):
        raise WorkflowError("invalid_workflow_config")
    if "tags" in result and (not isinstance(result["tags"], list) or len(result["tags"]) > 20
                            or any(not isinstance(tag, str) or not tag.strip() or len(tag) > 64 for tag in result["tags"])):
        raise WorkflowError("invalid_workflow_tags")
    if "published" in result and type(result["published"]) is not bool:
        raise WorkflowError("invalid_workflow_published")
    try:
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise WorkflowError("invalid_workflow") from None
    if len(serialized.encode()) > 65536:
        raise WorkflowError("workflow_too_large")
    return result, serialized


def dto(row):
    return {"id": row["id"], "name": row["name"], "description": row["description"],
            "config": json.loads(row["config"]), "category": row["category"], "tags": json.loads(row["tags"]),
            "published": bool(row["published"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}


def owned(conn, account, workflow_id):
    row = conn.execute("SELECT * FROM workflow_registry WHERE id=? AND owner_account_id=?", (workflow_id, account)).fetchone()
    if row is None:
        raise WorkflowError("workflow_not_found", 404)
    return dto(row)


def public(conn, workflow_id=None, *, limit=50, offset=0):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000:
        raise WorkflowError("invalid_pagination")
    query = "FROM workflow_registry w JOIN accounts a ON a.id=w.owner_account_id WHERE w.published=1 AND a.status='active'"
    if workflow_id is not None:
        row = conn.execute("SELECT w.* " + query + " AND w.id=?", (workflow_id,)).fetchone()
        if row is None:
            raise WorkflowError("workflow_not_found", 404)
        return dto(row)
    conn.execute("BEGIN")
    with conn:
        total = conn.execute("SELECT COUNT(*) " + query).fetchone()[0]
        rows = conn.execute("SELECT w.* " + query + " ORDER BY w.updated_at DESC,w.id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return {"workflows": [dto(row) for row in rows], "total": total, "limit": limit, "offset": offset}


def listing(conn, account, limit=50, offset=0):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000:
        raise WorkflowError("invalid_pagination")
    conn.execute("BEGIN")
    with conn:
        total = conn.execute("SELECT COUNT(*) FROM workflow_registry WHERE owner_account_id=?", (account,)).fetchone()[0]
        rows = conn.execute("SELECT * FROM workflow_registry WHERE owner_account_id=? ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?",
                            (account, limit, offset)).fetchall()
        return {"workflows": [dto(row) for row in rows], "total": total, "limit": limit, "offset": offset}


def create(conn, principal, key, data):
    values, serialized = payload(data)
    if not isinstance(key, str) or not key.strip() or len(key) > 128:
        raise WorkflowError("idempotency_key_required")
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        account = account_identity._account(conn, principal)
        previous = conn.execute("SELECT payload,result FROM account_workflow_requests WHERE account_id=? AND request_key=?", (account, key)).fetchone()
        if previous:
            if previous[0] != serialized:
                raise WorkflowError("idempotency_conflict", 409)
            return json.loads(previous[1])
        now = time.time()
        workflow_id = conn.execute("""INSERT INTO workflow_registry
            (creator_wallet,creator_account_id,owner_account_id,name,description,config,category,tags,published,created_at,updated_at)
            VALUES ('',?,?,?,?,?,?,?,?,?,?)""", (account, account, values["name"], values["description"],
            json.dumps(values["config"]), values["category"], json.dumps(values["tags"]), int(values["published"]), now, now)).lastrowid
        result = owned(conn, account, workflow_id)
        conn.execute("INSERT INTO account_workflow_requests VALUES (?,?,?,?)", (account, key, serialized, json.dumps(result)))
        return result


def update(conn, principal, workflow_id, data):
    values, _ = payload(data, partial=True)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        account = account_identity._account(conn, principal)
        owned(conn, account, workflow_id)
        for field in ("config", "tags"):
            if field in values:
                values[field] = json.dumps(values[field])
        values["updated_at"] = time.time()
        conn.execute("UPDATE workflow_registry SET " + ",".join(name + "=?" for name in values)
                     + " WHERE id=? AND owner_account_id=?", [*values.values(), workflow_id, account])
        return owned(conn, account, workflow_id)


def delete(conn, principal, workflow_id):
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        account = account_identity._account(conn, principal)
        owned(conn, account, workflow_id)
        conn.execute("DELETE FROM workflow_registry WHERE id=? AND owner_account_id=?", (workflow_id, account))

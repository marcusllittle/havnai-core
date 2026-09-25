"""Selected legacy music preferences; called within signed import transactions."""
import hashlib
import json


TABLES = {"likes": ("music_publication_likes", "account_music_likes", "created_at"),
          "saves": ("music_library_saves", "account_music_saves", "saved_at")}


class PreferenceImportError(ValueError):
    pass


def review(conn, account, wallet, kind, ids):
    if (kind not in TABLES or not isinstance(ids, list) or len(ids) > 100
            or any(not isinstance(value, str) or not value.strip() or len(value) > 200 for value in ids)
            or len(set(ids)) != len(ids)):
        raise PreferenceImportError("invalid_preference_selection")
    source, destination, timestamp = TABLES[kind]
    result = []
    for identifier in sorted(ids):
        rows = conn.execute(f"SELECT wallet,{timestamp} FROM {source} WHERE LOWER(wallet)=? AND publication_id=?",
                            (wallet, identifier)).fetchall()
        publication = conn.execute("SELECT title,state FROM music_publications WHERE id=?", (identifier,)).fetchone()
        if len(rows) != 1 or publication is None or publication[1] != "published":
            raise PreferenceImportError("import_preference_unavailable")
        current = conn.execute(f"SELECT created_at FROM {destination} WHERE account_id=? AND publication_id=?",
                               (account, identifier)).fetchone()
        state = {"wallet": rows[0][0], "created_at": rows[0][1], "account_created_at": current[0] if current else None,
                 "publication_id": identifier, "state": publication[1]}
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        result.append({"id": identifier, "title": publication[0], "already_in_account": current is not None,
                       "state_digest": digest})
    return result


def transfer(conn, account, wallet, kind, reviewed):
    """No commit: caller owns proof, resource changes, receipt and audit atomically."""
    if not conn.in_transaction:
        raise RuntimeError("preference import requires the enclosing write transaction")
    current = review(conn, account, wallet, kind, [row["id"] for row in reviewed])
    if current != reviewed:
        raise PreferenceImportError("import_preference_changed")
    source, destination, timestamp = TABLES[kind]
    for item in reviewed:
        identifier = item["id"]
        # Keep an existing account preference unchanged; otherwise preserve the
        # wallet preference's original date rather than making it look new.
        conn.execute(f"""INSERT OR IGNORE INTO {destination}(account_id,publication_id,created_at)
            SELECT ?,publication_id,{timestamp} FROM {source} WHERE LOWER(wallet)=? AND publication_id=?""",
            (account, wallet, identifier))
        removed = conn.execute(f"DELETE FROM {source} WHERE LOWER(wallet)=? AND publication_id=?", (wallet, identifier))
        if removed.rowcount != 1:
            raise PreferenceImportError("import_preference_changed")
        if kind == "likes":
            conn.execute("""UPDATE music_publications SET like_count=
                (SELECT COUNT(*) FROM music_publication_likes WHERE publication_id=?) +
                (SELECT COUNT(*) FROM account_music_likes WHERE publication_id=?) WHERE id=?""",
                (identifier, identifier, identifier))


def inventory(conn, account, wallet, kind, limit, offset):
    source, destination, _ = TABLES[kind]
    rows = conn.execute(f"""SELECT s.publication_id,p.title,p.state,COUNT(*) AS copies,
        EXISTS(SELECT 1 FROM {destination} a WHERE a.account_id=? AND a.publication_id=s.publication_id) AS existing
        FROM {source} s LEFT JOIN music_publications p ON p.id=s.publication_id
        WHERE LOWER(s.wallet)=? GROUP BY s.publication_id ORDER BY s.publication_id LIMIT ? OFFSET ?""",
        (account, wallet, limit, offset)).fetchall()
    total = conn.execute(f"SELECT COUNT(DISTINCT publication_id) FROM {source} WHERE LOWER(wallet)=?", (wallet,)).fetchone()[0]
    return [{"id": row[0], "title": row[1] if row[2] == "published" else None,
             "eligible": row[2] == "published" and row[3] == 1, "already_in_account": bool(row[4])} for row in rows], total

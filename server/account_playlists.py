"""Account-owned playlists; wallet provenance never grants management rights."""
import time
import uuid

import music_discover as music


class PlaylistError(ValueError):
    def __init__(self, code, status=422):
        self.code, self.status = code, status


def validate_metadata(data, *, creating=False):
    if not isinstance(data, dict) or set(data) - {"title", "description", "is_public", "id"}:
        raise PlaylistError("invalid_payload")
    if not creating and "id" in data:
        raise PlaylistError("invalid_payload")
    if creating and (not isinstance(data.get("id"), str) or not data["id"].startswith("playlist-")):
        raise PlaylistError("invalid_playlist_id")
    if creating:
        try:
            parsed = uuid.UUID(data["id"][9:])
            if data["id"] != f"playlist-{parsed}" or parsed.version != 4:
                raise ValueError()
        except ValueError:
            raise PlaylistError("invalid_playlist_id") from None
    for key, length in (("title", 120), ("description", 1000)):
        if key in data and (not isinstance(data[key], str) or len(data[key]) > length):
            raise PlaylistError("invalid_payload")
    if (creating or "title" in data) and not data.get("title", "").strip():
        raise PlaylistError("missing_title")
    if "is_public" in data and not isinstance(data["is_public"], bool):
        raise PlaylistError("invalid_payload")


def owned(conn, account_id, playlist_id):
    row = conn.execute("SELECT * FROM music_playlists WHERE id=? AND owner_account_id=?", (playlist_id, account_id)).fetchone()
    if not row:
        raise PlaylistError("playlist_not_found", 404)
    return row


def detail(account_id, playlist_id):
    conn = music.get_db()
    row = conn.execute("SELECT * FROM music_playlists WHERE id=? AND (owner_account_id=? OR is_public=1)",
                       (playlist_id, account_id)).fetchone()
    if not row:
        raise PlaylistError("playlist_not_found", 404)
    return music.playlist_to_dict(row, requester_account=account_id)


def listing(account_id):
    rows = music.get_db().execute("SELECT * FROM music_playlists WHERE owner_account_id=? ORDER BY updated_at DESC,id",
                                 (account_id,)).fetchall()
    return [music.playlist_to_dict(row, requester_account=account_id, include_items=False) for row in rows]


def create(account_id, data):
    validate_metadata(data, creating=True)
    conn = music.get_db()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        row = conn.execute("SELECT * FROM music_playlists WHERE id=?", (data["id"],)).fetchone()
        title, description, public = data["title"].strip(), data.get("description", ""), data.get("is_public", False)
        if row:
            if row["owner_account_id"] != account_id:
                raise PlaylistError("playlist_not_found", 404)
            if (row["title"], row["description"], bool(row["is_public"])) != (title, description, public):
                raise PlaylistError("playlist_request_conflict", 409)
        else:
            now = time.time()
            conn.execute("""INSERT INTO music_playlists
                (id,owner_wallet,owner_account_id,title,description,is_public,artwork_seed,created_at,updated_at)
                VALUES (?,'',?,?,?,?,?,?,?)""", (data["id"], account_id, title, description, public, uuid.uuid4().hex[:16], now, now))
            conn.execute("INSERT OR IGNORE INTO account_public_profiles VALUES (?,?,?,?)",
                (f"creator_{uuid.uuid4().hex}", account_id, f"Creator {uuid.uuid4().hex[:6]}", now))
        return detail(account_id, data["id"]), row is None


def modify(account_id, playlist_id, *, operation, data=None, publication_id=None):
    conn = music.get_db()
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        owned(conn, account_id, playlist_id)
        if operation == "delete":
            conn.execute("DELETE FROM music_playlist_items WHERE playlist_id=?", (playlist_id,))
            conn.execute("DELETE FROM music_playlists WHERE id=? AND owner_account_id=?", (playlist_id, account_id))
            return {"ok": True}
        if operation == "metadata":
            validate_metadata(data)
            if data.get("is_public") and music._playlist_has_adult_publications(playlist_id):
                raise PlaylistError("adult_content_restricted", 409)
            for key in ("title", "description", "is_public"):
                if key in data:
                    value = data[key].strip() if key == "title" else data[key]
                    conn.execute(f"UPDATE music_playlists SET {key}=? WHERE id=?", (value, playlist_id))
        elif operation == "add":
            if not music._published_publication_exists(publication_id):
                raise PlaylistError("publication_not_found", 404)
            position = conn.execute("SELECT COALESCE(MAX(position),-1)+1 FROM music_playlist_items WHERE playlist_id=?",
                                    (playlist_id,)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO music_playlist_items VALUES (?,?,?,?)", (playlist_id, publication_id, position, time.time()))
        elif operation == "remove":
            conn.execute("DELETE FROM music_playlist_items WHERE playlist_id=? AND publication_id=?", (playlist_id, publication_id))
            music._compact_playlist_positions(conn, playlist_id)
        elif operation == "reorder":
            if (not isinstance(data, dict) or set(data) != {"publication_ids"} or not isinstance(data["publication_ids"], list)
                    or any(not isinstance(item, str) for item in data["publication_ids"])):
                raise PlaylistError("invalid_payload")
            ids = data["publication_ids"]
            # Reorder the visible list; retain hidden unpublished entries at the end.
            rows = conn.execute("""SELECT i.publication_id,p.state,COALESCE(p.adult_content,0) AS adult_content FROM music_playlist_items i
                JOIN music_publications p ON p.id=i.publication_id WHERE playlist_id=? ORDER BY position,added_at""", (playlist_id,)).fetchall()
            visible = {row["publication_id"] for row in rows
                       if row["state"] == "published" and not row["adult_content"]}
            if len(set(ids)) != len(ids) or set(ids) != visible:
                raise PlaylistError("playlist_order_conflict", 409)
            ordered = ids + [row["publication_id"] for row in rows
                             if row["state"] != "published" or row["adult_content"]]
            for position, item in enumerate(ordered):
                conn.execute("UPDATE music_playlist_items SET position=? WHERE playlist_id=? AND publication_id=?", (position, playlist_id, item))
        else:
            raise PlaylistError("invalid_operation")
        conn.execute("UPDATE music_playlists SET updated_at=? WHERE id=?", (time.time(), playlist_id))
        return detail(account_id, playlist_id)

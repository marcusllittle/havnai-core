"""Public music publication and Discover helpers for HavnAI."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional


get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
WALLET_REGEX: Any
artifact_url: Callable[[str], Optional[str]]

PLAY_MIN_SECONDS = 5.0
PLAY_DEDUPE_SECONDS = 30 * 60
MAX_LIMIT = 100


def init_music_discover_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS music_publications (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            audio_artifact_id TEXT NOT NULL,
            creator_wallet TEXT NOT NULL,
            title TEXT NOT NULL,
            style TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            duration REAL,
            bpm INTEGER,
            song_key TEXT DEFAULT '',
            instrumental INTEGER NOT NULL DEFAULT 0,
            model TEXT DEFAULT '',
            cover_art_seed TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'published',
            play_count INTEGER NOT NULL DEFAULT 0,
            like_count INTEGER NOT NULL DEFAULT 0,
            published_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_music_publications_active_job
        ON music_publications(job_id)
        WHERE state = 'published'
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_music_publications_state_time ON music_publications(state, published_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_music_publications_style ON music_publications(style)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS music_publication_likes (
            publication_id TEXT NOT NULL REFERENCES music_publications(id),
            wallet TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(publication_id, wallet)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_music_publication_likes_wallet ON music_publication_likes(wallet)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS music_publication_plays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            publication_id TEXT NOT NULL REFERENCES music_publications(id),
            listener_key TEXT NOT NULL,
            seconds_listened REAL NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_music_publication_plays_recent ON music_publication_plays(publication_id, listener_key, created_at DESC)"
    )
    conn.commit()


def publish_song(
    *,
    job_id: str,
    creator_wallet: str,
    title: str,
    style: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    conn = get_db()
    wallet = creator_wallet.strip().lower()
    if not WALLET_REGEX.match(wallet):
        return {"ok": False, "error": "invalid_wallet"}

    job = conn.execute(
        "SELECT id, wallet, model, task_type, status, resolved_spec FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    if not job:
        return {"ok": False, "error": "job_not_found"}
    if str(job["wallet"] or "").strip().lower() != wallet:
        return {"ok": False, "error": "not_your_job"}
    if str(job["task_type"] or "").upper() != "MUSIC_GEN":
        return {"ok": False, "error": "not_music_job"}
    if _canonical_status(job["status"]) != "succeeded":
        return {"ok": False, "error": "job_not_completed"}

    artifact = conn.execute(
        """
        SELECT id, kind, filename, content_type, path, metadata
        FROM artifacts
        WHERE job_id=? AND kind='audio'
        ORDER BY created_at ASC
        LIMIT 1
        """,
        (job_id,),
    ).fetchone()
    if not artifact or not artifact_url(str(artifact["path"] or "")):
        return {"ok": False, "error": "audio_artifact_required"}

    existing = conn.execute(
        "SELECT * FROM music_publications WHERE job_id=? AND state='published'",
        (job_id,),
    ).fetchone()
    if existing:
        publication = publication_to_dict(existing, liked_by_wallet=wallet)
        publication["already_published"] = True
        return {"ok": True, "publication": publication}

    resolved = _parse_json(job["resolved_spec"])
    params = _parse_json(resolved.get("parameters"))
    metadata = _parse_json(artifact["metadata"])
    clean_title = _clean_title(title, params.get("prompt") or job_id)
    clean_style = _clean_style(style or params.get("style") or "")
    clean_tags = _clean_tags(tags if tags is not None else _tags_from_style(clean_style))
    now = time.time()
    publication_id = f"music-{hashlib.sha256(f'{job_id}:{wallet}:{now}'.encode()).hexdigest()[:24]}"
    duration = _number_or_none(metadata.get("duration"), params.get("duration"))
    bpm = _int_or_none(metadata.get("bpm"), params.get("bpm"))
    key = str(metadata.get("key") or metadata.get("keyscale") or params.get("key") or "").strip()[:64]
    instrumental = 1 if bool(params.get("instrumental")) else 0
    cover_seed = hashlib.sha256(f"{job_id}:{artifact['id']}:{clean_title}".encode()).hexdigest()[:16]

    conn.execute(
        """
        INSERT INTO music_publications (
            id, job_id, audio_artifact_id, creator_wallet, title, style, tags,
            duration, bpm, song_key, instrumental, model, cover_art_seed, state,
            play_count, like_count, published_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'published', 0, 0, ?, ?)
        """,
        (
            publication_id,
            job_id,
            str(artifact["id"]),
            wallet,
            clean_title,
            clean_style,
            json.dumps(clean_tags),
            duration,
            bpm,
            key,
            instrumental,
            str(job["model"] or ""),
            cover_seed,
            now,
            now,
        ),
    )
    conn.commit()
    log_event("Music publication created", publication_id=publication_id, job_id=job_id, wallet=wallet)
    publication = get_publication(publication_id, liked_by_wallet=wallet)
    return {"ok": True, "publication": publication}


def unpublish_song(publication_id: str, creator_wallet: str) -> Dict[str, Any]:
    wallet = creator_wallet.strip().lower()
    conn = get_db()
    cur = conn.execute(
        """
        UPDATE music_publications
           SET state='unpublished', updated_at=?
         WHERE id=? AND creator_wallet=? AND state='published'
        """,
        (time.time(), publication_id, wallet),
    )
    conn.commit()
    if cur.rowcount != 1:
        exists = conn.execute("SELECT id FROM music_publications WHERE id=?", (publication_id,)).fetchone()
        return {"ok": False, "error": "publication_not_found" if not exists else "not_your_publication"}
    log_event("Music publication unpublished", publication_id=publication_id, wallet=wallet)
    return {"ok": True}


def get_publication(publication_id: str, liked_by_wallet: Optional[str] = None) -> Optional[Dict[str, Any]]:
    row = get_db().execute(
        "SELECT * FROM music_publications WHERE id=? AND state='published'",
        (publication_id,),
    ).fetchone()
    if not row:
        return None
    return publication_to_dict(row, liked_by_wallet=liked_by_wallet)


def browse_publications(
    *,
    search: Optional[str] = None,
    style: Optional[str] = None,
    sort: str = "newest",
    limit: int = 24,
    offset: int = 0,
    liked_by_wallet: Optional[str] = None,
    creator_wallet: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_db()
    conditions = ["state='published'"]
    params: List[Any] = []
    if search:
        like = f"%{search.strip()}%"
        conditions.append("(title LIKE ? OR style LIKE ? OR tags LIKE ?)")
        params.extend([like, like, like])
    if style:
        like = f"%{style.strip()}%"
        conditions.append("(style LIKE ? OR tags LIKE ?)")
        params.extend([like, like])
    if creator_wallet:
        conditions.append("creator_wallet = ?")
        params.append(creator_wallet.strip().lower())
    where = f"WHERE {' AND '.join(conditions)}"
    total_row = conn.execute(f"SELECT COUNT(*) AS n FROM music_publications {where}", params).fetchone()
    if sort == "popular":
        order = "play_count DESC, like_count DESC, published_at DESC"
    elif sort == "liked":
        order = "like_count DESC, play_count DESC, published_at DESC"
    else:
        order = "published_at DESC"
    limit_int = max(1, min(int(limit), MAX_LIMIT))
    offset_int = max(0, int(offset))
    rows = conn.execute(
        f"SELECT * FROM music_publications {where} ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, limit_int, offset_int],
    ).fetchall()
    return {
        "publications": [publication_to_dict(row, liked_by_wallet=liked_by_wallet) for row in rows],
        "total": int(total_row["n"] if total_row else 0),
        "limit": limit_int,
        "offset": offset_int,
        "sort": sort,
    }


def set_like(publication_id: str, wallet: str, liked: bool = True) -> Dict[str, Any]:
    conn = get_db()
    normalized_wallet = wallet.strip().lower()
    if not WALLET_REGEX.match(normalized_wallet):
        return {"ok": False, "error": "invalid_wallet"}
    publication = conn.execute(
        "SELECT id FROM music_publications WHERE id=? AND state='published'",
        (publication_id,),
    ).fetchone()
    if not publication:
        return {"ok": False, "error": "publication_not_found"}
    now = time.time()
    if liked:
        conn.execute(
            """
            INSERT OR IGNORE INTO music_publication_likes (publication_id, wallet, created_at)
            VALUES (?, ?, ?)
            """,
            (publication_id, normalized_wallet, now),
        )
    else:
        conn.execute(
            "DELETE FROM music_publication_likes WHERE publication_id=? AND wallet=?",
            (publication_id, normalized_wallet),
        )
    count_row = conn.execute(
        "SELECT COUNT(*) AS n FROM music_publication_likes WHERE publication_id=?",
        (publication_id,),
    ).fetchone()
    like_count = int(count_row["n"] if count_row else 0)
    conn.execute(
        "UPDATE music_publications SET like_count=?, updated_at=? WHERE id=?",
        (like_count, now, publication_id),
    )
    conn.commit()
    return {"ok": True, "liked": liked, "like_count": like_count}


def record_play(
    publication_id: str,
    *,
    listener_key: str,
    seconds_listened: float = 0,
    completed: bool = False,
) -> Dict[str, Any]:
    conn = get_db()
    if not conn.execute(
        "SELECT id FROM music_publications WHERE id=? AND state='published'",
        (publication_id,),
    ).fetchone():
        return {"ok": False, "error": "publication_not_found"}
    seconds = max(0.0, float(seconds_listened or 0))
    if seconds < PLAY_MIN_SECONDS and not completed:
        return {"ok": True, "counted": False, "reason": "insufficient_listen"}
    key = hashlib.sha256(str(listener_key or "anonymous").encode()).hexdigest()
    now = time.time()
    recent = conn.execute(
        """
        SELECT id FROM music_publication_plays
        WHERE publication_id=? AND listener_key=? AND created_at >= ?
        LIMIT 1
        """,
        (publication_id, key, now - PLAY_DEDUPE_SECONDS),
    ).fetchone()
    if recent:
        return {"ok": True, "counted": False, "reason": "recent_duplicate"}
    conn.execute(
        """
        INSERT INTO music_publication_plays (publication_id, listener_key, seconds_listened, completed, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (publication_id, key, seconds, 1 if completed else 0, now),
    )
    conn.execute(
        "UPDATE music_publications SET play_count=play_count+1, updated_at=? WHERE id=?",
        (now, publication_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT play_count FROM music_publications WHERE id=?",
        (publication_id,),
    ).fetchone()
    return {"ok": True, "counted": True, "play_count": int(row["play_count"] if row else 0)}


def publication_to_dict(row: sqlite3.Row, liked_by_wallet: Optional[str] = None) -> Dict[str, Any]:
    conn = get_db()
    artifact = conn.execute(
        "SELECT id, filename, content_type, path, metadata FROM artifacts WHERE id=?",
        (row["audio_artifact_id"],),
    ).fetchone()
    audio_url = artifact_url(str(artifact["path"] or "")) if artifact else None
    liked_by_me = False
    normalized_wallet = (liked_by_wallet or "").strip().lower()
    if normalized_wallet and WALLET_REGEX.match(normalized_wallet):
        liked_by_me = bool(
            conn.execute(
                "SELECT 1 FROM music_publication_likes WHERE publication_id=? AND wallet=?",
                (row["id"], normalized_wallet),
            ).fetchone()
        )
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "audio_artifact_id": row["audio_artifact_id"],
        "creator_wallet": row["creator_wallet"],
        "creator": _format_wallet(row["creator_wallet"]),
        "title": row["title"],
        "style": row["style"] or "",
        "tags": _parse_tags(row["tags"]),
        "duration": row["duration"],
        "bpm": row["bpm"],
        "key": row["song_key"] or "",
        "instrumental": bool(row["instrumental"]),
        "model": row["model"] or "",
        "cover_art_seed": row["cover_art_seed"],
        "cover_art_url": f"/api/music/publications/{row['id']}/cover.svg",
        "audio_url": audio_url,
        "play_count": int(row["play_count"] or 0),
        "like_count": int(row["like_count"] or 0),
        "liked_by_me": liked_by_me,
        "published_at": row["published_at"],
        "updated_at": row["updated_at"],
    }


def _canonical_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return "succeeded" if status in {"success", "completed", "done", "succeeded"} else status


def _parse_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _parse_tags(value: Any) -> List[str]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [value]
    return _clean_tags(parsed if isinstance(parsed, list) else [])


def _clean_title(title: Any, fallback: Any) -> str:
    text = " ".join(str(title or "").split())[:120]
    if text:
        return text
    fallback_text = " ".join(str(fallback or "Untitled HavnAI Song").split())
    return fallback_text[:80] or "Untitled HavnAI Song"


def _clean_style(style: Any) -> str:
    return " ".join(str(style or "").split())[:160]


def _tags_from_style(style: str) -> List[str]:
    return [part.strip() for part in style.split(",") if part.strip()]


def _clean_tags(tags: Any) -> List[str]:
    cleaned: List[str] = []
    for tag in tags if isinstance(tags, list) else []:
        value = " ".join(str(tag or "").split())[:32]
        key = value.lower()
        if value and key not in {item.lower() for item in cleaned}:
            cleaned.append(value)
        if len(cleaned) >= 8:
            break
    return cleaned


def _number_or_none(*values: Any) -> Optional[float]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_or_none(*values: Any) -> Optional[int]:
    number = _number_or_none(*values)
    return int(number) if number is not None else None


def _format_wallet(wallet: Any) -> str:
    value = str(wallet or "")
    return f"{value[:6]}...{value[-4:]}" if len(value) >= 10 else value

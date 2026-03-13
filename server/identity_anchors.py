"""Wallet-scoped saved identity anchors for prompt-driven reference faces."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


class IdentityAnchorError(RuntimeError):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# Injected by app.py
get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]
ANCHORS_DIR: Path

MAX_SOURCE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
VALID_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")
DATA_URL_RE = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE | re.DOTALL)
ALLOWED_MIME_TYPES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}


def init_identity_anchor_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_anchors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            slug TEXT NOT NULL,
            display_name TEXT NOT NULL,
            storage_relpath TEXT NOT NULL,
            source_mime TEXT DEFAULT '',
            source_sha256 TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_used_at REAL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_anchors_wallet_slug "
        "ON identity_anchors(wallet, slug)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_identity_anchors_wallet_updated "
        "ON identity_anchors(wallet, updated_at DESC)"
    )
    conn.commit()


def normalize_slug(raw_slug: Any) -> str:
    slug = str(raw_slug or "").strip().lower()
    return slug if VALID_SLUG_RE.match(slug) else ""


def list_identity_anchors(wallet: str) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, wallet, slug, display_name, created_at, updated_at, last_used_at
        FROM identity_anchors
        WHERE wallet = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (wallet,),
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_identity_anchor(anchor_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM identity_anchors WHERE id = ?", (int(anchor_id),)).fetchone()
    if not row:
        return None
    return dict(row)


def get_identity_anchor_by_slug(wallet: str, slug: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM identity_anchors WHERE wallet = ? AND slug = ?",
        (wallet, slug),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def create_identity_anchor(
    wallet: str,
    slug: str,
    display_name: str,
    image_data_url: str,
) -> Dict[str, Any]:
    normalized_slug = normalize_slug(slug)
    if not normalized_slug:
        raise IdentityAnchorError(
            "invalid_identity_anchor_slug",
            "Anchor slug may contain only lowercase letters, numbers, underscores, and hyphens.",
            400,
        )
    name = str(display_name or "").strip()
    if not name:
        raise IdentityAnchorError("missing_display_name", "display_name is required", 400)
    if Image is None or ImageOps is None:
        raise IdentityAnchorError(
            "anchor_image_processing_unavailable",
            "Pillow is required for identity-anchor image processing.",
            500,
        )

    raw_bytes, source_mime = _decode_image_data_url(image_data_url)
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    normalized_bytes = _normalize_anchor_image(raw_bytes, source_mime)
    relative_path = _write_anchor_file(wallet, normalized_bytes)

    conn = get_db()
    now = time.time()
    try:
        cursor = conn.execute(
            """
            INSERT INTO identity_anchors (
                wallet, slug, display_name, storage_relpath, source_mime, source_sha256,
                created_at, updated_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                wallet,
                normalized_slug,
                name,
                relative_path.as_posix(),
                source_mime,
                source_sha256,
                now,
                now,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        _safe_delete_file(relative_path)
        raise IdentityAnchorError(
            "identity_anchor_slug_exists",
            "An identity anchor with that slug already exists for this wallet.",
            409,
        ) from exc

    anchor_id = int(cursor.lastrowid or 0)
    log_event("Identity anchor created", wallet=wallet, anchor_id=anchor_id, slug=normalized_slug)
    created = get_identity_anchor(anchor_id)
    return _row_to_dict(created) if created else {
        "id": anchor_id,
        "wallet": wallet,
        "slug": normalized_slug,
        "display_name": name,
        "created_at": now,
        "updated_at": now,
        "last_used_at": None,
    }


def delete_identity_anchor(anchor_id: int, wallet: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM identity_anchors WHERE id = ? AND wallet = ?",
        (int(anchor_id), wallet),
    ).fetchone()
    if not row:
        return False
    relpath = Path(str(row["storage_relpath"]))
    conn.execute("DELETE FROM identity_anchors WHERE id = ? AND wallet = ?", (int(anchor_id), wallet))
    conn.commit()
    _safe_delete_file(relpath)
    log_event("Identity anchor deleted", wallet=wallet, anchor_id=int(anchor_id), slug=str(row["slug"]))
    return True


def mark_identity_anchor_used(anchor_id: int) -> None:
    conn = get_db()
    now = time.time()
    conn.execute(
        "UPDATE identity_anchors SET last_used_at = ?, updated_at = ? WHERE id = ?",
        (now, now, int(anchor_id)),
    )
    conn.commit()


def anchor_image_to_data_url(anchor: Dict[str, Any]) -> str:
    relpath = Path(str(anchor.get("storage_relpath") or ""))
    if not relpath.as_posix():
        raise IdentityAnchorError("identity_anchor_missing_image", "Identity anchor file is missing.", 500)
    path = _resolve_storage_path(relpath)
    if not path.exists():
        raise IdentityAnchorError("identity_anchor_missing_image", "Identity anchor file is missing.", 500)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    return {
        "id": int(row["id"]),
        "wallet": str(row["wallet"]),
        "slug": str(row["slug"]),
        "display_name": str(row["display_name"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "last_used_at": float(row["last_used_at"]) if row["last_used_at"] is not None else None,
    }


def _decode_image_data_url(image_data_url: str) -> Tuple[bytes, str]:
    raw_value = str(image_data_url or "").strip()
    if not raw_value:
        raise IdentityAnchorError("missing_anchor_image", "image_data_url is required", 400)
    match = DATA_URL_RE.match(raw_value)
    if not match:
        raise IdentityAnchorError(
            "unsupported_anchor_image_type",
            "Identity anchor uploads must be PNG, JPEG, or WEBP data URLs.",
            415,
        )
    source_mime = match.group(1).lower()
    if source_mime not in ALLOWED_MIME_TYPES:
        raise IdentityAnchorError(
            "unsupported_anchor_image_type",
            "Only PNG, JPEG, and WEBP identity anchor uploads are supported.",
            415,
        )
    try:
        raw_bytes = base64.b64decode(match.group(2), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise IdentityAnchorError("invalid_anchor_image", "Identity anchor image payload is not valid base64.", 400) from exc
    if len(raw_bytes) > MAX_SOURCE_BYTES:
        raise IdentityAnchorError("anchor_image_too_large", "Identity anchor image exceeds the 10 MB limit.", 413)
    return raw_bytes, source_mime


def _normalize_anchor_image(raw_bytes: bytes, source_mime: str) -> bytes:
    assert Image is not None
    assert ImageOps is not None
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source_image:
            source_image.load()
            source_format = str(source_image.format or "").upper()
            expected_format = ALLOWED_MIME_TYPES.get(source_mime)
            if source_format not in {"PNG", "JPEG", "WEBP"} or (expected_format and source_format != expected_format):
                raise IdentityAnchorError(
                    "unsupported_anchor_image_type",
                    "Only PNG, JPEG, and WEBP identity anchor uploads are supported.",
                    415,
                )
            image = ImageOps.exif_transpose(source_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            if max(image.size) > MAX_IMAGE_SIDE:
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), resampling)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except IdentityAnchorError:
        raise
    except Exception as exc:
        raise IdentityAnchorError(
            "unsupported_anchor_image_type",
            "Identity anchor image must be a readable PNG, JPEG, or WEBP file.",
            415,
        ) from exc


def _write_anchor_file(wallet: str, normalized_bytes: bytes) -> Path:
    wallet_dir = ANCHORS_DIR / wallet.lower()
    wallet_dir.mkdir(parents=True, exist_ok=True)
    relative_path = Path(wallet.lower()) / f"{uuid.uuid4().hex}.png"
    abs_path = ANCHORS_DIR / relative_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(normalized_bytes)
    return relative_path


def _resolve_storage_path(relpath: Path) -> Path:
    candidate = (ANCHORS_DIR / relpath).resolve()
    root = ANCHORS_DIR.resolve()
    if candidate != root and root not in candidate.parents:
        raise IdentityAnchorError("identity_anchor_missing_image", "Identity anchor file is missing.", 500)
    return candidate


def _safe_delete_file(relpath: Path) -> None:
    try:
        path = _resolve_storage_path(relpath)
    except IdentityAnchorError:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

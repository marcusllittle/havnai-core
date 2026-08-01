"""
Astra Valkyries generative rewards.

Finish a run, and the network generates pilot art for your collection.
This is the bridge between the game and the platform's image pipeline.

Design constraints:
  - The game client submits IDs ONLY (run, pilot, outfit, map, grade).
    Prompts are composed server-side from the locked template tables
    below (seeded from docs/ASTRA_ACTIVE_PROMPT_PACK.md in the game
    repo). There is no user-controlled string, so there is no prompt
    injection surface.
  - Jobs go through the same enqueue + settlement machinery as
    /submit-job, via injected callables. Nothing in the stable
    submission path is modified.
  - One image per run: run_id is the primary key, so retries are
    idempotent by construction.
  - Free-for-launch economics, bounded by per-wallet and global daily
    caps (env-tunable). The real constraint is GPU capacity, not
    credits, so the caps apply regardless of CREDITS_ENABLED.
"""

import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional

# ─── Injected dependencies (set by app.py) ──────────────────

get_db: Callable[[], sqlite3.Connection]
log_event: Callable[..., None]

# ─── Caps (free launch tier) ─────────────────────────────────

DAILY_IMAGES_PER_WALLET = 10
DAILY_IMAGES_GLOBAL = 200
RUN_MAX_AGE_SECONDS = 86400.0   # a run older than a day cannot claim art
MIN_GRADE_FOR_IMAGE = {"S", "A", "B"}  # match the earn threshold: real play only

# Queue weight kept below typical paying traffic so a game launch spike
# cannot starve the core product.
ASTRA_JOB_WEIGHT = 5.0

# ─── Locked template tables ──────────────────────────────────
# Source of truth: Astra-Valkyries docs/ASTRA_ACTIVE_PROMPT_PACK.md.
# Update by editing here; the client can only pick, never write.

RENDER_SETTINGS: Dict[str, Any] = {
    "steps": 32,
    "guidance": 6.5,
    "width": 832,
    "height": 1216,
}

PREFERRED_MODELS: List[str] = [
    "amixillustrious_amix",
    "perfectdeliberate_v60",
    "copaxtimeless_xplus2bnsfw1",
]

NEGATIVE_PROMPT = (
    "extra fingers, malformed hands, bad hands, duplicate face, blurry, "
    "low quality, text, logo, watermark, distorted anatomy, messy cockpit "
    "clutter, fused body parts, robotic body merge"
)

PILOT_TEMPLATES: Dict[str, str] = {
    "pilot_nova": (
        "Nova, elite adult female starfighter pilot, seated in a futuristic "
        "spacecraft cockpit, long platinum-blonde hair in high glossy ponytail, "
        "piercing bright blue eyes, confident smile, athletic build, glossy "
        "metallic royal-blue and black pilot bodysuit, sleek reflective chest "
        "panel and shoulder plating, black gloves and futuristic wrist cuffs, "
        "premium command chair, glowing control panels, cinematic blue "
        "highlights, polished anime-inspired realism, sharp linework, "
        "ultra-clean render, centered full body portrait, premium game "
        "character splash art, high contrast"
    ),
    "pilot_rex": (
        "Rex, elite adult female tactical ace pilot, standing confidently "
        "inside a dark futuristic spacecraft cockpit, short sleek chin-length "
        "bob with dark roots fading to vivid neon teal tips, luminous cyan "
        "highlights framing her face, cool cyan eyes with a confident smirk, "
        "athletic build, glossy teal-black pilot bodysuit with subtle gold "
        "zipper detail, cropped jacket open, reflective form-fitting material, "
        "powerful stance, cockpit illuminated by teal accent lights, glowing "
        "interface panels, moody cyber-sci-fi atmosphere, cinematic low angle, "
        "ultra-detailed anime realism, premium character concept art, strong "
        "contrast, full body portrait"
    ),
    "pilot_yuki": (
        "Yuki, adult female space pilot, serene composed presence, seated "
        "elegantly in a bright futuristic spacecraft cockpit, long flowing "
        "pure white hair cascading softly over her shoulders, pale blue eyes, "
        "calm expression, slim build, glossy white pilot bodysuit with black "
        "seam lines and subtle cyan insignia, sleek silhouette, soft hands "
        "resting on her thighs, bright clean cockpit with white framing, cool "
        "luminous lighting accenting her silhouette and reflective suit "
        "texture, polished anime-inspired realism, highly detailed facial "
        "features, centered three-quarter full body portrait, premium game "
        "character art, soft celestial palette, crisp render quality"
    ),
}

OUTFIT_FLAVOR: Dict[str, str] = {
    "outfit_01": "wearing the Standard Flight Suit, regulation navy and steel trim",
    "outfit_02": "wearing the Neon Vanguard suit, electric magenta and cyan accent lines",
    "outfit_03": "wearing the Desert Storm suit, sand-gold plating with burnt-orange trim",
    "outfit_04": "wearing the Iron Hawk suit, gunmetal armor plating with bronze detailing",
    "outfit_05": "wearing the Cloud Walker suit, soft white and sky-blue gradient panels",
    "outfit_06": "wearing the Shadow Pulse suit, matte black with violet pulse lines",
    "outfit_07": "wearing the Ocean Drift suit, deep teal with flowing aqua accents",
    "outfit_08": "wearing the Crimson Wing suit, scarlet plating with winged shoulder flares",
    "outfit_09": "wearing the Frost Nova suit, ice-white armor with crystalline blue edges",
    "outfit_10": "wearing the Thunder Strike suit, storm-grey with crackling gold accents",
    "outfit_11": "wearing the Emerald Gale suit, jade-green panels with silver wind lines",
    "outfit_12": "wearing the Violet Tempest suit, royal purple with swirling lavender trim",
    "outfit_13": "wearing the Solar Flare suit, blazing orange-gold with radiant plating",
    "outfit_14": "wearing the Lunar Eclipse suit, midnight silver with pale crescent accents",
    "outfit_15": "wearing the Cosmic Surge suit, iridescent nebula-dyed panels",
    "outfit_16": "wearing the Starfall Armor, gleaming star-forged plate with light trails",
    "outfit_17": "wearing the Aurora Borealis suit, shimmering polar-light gradient armor",
    "outfit_18": "wearing the Void Reaper suit, obsidian armor with abyssal red seams",
}

MAP_FLAVOR: Dict[str, str] = {
    "nebula-runway": (
        "wide cockpit windows showing deep space and vibrant blue-violet "
        "nebulae drifting past the runway lights"
    ),
    "solar-rift": (
        "cockpit windows blazing with the golden fire of a solar rift, "
        "ember light washing across the instruments"
    ),
    "abyss-crown": (
        "cockpit windows opening onto the cold darkness of the abyss, "
        "faint teal starlight and drifting ice crystals"
    ),
}

GRADE_FLAVOR: Dict[str, str] = {
    "S": "triumphant victorious mood, celebratory lighting, ace-of-the-fleet energy",
    "A": "confident post-victory mood, warm triumphant lighting",
    "B": "steady professional mood, mission-complete calm",
}


def compose_prompt(pilot_id: str, outfit_id: str, map_id: str, grade: str) -> Optional[str]:
    """Compose the render prompt from locked tables. Returns None if any ID
    is outside its closed set — unknown IDs never reach the pipeline."""
    base = PILOT_TEMPLATES.get(pilot_id)
    outfit = OUTFIT_FLAVOR.get(outfit_id)
    scene = MAP_FLAVOR.get(map_id)
    mood = GRADE_FLAVOR.get(grade)
    if not base or not outfit or not scene or not mood:
        return None
    return f"{base}, {outfit}, {scene}, {mood}"


# ─── Table setup ─────────────────────────────────────────────

def init_astra_gen_tables(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS astra_reward_images (
            run_id      TEXT PRIMARY KEY,
            job_id      TEXT NOT NULL,
            wallet      TEXT NOT NULL,
            pilot_id    TEXT NOT NULL,
            outfit_id   TEXT NOT NULL,
            map_id      TEXT NOT NULL,
            grade       TEXT NOT NULL,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_astra_reward_images_wallet
            ON astra_reward_images(wallet);
    """)
    db.commit()


# ─── Core operations ─────────────────────────────────────────

def _daily_count(db: sqlite3.Connection, wallet: Optional[str] = None) -> int:
    day_start = float(int(time.time() // 86400) * 86400)
    if wallet is None:
        row = db.execute(
            "SELECT COUNT(*) FROM astra_reward_images WHERE created_at >= ?",
            (day_start,),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT COUNT(*) FROM astra_reward_images WHERE wallet = ? AND created_at >= ?",
            (wallet, day_start),
        ).fetchone()
    return int(row[0])


def request_reward_image(
    wallet: str,
    run_id: str,
    pilot_id: str,
    outfit_id: str,
    map_id: str,
    enqueue_fn: Callable[..., str],
    ticket_fn: Callable[..., None],
    safety_fn: Callable[[str, str], Optional[str]],
    select_model_fn: Callable[[], Optional[str]],
    job_payload_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Any]:
    """
    Validate the run and enqueue one reward image for it.

    All heavy dependencies are injected so this module stays isolated
    from app.py and directly testable.
    """
    db = get_db()

    # The run must exist, belong to this wallet, be recent, and reflect
    # real play (same grade floor as the credit earn threshold).
    run = db.execute(
        "SELECT wallet, grade, map_id, created_at FROM astra_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return {"ok": False, "reason": "unknown_run"}
    if str(run["wallet"]) != wallet:
        return {"ok": False, "reason": "run_wallet_mismatch"}
    if time.time() - float(run["created_at"]) > RUN_MAX_AGE_SECONDS:
        return {"ok": False, "reason": "run_expired"}

    grade = str(run["grade"] or "").strip().upper()
    if grade not in MIN_GRADE_FOR_IMAGE:
        return {"ok": False, "reason": "grade_too_low", "grade": grade}

    # Idempotent: one image per run, retries return the original job.
    existing = db.execute(
        "SELECT job_id FROM astra_reward_images WHERE run_id = ?", (run_id,)
    ).fetchone()
    if existing is not None:
        return {"ok": True, "job_id": existing["job_id"], "status": "existing"}

    # Free-tier caps. GPU capacity is the real budget here.
    if _daily_count(db, wallet) >= DAILY_IMAGES_PER_WALLET:
        return {"ok": False, "reason": "daily_image_cap_reached", "cap": DAILY_IMAGES_PER_WALLET}
    if _daily_count(db) >= DAILY_IMAGES_GLOBAL:
        return {"ok": False, "reason": "generation_busy"}

    prompt = compose_prompt(pilot_id, outfit_id, map_id, grade)
    if prompt is None:
        return {"ok": False, "reason": "invalid_ids"}

    # Defense in depth: the prompt is entirely server-authored, but it
    # still goes through the same safety gate as public submissions.
    block_reason = safety_fn(prompt, NEGATIVE_PROMPT)
    if block_reason:
        log_event(f"Astra gen blocked by safety: {block_reason}", level="warning")
        return {"ok": False, "reason": "generation_unavailable"}

    model = select_model_fn()
    if not model:
        return {"ok": False, "reason": "no_capacity"}

    job_settings: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "sfw_mode": True,
        "source": "astra_reward",
        "astra_run_id": run_id,
        **RENDER_SETTINGS,
    }
    job_data = job_payload_fn(job_settings)

    job_id = enqueue_fn(wallet, model, "IMAGE_GEN", job_data, ASTRA_JOB_WEIGHT, None)
    try:
        ticket_fn(
            job_id=job_id,
            wallet=wallet,
            job_type="IMAGE_GEN",
            model=model,
            job_data=job_data,
        )
    except Exception as exc:  # ticket failure is non-fatal, same as /submit-job
        log_event("Astra gen settlement ticket failed (non-fatal)", job_id=job_id, error=str(exc))

    db.execute(
        """INSERT INTO astra_reward_images
           (run_id, job_id, wallet, pilot_id, outfit_id, map_id, grade, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, job_id, wallet, pilot_id, outfit_id, map_id, grade, time.time()),
    )
    db.commit()

    log_event(f"Astra reward image queued: {wallet[:10]}… run {run_id} -> {job_id}")
    return {"ok": True, "job_id": job_id, "status": "queued"}


def get_recent_creations(
    attach_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    limit: int = 12,
) -> Dict[str, Any]:
    """Latest completed reward images across all players, for the public
    'recent Astra creations' strip on joinhavn.io. Wallets are truncated —
    this is a showcase, not an account view."""
    db = get_db()
    rows = db.execute(
        """SELECT r.run_id, r.job_id, r.wallet, r.pilot_id, r.outfit_id,
                  r.map_id, r.grade, r.created_at
           FROM astra_reward_images r
           JOIN jobs j ON j.id = r.job_id
           WHERE j.status IN ('completed', 'done')
           ORDER BY r.created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    creations: List[Dict[str, Any]] = []
    for row in rows:
        wallet = str(row["wallet"])
        record: Dict[str, Any] = {
            "job_id": row["job_id"],
            "pilot_id": row["pilot_id"],
            "outfit_id": row["outfit_id"],
            "map_id": row["map_id"],
            "grade": row["grade"],
            "created_at": row["created_at"],
            "pilot_short": f"{wallet[:6]}…{wallet[-4:]}",
        }
        record = attach_fn(record)
        if record.get("image_url") or record.get("preview_url"):
            creations.append(record)

    return {"creations": creations}


def get_gallery(
    wallet: str,
    attach_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    limit: int = 50,
) -> Dict[str, Any]:
    """Player's generated reward images, newest first, with job status and
    result URLs attached via the platform's own resolver."""
    db = get_db()
    rows = db.execute(
        """SELECT r.run_id, r.job_id, r.wallet, r.pilot_id, r.outfit_id,
                  r.map_id, r.grade, r.created_at, j.status AS job_status
           FROM astra_reward_images r
           LEFT JOIN jobs j ON j.id = r.job_id
           WHERE r.wallet = ?
           ORDER BY r.created_at DESC
           LIMIT ?""",
        (wallet, limit),
    ).fetchall()

    images: List[Dict[str, Any]] = []
    for row in rows:
        job_status = str(row["job_status"] or "unknown")
        if job_status in {"completed", "done"}:
            status = "completed"
        elif job_status in {"failed", "error", "cancelled"}:
            status = "failed"
        else:
            status = "pending"
        record: Dict[str, Any] = {
            "run_id": row["run_id"],
            "job_id": row["job_id"],
            "pilot_id": row["pilot_id"],
            "outfit_id": row["outfit_id"],
            "map_id": row["map_id"],
            "grade": row["grade"],
            "status": status,
            "created_at": row["created_at"],
        }
        if status == "completed":
            record = attach_fn(record)
        images.append(record)

    return {"images": images}

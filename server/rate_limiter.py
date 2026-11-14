"""Sliding-window rate limiter with Redis backend and in-memory fallback.

Usage
-----
    from server.rate_limiter import check_rate_limit

    if not check_rate_limit(key=f"submit-job:{wallet}", limit=20, window_seconds=60):
        abort(429)

Set REDIS_URL (e.g. ``redis://localhost:6379/0``) to use Redis sorted sets.
Falls back to in-memory deques (current app.py behaviour) when Redis is not
configured or temporarily unreachable, so deployment is zero-config for dev.

The existing ``rate_limit()`` function in app.py continues to work unchanged;
this module is available as a drop-in replacement that survives restarts and
scales across multiple coordinator processes.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, Optional

REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "").strip() or None

_logger = logging.getLogger("havnai.rate_limiter")
_BUCKETS: Dict[str, deque] = defaultdict(deque)

# Lazy Redis client — initialised once on first use.
_redis_client = None
_redis_ready: Optional[bool] = None  # None = not yet attempted


def _get_redis():
    global _redis_client, _redis_ready
    if _redis_ready is not None:
        return _redis_client
    _redis_ready = False
    if not REDIS_URL:
        return None
    try:
        import redis  # type: ignore
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        _redis_ready = True
        _logger.info("Rate limiter: Redis backend active (%s)", REDIS_URL)
    except Exception as exc:
        _logger.warning("Rate limiter: Redis unavailable, using in-memory fallback (%s)", exc)
    return _redis_client


def _in_memory_check(key: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    window_start = now - window_seconds
    bucket = _BUCKETS.setdefault(key, deque())
    while bucket and bucket[0] < window_start:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def _redis_check(r, key: str, limit: int, window_seconds: int) -> bool:
    try:
        now = time.time()
        window_start = now - window_seconds
        rkey = f"rl:{key}"
        pipe = r.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", window_start)
        pipe.zcard(rkey)
        pipe.zadd(rkey, {str(now): now})
        pipe.expire(rkey, window_seconds + 10)
        results = pipe.execute()
        count_before = results[1]
        if count_before >= limit:
            # Roll back the zadd so the count stays accurate
            r.zrem(rkey, str(now))
            return False
        return True
    except Exception as exc:
        _logger.warning("Redis rate-limit check failed, falling back to in-memory: %s", exc)
        return _in_memory_check(key, limit, window_seconds)


def check_rate_limit(key: str, limit: int, window_seconds: int = 60) -> bool:
    """Return True if the request is within the rate limit, False if it should be rejected."""
    r = _get_redis()
    if r is not None:
        return _redis_check(r, key, limit, window_seconds)
    return _in_memory_check(key, limit, window_seconds)


def using_redis() -> bool:
    """Return True if the Redis backend is active and healthy."""
    return _get_redis() is not None

# app.py wiring guide — Phase 2 + 3 modules

The modules added in Phase 2 and 3 (`stale_job_recovery`, `health`, `rate_limiter`,
`alerting`) follow the same dependency-injection pattern already used throughout
`app.py`. This document shows exactly where to add the wiring.

---

## 1. stale_job_recovery — continuous stale job reset + node dropout alerts

**Location:** right after `_inject_module_dependencies()` (around line 976)

```python
import stale_job_recovery

# ... existing injection block ...
_inject_module_dependencies()

# Phase 2: start background stale-job recovery thread
stale_job_recovery.start(get_db, NODES, log_event)
```

That single `start()` call:
- Resets jobs stuck in `running` beyond `HAVNAI_STALE_JOB_TIMEOUT_SECONDS` (default 600s) every 60s
- Fires Slack alerts for node dropouts (requires `HAVNAI_ALERT_WEBHOOK_URL`)

---

## 2. health — structured /health endpoint

**Location:** after `_inject_module_dependencies()`, then update (or replace) the existing `/health` route.

```python
import health as health_module
import time as _time

_STARTUP_TIME = _time.time()  # add near top of file

# after _inject_module_dependencies():
health_module.start(get_db, NODES, ONLINE_THRESHOLD, startup_time=_STARTUP_TIME)
```

Then update the `/health` route:

```python
@app.route("/health")
def healthz():
    payload, status = health_module.check()
    return jsonify(payload), status
```

---

## 3. alerting — coordinator startup notification

Add one call near the bottom of the startup block (after `init_db`):

```python
import alerting

# after log_event(f"Telemetry online..."):
alerting.coordinator_started(version=APP_VERSION)
```

Set `HAVNAI_ALERT_WEBHOOK_URL` in `.env` to activate (Slack incoming webhook URL).

---

## 4. rate_limiter — Redis-backed rate limiting (optional migration)

The existing `rate_limit()` in `app.py` continues to work. To migrate a call site
to the Redis-backed version, replace:

```python
# before
if not rate_limit(f"submit-job:{request.remote_addr}", limit=30):
    abort(429)

# after
from rate_limiter import check_rate_limit
if not check_rate_limit(f"submit-job:{request.remote_addr}", limit=30, window_seconds=60):
    abort(429)
```

Set `REDIS_URL=redis://localhost:6379/0` in `.env` to use Redis. Falls back to
in-memory automatically if Redis is down, so migration can be done incrementally.

---

## Summary of new env vars

| Variable | Module | Effect |
|---|---|---|
| `HAVNAI_ALERT_WEBHOOK_URL` | alerting | Slack webhook URL for operational alerts |
| `HAVNAI_STALE_JOB_TIMEOUT_SECONDS` | stale_job_recovery | Seconds before a running job is requeued (default: 600) |
| `HAVNAI_RECOVERY_INTERVAL_SECONDS` | stale_job_recovery | How often the recovery thread runs (default: 60) |
| `HAVNAI_NODE_ALERT_COOLDOWN_SECONDS` | stale_job_recovery | Min seconds between node-offline alerts (default: 300) |
| `REDIS_URL` | rate_limiter | Redis connection string; omit for in-memory fallback |

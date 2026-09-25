# Account video: implementation and remaining work

HAVN-11 Video Studio uses owned image/audio uploads and durable `/v2/jobs`
submissions with a persisted idempotency key. Recovery and private result playback
use the account session. Public wallet-era jobs are not silently imported.

The image-to-video resolver preserves and validates numeric controls before credit
reservation. Steps are integral 1–150, guidance 0–20, FPS integral 8–30, and frames
integral 9–257. Frames are normalized down to an `8n+1` count and the resolved
duration reports `(frames - 1) / fps`. Seeds are integral -1 through 2^32-1;
-1 requests a generated seed. Boolean, nonfinite, malformed, fractional integer,
and out-of-range inputs return a validation error rather than a queued job.

Optional width and height must be supplied together, each 256–1280 and divisible
by 32. These produce a custom render size with matching delivery dimensions.
Without overrides, existing fast-upscaled/native-quality preset dimensions remain.
Motion strength and explicit source strength are each 0–1. An omitted source
strength remains absent, preserving the worker's motion-strength fallback.

Invalid controls do not reserve credits or create jobs. The web submission helper
discards a pending intent only for explicit pre-enqueue validation errors, allowing
the user to correct their request. Ambiguous failures retain the original intent.

These coordinator limits are not a guarantee that every video model supports every
combination. Live GPU verification is outstanding. Create's legacy video controls,
reference-sheet workflows, multi-clip continuation, private last-frame extraction,
stitched artifacts, and chain recovery still need account integration; they are
not replaced by the narrower Video Studio flow.

Tests: `tests/test_account_jobs.py`, `tests/test_platform_v1.py`, and web
`lib/__tests__/accountJobSubmission.test.ts`.

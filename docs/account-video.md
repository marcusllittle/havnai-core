# Account video: implementation and remaining work

HAVN-11 Video Studio and single-clip Create use owned uploads and durable `/v2/jobs`
submissions with a persisted idempotency key. Recovery and private result playback
use the account session. Public wallet-era jobs are not silently imported.

Create chooses `image_to_video` when a starting image is supplied and
`text_to_video` otherwise. Core requires a manifest `text_to_video` capability for
the latter; image-only models fail before reservation. Text-only jobs reject a
source image rather than ignoring it. Raw input paths/URLs are rejected for account
video jobs. Source images go through private uploads, and safe-content negatives
are included when requested. The resolved spec retains the prompt and input asset
IDs for owned history/recovery. The worker task type matches the model task type.

Create stores its pending video intent separately from Video Studio. A retry uses
the original request body/key and uploaded asset, even if the response was lost.
Account change/unmount aborts requests; recovery and collection saves are scoped to
the account. Create workflow presets populate the numeric controls sent to core.

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
combination. Live GPU verification is outstanding. Reference-sheet workflows,
multi-clip orchestration,
stitched artifacts, and chain recovery still need account integration. Create
explicitly blocks account multi-clip/reference-sheet submissions until those are
connected; single clips are available. This does not complete the full video scope.

## Private continuation input

`POST /v2/jobs/<job_id>/last-frame` requires an owned, successful job with a stored
video artifact. Core validates that its path is inside the outputs directory,
extracts the last frame into a private image asset, and returns its ID/kind/hash.
There is no credit charge for this derivation. A new video job using that asset
uses the normal generation reservation. No public image URL is returned.

The account/artifact pair uniquely identifies the derived asset. Retries reuse
it. Concurrent extraction results converge on one saved asset; unused temporary
files are cleaned up. Ownership and artifact identity are checked again in the
save transaction after decoding. The uploaded image belongs to the current account
and is subject to the same download/reuse authorization as other private inputs.

FFmpeg decodes a local MP4/MOV container with network protocols disabled, a bounded
60-second subprocess timeout, and no shell. Missing tooling, incomplete videos,
failed decoding, and ownership changes return structured errors without creating
a derived asset or reserving credits. The web helper aborts on account changes and
can supply the resulting asset ID directly to the next video request.

## Durable clip-chain API

`POST /v2/video-chains` accepts an immutable `template` video request, `total`
(integer 2–7), and boolean `auto_stitch`, with an account-scoped Idempotency-Key.
Creation costs no credits. Repeating the same key/body recovers the same chain;
changing its body conflicts. Its initial seed is stored rather than regenerated
after a refresh. Raw image URLs, wallet fields, and unrecognized controls cannot
enter the stored template. Starting image/audio assets must belong to the account.

`GET /v2/video-chains` lists account chains (50 per page, `offset`); GET on
`/v2/video-chains/<id>` returns the plan and its clip IDs/statuses. Another account
cannot read, advance, or stop it. This allows a different authenticated browser
session to recover the plan without relying on localStorage.

`POST /v2/video-chains/<id>/next` returns the current clip while it is running.
After success it derives the private last frame, increments the stored seed, and
submits the next image-to-video job through the shared account validation path.
The per-clip request key is deterministic. The clip reference and normal credit
reservation commit together under the same database lock; a retry cannot reserve
again or orphan a billed job outside the chain. Failed clips stop advancement.
When all clips succeed, the API reports `rendered`; this does not mean stitched.

DELETE on the chain stops further submissions. It does not cancel an already
accepted clip. Stop/ownership/status are rechecked under the enqueue lock, including
when stopping overlaps last-frame extraction. No new reservation occurs after that
stop wins the transaction. The front-end chain runner and stitching endpoint remain
to be implemented; this API alone does not enable the multi-clip control.

Tests: `tests/test_account_video_chains.py`, `tests/test_account_video.py` (including real local FFmpeg when installed),
`tests/test_account_jobs.py`, `tests/test_platform_v1.py`, and web
`lib/__tests__/accountJobSubmission.test.ts` and `accountVideoCreate.test.ts`.

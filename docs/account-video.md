# Account video: implementation and remaining work

HAVN-11 Video Studio and Create use owned uploads and durable `/v2/jobs`
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
combination. Live GPU verification and reference-sheet workflows remain outstanding.
Create supports single clips and 2–7-clip account sequences. It still explicitly
blocks reference-sheet submissions until that runtime path is connected.

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
stop wins the transaction.

Create's Total clips and automatic merge controls now use this API for accounts.
The client saves a chain-creation intent before submitting it, reuses the key and
uploaded starting image after a lost response, and leaves the durable plan on the
server. It polls each accepted clip before advancing. Account change/unmount aborts
the local runner and prevents subsequent submissions from that browser.

Saved video sequences lists account plans on demand, with pagination and explicit
resume/stop/result actions. Merely opening Create or its recovery list does not
submit remaining clips. Resume reads the server state, so another device can resume
without copying localStorage. Stop remaining clips commits the server stop before
aborting the local runner; the accepted clip keeps running. Merge failures leave
the rendered sequence available for a later retry. Foreign account responses are
rejected before advancing or displaying private results.

## Private merged results

`POST /v2/video-chains/<id>/stitch` requires all planned clips to have succeeded and
remain owned by the account. FFprobe validates compatible video/audio streams;
FFmpeg then concatenates the clips in chain order without re-encoding. Mismatched
stream formats fail explicitly, leaving all originals intact. Local tests exercise
both silent and audio-bearing MP4 clips. Missing tools return a structured error.

The result is a completed, zero-cost `video_stitch` job with a private video artifact.
Its resolved spec and artifact metadata record the chain and source job/artifact
IDs. It appears in the account Collection and plays through the existing private
artifact route. Its output directory is denied on public static routes, including
before the artifact row is inserted. Intermediate merge files are outside web roots.

The chain has one output record. Retries return the same job; interleaved merges
converge on one record and clean up discarded files. Source ownership and artifact
identity are checked again before publishing the result. No credit reservation is
made for stitching, and source clips remain unchanged. A chain with a merged result
reports `complete`; `rendered` means clips are ready but have not been merged.

Tests: `tests/test_account_video_stitch.py`, `tests/test_account_video_chains.py`, `tests/test_account_video.py` (including real local FFmpeg when installed),
`tests/test_account_jobs.py`, `tests/test_platform_v1.py`, and web
`lib/__tests__/accountJobSubmission.test.ts` and `accountVideoCreate.test.ts`.
Frontend sequence coverage: `accountVideoChains.test.ts`, `AccountVideoSequences.test.tsx`,
and `AccountCreate.test.tsx` (two clips through merging and stop during a running clip).

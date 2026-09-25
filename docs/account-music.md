# Account Music Studio

Configured account deployments use `/v2/capabilities`, `/v2/assets`, and `/v2/jobs`
for creation, polling, cancellation, and recovery. Browser drafts and unfinished
submission intents are scoped to the immutable account ID. Requests contain no
wallet and cannot fall back to the owner proxy. A lost submission response keeps
its original payload and idempotency key; the studio offers an explicit resume.
Core replays an existing intent before validating model settings that might have
changed since the original submission. Credit reservation and job creation remain
one transaction.

`GET /v2/music/publications` lists the account's active publications with private
job references. `POST` accepts `job_id`, `title`, optional `artifact_id`, `style`,
and `tags`; core derives account ownership from the session. The selected audio
must belong to a completed music job owned by that account. Concurrent/repeated
publication requests return one active publication per job. No wallet signature
or additional credit charge is made for publication.

`DELETE /v2/music/publications/:id` unpublishes only the owner's publication and
is idempotent. The public audio endpoint checks publication state on each request;
unpublishing removes public playback access but retains the owner's private audio.
Audio already downloaded by a listener cannot be recalled.

Public attribution uses a separate `creator_<random-id>` profile and generated
display name, never a Clerk subject, email, session, or internal account ID.
`GET /music/creators/:profile_id` exposes only that creator's published music.
Existing anonymous Discover/audio/cover routes include account publications while
omitting source job IDs, artifact IDs, prompts, and account IDs. Direct static URLs
for account-owned audio remain blocked even after publication. Legacy wallet
authorization cannot publish or unpublish account-owned jobs/publications.

The web `/api/account-media/:artifact` route serves authenticated private audio
with range support, no redirects, no public caching, and no forwarded browser
cookies/owner tokens. Its verified session token reaches core's per-artifact ACL.
The player and draft views remount on account changes; saved queues are account
scoped so another account or guest cannot restore a private queue.

## Evidence and remaining work

Backend tests exercise real SQLite, RSA-signed session tokens, concurrent publish,
private artifact ranges, public playback/unpublish, and cross-account/legacy denial.
Web tests exercise account music creation/publication without wallet calls, draft
and player isolation, StrictMode, lost-response retries, and the private streaming
proxy. Audio bytes and jobs in these tests are fixtures, not GPU-generated music.

Account library saves, likes, playlists, profile editing, migration, and the full
real-provider paid generation/publication acceptance remain separate work. The
isolated preview has no GPU nodes or card checkout; an offline message there is
expected and is not evidence that live generation has passed.

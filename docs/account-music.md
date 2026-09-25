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
`GET /music/creators/:profile_id` exposes only that creator's published music and public playlists.
Existing anonymous Discover/audio/cover routes include account publications while
omitting source job IDs, artifact IDs, prompts, and account IDs. Direct static URLs
for account-owned audio remain blocked even after publication. Legacy wallet
authorization cannot publish or unpublish account-owned jobs/publications.

The web `/api/account-media/:artifact` route serves authenticated private audio
with range support, no redirects, no public caching, and no forwarded browser
cookies/owner tokens. Its verified session token reaches core's per-artifact ACL.
The player and draft views remount on account changes; saved queues are account
scoped so another account or guest cannot restore a private queue.

## Account library API

`GET /v2/music/library` returns the authenticated account's saved publications,
recent likes, and playlists. Optional `limit`, `offset`, and `search` apply to
saved publications. `PUT /v2/music/publications/:id/save` accepts a boolean
`saved`; the corresponding `/like` endpoint accepts a boolean `liked`. These
are explicit idempotent states, not toggles. Account preferences are stored
separately from legacy wallet preferences. Public like totals include both;
wallet actions cannot remove an account's like. Unpublished tracks are hidden,
but users can still remove their own stale preferences.

`GET/POST /v2/music/playlists` lists or creates account-owned playlists. Creation
requires a client-generated `playlist-<UUIDv4>` ID, retained for ambiguous retries;
the same ID and metadata returns the existing playlist, while conflicting
metadata returns 409. New playlists are private unless explicitly made public.
`GET/PATCH/DELETE /v2/music/playlists/:id` reads, edits, or deletes a playlist.
`PUT/DELETE .../:id/items/:publication_id` sets membership without a request body.
`PUT .../:id/reorder` accepts `publication_ids`, requiring exactly the visible
membership without duplicates, and preserves unpublished entries at the end.
All writes check session-derived ownership inside a SQLite write transaction.
Public read access never grants editing rights. Existing anonymous playlist and
cover routes support explicitly shared account playlists without exposing the
account ID. A preserved provenance wallet cannot read a private account playlist
or manage it. No library operation touches the credit ledger.

Configured web deployments use these account APIs for music libraries, saves,
likes, playlists, and editing shared playlists. `POST /v2/music/preferences`
accepts up to 100 publication IDs and returns only the current account's saved/
liked flags and public like counts, so public Discover/creator pages can show
accurate controls without wallet signatures. Unknown/unpublished IDs are omitted.
The signed-out experience remains public, and private actions offer sign-in.
Account changes remount private views and discard pending responses. Playlist
creation IDs persist through the initial add-song action; ambiguous failures
reuse the same ID. Legacy wallet flows remain only when accounts are unconfigured.

## Evidence and remaining work

Backend tests exercise real SQLite, RSA-signed session tokens, concurrent publish,
private artifact ranges, public playback/unpublish, and cross-account/legacy denial.
Web tests exercise account music creation/publication without wallet calls, draft
and player isolation, StrictMode, lost-response retries, and the private streaming
proxy. Audio bytes and jobs in these tests are fixtures, not GPU-generated music.

Additional backend tests cover account library isolation, concurrent like/create
retries, public/private playlist transitions, reorder conflicts, deletion, hidden
unpublished tracks, and wallet provenance bypass attempts.

Profile editing, migration, and the full real-provider paid generation/publication
acceptance remain work. The
isolated preview has no GPU nodes or card checkout; an offline message there is
expected and is not evidence that live generation has passed.

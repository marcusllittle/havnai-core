# HavnAI production architecture and ownership map

Work item: HAVN-4 under HAVN-17. Last reviewed: 2026-09-26.

This map identifies the production components, trust boundaries, data stores,
operational owners, and evidence hooks used for commercial launch hardening. It
is an operations map, not launch approval. The drill evidence in
`production-operations-runbook.md` and `production-observability.md` remains the
authority for HAVN-17 acceptance.

## System boundary

| Component | Production role | Primary owner | Source of truth |
| --- | --- | --- | --- |
| `havnai-web` | Public Next.js application at `joinhavn.io`; account UI, create studios, pricing/policy pages, library, marketplace, Discover, operator pages, and same-origin API proxy routes. | Platform web owner | Web GitHub repo, Vercel production deployment, web environment variables |
| `havnai-core` coordinator | Flask API behind `https://api.joinhavn.io`; account auth, job queue, node control plane, account ledger, payments, publications, marketplace, observability, backup/restore tooling. | Platform core owner | Core GitHub repo, coordinator release directory/`RELEASE_SHA`, private systemd environment |
| Creator worker node(s) | Polls/leases queued jobs, runs image/video/music model engines, uploads progress and artifacts back to coordinator. | Node operator with platform runbook ownership | Worker install bundle, `~/.havnai/current`, node wallet/config, coordinator node telemetry |
| Clerk | Managed account identity provider. Core verifies short-lived bearer tokens and lifecycle webhooks. | Platform account owner | Clerk dashboard, core private auth environment, lifecycle webhook delivery logs |
| Stripe | Account credit Checkout, webhook events, refunds/disputes, provider reconciliation. | Platform payments owner | Stripe dashboard, `/v2/payments/stripe/webhook`, account payment tables |
| Cloudflare/DNS/ingress | Public DNS and TLS routing for `joinhavn.io` and `api.joinhavn.io`. | Platform operations owner | DNS/provider dashboard, public health probes |
| HavnAI static/media roots | Coordinator-side uploaded assets and generated artifacts used by private account media, public publications, marketplace previews, and legacy result routes. | Platform operations owner | Private host filesystem plus artifact rows in SQLite |
| Astra game client | Separate game repository and frontend owned by Claude for game work. It consumes bounded HavnAI account/reward APIs only. | Astra owner for client/game quality; platform owner for core API contract | Astra repo for client, core Jira/API docs for backend contract |

## Request paths

| Flow | Path | Auth boundary | Persistence and evidence |
| --- | --- | --- | --- |
| Public web browsing | Browser -> `joinhavn.io` -> Vercel pages and same-origin proxy routes | Public pages; authenticated pages use Clerk session through web | Web deployment ID, route smoke checks, browser tests |
| Account API | Browser -> `joinhavn.io/api/...` -> `api.joinhavn.io/v2/...` | Clerk session token forwarded as `Authorization: Bearer ...`; core validates issuer, audience, origin, expiry, and local account status | Account tables, account audit events, private/no-store responses |
| Legacy wallet generation | Browser/client -> `POST /submit-job` | Wallet field plus legacy invite/rate-limit policy; invite gating is opt-in by `INVITE_GATING` | `jobs`, settlement rows, legacy feed/result routes |
| Account generation | Browser/client -> `POST /v2/jobs` | Verified account bearer token plus `Idempotency-Key`; no wallet or owner-token fallback | Account job row, credit reservation, artifacts, account ledger |
| Worker execution | Worker -> `/v1/node/jobs/...` and `/v1/node/artifact-purges` | Node token via `require_node`; lease and attempt IDs bind progress/artifact uploads to assigned work | Job progress, attempt history, artifacts, settlement, node telemetry |
| Account private media | Browser -> web `/api/account-media/...` -> core artifact route | Web verifies Clerk cookie and forwards short-lived bearer token; core checks per-artifact ACL | Private/no-store media response, artifact row |
| Public music/marketplace | Browser -> `/music/...`, `/v2/marketplace/...`, `/gallery/...` | Public reads only return explicitly published/listed non-adult rows; writes require owner account or legacy wallet boundary by route | `music_publications`, playlists, gallery/listing/sale tables, lifecycle audit |
| Stripe funding | Stripe -> `/v2/payments/stripe/webhook`; operator -> reconciliation CLI | Stripe signature and local purchase/provider validation; no account bearer token on webhook | Account payment events, receipts, credit ledger, reconciliation checks |
| Clerk lifecycle | Clerk -> `/v2/auth/clerk/webhook` | Svix signature, expected issuer/instance, bounded body size | Account identity/provider state, lifecycle audit, access denial checks |
| Astra economy/rewards | Astra client -> `/astra/...` | Current legacy routes are wallet based; launch account/reward integration changes must document request/response, authorization, idempotency, and compatibility in Jira before client reliance | Astra rewards/receipts tables, account/reward contract evidence |

## Data ownership

| Data set | Storage | Owner and access rule | Backup/recovery note |
| --- | --- | --- | --- |
| Accounts and identities | Coordinator SQLite | Immutable `acct_...`; derived only from verified provider issuer/subject. Email, wallet, owner token, and request body fields are never account identity. | Included in SQLite backup/restore drill |
| Wallet links and challenges | Coordinator SQLite | Optional wallet proof is account-scoped and never moves content or credits implicitly. | Included in SQLite backup/restore drill |
| Account credit balances, reservations, ledger | Coordinator SQLite | Integer units; reservations/captures/refunds commit under core write locks. | Included in SQLite restore and accepted-work restart drills |
| Stripe purchases/events/receipts | Coordinator SQLite plus Stripe provider state | Core validates provider truth and deduplicates by event and economic payment identity. | SQLite restore plus provider reconciliation procedure |
| Jobs, attempts, settlement, rewards, payout claims | Coordinator SQLite | `job_id` is the correlation key across request, queue, worker, artifact, account charge, settlement, and receipts. | Accepted-work restart drill must prove no duplicate charge/payout/receipt |
| Generated artifacts and uploads | Coordinator filesystem plus artifact rows | Private account artifacts require owner authorization; public publication/listing routes use allowlisted transformed previews or published media. | Separate media/artifact recovery sample required; DB backup alone is insufficient |
| Music publications, playlists, library preferences | Coordinator SQLite plus media artifacts | Account-owned rows use account IDs; public Discover/audio/cover omit private prompts, job IDs, artifact IDs, and account IDs. | SQLite restore plus representative media sample |
| Marketplace listings, sales, ownership logs | Coordinator SQLite plus preview/artifact files | Account sales transfer current ownership while preserving creator provenance and immutable receipts. | SQLite restore plus preview/artifact sample |
| Workflow templates | Coordinator SQLite | Account-owned drafts stay private; public browse exposes only published active rows and omits identity internals. | SQLite restore |
| Operator logs/metrics | Coordinator process, systemd journal, `/metrics`, `/v1/network/*` | Secrets and private prompts must be redacted before Jira evidence. | Evidence packet only; not customer data recovery |

## Public/private content boundary

Public routes may expose only deliberately published or listed content. The
following invariants are launch gates:

- Account jobs and artifacts are private by default and excluded from legacy
  wallet feed/history and public static/result routes.
- Adult-restricted content stays private to the owner and is blocked from
  Discover, public audio/cover, playlists, marketplace, previews, social cards,
  and Astra-visible public galleries.
- Direct static URLs must not bypass account authorization for account-owned
  artifacts. Web private-media routes stream through verified account access.
- Public music and marketplace responses use field allowlists and omit internal
  account IDs, Clerk identifiers, bearer tokens, private prompts, source paths,
  unlisted media URLs, and artifact IDs unless the route is owner-authenticated.
- Legacy wallet compatibility never grants access to rows that have account
  ownership.

## Operational control plane

| Surface | Purpose | Owner action |
| --- | --- | --- |
| `GET /health` | Public liveness and coarse node/queue count | Smoke after deploy, restart, rollback, DNS changes |
| `GET /healthz` | DB touch plus concurrent job summary exercise | Local and public deploy/rollback gate |
| `GET /metrics` | Admin-gated Prometheus-style job, worker, artifact, failure, and disk metrics | Scrape and alert; redact token in evidence |
| `GET /nodes` | Public/legacy dashboard feed and worker summary | Do not run rough QA through visible legacy route without explicit approval |
| `GET /v1/network/control-plane` | Queue, worker, claim, receipt, and health state | Primary restart/recovery and alert evidence |
| `GET /v1/network/alerts/dry-run` | Admin-gated alert rule evaluation without notification delivery | Prove alert matchers and injected model/GPU cases |
| `GET /v1/account/readiness` | Admin-gated account auth/payment readiness booleans | Capture redacted readiness after auth/payment config changes |
| Systemd `havnai-coordinator.service` | Runs the coordinator release | Restart/rollback only through documented runbook unless emergency |
| Worker `~/.havnai/current` | Runs node bundle and engine environment | Roll forward/back using `RUN_A_NODE.md` and heartbeat checks |

## Release and rollback ownership

Coordinator releases are built from a Git commit into a host release directory
with a `RELEASE_SHA`. `scripts/deploy_coordinator_release.sh` requires a commit
on the configured branch, creates a verified SQLite backup, switches the
`/opt/havnai/current` symlink, restarts the coordinator, and rolls back the
symlink if local `/healthz` does not recover. Web rollback is a Vercel deployment
promotion/restoration. Worker rollback uses `~/.havnai/previous` and must prove
heartbeat/model capability after restart.

Rollback is not automatically safe after schema, provider, or filesystem changes.
Every rollback drill must record whether a backward move is safe, or whether a
forward fix/compensating migration is required.

As of 2026-09-26, coordinator rollback has been exercised once in production-like
conditions: release `2f8c8e38cd98da991e71a64de89ba9a2b85060b1` was rolled back
to `ad7164dbacc5a0351546b88ca7b1558a0b34f36e` and then restored to
`2f8c8e38cd98da991e71a64de89ba9a2b85060b1`, with local/public health and
control-plane checks green. Web rollback has not been executed, but Vercel
deployment inventory identified production rollback candidates under project
`havnai-web` (`prj_41KTJsGstZ9bHYJpajM4qH9d8wAl`) and production route smoke
checks returned HTTP 200 for `/`, `/pricing`, `/support`, `/marketplace`,
`/discover`, `/terms/credits-v1`, `/refunds/credits-v1`, and `/create`.

## Evidence ownership matrix

| HAVN ticket | Evidence owner | Required production or production-like proof |
| --- | --- | --- |
| HAVN-4 | Platform core owner | This architecture/ownership map linked from HAVN-17/HAVN-4 and kept current as launch topology changes |
| HAVN-43 | Platform ops owner | Public health, metrics/control-plane, alert dry-run, dashboard/log evidence, external alert status |
| HAVN-44 | Platform ops owner | SQLite restore report plus representative media/artifact recovery sample, with RPO/RTO/retention/access-control evidence |
| HAVN-45 | Platform ops owner | Accepted-work restart drill with `job_id`, ledger, settlement, reservation, payout/reward, receipt checks |
| HAVN-46 | Platform ops owner | Coordinator/web/node rollback drill with version IDs, health checks, smoke checks, and rollback-safety review |
| HAVN-49 | Platform ops owner | Reuse HAVN-45 worker/coordinator restart drill; add worker heartbeat/model capability and no duplicate reward evidence |
| HAVN-14 | Platform core/web owner plus Claude for Astra HAVN-58 | Platform public/private/adult isolation evidence; Astra evidence is consumed from HAVN-58, not produced by platform edits |
| HAVN-56/HAVN-57 | Platform core owner for API contract; Claude for Astra client | Core request/response, authorization, idempotency, compatibility contract in Jira before Astra client reliance |

## Current launch risks tracked outside this map

- HAVN-47/HAVN-12 still need a private account-token mixed-model drill; public
  legacy drill artifacts must not appear on the production dashboard.
- HAVN-44 has production/prod-like SQLite restore, orphan-attempt repair, local
  backup timer, and representative media/artifact sample evidence. It still
  needs remote backup/retention/encryption/access-control approval or waiver.
- HAVN-45/HAVN-49 still need a controlled restart drill with accepted work and
  duplicate-charge/reward/receipt evidence.
- HAVN-46 has coordinator rollback evidence and Vercel rollback inventory. It
  still needs web/Vercel and node-runtime rollback exercise or waiver.
- HAVN-43 still needs external alert delivery evidence or explicit launch
  waivers for thresholds not wired to notifications.
- HAVN-14 consumed Claude's Astra HAVN-58 evidence; remaining platform gap is
  final deployed owner/private adult generation, cross-account denial, and
  public cache/social-preview proof or waiver.

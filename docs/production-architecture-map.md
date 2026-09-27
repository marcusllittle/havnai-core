# HavnAI Production Architecture and Ownership Map

This is the launch-readiness map for HAVN-4 and the HAVN-17 operations rollup.
It records the production path from public web traffic through the coordinator,
workers, storage, identity, payments, and economy ledger. It is intentionally
short: use it to identify ownership, monitoring, and blast radius during a
release decision or incident.

## System Path

```text
Users
  -> joinhavn.io web app
  -> Clerk account session / optional wallet signature
  -> api.joinhavn.io coordinator
  -> SQLite account, job, receipt, and reward ledger
  -> GPU creator nodes polling for accepted work
  -> coordinator artifact/music/gallery APIs
  -> public or private web surfaces
```

The game repository is a separate client surface. Core owns account, credit,
reward, and receipt APIs consumed by Astra. Astra gameplay, art, levels,
cutscenes, and client integration evidence are owned outside this repository.

## Component Map

| Component | Production location | Deploy / change path | Monitoring and evidence | Owner | If it fails |
| --- | --- | --- | --- | --- | --- |
| Web frontend | `https://joinhavn.io` on Vercel | `havnai-web` production deployment from release branch/commit | Public route smoke for `/`, `/create`, pricing/support/legal pages, browser auth handoff checks | Platform web | Users cannot sign in, buy credits, start jobs, or view account surfaces even if core is healthy |
| Public DNS and edge routing | Domain DNS / edge provider in front of Vercel and coordinator | DNS/provider console; avoid release changes without rollback inventory | Public HTTPS checks for `joinhavn.io` and `api.joinhavn.io` | Platform ops | Traffic cannot reach healthy services; symptoms look like broad outage |
| Core API / coordinator | `https://api.joinhavn.io`; current host runs `havnai-coordinator.service` as `marcus` from `/home/marcus/Downloads/source-code/havnai-core` | Fast-forward `feat/havn-11-commercial-accounts`, preserve runtime `nodes.json`, restart systemd | `/health`, `/healthz`, `/v1/network/control-plane`, `/metrics`, alert dry-run collector | Platform core | New jobs, account APIs, funding, private reads, node assignment, and public content APIs fail |
| Coordinator configuration | `/home/marcus/.havnai/.env` plus systemd drop-ins | Operator-only env edits followed by `systemctl daemon-reload` and restart | `systemctl cat havnai-coordinator.service`; redacted evidence in Jira | Platform ops | Wrong DB/static/node paths, auth failures, broken Stripe/Clerk/payment behavior |
| SQLite ledger | `/home/marcus/Downloads/source-code/havnai-core/db/ledger.db` | Schema migrations through reviewed core code; backup before operational recovery | Verified local backup and restore drill, integrity check, foreign-key check, table counts | Platform core/ops | Account balances, purchases, jobs, rewards, receipts, and content ownership become unavailable or inconsistent |
| Local backups | `/home/marcus/Downloads/source-code/havnai-core/db/backups` | `scripts/backup_coordinator.py` with `HAVNAI_DB_PATH` and `HAVNAI_BACKUP_DIR` | `scripts/backup_evidence_audit.py`; restore drill via `scripts/account_restore_drill.py` | Platform ops | Local DB recovery point is missing; launch still requires remote/offsite proof or waiver |
| Artifact and media storage | Coordinator `static` tree plus worker-local output/cache paths | Coordinator writes public/private outputs; worker cleanup follows authorized purge evidence | Artifact lifecycle audits, public/private route checks, backup/media inventory | Platform core/ops | Outputs may fail to render, private/public isolation may be unverifiable, or purge/restore may be incomplete |
| GPU creator nodes | Registered nodes polling `api.joinhavn.io`; live control-plane requires at least one ready creator node | Node installer/runtime bundle; node operator updates and rollback via previous runtime | `/nodes`, `/v1/network/control-plane`, `/models/list`, worker drill evidence | Node ops / platform coordination | Jobs queue or fail; mixed-model acceptance and restart recovery cannot pass |
| Model registry and routing | `server/manifests/registry.json` loaded by coordinator and node bundle | Reviewed manifest changes; node model availability must match launch-critical task types | `/models/list`, mixed-model worker drill, node doctor/preflight | Platform core/node ops | Wrong model routing, unsupported jobs, degraded quality, or repeated worker failures |
| Account auth | Clerk plus coordinator account-token verification | Clerk production config and core auth code; no invite-code requirement for launch | Browser sign-in checks, account API token checks, private route denial checks | Platform web/core | Users cannot authenticate or private account isolation cannot be proven |
| Optional wallet identity | Web wallet signature routes plus coordinator wallet/account linking | Wallet UI in web, account identity APIs in core | Wallet lifecycle tests and live account evidence where required | Platform web/core | Wallet-backed flows fail, but account-first usage should continue when wallet is optional |
| Stripe account funding | Stripe Checkout and `/v2/payments/stripe/webhook` | Core payment code and private Stripe env; webhook endpoint configured in Stripe dashboard | HAVN-25 webhook delivery, idempotent replay, receipt/funding/balance checks | Platform core/ops | Paid credits do not fund, duplicate credits risk appears, or receipts cannot reconcile |
| Credit and reward ledger | Coordinator SQLite tables for purchases, credits, jobs, node rewards, Astra receipts | Reviewed core code; append-only ledger/receipt invariants | Account payment tests, Stripe replay evidence, reward receipt tests, restart drill | Platform core | Balances, node payouts, or Astra rewards become wrong or unverifiable |
| Public gallery/music surfaces | Coordinator APIs consumed by web | Core content policy and web presentation changes | Public browse/discover smoke, no-store denial checks, owner/cross-account isolation evidence | Platform core/web | Private/adult/legacy content may leak or public product appears low-quality |
| Observability and alerts | Coordinator `/metrics`, control-plane JSON, dry-run alert route; external alerting/dashboard evidence pending | Core metrics code plus operator alert/dashboard setup | `scripts/collect_observability_evidence.py`; saved dashboard/alert delivery evidence before launch | Platform ops | Incidents are detected late; HAVN-17/HAVN-43 cannot close without evidence or waiver |
| Rollback paths | Vercel deployment history, coordinator git/service restart, node previous runtime | Surface-specific rollback drills or explicit waivers | `scripts/rollback_evidence_audit.py` and linked drill evidence | Platform ops | Bad releases take longer to unwind and launch rollback gate remains open |

## Current Production Notes

- The coordinator host does not use the older `/opt/havnai/releases` layout.
  Current service evidence shows the live systemd unit runs as `marcus` from the
  repo checkout with the repo `.venv`.
- The production SQLite path is `db/ledger.db` under the coordinator checkout,
  not `/var/lib/havnai/ledger.db`.
- Runtime `nodes.json` is modified by the coordinator. Preserve it across
  release-branch fast-forwards unless the change intentionally replaces node
  state.
- Public gallery and music/publication routes are separate surfaces. Keep
  `gallery_listings` evidence separate from `music_publications` evidence.
- Launch acceptance must distinguish implementation evidence, local/sandbox
  evidence, and production evidence. Missing external dashboards, offsite
  backups, restart drills, rollback drills, or Astra-side evidence stay open
  until proven or waived.

## Release Gate Checklist

| Gate | Required evidence |
| --- | --- |
| Funding | Stripe webhook delivery and idempotent replay show one receipt/funding row and no duplicate credits |
| Content isolation | Public browse/discover no-leak checks, private owner access, cross-account denial, social preview denial, and Astra HAVN-58 evidence |
| Worker stability | Mixed-model accepted-work drill against live coordinator and node fleet |
| Restart recovery | Accepted private work survives coordinator/node restart without duplicate settlement; shared by HAVN-45 and HAVN-49 |
| Observability | Health/control-plane/metrics/alert dry-run plus external dashboard or waiver |
| Backup/restore | Verified SQLite restore plus offsite backup/media/provider evidence or waiver |
| Rollback | Web, coordinator, and node rollback drill evidence or explicit waiver |
| Game integration | Core API request/response/auth contract documented; Astra client acceptance supplied by Claude |

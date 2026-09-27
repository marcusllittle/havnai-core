# HAVN-72 Release Evidence Map

Updated: 2026-09-27. Owner: Marcus Little. Platform execution: Codex for
`havnai-core` and `havnai-web`. Astra execution: Claude for the Astra game
repository.

This map is the HAVN-72 GO/NO-GO audit surface. It does not approve launch.
Launch requires Marcus to record a dated GO, NO-GO, or CONDITIONAL GO in
HAVN-72 after every open blocker below is either proven closed or explicitly
waived with mitigation.

## Release Candidate Inventory

| Surface | Current candidate | Evidence | Status |
| --- | --- | --- | --- |
| Core baseline | Production `https://api.joinhavn.io` reports `/health.version=215afa5`, one ready node, empty queue. The active systemd service runs from `/home/marcus/Downloads/source-code/havnai-core` as user `marcus`; the production checkout contains `origin/main` through PR #164 plus local merge `ceaa458`. | 2026-09-27 live probes; HAVN-17 Jira comments through `11014`; HAVN-72 comments through `11035`; tracked smoke/preflight reports under `docs/evidence/`. | Healthy but not launch complete. |
| Core no-invite, Stripe funding, gallery guard | Core production has invite gating opt-in only. `JOIN_TOKEN` can exist without requiring public invite codes unless `HAVNAI_INVITE_GATING` is explicitly enabled. HAVN-25 automatic Stripe funding/idempotent replay is accepted. | Core PR #134 merged (`a324ee9`) for invite-gating default tests; HAVN-25 live DB evidence shows one Stripe event, one receipt, one funding ledger row, no duplicate operation keys, and balance moved only by later generation spend; HAVN-25 Done. | Invite-code gate removed as blocker; full private mixed-model acceptance still open. |
| Web account launch surfaces and no invite/access-code copy | Production `https://joinhavn.io` deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`, from `havnai-web` commercial branch merge `82e1e9f` after PRs #112/#113/#119. | Local validation: `npx tsc --noEmit`, `npm test` (`80 files / 484 tests`), `npm run build`. Live validation: trust surface audit ok, creator public audit ok across desktop/iPhone/Android user agents, direct scan of `/`, `/create`, `/music`, `/video-studio`, `/pricing`, `/library`, `/privacy`, `/terms`, `/sign-in` found no invite/access-code/operator-key copy; sign-in CSP still includes Google/Clerk. | Deployed for account-first public surfaces; signed-in funded journey still requires account token evidence. |
| Public/private content isolation | Origin behavior denies private artifact URLs with `404`, `Cache-Control: private, no-store`, `Vary: Authorization`, and `cf-cache-status: BYPASS`. The exact stale CDN URL that previously served cached private media now rechecks as a denied/not-found response and the formal forbidden-URL production smoke passes. A follow-up anonymous leakage probe found no restricted publication ID or sensitive-term leakage in public Discover/search, direct public music media routes, public `/logs`, or web page body/meta surfaces. | HAVN-14 comments through `11025`; HAVN-42 comments through `11026`; HAVN-72 comments through `11027`; failing historical report `docs/evidence/havn-14-forbidden-url-smoke-tracked-20260927T124624Z.json`; resolved production report `docs/evidence/havn-14-forbidden-url-smoke-resolved-20260927T130022Z.json` with `ok=true`, `forbidden_public_url_1` status `404`, `content-type=application/json`, `cache-control=private, no-store`, `cf-cache-status=BYPASS`; anonymous leakage probe `docs/evidence/havn-42-public-leakage-probe-20260927T1305Z.json` with `ok=true`. | Platform stale-CDN and anonymous public leakage blockers are resolved. HAVN-14 final closure still waits for accepted Claude-owned HAVN-58 Astra evidence; signed-in/cross-account checks or waivers remain separate HAVN-42 evidence. |
| Observability and alerting | Coordinator health/control-plane and admin alert evaluation are deployed. `SERVER_JOIN_TOKEN` is now configured from the existing node token and verified by collector plus register smoke checks. The latest report proves metrics scrape and alert dry-run evaluation pass, but `HAVNAI_ALERT_WEBHOOK` is not configured. | HAVN-43 comments through `11063`; HAVN-17 comments through `11062`; latest report `docs/evidence/havn-43-observability-join-token-configured-20260927T1321Z.json` shows health/control-plane/metrics/dry-run checks pass and `join_token_config.configured=true`. Earlier admin report: `docs/evidence/havn-43-observability-admin-token-20260927T1313Z.json`. | External alert delivery receipt or dated waiver still open. |
| Backup/restore repair | Production local backups are scheduled and verified. The tracked backup audit report shows backup manifest, SQLite restore report, and representative media restore reports all pass. A fresh pre-restart production backup was also created with the active live-checkout topology. Remote/offsite backup remains unconfigured; PR #161 requires any remote/offsite waiver to include `approved_by`, `expires_at`, `mitigation`, and `reason`. | HAVN-44 comments through `11004`; HAVN-17 comments through `11005`; evidence report `docs/evidence/havn-44-backup-audit-20260927T115159Z.json` shows local checks pass and `remote_configured=false`; fresh backup/restart packet `docs/evidence/havn-17-prod-backup-restart-20260927T125611Z.json` records backup `ledger-20260927T125611Z.sqlite.gz`, integrity `ok`, mode `0o600`, retention count `7`, and `remote.configured=false`. | Remote/offsite backup target is not configured (`HAVNAI_BACKUP_REMOTE=missing`); offsite proof or explicit waiver is still required. |
| Restart recovery and worker stability | Production coordinator was restarted cleanly after verified live DB backup and came back healthy. Drill tooling is present on production checkout and now emits structured missing-token blocker reports, but `HAVNAI_DRILL_ACCOUNT_TOKEN` is still missing. | HAVN-12 comment `11032`; HAVN-45 comment `11033`; HAVN-49 comment `11034`; HAVN-72 comment `11035`; tracked reports `docs/evidence/havn-12-mixed-model-preflight-missing-token-tracked-20260927T125005Z.json`, `docs/evidence/havn-45-restart-preflight-missing-token-tracked-20260927T125005Z.json`, and `docs/evidence/havn-17-prod-backup-restart-20260927T125611Z.json`. | Needs funded bearer token for mixed-model and accepted-work restart acceptance; coordinator restart smoke alone is not enough. |
| Rollback and operations map | Deploy/runbook docs were corrected to the active live-checkout topology. The coordinator rollback packet is tracked and passes the rollback audit. Web rollback inventory identifies the current production deployment and prior READY rollback candidate, but no Vercel alias promotion/rollback was executed. Node rollback blocker evidence now records one ready node plus missing authoritative node host/service/runtime paths. | Coordinator packet `docs/evidence/havn-46-coordinator-rollback-packet-20260926T211624Z.json`; web inventory `docs/evidence/havn-46-web-rollback-inventory-20260927T121802Z.json`; node blocker `docs/evidence/havn-46-node-rollback-blocker-20260927T122541Z.json`; refreshed rollback report `docs/evidence/havn-46-rollback-audit-20260927T121000Z.json`; HAVN-46 comments through `10993`; HAVN-17 comments through `10994`; HAVN-72 comments through `10995`. | Exercised web/Vercel rollback packet or dated waiver, plus node-runtime rollback packet or dated waiver, still open for final HAVN-17. |
| Astra account/reward core API | Core PR #129 merged (`17e2df8`) and deployed account Astra economy routes; HAVN-56/HAVN-57 moved to In Review. | Validation suite `156 passed` for account/Astra/economy/ledger/jobs; anonymous `/v2/account/astra/stats` returns `401 account_required`; request/response/auth/compatibility contract recorded on Jira. | Claude owns Astra client/game evidence; platform must not close HAVN-14/HAVN-72 until required Astra evidence is accepted. |

## Gate Matrix

| HAVN-72 gate | Owner | Evidence recorded | Remaining blocker |
| --- | --- | --- | --- |
| Map each HAVN-1 gate to owner, issue, and dated evidence | Codex maintains this document; Marcus owns final decision | This file plus HAVN-72 Jira comments through `11035` | Keep updated until final GO/NO-GO |
| Exact commits, deployments, schema/config compatibility, CI, URLs | Codex for web/core; Marcus for production deploy approval | Web production deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`; core runtime `/health.version=215afa5`; core checkout includes `origin/main` through PR #164 plus production local merge `ceaa458`; live API health probes | Web/node rollback evidence or waiver still open; final GO/NO-GO not recorded |
| Account sign-in, automatic Stripe funding/receipt delivery, idempotent replay | Codex platform | HAVN-25 Done. Corrected Stripe event replay remained idempotent: one provider event, one receipt, one funding ledger row, no duplicate operation keys. Web sign-in live CSP includes Clerk/Google after production deploy. | Signed-in production generation/publication journey still needs funded account token. |
| Account-funded generation, artifact delivery, publication, recovery | Codex platform | HAVN-11/HAVN-14/HAVN-44 evidence across account images, deletion/restore tooling, and restore reports | Full signed-in production journey and private mixed-model drill need funded account bearer token |
| Public/private content isolation | Codex platform; Claude provides Astra HAVN-58 evidence | Production forbidden-URL smoke now passes with the exact stale private-media URL returning `404`, `private, no-store`, and Cloudflare `BYPASS`. Anonymous leakage probe passes for known restricted music ID, sensitive search term, public media routes, public logs denial, and web body/meta surfaces. Web public account-studio audit passes with no invite/access-code gating copy. | HAVN-42 still needs signed-in/cross-account evidence or waiver; HAVN-14 final closure still waits for accepted HAVN-58 Astra evidence. |
| Worker stability, mixed-model switching | Codex platform | Drill harness emits structured blocker evidence when the funded token is missing; no public rough outputs are submitted. | Acceptance requires funded account bearer token for private `/v2/jobs`; public rough outputs excluded. |
| Monitoring and alerting | Codex platform | Live observability reports tracked; latest report proves health/control-plane/metrics scrape, alert dry-run, and join-token config are passing. | `HAVNAI_ALERT_WEBHOOK` is missing, so external delivery proof or dated waiver is still open. |
| Backup and restore | Codex platform; Marcus approves remote backup target/waiver | Local backup/restore/media audit report tracked and passing for local evidence; fresh production backup/restart packet records active DB backup integrity before service restart; waiver schema now requires reason. | `HAVNAI_BACKUP_REMOTE` missing; remote target, retention, encryption/access-control evidence or waiver still open. |
| Restart recovery | Codex platform; operator supplies live restart action | Coordinator backup/restart smoke passed and is tracked in `docs/evidence/havn-17-prod-backup-restart-20260927T125611Z.json`; drill tooling deployed; structured no-token preflight blocks accepted-work execution. | `HAVNAI_DRILL_ACCOUNT_TOKEN` missing; needs funded bearer token to prove no lost accepted work or duplicate charge/receipt. |
| Rollback | Codex platform; operator owns web/node rollback action or waiver | Coordinator rollback packet tracked and audit-passing; web rollback inventory tracked with current/prior Vercel deployments and public route smoke; node rollback blocker tracked with required operator inputs; refreshed rollback audit still fails with explicit missing web/node exercise or waiver surfaces. | Web/Vercel and node-runtime rollback evidence packets or dated waivers still open. |
| Public Astra, marketplace, reward paths | Claude for Astra game/client; Codex for core APIs | Core Astra account/reward APIs deployed and contract recorded; HAVN-56/HAVN-57 In Review. | Claude-owned Astra client/browser acceptance and game-quality gates remain open. |
| Public content quality | Codex platform for web/core; Claude for Astra assets | Web public creator audit passed across desktop/iPhone/Android; trust/privacy surfaces deployed; production dashboard/gallery invite copy removed. | Claude-owned Astra asset/game-quality gates remain open. |
| Invite/access-code launch removal | Codex platform | Core invite gating is opt-in; web direct route scan found no invite/access-code/operator-key copy on checked public routes. | No longer a platform launch blocker. |
| Security, trust/privacy, licensing, accessibility/device | Marcus/Codex/Claude by issue | HAVN-68 In Review with live public audit; HAVN-70 trust/privacy surfaces deployed; HAVN-69 bundle/static asset audit merged in PR #137 with 46 production assets scanned and no findings; HAVN-71 licensing inventory merged in PR #138 with model/web/Astra provenance gaps recorded. | Do not silently waive unresolved security/licensing/accessibility gates. HAVN-69 still needs credentialed/session/upload/rate-limit/dependency evidence or exceptions; HAVN-71 still needs model and public-asset clearance evidence or waivers. |
| Launch-day owner, stop/rollback triggers, known limitations, post-launch smoke | Marcus final owner; Codex supplies operations material | `docs/production-operations-runbook.md` evidence template and rollback/restore procedures; `docs/havn-72-launch-day-checklist.md` signoff template | Final owner roster, support owner, monitoring links, stop triggers, and smoke checklist not yet signed off |

## Public Launch Smoke Command

After deploying the release candidate, run the public smoke script and attach the
redacted output to HAVN-72:

```bash
python3 scripts/launch_public_smoke.py --json \
  --forbid-public-url https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3
```

The command uses only public endpoints. It must pass without
`--allow-legacy-gallery` and with all `--forbid-public-url` checks passing
before any GO or CONDITIONAL GO.

Latest live platform checks, 2026-09-27:

- Core runtime `/health.version=215afa5` and coordinator health
  `{"nodes":1,"queue_depth":0,"status":"ok","version":"215afa5"}` after the
  latest production sync through PR #164.
- Tracked public smoke report
  `docs/evidence/havn-72-public-smoke-20260927T115706Z.json` passed with
  `ok=true`: health, healthz, control-plane, hidden legacy gallery, music
  Discover, key web routes, and no invite/access-code copy.
- Web production deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME` is aliased to
  `https://joinhavn.io`.
- Live `trust_surface_audit.mjs` passed against `https://joinhavn.io`.
- Live `creator_journey_public_audit.mjs` passed against `https://joinhavn.io`
  across desktop Chrome, iPhone Safari, and Android Chrome user agents.
- Direct live scan of `/`, `/create`, `/music`, `/video-studio`, `/pricing`,
  `/library`, `/privacy`, `/terms`, and `/sign-in` found no
  invite/access-code/operator-key copy; `/sign-in` CSP still includes
  Google/Clerk.
- Historical forbidden-URL smoke report
  `docs/evidence/havn-14-forbidden-url-smoke-tracked-20260927T124624Z.json`
  captured the stale private-media Cloudflare `HIT` failure. The resolved
  production smoke report
  `docs/evidence/havn-14-forbidden-url-smoke-resolved-20260927T130022Z.json`
  now passes with `ok=true`; the exact URL
  `https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3` returns
  `status=404`, `content-type=application/json`,
  `cache-control=private, no-store`, and `cf-cache-status=BYPASS`.
- Anonymous public leakage probe
  `docs/evidence/havn-42-public-leakage-probe-20260927T1305Z.json` passes
  with `ok=true` for restricted publication
  `music-e73798ff0a7bf6a1305a4ba1` and sensitive term `sexual`: public
  Discover excludes the blocked ID, search returns zero publications, direct
  public music publication/audio/cover routes deny without leaked terms,
  `/logs` returns `401` without leaked terms, and web home/music/create/pricing/
  support/terms/refunds body/meta surfaces contain no restricted ID or term.
- Backup-before-restart evidence is now tracked in
  `docs/evidence/havn-17-prod-backup-restart-20260927T125611Z.json`: live
  service topology is `User=marcus`,
  `WorkingDirectory=/home/marcus/Downloads/source-code/havnai-core`,
  `ExecStart=/home/marcus/Downloads/source-code/havnai-core/.venv/bin/python server/app.py`,
  active DB `/home/marcus/Downloads/source-code/havnai-core/db/ledger.db`;
  backup `ledger-20260927T125611Z.sqlite.gz` passed integrity with mode
  `0o600`, then `havnai-coordinator.service` restarted successfully and
  returned `/health` status `ok`, one node, empty queue, version `07aa5c4`.
  Structured missing-token preflight reports are also tracked for worker
  stability and accepted-work restart:
  `docs/evidence/havn-12-mixed-model-preflight-missing-token-tracked-20260927T125005Z.json`
  and
  `docs/evidence/havn-45-restart-preflight-missing-token-tracked-20260927T125005Z.json`.
- Live observability evidence is tracked in
  `docs/evidence/havn-43-observability-join-token-configured-20260927T1321Z.json`;
  it records passing health/control-plane/admin metrics scrape, alert dry-run,
  and join-token config checks. The remaining blocker is
  `alert_webhook_not_configured`. Earlier admin-token report:
  `docs/evidence/havn-43-observability-admin-token-20260927T1313Z.json`.
- Backup evidence audit is tracked in
  `docs/evidence/havn-44-backup-audit-20260927T115159Z.json`; local
  backup/restore/media evidence passes and remote/offsite proof or waiver
  remains missing.
- Coordinator rollback evidence packet is tracked in
  `docs/evidence/havn-46-coordinator-rollback-packet-20260926T211624Z.json`;
  web rollback inventory is tracked in
  `docs/evidence/havn-46-web-rollback-inventory-20260927T121802Z.json` with
  current production `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`, prior READY candidate
  `dpl_DfqBCeKmRnj7ouZZE5o3sXbM2zy3`, and public route smoke all `200`;
  node rollback blocker evidence is tracked in
  `docs/evidence/havn-46-node-rollback-blocker-20260927T122541Z.json` and
  records one ready node plus missing node host/service/current/previous
  runtime path evidence;
  refreshed rollback audit
  `docs/evidence/havn-46-rollback-audit-20260927T121000Z.json` now passes the
  coordinator surface and remains `passed=false` because exercised web and node
  rollback packets or waivers are missing.

Deployment correction notes, 2026-09-27:

- Coordinator was reachable at `marcus@192.168.4.105`.
- Active service topology is `User=marcus`,
  `WorkingDirectory=/home/marcus/Downloads/source-code/havnai-core`,
  `ExecStart=/home/marcus/Downloads/source-code/havnai-core/.venv/bin/python server/app.py`,
  `HAVNAI_DB_PATH=/home/marcus/Downloads/source-code/havnai-core/db/ledger.db`.
- Old `/opt/havnai/current`, `/var/lib/havnai/ledger.db`, and Linux user
  `havnai` deploy commands are stale for this launch host.
- Core PR #135 updated the production architecture map and operations runbook
  to match this topology; the coordinator has since been backed up and
  restarted successfully, and current production has since advanced to local
  merge `ceaa458`.
- Core PR #151 added `docs/havn-72-launch-day-checklist.md`; it is the launch
  signoff packet template, not a launch approval.

## Final Platform Evidence Commands

Run these from the deployed release candidate or operator context before any
GO/CONDITIONAL GO. Attach redacted JSON output or a dated waiver to the linked
ticket.

| Gate | Command | Passing signal |
| --- | --- | --- |
| Public smoke | `python3 scripts/launch_public_smoke.py --json --forbid-public-url https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3` | `ok=true` without `--allow-legacy-gallery`; exact stale private-media URLs return 401/403/404/410, not public media |
| Legacy gallery cleanup, if regression returns | `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --json` then `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --apply --include-job-ids --json` | Audit/apply packet schema `havn-72-legacy-gallery-cleanup.v1`; active legacy rows delisted, deleted rows `0`, account-owned rows excluded; public smoke passes after cleanup |
| Content isolation | `python3 scripts/launch_public_smoke.py --json --forbid-public-url <exact stale URL>` plus signed-in owner/cross-account checks where available | `passed=true`; public denial for exact stale/private URLs, no public media cache hit, owner access and cross-account denial evidence recorded separately |
| Mixed-model stability | `python3 scripts/mixed_model_worker_drill.py --preflight --account-token <redacted>` then private 30-job drill | funded account preflight passes; private `/v2/jobs` drill completes without public rough outputs; missing-token preflight emits a structured blocker report |
| Monitoring | `HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py --alert-waiver <waiver.json> --join-token-waiver <waiver.json>` when no webhook or join token is configured | `passed=true`; required metrics present; alert dry-run schema returned; webhook delivery sent or dated alert waiver accepted; `SERVER_JOIN_TOKEN` configured or dated join-token waiver accepted |
| Backup/restore | `python3 scripts/backup_evidence_audit.py --backup-manifest <manifest.json> --restore-report <report.json> --media-report <media-report.json> --remote-waiver <waiver.json> --json` | `passed=true`; local backup plus remote retention evidence or dated waiver containing `approved_by`, `expires_at`, `mitigation`, and `reason` |
| Restart recovery | `python3 scripts/account_restart_recovery_drill.py --preflight --account-token <redacted> --restart-command '<approved command>'` then live drill | preflight passes; accepted job survives restart without duplicate charge, receipt, or payout; missing-token preflight emits a structured blocker report |
| Rollback | `python3 scripts/rollback_evidence_audit.py --coordinator-packet <packet.json> --web-packet <packet.json> --node-waiver <waiver.json> --json` | `passed=true`; coordinator, web, and node rollback evidence or dated waivers present |

## Current NO-GO Conditions

- HAVN-12 mixed-model acceptance still lacks a funded commercial account bearer
  token for private `/v2/jobs`; public `/submit-job` rough outputs do not count.
- HAVN-14 cannot close until Claude-owned HAVN-58 Astra evidence is accepted.
  The platform stale private-media CDN URL blocker is resolved by the
  production forbidden-URL smoke report
  `docs/evidence/havn-14-forbidden-url-smoke-resolved-20260927T130022Z.json`.
  Anonymous public leakage checks are also tracked in
  `docs/evidence/havn-42-public-leakage-probe-20260927T1305Z.json`; signed-in
  owner/cross-account evidence or explicit waiver remains separate.
- HAVN-17 cannot close until remote backup/offsite waiver, accepted-work
  restart recovery, rollback evidence or waiver, alert webhook receipt or
  waiver, join-token configuration proof or waiver, and final operations owner
  checklist are recorded.
- Claude-owned Astra game-quality gates and client integration evidence are
  outside Codex's platform branch and remain release blockers.

## Final Decision Checklist

Marcus should not record GO until this document points to dated evidence for:

- release commits and deployed versions for web, core, workers, and config;
- clean public gallery/discover/marketplace probes;
- Stripe funding and idempotent replay evidence accepted for production scope;
- private account generation, artifact delivery, publication, deletion/recovery,
  and mixed-model drill;
- content-isolation tests plus deployed adult/private denial proof;
- monitoring, backup, restore, restart, and rollback packets;
- Astra client/game-quality acceptance from Claude-owned tickets;
- security, trust/privacy, licensing, accessibility/device acceptance;
- launch-day operator, support owner, stop triggers, rollback owner, and
  post-launch smoke route list in `docs/havn-72-launch-day-checklist.md`.

Any CONDITIONAL GO must list the exception owner, expiration date, mitigation,
disabled public surface if applicable, and rollback trigger.

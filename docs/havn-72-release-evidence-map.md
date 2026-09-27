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
| Core baseline | Production `https://api.joinhavn.io` reports `/health.version=2a61013`, one ready node, empty queue. The active systemd service runs from `/home/marcus/Downloads/source-code/havnai-core` as user `marcus`; the production checkout contains `origin/main` through PR #141 plus local merge `2a61013`. | 2026-09-27 live probes; HAVN-17 Jira comments `10864`, `10870`, `10873`, `10876`, `10911`; HAVN-72 comments `10867`, `10871`, `10874`, `10877`, `10915`. | Healthy but not launch complete. |
| Core no-invite, Stripe funding, gallery guard | Core production has invite gating opt-in only. `JOIN_TOKEN` can exist without requiring public invite codes unless `HAVNAI_INVITE_GATING` is explicitly enabled. HAVN-25 automatic Stripe funding/idempotent replay is accepted. | Core PR #134 merged (`a324ee9`) for invite-gating default tests; HAVN-25 live DB evidence shows one Stripe event, one receipt, one funding ledger row, no duplicate operation keys, and balance moved only by later generation spend; HAVN-25 Done. | Invite-code gate removed as blocker; full private mixed-model acceptance still open. |
| Web account launch surfaces and no invite/access-code copy | Production `https://joinhavn.io` deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`, from `havnai-web` commercial branch merge `82e1e9f` after PRs #112/#113/#119. | Local validation: `npx tsc --noEmit`, `npm test` (`80 files / 484 tests`), `npm run build`. Live validation: trust surface audit ok, creator public audit ok across desktop/iPhone/Android user agents, direct scan of `/`, `/create`, `/music`, `/video-studio`, `/pricing`, `/library`, `/privacy`, `/terms`, `/sign-in` found no invite/access-code/operator-key copy; sign-in CSP still includes Google/Clerk. | Deployed for account-first public surfaces; signed-in funded journey still requires account token evidence. |
| Public/private content isolation | Origin behavior denies cache-busted private artifact URL with `404`, `Cache-Control: private, no-store`, `Vary: Authorization`, and `cf-cache-status: BYPASS`. Exact stale CDN URL still serves a cached private artifact as `HTTP/2 200`, `cf-cache-status: HIT`, `content-length: 960813` at 2026-09-27 11:33 UTC. | HAVN-14 comments `10865`, `10879`, `10882`, `10909`; HAVN-42 comments `10866`, `10881`, `10908`; HAVN-72 comment `10915`; Cloudflare purge attempt failed with API auth error 10000. | NO-GO until purge-capable Cloudflare action succeeds or the exact stale URL expires and rechecks as private/not found. HAVN-58 Astra evidence must also remain consumed before final closure. |
| Observability and alerting | Coordinator health/control-plane and admin alert evaluation are deployed. Admin alert send works but `HAVNAI_ALERT_WEBHOOK` is not configured. | HAVN-43 comment `10850`; HAVN-17 comments `10851`, `10864`, `10873`; env recheck 2026-09-27 shows `HAVNAI_ADMIN_TOKEN=configured`, `HAVNAI_ALERT_WEBHOOK=missing`. | External alert delivery receipt or dated waiver still open. |
| Backup/restore repair | Production local backups are scheduled and verified. `havnai-backup.timer` is active; systemd backup run created `ledger-20260927T104900Z.sqlite.gz` plus manifest with `integrity_check=ok`, `local_mode_octal=0o600`, `local_retention_count=5`; `gzip -t` passed. A backup-before-restart run created `ledger-20260927T113154Z.sqlite.gz` with manifest `integrity_check=ok`, `local_mode_octal=0o600`, `local_retention_count=6`, and sha256 `056fb176d60b56844f5d7f1b23e879bf21941b8cf03e587eaaeebe341b8fee2e`. Representative local media restore samples verified DB/source/restored SHA-256 matches for a manifest artifact, an audio asset, and a generated image artifact. | HAVN-44 comments `10872`, `10875`, `10912`; HAVN-17 comments `10873`, `10876`, `10911`. Older local backup file modes were tightened to `600`; restore drill outputs are `700` dirs / `600` files. | Remote/offsite backup target is not configured (`HAVNAI_BACKUP_REMOTE=missing`); offsite proof or explicit waiver is still required. |
| Restart recovery and worker stability | Production coordinator was restarted cleanly after verified live DB backup and came back healthy on version `2a61013`. Drill tooling is present on production checkout (`scripts/mixed_model_worker_drill.py`, `scripts/account_restart_recovery_drill.py`), but env recheck shows `HAVNAI_DRILL_ACCOUNT_TOKEN=missing`. | HAVN-17 comment `10911`; HAVN-45 comment `10913`; HAVN-49 comment `10914`; HAVN-12 comment `10916`. Prior preflights refuse live execution without a funded account token; same accepted-work drill can cover HAVN-45 and HAVN-49 once the token is available. | Needs funded bearer token for mixed-model and accepted-work restart acceptance; coordinator restart smoke alone is not enough. |
| Rollback and operations map | Deploy/runbook docs were corrected to the active live-checkout topology. PR #135 merged to `main` (`e510c69`); production checkout is now synced through PR #141 and the coordinator was restarted successfully on local merge `2a61013`. | Tests `pytest tests/test_deploy_scripts.py tests/test_backup_coordinator.py` -> `6 passed`; HAVN-17 comments `10868`, `10870`, `10911`; HAVN-72 comments `10869`, `10871`, `10915`. | Broader rollback evidence or waiver still open for final HAVN-17. |
| Astra account/reward core API | Core PR #129 merged (`17e2df8`) and deployed account Astra economy routes; HAVN-56/HAVN-57 moved to In Review. | Validation suite `156 passed` for account/Astra/economy/ledger/jobs; anonymous `/v2/account/astra/stats` returns `401 account_required`; request/response/auth/compatibility contract recorded on Jira. | Claude owns Astra client/game evidence; platform must not close HAVN-14/HAVN-72 until required Astra evidence is accepted. |

## Gate Matrix

| HAVN-72 gate | Owner | Evidence recorded | Remaining blocker |
| --- | --- | --- | --- |
| Map each HAVN-1 gate to owner, issue, and dated evidence | Codex maintains this document; Marcus owns final decision | This file plus HAVN-72 Jira comments through `10915` | Keep updated until final GO/NO-GO |
| Exact commits, deployments, schema/config compatibility, CI, URLs | Codex for web/core; Marcus for production deploy approval | Web production deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`; core runtime `/health.version=2a61013`; core checkout includes `origin/main` through PR #141; live API health probes | Node/core/web rollback evidence or waiver still open; final GO/NO-GO not recorded |
| Account sign-in, automatic Stripe funding/receipt delivery, idempotent replay | Codex platform | HAVN-25 Done. Corrected Stripe event replay remained idempotent: one provider event, one receipt, one funding ledger row, no duplicate operation keys. Web sign-in live CSP includes Clerk/Google after production deploy. | Signed-in production generation/publication journey still needs funded account token. |
| Account-funded generation, artifact delivery, publication, recovery | Codex platform | HAVN-11/HAVN-14/HAVN-44 evidence across account images, deletion/restore tooling, and restore reports | Full signed-in production journey and private mixed-model drill need funded account bearer token |
| Public/private content isolation | Codex platform; Claude provides Astra HAVN-58 evidence | Cache-busted private artifact denial proves origin fix, but exact stale CDN object remains public HIT at 2026-09-27 11:33 UTC. Web public account-studio audit now passes with no invite/access-code gating copy. | Cloudflare purge/expiry proof required before HAVN-14/HAVN-42 can close; HAVN-58 evidence must remain accepted. |
| Worker stability, mixed-model switching | Codex platform | Drill harness exists and refuses unsafe live execution without token. | Acceptance requires funded account bearer token for private `/v2/jobs`; public rough outputs excluded. |
| Monitoring and alerting | Codex platform | Live coordinator health ok; admin alert send schema exercised; admin token configured. | `HAVNAI_ALERT_WEBHOOK` missing, so external alert delivery receipt or waiver still open. |
| Backup and restore | Codex platform; Marcus approves remote backup target/waiver | Local backup timer/service verified, backup-before-restart manifest integrity ok, gzip ok, local files mode `600`, SQLite restore drill ok, representative media restore samples ok. | `HAVNAI_BACKUP_REMOTE` missing; remote target, retention, encryption/access-control evidence or waiver still open. |
| Restart recovery | Codex platform; operator supplies live restart action | Coordinator backup/restart smoke passed on `2a61013`; drill tooling deployed; no-token preflight blocks accepted-work execution. | `HAVNAI_DRILL_ACCOUNT_TOKEN` missing; needs funded bearer token to prove no lost accepted work or duplicate charge/receipt. |
| Rollback | Codex platform; operator owns web/node rollback action or waiver | Deploy/runbook topology corrected in PR #135 and production checkout synced. | Coordinator/web/node rollback evidence packets or dated waivers still open. |
| Public Astra, marketplace, reward paths | Claude for Astra game/client; Codex for core APIs | Core Astra account/reward APIs deployed and contract recorded; HAVN-56/HAVN-57 In Review. | Claude-owned Astra client/browser acceptance and game-quality gates remain open. |
| Public content quality | Codex platform for web/core; Claude for Astra assets | Web public creator audit passed across desktop/iPhone/Android; trust/privacy surfaces deployed; production dashboard/gallery invite copy removed. | Claude-owned Astra asset/game-quality gates remain open. |
| Invite/access-code launch removal | Codex platform | Core invite gating is opt-in; web direct route scan found no invite/access-code/operator-key copy on checked public routes. | No longer a platform launch blocker. |
| Security, trust/privacy, licensing, accessibility/device | Marcus/Codex/Claude by issue | HAVN-68 In Review with live public audit; HAVN-70 trust/privacy surfaces deployed; HAVN-69 bundle/static asset audit merged in PR #137 with 46 production assets scanned and no findings; HAVN-71 licensing inventory merged in PR #138 with model/web/Astra provenance gaps recorded. | Do not silently waive unresolved security/licensing/accessibility gates. HAVN-69 still needs credentialed/session/upload/rate-limit/dependency evidence or exceptions; HAVN-71 still needs model and public-asset clearance evidence or waivers. |
| Launch-day owner, stop/rollback triggers, known limitations, post-launch smoke | Marcus final owner; Codex supplies operations material | `docs/production-operations-runbook.md` evidence template and rollback/restore procedures | Final owner roster, support owner, monitoring links, stop triggers, and smoke checklist not yet signed off |

## Public Launch Smoke Command

After deploying the release candidate, run the public smoke script and attach the
redacted output to HAVN-72:

```bash
python3 scripts/launch_public_smoke.py --json
```

The command uses only public endpoints. It must pass without
`--allow-legacy-gallery` before any GO or CONDITIONAL GO.

Latest live platform checks, 2026-09-27:

- Core runtime `/health.version=2a61013` and coordinator health
  `{"nodes":1,"queue_depth":0,"status":"ok","version":"2a61013"}` after a
  verified backup-before-restart run.
- Web production deployment `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME` is aliased to
  `https://joinhavn.io`.
- Live `trust_surface_audit.mjs` passed against `https://joinhavn.io`.
- Live `creator_journey_public_audit.mjs` passed against `https://joinhavn.io`
  across desktop Chrome, iPhone Safari, and Android Chrome user agents.
- Direct live scan of `/`, `/create`, `/music`, `/video-studio`, `/pricing`,
  `/library`, `/privacy`, `/terms`, and `/sign-in` found no
  invite/access-code/operator-key copy; `/sign-in` CSP still includes
  Google/Clerk.
- Exact stale private-media URL still returns a public Cloudflare `HIT`, so
  public/private isolation is still NO-GO until purge or expiry proof.
- Backup-before-restart evidence is recorded in Jira: HAVN-17 comment `10911`,
  HAVN-44 comment `10912`, HAVN-45 comment `10913`, HAVN-49 comment `10914`,
  HAVN-72 comment `10915`, and HAVN-12 blocker comment `10916`.

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
  restarted successfully on local merge `2a61013`.

## Final Platform Evidence Commands

Run these from the deployed release candidate or operator context before any
GO/CONDITIONAL GO. Attach redacted JSON output or a dated waiver to the linked
ticket.

| Gate | Command | Passing signal |
| --- | --- | --- |
| Public smoke | `python3 scripts/launch_public_smoke.py --json` | `ok=true` without `--allow-legacy-gallery` |
| Legacy gallery cleanup, if regression returns | `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --json` then `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --apply --include-job-ids --json` | Audit/apply packet schema `havn-72-legacy-gallery-cleanup.v1`; active legacy rows delisted, deleted rows `0`, account-owned rows excluded; public smoke passes after cleanup |
| Content isolation | `python3 scripts/content_isolation_evidence.py ...` | `passed=true`; public denial, no-store cache, owner access, cross-account denial, and social-preview denial proven |
| Mixed-model stability | `python3 scripts/mixed_model_worker_drill.py --preflight --account-token <redacted>` then private 30-job drill | funded account preflight passes; private `/v2/jobs` drill completes without public rough outputs |
| Monitoring | `HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py` | `passed=true`; required metrics present; alert dry-run schema returned |
| Backup/restore | `python3 scripts/backup_evidence_audit.py ...` | `passed=true`; local backup plus remote retention evidence or dated waiver |
| Restart recovery | `python3 scripts/account_restart_recovery_drill.py --preflight --account-token <redacted> --restart-command '<approved command>'` then live drill | preflight passes; accepted job survives restart without duplicate charge, receipt, or payout |
| Rollback | `python3 scripts/rollback_evidence_audit.py ...` | `passed=true`; coordinator, web, and node rollback evidence or dated waivers present |

## Current NO-GO Conditions

- HAVN-12 mixed-model acceptance still lacks a funded commercial account bearer
  token for private `/v2/jobs`; public `/submit-job` rough outputs do not count.
- HAVN-14/HAVN-42 cannot close while the exact stale private-media CDN URL
  serves `HTTP/2 200` from Cloudflare cache; purge-capable access or cache
  expiry proof is required. Claude-owned HAVN-58 evidence must also remain
  accepted before final closure.
- HAVN-17 cannot close until remote backup/offsite waiver, accepted-work
  restart recovery, rollback evidence or waiver, alert webhook receipt or
  waiver, and final operations owner checklist are recorded.
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
  post-launch smoke route list.

Any CONDITIONAL GO must list the exception owner, expiration date, mitigation,
disabled public surface if applicable, and rollback trigger.

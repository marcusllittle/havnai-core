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
| Core baseline | Production `https://api.joinhavn.io` release `97290aa` reports `/health` ok, one ready node, empty queue, no control-plane alerts | 2026-09-27 live probes; control-plane schema `network-control-plane.v1`; HAVN-72 Jira comments `10532`, `10536`; HAVN-17 comment `10538` | Healthy but not launch complete |
| Core no-invite, worker drill, gallery guard | Production release branch `feat/havn-11-commercial-accounts` at `97290aaf017e9fcef74ee475910d841793c7bc4d`; PR #91 `codex/havn-47-mixed-model-drill` at `16f3c2942cf9be6da028acce190d84f55b1b36ad` remains the broader mixed-model/no-invite harness stack | Public smoke `ok=true`; legacy gallery hidden with `total=0`; release SHA exposed via `/health.version=97290aa`; PR #91 clean with tests `25 passed, 2 subtests passed` and mixed-model preflight tests `8 passed` | Gallery guard deployed; broader private mixed-model acceptance still open |
| Web no-invite, dashboard preview hardening, Google auth restore | Production `https://joinhavn.io` build `kSKPK1O_mEzUFC5yLDaPB`, deployment `dpl_EveZREW24UNrXUBfDt9aryD2L1Dq`, from production branch commit `5c6c1863106a2a73d5f0dea6534b2557d27feb8d` | Public `/create` no-invite smoke passes; live bundle no longer contains hidden-social override; browser proof shows `Continue with Google` and Google handoff includes real `client_id`; HAVN-72 Jira comments `10506`, `10507` | Deployed for web auth/no-invite; wider launch still blocked by core/Astra/ops gates |
| Private content cache hardening | Production release `97290aa`; PR #101 no-store music denials; PR #96 evidence collector/docs at `70096e8aa6d9f992cb770168d269b5c6d620893f` | Public content-isolation collector now passes Discover no-leak and audio/cover denial no-store slices; owner/cross-account/social-preview inputs still missing; HAVN-14 comments `10535`, `10533` | Public denial slice deployed; tokenized private/cross-account/social-preview proof plus HAVN-58 still open |
| Observability and rollback evidence | Production release `97290aa`; PR #97 docs/helpers at `0f59f43782cc9e8700958695e2082e0001c2a2b6` | Live public/local health and control-plane healthy; admin `/metrics` 200; alert dry-run 200 schema `network-alert-dry-run.v1`; HAVN-17 `10538`, HAVN-43 `10539` | External dashboard/alert delivery and web/node rollback exercise or waiver still open |
| Backup/restore repair | Production backup timer plus PR #93 docs/helpers at `1e5bf04f8bef2fae116904bf4e0f9c0e0a3625bc` | `havnai-backup.timer` active/enabled; local backup dir mode `0700`; latest gzip backup integrity read ok; HAVN-44 comment `10540` | Remote backup/30-day retention/encryption/access-control evidence or waiver still open |
| Restart recovery | PR #95 `codex/havn-45-restart-drill` at `8a89d91f35d8d1dc3fda798a85511fb22faec408` | PR #95 clean; restart drill tests `7 passed`; py_compile passed; 2026-09-27 no-token preflight emits redacted plan and refuses live execution; HAVN-45/HAVN-49 comments `10516`, `10517` | Implemented, live funded-account drill still open |
| Astra account/reward core API | PR #94 `codex/havn-56-astra-account-core` at `2957d2298faa4ec9cd2e1711935b3b5f22356a42` | Account-auth session/start/reward/generate-preflight/spend/stats endpoints; tests `26 passed`; py_compile passed; API contract recorded | Core implemented; Claude owns Astra client acceptance |

## Gate Matrix

| HAVN-72 gate | Owner | Evidence recorded | Remaining blocker |
| --- | --- | --- | --- |
| Map each HAVN-1 gate to owner, issue, and dated evidence | Codex maintains this document; Marcus owns final decision | This file plus HAVN-72 Jira comments through `10536` | Keep updated until final GO/NO-GO |
| Exact commits, deployments, schema/config compatibility, CI, URLs | Codex for web/core; Marcus for production deploy approval | Web production deployment `dpl_EveZREW24UNrXUBfDt9aryD2L1Dq`; core production release `97290aa`; live API health/control-plane probes; PR heads above | Node/web rollback evidence or waiver still open; final GO/NO-GO not recorded |
| Account sign-in, automatic Stripe funding/receipt delivery, idempotent replay | Codex platform | HAVN-25 evidence: corrected endpoint `we_1UK1YUFWBrjj49xVEhwYtzEL`, event `evt_1UK1SqFWBrjj49xVbrD9L4AS`, purchase `pur_889711ac10204534b7620d69a7b2ea4d`, duplicate replay stable at one receipt and one funding ledger row | Conventional production checkout acceptance and remaining browser publication/management proof still open |
| Account-funded generation, artifact delivery, publication, recovery | Codex platform | HAVN-11/HAVN-14/HAVN-44 evidence across account images, deletion/restore tooling, and restore reports | Full signed-in production journey and private mixed-model drill need funded account bearer token |
| Public/private content isolation | Codex platform; Claude provides Astra HAVN-58 evidence | Public adult legacy music probe: Discover 200 with no leak; public audio/cover denials 404 with `Cache-Control: no-store`; HAVN-58 currently `In Review`; `content_isolation_evidence.py` records owner/cross-account/social-preview packet requirements | Signed-in owner/private adult generation, cross-account denial, social-preview proof, and HAVN-58 completion still open |
| Worker stability, mixed-model switching | Codex platform | PR #91 adds 30-job private mixed-model drill harness and artifact-only success handling; 2026-09-27 preflight generated 30-job image/video/music plan | Acceptance requires funded account bearer token for private `/v2/jobs`; public rough outputs excluded |
| Monitoring and alerting | Codex platform | Live public/local health pass; control-plane healthy; protected `/metrics` 200; protected alert dry-run 200 schema `network-alert-dry-run.v1`; admin token configured and redacted | External dashboard/alert delivery evidence or waiver still open |
| Backup and restore | Codex platform; Marcus approves remote backup target/waiver | Local backup timer active/enabled; backup dir `0700`; latest gzip backup integrity read ok | Remote target, 30-day retention, encryption/access-control evidence or waiver still open |
| Restart recovery | Codex platform; operator supplies live restart action | PR #95 tests `7 passed`; no-token preflight emits redacted `havn-45-account-restart-recovery-drill-plan.v1`; one restart drill can cover HAVN-45/HAVN-49 | Needs funded bearer token and explicit operator restart action |
| Rollback | Codex platform; operator owns web/node rollback action or waiver | PR #97 rollback audit helper emits `havn-46-rollback-evidence-audit.v1` and identifies coordinator/web/node evidence as missing without packet inputs | Coordinator/web/node rollback evidence packets or dated waivers still open |
| Public Astra, marketplace, reward paths | Claude for Astra game/client; Codex for core APIs | PR #94 records account bearer auth, server-side reward validation, idempotent spend, disabled account reward-image preflight, and wallet-era compatibility boundaries | Claude-owned HAVN-58 remains `In Review`; Astra client/browser acceptance and game-quality gates HAVN-65/HAVN-73-HAVN-78 remain open |
| Public content quality | Codex platform for web/core; Claude for Astra assets | PR #108 withholds dashboard job previews, removes public dashboard media URL exposure, and filters non-account-backed legacy gallery rows in the web fallback; production `/gallery/browse` now returns `total=0`; public smoke `ok=true` | Claude-owned Astra asset/game-quality gates remain open |
| Invite/access-code launch removal | Codex platform | Core PR #91 makes invite gating opt-in; web production `/create` no-invite copy smoke passes with `legacy_prompts=[]` | Coordinator core refresh still needed for final account/private generation paths |
| Security, trust/privacy, licensing, accessibility/device | Marcus/Codex/Claude by issue | HAVN-68, HAVN-69, HAVN-70, HAVN-71 linked to HAVN-72 | All four remain To Do and cannot be silently waived |
| Launch-day owner, stop/rollback triggers, known limitations, post-launch smoke | Marcus final owner; Codex supplies operations material | `docs/production-operations-runbook.md` evidence template and rollback/restore procedures | Final owner roster, support owner, monitoring links, stop triggers, and smoke checklist not yet signed off |

## Public Launch Smoke Command

After deploying the release candidate, run the public smoke script and attach the
redacted output to HAVN-72:

```bash
python3 scripts/launch_public_smoke.py --json
```

The command uses only public endpoints. It must pass without
`--allow-legacy-gallery` before any GO or CONDITIONAL GO.

Latest live run, 2026-09-27:

- Core release `97290aa`.
- `python3 scripts/launch_public_smoke.py --json` -> exit `0`, `ok=true`.
- Passing checks: `api_health`, `api_healthz`, `control_plane`,
  `legacy_gallery_hidden` with `total=0`, `music_discover_public`, web
  home/create/pricing/support/terms/refunds, and `create_no_invite_copy` with
  `legacy_prompts=[]`.

Deployment correction note, 2026-09-27:

- Coordinator was reachable at `marcus@192.168.4.105`.
- `havnai-coordinator.service` had been pinned to stale working directory
  `/home/marcus/Downloads/source-code/havnai-core-havn47-2f8c8e3`; the systemd
  drop-in was backed up and corrected to
  `/home/marcus/Downloads/source-code/havnai-core`.
- The release branch was advanced and restarted through commits `d7e79f7` and
  `97290aa`; public `/health.version` now reports `97290aa`.

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
- HAVN-14 cannot close until owner/private adult generation, cross-account
  denial, social-preview behavior, and Claude-owned HAVN-58 are
  completed/consumed.
- HAVN-17 cannot close until remote backup/waiver, accepted-work restart
  recovery, web/node rollback evidence or waiver, external alert/dashboard
  evidence or waiver, and final operations owner checklist are recorded.
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

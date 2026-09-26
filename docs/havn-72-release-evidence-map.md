# HAVN-72 Release Evidence Map

Updated: 2026-09-26. Owner: Marcus Little. Platform execution: Codex for
`havnai-core` and `havnai-web`. Astra execution: Claude for the Astra game
repository.

This map is the HAVN-72 GO/NO-GO audit surface. It does not approve launch.
Launch requires Marcus to record a dated GO, NO-GO, or CONDITIONAL GO in
HAVN-72 after every open blocker below is either proven closed or explicitly
waived with mitigation.

## Release Candidate Inventory

| Surface | Current candidate | Evidence | Status |
| --- | --- | --- | --- |
| Core baseline | Production `https://api.joinhavn.io` reports `/health` ok, one ready node, empty queue, no control-plane alerts | 2026-09-26 live probes; control-plane schema `network-control-plane.v1` | Healthy but not launch complete |
| Core no-invite, worker drill, gallery guard | PR #91 `codex/havn-47-mixed-model-drill` at `16f3c2942cf9be6da028acce190d84f55b1b36ad` | PR #91 clean; tests `25 passed, 2 subtests passed`; mixed-model preflight tests `8 passed`; legacy gallery cleanup tests `5 passed`; py_compile passed | Implemented, not deployed |
| Web no-invite and dashboard preview hardening | PR #108 `codex/havn-web-no-invite-required` at `c41e102acf5b48fb04bf5420789f807ce7e5bf32` | Focused web/marketplace/create/studio tests `32 passed`; `npx tsc --noEmit`; embedded dashboard scripts parse; dashboard media URL exposure grep clean; GitHub web checks and Vercel green | Implemented, not deployed |
| Private content cache hardening | PR #96 `codex/havn-14-cache-isolation` at `70096e8aa6d9f992cb770168d269b5c6d620893f` | Focused/private denial cache tests, content-isolation collector tests, and broad marketplace/music tests in PR evidence | Implemented, not deployed |
| Observability and rollback evidence | PR #97 `codex/havn-43-observability-doc-refresh` at `0f59f43782cc9e8700958695e2082e0001c2a2b6` | `/health`, control-plane, metrics/alert collector, rollback audit helper, and ops runbook evidence recorded in PR/Jira | Implemented, pending final token-backed refresh and web/node rollback exercise or waiver |
| Backup/restore repair | PR #93 `codex/havn-44-restore-repair` at `1e5bf04f8bef2fae116904bf4e0f9c0e0a3625bc` | Restore repair, restore drill, media sample reports, and backup evidence audit helper recorded in Jira | Implemented, remote-backup evidence still open |
| Restart recovery | PR #95 `codex/havn-45-restart-drill` at `8a89d91f35d8d1dc3fda798a85511fb22faec408` | Private drill harness, account preflight, and duplicate-charge checks pass | Implemented, live funded-account drill still open |
| Astra account/reward core API | PR #94 `codex/havn-56-astra-account-core` at `b21bdf96a2408936f26fe7ce4084394e4eccf128` | Account-auth endpoints, tests, and API contract recorded | Core implemented; Claude owns Astra client acceptance |

## Gate Matrix

| HAVN-72 gate | Owner | Evidence recorded | Remaining blocker |
| --- | --- | --- | --- |
| Map each HAVN-1 gate to owner, issue, and dated evidence | Codex maintains this document; Marcus owns final decision | This file plus HAVN-72 Jira comments `10431`, `10433`, `10437` | Keep updated until final GO/NO-GO |
| Exact commits, deployments, schema/config compatibility, CI, URLs | Codex for web/core; Marcus for production deploy approval | PR heads above; live API health/control-plane probes; PR #108 GitHub/Vercel checks green | Core PRs and web PR #108 not deployed |
| Account sign-in, automatic Stripe funding/receipt delivery, idempotent replay | Codex platform | HAVN-25 evidence: corrected endpoint `we_1UK1YUFWBrjj49xVEhwYtzEL`, event `evt_1UK1SqFWBrjj49xVbrD9L4AS`, purchase `pur_889711ac10204534b7620d69a7b2ea4d`, duplicate replay stable at one receipt and one funding ledger row | Conventional production checkout acceptance and remaining browser publication/management proof still open |
| Account-funded generation, artifact delivery, publication, recovery | Codex platform | HAVN-11/HAVN-14/HAVN-44 evidence across account images, deletion/restore tooling, and restore reports | Full signed-in production journey and private mixed-model drill need funded account bearer token |
| Public/private content isolation | Codex platform; Claude provides Astra HAVN-58 evidence | Platform public/private tests passed; live adult music probes absent/404; HAVN-58 merged evidence consumed; PR #96 adds no-store denied media headers and `content_isolation_evidence.py` final packet tooling | Final deployed signed-in owner/private adult generation plus cross-account denial/cache/social-preview proof still open |
| Worker stability, mixed-model switching | Codex platform | PR #91 adds 30-job private mixed-model drill harness and artifact-only success handling | Acceptance requires funded account bearer token for private `/v2/jobs`; public rough outputs excluded |
| Monitoring and alerting | Codex platform | PR #97 and live control-plane snapshot: one ready node, zero queued/running, no alerts; `collect_observability_evidence.py` gives the final redacted packet command | Final `/metrics` and alert dry-run refresh need admin/node token or waiver; external disk/payment alert delivery still open |
| Backup and restore | Codex platform; Marcus approves remote backup target/waiver | PR #93 restore and media sample reports; local backup timer evidence; `backup_evidence_audit.py` gives the final redacted packet command | Remote backup target, 30-day retention, encryption/access-control evidence or waiver still open |
| Restart recovery | Codex platform; operator supplies live restart action | PR #95 private restart harness and tests; one restart drill can cover HAVN-45/HAVN-49 | Needs funded bearer token and explicit operator restart action |
| Rollback | Codex platform; operator owns web/node rollback action or waiver | Coordinator rollback report completed; Vercel rollback inventory captured; `rollback_evidence_audit.py` gives the final redacted packet command | Web/Vercel rollback exercise or waiver and node-runtime rollback exercise or waiver still open |
| Public Astra, marketplace, reward paths | Claude for Astra game/client; Codex for core APIs | PR #94 records core Astra account/reward/spend/stats contract; HAVN-58 evidence consumed | Claude-owned Astra client and game-quality gates HAVN-65/HAVN-73-HAVN-78 remain open |
| Public content quality | Codex platform for web/core; Claude for Astra assets | PR #108 withholds dashboard job previews, removes public dashboard media URL exposure, and filters non-account-backed legacy gallery rows in the web fallback; PR #91 defaults legacy public gallery closed and adds `scripts/legacy_gallery_cleanup.py` for audited delisting of active wallet-era rows | Production still returns `total: 7` from `/gallery/browse`; PR #91 deploy or coordinator DB cleanup evidence must land before launch |
| Invite/access-code launch removal | Codex platform | Core PR #91 makes invite gating opt-in; Web PR #108 removes visible legacy access-code/operator-key prompts | Deploy/reprobe required; stale coordinators may still return `invite_required` until refreshed |
| Security, trust/privacy, licensing, accessibility/device | Marcus/Codex/Claude by issue | HAVN-68, HAVN-69, HAVN-70, HAVN-71 linked to HAVN-72 | All four remain To Do and cannot be silently waived |
| Launch-day owner, stop/rollback triggers, known limitations, post-launch smoke | Marcus final owner; Codex supplies operations material | `docs/production-operations-runbook.md` evidence template and rollback/restore procedures | Final owner roster, support owner, monitoring links, stop triggers, and smoke checklist not yet signed off |

## Public Launch Smoke Command

After deploying the release candidate, run the public smoke script and attach the
redacted output to HAVN-72:

```bash
python3 scripts/launch_public_smoke.py --json
```

The command uses only public endpoints. It should fail while live
`/gallery/browse` returns unreviewed legacy rows. Use `--allow-legacy-gallery`
only if Marcus has explicitly approved those rows after product-quality review
and the waiver is recorded in HAVN-72.

Latest live run, 2026-09-26:

- `python3 scripts/launch_public_smoke.py --json` -> exit `1`.
- Passing checks: `api_health`, `api_healthz`, `control_plane`,
  `music_discover_public`, web home/create/pricing/support/terms/refunds, and
  `create_no_invite_copy` with `legacy_prompts=[]`.
- Failing check: `legacy_gallery_hidden` with
  `status=200 total=7 allow_legacy_gallery=False`.

## Final Platform Evidence Commands

Run these from the deployed release candidate or operator context before any
GO/CONDITIONAL GO. Attach redacted JSON output or a dated waiver to the linked
ticket.

| Gate | Command | Passing signal |
| --- | --- | --- |
| Public smoke | `python3 scripts/launch_public_smoke.py --json` | `ok=true` without `--allow-legacy-gallery` |
| Legacy gallery cleanup, if deploy is not immediate | `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --json` then `python3 scripts/legacy_gallery_cleanup.py --db-path <coordinator-db> --apply --include-job-ids --json` | Audit/apply packet schema `havn-72-legacy-gallery-cleanup.v1`; active legacy rows delisted, deleted rows `0`, account-owned rows excluded; public smoke passes after cleanup |
| Content isolation | `python3 scripts/content_isolation_evidence.py ...` | `passed=true`; public denial, no-store cache, owner access, cross-account denial, and social-preview denial proven |
| Mixed-model stability | `python3 scripts/mixed_model_worker_drill.py --preflight --account-token <redacted>` then private 30-job drill | funded account preflight passes; private `/v2/jobs` drill completes without public rough outputs |
| Monitoring | `HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py` | `passed=true`; required metrics present; alert dry-run schema returned |
| Backup/restore | `python3 scripts/backup_evidence_audit.py ...` | `passed=true`; local backup plus remote retention evidence or dated waiver |
| Restart recovery | `python3 scripts/account_restart_recovery_drill.py --preflight --account-token <redacted> --restart-command '<approved command>'` then live drill | preflight passes; accepted job survives restart without duplicate charge, receipt, or payout |
| Rollback | `python3 scripts/rollback_evidence_audit.py ...` | `passed=true`; coordinator, web, and node rollback evidence or dated waivers present |

## Current NO-GO Conditions

- Live production `/gallery/browse?limit=1` still returns `total: 7` and
  legacy listing `id=15` (`job-6aa52ed839c8`). Public launch remains blocked
  until PR #91 is deployed or the coordinator DB cleanup helper delists the
  active legacy rows and public smoke is rerun clean.
- HAVN-12 mixed-model acceptance still lacks a funded commercial account bearer
  token for private `/v2/jobs`; public `/submit-job` rough outputs do not count.
- HAVN-14 cannot close until deployed owner/private adult generation,
  cross-account denial, cache headers, and social-preview behavior are proven.
- HAVN-17 cannot close until remote backup/waiver, monitoring token-backed
  refresh or waiver, restart recovery, web/node rollback evidence, and final
  operations owner checklist are recorded.
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

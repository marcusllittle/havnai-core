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
| Core no-invite, worker drill, gallery guard | PR #91 `codex/havn-47-mixed-model-drill` at `8171176ba5a737d430ebda644ec57b07c2cb57e4` | PR #91 clean; tests `23 passed, 2 subtests passed`; py_compile passed | Implemented, not deployed |
| Web no-invite and dashboard preview hardening | PR #108 `codex/havn-web-no-invite-required` at `e28727d2ab9f1a07616a7f861f4e0b43f12c8636` | Focused web tests `17 passed`; `npx tsc --noEmit`; GitHub web checks and Vercel green | Implemented, not deployed |
| Private content cache hardening | PR #96 `codex/havn-14-cache-isolation` at `84bdb521b671b7ba385076a44bf92d4e5383f625` | Focused/private denial cache tests and broad marketplace/music tests in PR evidence | Implemented, not deployed |
| Observability snapshot | PR #97 `codex/havn-43-observability-doc-refresh` at `82c47185153c6b9ddb0480d589e4744fbd86fb7c` | `/health`, control-plane, `/metrics`, alert dry-run evidence recorded in PR/Jira | Implemented, pending final token-backed refresh or waiver |
| Backup/restore repair | PR #93 `codex/havn-44-restore-repair` at `c35f1dfe16a6e10e17a3d779f73ce662b72b99e3` | Restore repair, restore drill, and media sample reports recorded in Jira | Implemented, remote-backup evidence still open |
| Restart recovery | PR #95 `codex/havn-45-restart-drill` at `88c529412c6d609a21d649cf9d7b97c7070776cb` | Private drill harness and duplicate-charge checks pass | Implemented, live funded-account drill still open |
| Astra account/reward core API | PR #94 `codex/havn-56-astra-account-core` at `b21bdf96a2408936f26fe7ce4084394e4eccf128` | Account-auth endpoints, tests, and API contract recorded | Core implemented; Claude owns Astra client acceptance |

## Gate Matrix

| HAVN-72 gate | Owner | Evidence recorded | Remaining blocker |
| --- | --- | --- | --- |
| Map each HAVN-1 gate to owner, issue, and dated evidence | Codex maintains this document; Marcus owns final decision | This file plus HAVN-72 Jira comments `10431`, `10433`, `10437` | Keep updated until final GO/NO-GO |
| Exact commits, deployments, schema/config compatibility, CI, URLs | Codex for web/core; Marcus for production deploy approval | PR heads above; live API health/control-plane probes; PR #108 GitHub/Vercel checks green | Core PRs and web PR #108 not deployed |
| Account sign-in, automatic Stripe funding/receipt delivery, idempotent replay | Codex platform | HAVN-25 evidence: corrected endpoint `we_1UK1YUFWBrjj49xVEhwYtzEL`, event `evt_1UK1SqFWBrjj49xVbrD9L4AS`, purchase `pur_889711ac10204534b7620d69a7b2ea4d`, duplicate replay stable at one receipt and one funding ledger row | Conventional production checkout acceptance and remaining browser publication/management proof still open |
| Account-funded generation, artifact delivery, publication, recovery | Codex platform | HAVN-11/HAVN-14/HAVN-44 evidence across account images, deletion/restore tooling, and restore reports | Full signed-in production journey and private mixed-model drill need funded account bearer token |
| Public/private content isolation | Codex platform; Claude provides Astra HAVN-58 evidence | Platform public/private tests passed; live adult music probes absent/404; HAVN-58 merged evidence consumed; PR #96 adds no-store denied media headers | Final deployed signed-in owner/private adult generation plus cross-account denial/cache/social-preview proof still open |
| Worker stability, mixed-model switching | Codex platform | PR #91 adds 30-job private mixed-model drill harness and artifact-only success handling | Acceptance requires funded account bearer token for private `/v2/jobs`; public rough outputs excluded |
| Monitoring and alerting | Codex platform | PR #97 and live control-plane snapshot: one ready node, zero queued/running, no alerts | Final `/metrics` and alert dry-run refresh need admin/node token or waiver; external disk/payment alert delivery still open |
| Backup and restore | Codex platform; Marcus approves remote backup target/waiver | PR #93 restore and media sample reports; local backup timer evidence | Remote backup target, 30-day retention, encryption/access-control evidence or waiver still open |
| Restart recovery | Codex platform; operator supplies live restart action | PR #95 private restart harness and tests; one restart drill can cover HAVN-45/HAVN-49 | Needs funded bearer token and explicit operator restart action |
| Rollback | Codex platform; operator owns web/node rollback action or waiver | Coordinator rollback report completed; Vercel rollback inventory captured | Web/Vercel rollback exercise or waiver and node-runtime rollback exercise or waiver still open |
| Public Astra, marketplace, reward paths | Claude for Astra game/client; Codex for core APIs | PR #94 records core Astra account/reward/spend/stats contract; HAVN-58 evidence consumed | Claude-owned Astra client and game-quality gates HAVN-65/HAVN-73-HAVN-78 remain open |
| Public content quality | Codex platform for web/core; Claude for Astra assets | PR #108 withholds dashboard job previews; PR #91 defaults legacy public gallery closed | Production still returns `total: 7` from `/gallery/browse`; PR #91 or row delist/review must deploy before launch |
| Invite/access-code launch removal | Codex platform | Core PR #91 makes invite gating opt-in; Web PR #108 removes visible legacy access-code/operator-key prompts | Deploy/reprobe required; stale coordinators may still return `invite_required` until refreshed |
| Security, trust/privacy, licensing, accessibility/device | Marcus/Codex/Claude by issue | HAVN-68, HAVN-69, HAVN-70, HAVN-71 linked to HAVN-72 | All four remain To Do and cannot be silently waived |
| Launch-day owner, stop/rollback triggers, known limitations, post-launch smoke | Marcus final owner; Codex supplies operations material | `docs/production-operations-runbook.md` evidence template and rollback/restore procedures | Final owner roster, support owner, monitoring links, stop triggers, and smoke checklist not yet signed off |

## Current NO-GO Conditions

- Live production `/gallery/browse?limit=1` still returns `total: 7` and
  legacy listing `id=15` (`job-6aa52ed839c8`). Public launch remains blocked
  until PR #91 is deployed or those rows are curated/delisted and reprobed.
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

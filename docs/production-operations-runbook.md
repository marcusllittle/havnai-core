# HavnAI Production Operations Runbook

This runbook is the operator evidence checklist for HAVN-17, HAVN-44,
HAVN-45, and HAVN-46. It records what can be verified from committed tooling,
what must be captured during a production or production-like drill, and what
must be redacted before evidence is attached to Jira.

The deployed component, data ownership, trust-boundary, and evidence ownership
map for HAVN-4 lives in
[`production-architecture-map.md`](production-architecture-map.md).

## Scope

The commercial launch state includes:

- coordinator SQLite database at the private `HAVNAI_DB_PATH`;
- uploaded/generated artifacts under the coordinator output and asset roots;
- account balances, account credit reservations, purchases, payment receipts,
  import receipts, publications, playlists, jobs, and settlement rows;
- worker runtime state under `~/.havnai/current` and `~/.havnai/previous`;
- web deployment, coordinator release, private service environment, and node
  runtime configuration.

Never restore over a live production database. Every drill restores into a new
private directory or isolated host, and every evidence packet must redact
secrets, private prompts, internal account IDs, full filesystem paths, payment
provider IDs, bearer tokens, admin tokens, Clerk tokens, Stripe secrets, wallet
private keys, and unlisted media URLs.

## Targets

| Area | Target | Evidence required |
| --- | --- | --- |
| Coordinator database RPO | 24 hours until an automated timer proves a tighter cadence | Latest backup timestamp, schedule owner, and failed-backup alert path |
| Coordinator database RTO | 4 hours for operator-led restore to a private replacement coordinator | Restore report, smoke checks, and operator timeline |
| Local backup retention | Seven most recent `ledger-*.sqlite.gz` snapshots | `scripts/backup_coordinator.py` output and directory listing with paths redacted |
| Remote backup retention | 30 days when `HAVNAI_BACKUP_REMOTE` is configured | Remote listing with host/path redacted |
| Artifact/media recovery | Same retention window as the artifact lifecycle policy unless deletion has legally expired | Sample restored artifact checksums and deletion-hold notes |
| Accepted-work recovery | No duplicate charge, payout, reward, or receipt for one accepted `job_id` | Job attempt, ledger, payout, and receipt queries before and after restart |
| Rollback | Prior web/coordinator/node version can be restored or a forward fix is chosen with written rationale | Version IDs, actions, health checks, and smoke results |

## Backup Procedure

Before a coordinator release, `scripts/deploy_coordinator_release.sh` runs:

```bash
sudo -u havnai HAVNAI_DB_PATH=/var/lib/havnai/ledger.db \
  /opt/havnai/venv/bin/python "$release/scripts/backup_coordinator.py"
```

Manual backup uses the same script:

```bash
sudo -u havnai \
  HAVNAI_DB_PATH=/var/lib/havnai/ledger.db \
  HAVNAI_BACKUP_DIR=/var/lib/havnai/backups \
  HAVNAI_BACKUP_REMOTE='backup-host:/redacted/havnai/backups' \
  /opt/havnai/venv/bin/python /opt/havnai/current/scripts/backup_coordinator.py
```

The backup script opens the source SQLite database read-only, uses the SQLite
online backup API, runs `PRAGMA integrity_check`, writes a compressed
`ledger-YYYYMMDDTHHMMSSZ.sqlite.gz` snapshot, keeps the seven newest local
snapshots, and prunes optional remote snapshots older than 30 days.

Launch evidence must record:

- command timestamp in UTC;
- backup file timestamp and size;
- local retained backup count;
- remote retained backup count, when configured;
- owner of the backup schedule and failed-backup alert path;
- redacted storage location and access-control group.

## Remote Backup Approval Or Waiver

HAVN-44 cannot close for launch until either remote backups are enabled and
verified or an explicit launch waiver is attached to Jira. The decision must be
written in the HAVN-44 evidence packet and include:

- selected remote target type and retention period, or waiver rationale;
- encryption-at-rest owner and key-rotation owner;
- access-control group with at least two named operator roles;
- restore-read permission test from an operator account that is not the
  coordinator service account;
- redacted remote listing showing the newest backup and retention boundary;
- failed-upload alert path and on-call owner;
- statement that provider/payment secrets, bearer tokens, and private prompts
  are not included in the Jira evidence packet.

If the decision is a waiver, record the temporary compensating controls, expiry
date, and the person accountable for enabling remote retention after launch.

Use the read-only audit helper to create the redacted evidence packet:

```bash
python scripts/backup_evidence_audit.py \
  --backup-dir /redacted/local/backups \
  --remote-listing /redacted/remote-listing.json \
  --remote-retention-days 30 \
  --encryption-owner platform-ops \
  --access-group havnai-backup-operators \
  --restore-read-test operator-read-YYYYMMDD \
  --alert-owner on-call
```

If remote retention is waived, replace the remote fields with
`--waiver-id <jira-comment-or-approval-id> --waiver-expires YYYY-MM-DD`.
The helper reports `passed=false` unless local backups are present and either
remote evidence or an explicit waiver is supplied. It hashes paths before
printing evidence for Jira.

## Restore Drill

Run restore verification only into a new private output directory:

```bash
python scripts/account_restore_drill.py \
  --source /redacted/coordinator/ledger.db \
  --output /redacted/private/restore-drill-YYYYMMDDTHHMMSSZ
```

The restore drill captures committed WAL contents through SQLite online backup,
compresses the snapshot, restores it to a separate file, compares snapshot and
restored SHA-256 values, runs integrity and foreign-key checks, counts every
table, and writes `report.json`.

Launch evidence must include a redacted report with:

- `verified:true`;
- integrity check `ok`;
- zero foreign-key violations;
- table count summary;
- matching snapshot/restored hashes;
- selected account, wallet-link, balance, reservation, job, artifact,
  publication, playlist, payment receipt, marketplace receipt, and import
  receipt smoke checks.

The 2026-09-26 production/prod-like evidence includes a verified SQLite restore
report with zero foreign-key violations, selected account/ledger/job/artifact
table counts, an orphan-attempt repair report, a local backup timer, and a
representative media/artifact recovery sample. HAVN-44 still needs either a
configured remote backup target with retention/access-control evidence or an
explicit waiver for remote retention/encryption before launch closeout.

## Media And Artifact Recovery

Database backup does not copy generated media, source uploads, or worker-local
cache. A production-like recovery drill must choose representative artifacts:

- one generated image or video artifact;
- one music audio artifact plus cover;
- one source/reference asset still inside the recovery window;
- one deleted item protected by a legal/support hold, if any exists.

For each selected artifact, record its redacted `job_id`, artifact kind,
expected owner account, stored byte size, SHA-256 when available, restored byte
size, restored SHA-256, and whether deletion policy permits restoration. Do not
include private prompts or direct unlisted media URLs in Jira.

## Accepted-Work Restart Drill

The drill must prove one accepted job remains correlated by a single `job_id`
and cannot charge or pay twice across restart. Use a staging or production-like
coordinator and worker.

1. Capture pre-drill health:

   ```bash
   curl -fsS https://api.joinhavn.io/health
   curl -fsS https://api.joinhavn.io/healthz
   curl -fsS -H "X-HavnAI-Token: <admin token>" \
     https://api.joinhavn.io/v1/network/control-plane
   ```

2. Submit account-funded work and record the `job_id`, account ID hash, model,
   and UTC timestamp.
3. Restart either `havnai-coordinator.service` or the worker while the job is
   pending or running.
4. Wait for terminal success, terminal failure, or a documented safe retry path.
5. Run read-only checks for the same `job_id`:

   ```sql
   SELECT id, status, owner_account_id FROM jobs WHERE id = '<job_id>';
   SELECT job_id, attempt_count, execution_status, settlement_outcome
     FROM job_settlement WHERE job_id = '<job_id>';
   SELECT event_type, amount, reason FROM account_credit_ledger
     WHERE job_id = '<job_id>' ORDER BY id;
   SELECT COUNT(*) FROM account_credit_reservations WHERE job_id = '<job_id>';
   SELECT COUNT(*) FROM account_payment_receipts WHERE purchase_id IN (
     SELECT purchase_id FROM account_credit_ledger WHERE job_id = '<job_id>'
   );
   SELECT COUNT(*) FROM node_payouts WHERE job_id = '<job_id>';
   ```

Acceptance requires one intended account charge/capture path, no duplicate
reservation capture, no duplicate payout/reward, no duplicate receipt, and
artifact ownership still attached to the same account.

## Rollback Drill

The coordinator deploy helper records the previous release and rolls back on a
failed local `/healthz` check:

```bash
bash scripts/deploy_coordinator_release.sh <main-sha>
```

The drill evidence must record:

- source version, target version, and previous version;
- exact command or platform action;
- release timestamp and rollback timestamp in UTC;
- `RELEASE_SHA` before and after rollback;
- local and public `/healthz` results;
- queue/control-plane state;
- account auth, credit-package, account library, pricing, support, static, and
  media smoke checks;
- migration or environment-change review explaining whether rollback is safe or
  whether a forward fix/compensating migration is required.

For web rollback, record the Vercel deployment ID promoted or restored and
smoke `joinhavn.io` routes after the action. For node rollback, follow
`docs/RUN_A_NODE.md`: stop the node, replace `~/.havnai/current` from
`~/.havnai/previous`, start the node, then verify heartbeat and model
capabilities.

As of 2026-09-26, coordinator rollback was exercised and restored with
local/public health checks and control-plane smoke evidence. Vercel deployment
inventory for `havnai-web` identified production rollback candidates and
production route smoke checks, but no Vercel promotion/rollback was executed.
Node-runtime rollback is documented in `docs/RUN_A_NODE.md` and still requires a
heartbeat/model-capability exercise or waiver.

Use the read-only rollback audit helper to assemble the redacted HAVN-46 packet:

```bash
python scripts/rollback_evidence_audit.py \
  --coordinator-evidence /redacted/coordinator-rollback.json \
  --web-evidence /redacted/vercel-rollback.json \
  --node-evidence /redacted/node-rollback.json
```

For any surface that is waived instead of exercised, provide
`--web-waiver-id`, `--web-waiver-expires`, `--node-waiver-id`, or
`--node-waiver-expires` with the approval recorded in Jira. The helper reports
`passed=false` until coordinator, web, and node rollback evidence are each
exercised or explicitly waived.

## Evidence Packet Template

Each Jira evidence packet should include:

```text
Drill:
Environment:
Operator:
UTC start:
UTC end:
Versions:
Commands/actions:
Job IDs or artifact IDs, redacted:
Health checks:
Read-only database checks:
Outcome:
Residual risks:
Redactions applied:
```

Attach or link reports only after removing secrets, internal paths, private
prompts, account emails, raw account IDs, provider IDs, unlisted media URLs, and
tokens. Keep the unredacted packet in the private operations store.

## Launch-Day Operations Checklist

HAVN-72 final acceptance needs a dated owner roster, stop triggers, rollback
owners, and post-launch smoke checks. Fill this table in Jira before GO or
CONDITIONAL GO. Do not put secrets, personal phone numbers, tokens, private
paths, raw account IDs, or unlisted media URLs in the public ticket.

| Role | Named owner required before GO | Evidence or handoff required |
| --- | --- | --- |
| Release decision owner | Marcus Little | HAVN-72 dated GO, NO-GO, or CONDITIONAL GO |
| Coordinator deploy owner | Platform operator | Core commit, deploy timestamp, health checks, rollback owner |
| Web deploy owner | Platform web operator | Vercel deployment ID, production URL smoke checks, rollback candidate |
| Worker/node owner | Node operator | Ready heartbeat, model capability smoke, node rollback path |
| Payments owner | Platform/payment operator | Stripe webhook health, funding replay evidence, reconciliation contact |
| Support owner | Marcus or named support lead | Support intake URL, escalation path, known-limitations note |
| Astra owner | Claude/Marcus coordination | Astra acceptance links and any disabled/deferred public surfaces |

### Pre-Launch Stop Triggers

Any of the following should block GO until fixed or explicitly waived by Marcus
with mitigation and an expiration:

- public gallery, Discover, marketplace, social preview, or game route exposes
  unreviewed, private, adult-restricted, or account-owned content;
- `/health` is non-200, control-plane status is not healthy, no ready worker is
  available for launch-critical task types, or queue/running jobs cannot drain;
- account sign-in, credits, Stripe webhook funding, receipts, account library,
  artifact delivery, deletion/recovery, publication, or download fails in the
  signed-in smoke path;
- duplicate funding, duplicate receipt, duplicate account charge/capture,
  duplicate node payout, or duplicate reward is observed;
- backup is older than the approved RPO, restore drill is unverified, or remote
  backup is neither configured nor waived;
- rollback owner cannot identify the previous web/coordinator/node version;
- monitoring/alert dry-run cannot be refreshed and no waiver exists;
- Claude-owned Astra quality/client gates are still public but unaccepted.

### Post-Launch Smoke Route List

Run these checks after every production deploy, rollback, DNS change, or
launch-day configuration change. Record UTC timestamp, route, status, owner, and
redacted response summary.

| Surface | Smoke check | Healthy signal |
| --- | --- | --- |
| Core liveness | `GET https://api.joinhavn.io/health` | 200, `status=ok`, expected version |
| Core readiness | `GET https://api.joinhavn.io/healthz` | 200, ready |
| Control plane | `GET /v1/network/control-plane` with admin token when required | healthy, no critical alerts, ready node present |
| Monitoring evidence | `HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py` | `passed=true`, required metrics present, alert dry-run schema returned |
| Model catalog | `GET /models/list` | launch-critical image/video/music models mapped to ready capacity |
| Public gallery | `GET /gallery/browse?limit=1` | zero unreviewed legacy rows unless explicitly reviewed/opted in |
| Music Discover | `GET /music/discover` | public-only rows; no private/adult media leaks |
| Web home | `GET https://joinhavn.io/` | 200 and current production deployment |
| Create | `GET https://joinhavn.io/create` | 200, account/credits launch copy, no invite-code gate |
| Pricing/policies | `/pricing`, `/terms/credits-v1`, `/refunds/credits-v1`, `/support` | 200 and policy links current |
| Account auth | signed-in browser smoke | account loads, balance visible, no wallet prompt for account generation |
| Account generation | signed-in private job smoke | job reaches terminal state or documented safe retry without duplicate charge |
| Artifact access | owner and cross-account smoke | owner can view/download; other account denied with no-store denial headers |
| Restart/rollback | drill-specific smoke | same `job_id` retains ownership and has no duplicate charge/payout/receipt |

### Known-Limitations Template

Use this text shape for HAVN-72 if any CONDITIONAL GO is requested:

```text
Limitation:
Public surface affected:
Owner:
Expires:
Mitigation:
Disabled feature or reduced scope:
Customer/support impact:
Rollback or stop trigger:
Evidence link:
```

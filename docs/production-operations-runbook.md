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

The committed local evidence as of 2026-09-26 is useful implementation proof,
but it is not sufficient for launch by itself. HAVN-44 needs a production or
production-like restore report and media/artifact recovery sample.

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

# HAVN-45 restart acceptance evidence

`scripts/account_restart_recovery_drill.py` defaults to a plan only. Account
preflight can pass its own identity/balance check, but both outputs explicitly
set `acceptance_passed=false`. They do not prove recovery. Historical legacy
SQL evidence returned by `readonly_db_checks` is likewise non-accepting.

Full `--execute` requires a usable `--db-path` before job submission. Connections
use SQLite `mode=ro`; each evidence read uses one consistent transaction. The
drill never migrates or repairs the evidence database.

Immediately before `restart_marker`, the authenticated job read must explicitly
identify the submitted job and the same owner as its reserved database row,
in `queued`, `leased`, `running`, or `uploading` state. Unknown or terminal states
abort without invoking the restart command. This is an observation immediately
before the command, not a lock preventing worker completion during that interval.

Current-main contracts used by the acceptance checks:

- `server/account_ledger.py`: generation entries use `resource_id=job_id`,
  `operation`, `settled_delta`, and `reserved_delta`. Before restart there must
  be exactly one positive reservation in `reserved` state and one matching
  reserve entry, with no capture. After terminal success the same reservation
  must be `captured`, with exactly one capture debiting its units and releasing
  the same reserved units. Unexpected charge/debit aliases or other generation
  operations fail. Ownership and the reserved amount must remain unchanged.
- `server/account_payments.py`: receipts belong to funding purchases, not
  generation jobs. There is no job-to-purchase join in the current account
  ledger. Zero job receipts is normal. Multiple distinct purchases/payments are
  valid; duplicate purchase/payment receipt groups for this owner fail.
- `server/settlement.py` and `server/rewards.py`: at most one legacy payout and
  reward record may exist per job. Zero is valid: the account enqueue route does
  not create a legacy job-settlement record, payout creation is conditional on
  positive reward, and `record_reward` returns for an empty wallet. Both counts
  must be zero before restart and at most one after success. A legacy
  `job_settlement` row is therefore not mandatory for account-job acceptance.

Missing required tables/columns, missing jobs, inconsistent ownership, duplicate
counts, or unsettled reservations fail closed. The script accepts only terminal
success; failed/cancelled jobs remain non-accepting even when safely released.

Synthetic tests exercise the real drill orchestration with a fake account client,
a fake restart callback, and disposable SQLite databases. They prove harness
behavior, not GPU restart recovery or production settlement atomicity. Production
acceptance still requires the funded private account, approved bounded restart,
actual service/worker recovery evidence, and artifact ownership/access checks
specified by `production-operations-runbook.md`.

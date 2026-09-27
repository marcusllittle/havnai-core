# HAVN-44 Backup And Media Evidence - 2026-09-27

This evidence refresh records the current coordinator backup timer, scheduled
SQLite backup inventory, and local media scope for the HAVN-17 operations rollup.

## Proven

- `havnai-backup.timer` is enabled and active on the coordinator host.
- `havnai-backup.service` runs as `marcus` with a restricted systemd service.
- The timer runs `scripts/backup_coordinator.py` from the live coordinator
  checkout and writes to `/home/marcus/havnai-backups/coordinator`.
- Recent journal evidence shows successful scheduled backup runs on
  2026-09-26 and 2026-09-27.
- The generated backup audit found two local scheduled backup archives.
- Newest scheduled backup:
  `ledger-20260927T060004Z.sqlite.gz`, 43,958,099 bytes.
- Local media inventory exists for:
  - `static/assets`: 10 files, 16,237,527 bytes.
  - `static/outputs`: 3,338 files, 4,958,931,848 bytes.

## Not Proven

- Remote/offsite backup listing.
- Remote retention period.
- Backup encryption owner and access group.
- Remote restore-read test.
- External backup alert owner.
- Offsite media/artifact backup and restore-read scope.

The generated `docs/evidence/havn-44-backup-audit-20260927.json` report
therefore intentionally has `"passed": false` with
`remote_backup_evidence_or_waiver` missing.

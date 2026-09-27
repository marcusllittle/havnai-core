# HAVN-44 Production Media And Restore Evidence

Collected from coordinator host `marcus@192.168.4.105` on 2026-09-27T10:36:15Z while the deployed coordinator reported version `5c92891`.

## Proven

- Production SQLite restore drill passed and did not overwrite the source database.
- Restore report: `db/restore-drills/restore-20260927T103332Z/report.json`.
- Restore result: `verified=true`, `table_count=84`, `integrity=ok`, `foreign_key_violations=0`.
- Snapshot and restored database hashes matched: `f9c56190246c338420fecc78fa21cac69fb74ce81a07e788907553ca6af8419a`.
- Latest backup manifest was present at `db/backups/ledger-20260927T100837Z.manifest.json` with `integrity_check=ok`.
- Local media inventory was captured:
  - `static/outputs`: 3,338 files, 4,958,931,848 bytes.
  - `static/assets`: 10 files, 16,237,527 bytes.
  - Database rows: 755 `artifacts` rows and 10 `assets` rows.

## Not Proven

- Offsite backup is not configured in the captured manifest: `remote.configured=false`.
- Media file backup/restore is not verified by the SQLite restore drill.
- The database inventory observed 10 artifact rows whose files were not present on disk.
- Alert webhook delivery remains unconfigured.
- Funded account restart/mixed-model drill token remains unconfigured.

## Evidence File

The structured evidence snapshot is in `docs/evidence/havn-44-production-media-inventory-20260927.json`.

## Launch Conclusion

This evidence strengthens HAVN-44 for local production database restore and media inventory. It does not close the launch gate. HAVN-44 still needs offsite/media-file restore proof or an explicit waiver.

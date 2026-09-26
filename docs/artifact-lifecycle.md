# Artifact lifecycle: HAVN-26/27/28

Account owners can `DELETE /v2/jobs/:id` to soft-delete a completed generation
and all its output artifacts. Active jobs must be cancelled and reach a terminal
state first. The action is distinct from reversible Collection hiding.

Deletion and publication/marketplace mutations serialize through SQLite write
transactions. Deletion unpublishes music, delists marketplace entries and detaches
playlist placements. History and private artifact routes hide deleted outputs;
republishing and selling are rejected. Likes, plays, financial receipts, ledger
entries, ownership logs and import records are retained. An already-downloaded
copy or a stream authorized before deletion cannot be recalled.

`GET /v2/account/deleted-generations` returns minimal owner-only recovery metadata.
`POST /v2/jobs/:id/restore` restores private visibility within 30 days; it never
republishes, relists or restores playlist placement. Retry does not reset the
original deadline. A new deletion after restoration starts a new window.

The music studio and Collection expose confirmation actions; Collection keeps
its existing hide action separate. `/account/deleted` exposes private restoration.
No wallet operation is used. The video source, extracted-frame and sequence paths
also reject deleted generations, including deletion during frame extraction.
Stale submission recovery returns `generation_deleted` without re-enqueueing or
charging again. Other studio delete controls, comprehensive cross-surface review
and live browser acceptance remain part of HAVN-28; this implementation checkpoint
does not close that ticket.

## Purge operation

`artifact_lifecycle_holds` records admin/legal/support/dispute/settlement holds.
An unreleased hold prevents physical purge, but does not extend owner recovery.
Hold administration must remain operator-only; no customer endpoint grants it.

An operator with coordinator database access can place and release a named hold:

```sh
python server/artifact_lifecycle.py --database /path/to/coordinator.db --hold job-ID --hold-id support-case-123 --actor operator-id --reason support
python server/artifact_lifecycle.py --database /path/to/coordinator.db --release-hold job-ID --hold-id support-case-123 --actor operator-id
```

Allowed reasons are `admin`, `legal`, `support`, `dispute` and `settlement`.
Do not put customer details or secrets in these identifiers. Each action records
the operator and hold ID in the lifecycle audit; identical retries add no event.
Released hold IDs cannot be reused. New holds after completed purge are rejected.
Holds and purge serialize under the same database lock.

After migration and a backup/restore check, the scheduler can invoke:

```sh
python server/artifact_lifecycle.py --database /path/to/coordinator.db --outputs-dir /path/to/outputs --limit 25
```

The service/timer templates in `deploy/systemd/havnai-artifact-purge.*` are not
installed or enabled by this change. Configure both paths explicitly. Purge
rechecks expiry and holds under a write lock, refuses shared or out-of-root
paths (including upload references), removes registered artifact files and retains
database audit rows. Missing files are safe on retry. Failures return a nonzero
exit status and must be monitored. Last-attempt ordering prevents a failed oldest
item from starving the remainder of the batch. Apply the additive schema migration
before running the scheduler. Unregistered worker caches/original copies require a separate inventory
and cleanup review before claiming complete physical reclamation.

Production activation, storage inventory, operational hold procedures and live recovery
evidence remain open. Do not infer launch readiness from fixture tests.

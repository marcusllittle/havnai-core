# HAVN-46 Rollback Inventory Refresh - 2026-09-27

This evidence refresh records the current web rollback inventory for the HAVN-17
operations rollup.

## Web

- Current production deployment: `dpl_EveZREW24UNrXUBfDt9aryD2L1Dq`
- Current production URL: `https://havnai-4hyjn1khu-marcus-littles-projects.vercel.app`
- Previous ready production deployment: `dpl_GS1V5fev2NeNCmUieuGiB4aj8aau`
- Previous ready production URL: `https://havnai-1ewjauwyz-marcus-littles-projects.vercel.app`
- Public smoke routes checked on `https://joinhavn.io`: `/`, `/create`, `/pricing`, `/support`
- Result: web rollback inventory passes as `inventory_only`.

## Still Missing

- Coordinator rollback exercise or accepted waiver.
- Node runtime rollback exercise or accepted waiver.

`docs/evidence/havn-46-rollback-audit-20260927.json` is the generated audit
report. It intentionally has `"passed": false` until coordinator and node
surfaces have evidence or waivers.

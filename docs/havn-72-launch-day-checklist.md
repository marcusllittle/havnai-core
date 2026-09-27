# HAVN-72 Launch-Day Checklist

This checklist is the dated GO, NO-GO, or CONDITIONAL GO packet for Marcus
Little to complete before any public launch announcement or expanded promotion.
It does not approve launch by itself. Every item needs dated evidence or an
explicit owner-approved waiver with mitigation and expiration.

## Decision

| Field | Value |
| --- | --- |
| Decision | GO / NO-GO / CONDITIONAL GO |
| Decision timestamp UTC | |
| Release owner | Marcus Little |
| Platform operator | |
| Web operator | |
| Coordinator/API operator | |
| Worker/node operator | |
| Support owner | |
| Astra/game acceptance owner | Claude-owned evidence; Marcus final acceptance |

## Required Evidence Links

| Gate | Evidence or waiver |
| --- | --- |
| Release commits and deployments | |
| Public smoke report | `docs/evidence/havn-72-public-smoke-20260927T115706Z.json` |
| Public/private content isolation | |
| Stripe funding and idempotent replay | HAVN-25 Done evidence |
| Signed-in funded account generation | |
| Mixed-model worker drill | |
| Backup/restore audit | `docs/evidence/havn-44-backup-audit-20260927T115159Z.json` |
| Observability/alert evidence | `docs/evidence/havn-43-observability-20260927T114343Z.json` |
| Rollback audit | `docs/evidence/havn-46-rollback-audit-20260927T115508Z.json` |
| Security and abuse controls | |
| Licensing/provenance | |
| Astra client/game-quality acceptance | |

## Stop Triggers

Stop launch or roll back/disable affected surfaces when any of these are true:

- exact private-media URLs are publicly reachable without authorization;
- public smoke fails on health, control plane, web core routes, or hidden legacy
  gallery checks;
- account sign-in, pricing, funding, or receipt creation fails for production
  users;
- accepted jobs are lost, duplicate-charged, duplicate-paid, or lose account
  ownership after restart;
- alert delivery is unavailable without a dated waiver and manual monitoring
  owner;
- remote/offsite backup is unavailable without a dated waiver and local-retention
  mitigation;
- a rollback attempt fails and no forward-fix rationale is recorded;
- unresolved security, licensing, or Astra quality gates are not explicitly
  waived by owner.

## Rollback Actions

| Surface | Primary action | Evidence required after action |
| --- | --- | --- |
| Web | Promote prior Vercel deployment or disable affected route/surface | Deployment ID, route smoke, owner timestamp |
| Coordinator/API | Use documented checkout/deploy rollback or forward fix | Version IDs, `/health`, `/healthz`, control-plane, account smoke |
| Worker/node | Restore `~/.havnai/current` from `~/.havnai/previous` or stop node | Heartbeat, model capability, queue state |
| Cloudflare cache | Purge exact file URL or wait for TTL expiry and recheck exact URL | Headers showing private/not-found without query params |
| Payments/account funding | Disable purchase entry points or switch to support-only recovery | Policy page route smoke, Stripe event audit |

## Post-Launch Smoke

Run and attach these within the launch window:

```bash
python3 scripts/launch_public_smoke.py --json
HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py --json
```

Also record:

- `https://joinhavn.io` deployment ID and alias;
- `https://api.joinhavn.io/health` version;
- exact Cloudflare private-media isolation check;
- funded account generation job ID hash and ledger/receipt summary;
- support inbox/contact path;
- manual monitoring owner while any alert waiver is active.

## Conditional GO Exceptions

| Exception | Owner | Expires | Mitigation | Disabled surface | Rollback trigger |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

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
| Public smoke report | Latest tracked smoke with forbidden URL: `docs/evidence/havn-14-forbidden-url-smoke-resolved-20260927T130022Z.json`; historical failing stale-cache report: `docs/evidence/havn-14-forbidden-url-smoke-tracked-20260927T124624Z.json` |
| Public/private content isolation | Platform forbidden-URL smoke now returns `ok=true`; anonymous public leakage probe: `docs/evidence/havn-42-public-leakage-probe-20260927T1305Z.json`; HAVN-14 final closure still waits for accepted Claude-owned HAVN-58 Astra evidence |
| Stripe funding and idempotent replay | HAVN-25 Done evidence |
| Signed-in funded account generation | |
| Mixed-model worker drill | Missing-token blocker: `docs/evidence/havn-12-mixed-model-preflight-missing-token-tracked-20260927T125005Z.json`; final drill still required |
| Backup/restore audit | `docs/evidence/havn-44-backup-audit-20260927T115159Z.json`; fresh production backup/restart packet `docs/evidence/havn-17-prod-backup-restart-20260927T125611Z.json` |
| Observability/alert evidence | `docs/evidence/havn-43-observability-admin-token-20260927T1313Z.json`; metrics scrape and alert dry-run pass with admin token; attach real webhook receipt / join-token config proof or approved waivers |
| Rollback audit | `docs/evidence/havn-46-rollback-audit-20260927T121000Z.json`; web/node rollback packet or waiver still required |
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
python3 scripts/launch_public_smoke.py --json \
  --forbid-public-url https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3
HAVNAI_ADMIN_TOKEN=<redacted> python3 scripts/collect_observability_evidence.py --json \
  --alert-waiver docs/evidence/<approved-alert-waiver>.json \
  --join-token-waiver docs/evidence/<approved-join-token-waiver>.json
```

Also record:

- `https://joinhavn.io` deployment ID and alias;
- `https://api.joinhavn.io/health` version;
- latest production backup/restart packet if restart occurs during launch;
- exact Cloudflare private-media isolation check;
- funded account generation job ID hash and ledger/receipt summary;
- support inbox/contact path;
- manual monitoring owner while any alert waiver is active.
- node enrollment owner and mitigation while any join-token waiver is active.

Omit `--alert-waiver` only when `HAVNAI_ALERT_WEBHOOK` is configured and the
collector records a real webhook delivery receipt. Omit `--join-token-waiver`
only when `SERVER_JOIN_TOKEN` is configured and the collector records it as
present.

## Conditional GO Exceptions

| Exception | Owner | Expires | Mitigation | Disabled surface | Rollback trigger |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

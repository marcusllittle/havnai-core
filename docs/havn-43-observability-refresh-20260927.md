# HAVN-43 Observability Evidence Refresh - 2026-09-27

This evidence refresh records the live coordinator observability collector output
for the HAVN-17 operations rollup.

## Proven

- `/health` returned HTTP 200 with coordinator version `cc69a24`.
- `/healthz` returned HTTP 200 with `ok: true`.
- `/v1/network/control-plane` returned HTTP 200 with:
  - schema `network-control-plane.v1`
  - health `healthy`
  - 1 ready node
  - 0 queued jobs
  - 0 running jobs
  - 0 active alerts
- `/metrics` returned HTTP 200 with required metric groups present:
  - jobs: `havnai_jobs`
  - workers online: `havnai_nodes_online`
  - disk free: `havnai_output_disk_free_bytes`
- `/v1/network/alerts/dry-run` returned schema
  `network-alert-dry-run.v1`.
- Injected alert dry-run matched 3 rules for
  `model_load_failures,gpu_vram_exhaustion`.
- Dry-run delivery mode was `dry_run`; no external notification was sent.

## Not Proven

- External alert delivery to a real on-call/webhook destination.
- Saved external dashboard link, screenshot, or provider evidence.
- Launch waiver for external alert/dashboard delivery.

The generated `docs/evidence/havn-43-observability-20260927.json` report passes
the internal collector, but HAVN-17 still needs external delivery/dashboard
evidence or an accepted waiver before launch readiness can close.

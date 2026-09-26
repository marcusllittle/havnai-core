# HavnAI Production Observability

This document records the launch-readiness observability surface for HAVN-43.
It separates production endpoints from local/test evidence so acceptance can be
audited without exposing secrets.

Backup, restore, accepted-work restart, rollback, RPO/RTO, and redacted
evidence capture procedures live in
[`production-operations-runbook.md`](production-operations-runbook.md).
The production component and ownership map lives in
[`production-architecture-map.md`](production-architecture-map.md).

## Production Health Checks

| Area | Endpoint or check | Expected healthy signal |
| --- | --- | --- |
| Public ingress | `GET https://api.joinhavn.io/health` | HTTP 200, `status=ok`, current deployed `version` |
| Coordinator queue | `GET https://api.joinhavn.io/v1/network/control-plane` | `queue.queued=0`, `queue.running=0` when idle, no critical alerts |
| Worker fleet | `GET https://api.joinhavn.io/nodes` and `/v1/network/control-plane` | At least one online ready creator node, zero stale/offline launch nodes |
| Model catalog | `GET https://api.joinhavn.io/models/list` | Expected models listed with available nodes for launch-critical task types |
| Metrics scrape | `GET /metrics` with admin token | Prometheus gauges/counters for jobs, artifacts, failures, workers, disk |
| Storage/disk | `/metrics` | `havnai_output_disk_free_bytes` above operator threshold |
| Public music surface | `GET /music/discover` | HTTP 200 with public-only rows; adult/private content omitted |
| Alert route | `GET /v1/network/alerts/dry-run` with admin token | Dry-run payload reports matched alerts without sending external notifications |

## Metrics Coverage

The production `/metrics` endpoint exports:

- job state counts by `queued`, `running`, `succeeded`, `failed`, and
  `cancelled`;
- artifact count and bytes;
- categorized failures through `havnai_failures_total`;
- worker online/readiness, busy state, and last heartbeat age;
- output disk free bytes;
- worker model failure and unhealthy gauges when workers report model-health
  errors.

The control-plane JSON endpoint complements Prometheus with:

- queue depth, oldest wait, running count, completed count, and failed count;
- node online/ready/busy/offline counts;
- claim and receipt health;
- normalized health status and alert list.

## Alert Coverage

The admin-gated dry-run endpoint supports production-safe alert verification.
It can evaluate real control-plane state and inject synthetic conditions for
model/GPU paths without sending notifications.

Required alert classes and current evidence path:

| Alert class | Evidence path |
| --- | --- |
| Offline/no creator workers | `/v1/network/control-plane` health alerts and dry-run evaluation |
| Stuck queue/jobs | control-plane queue fields and dry-run stuck-queue matcher |
| Error spikes | dry-run `job_error_spike` matcher from recent failed/expired jobs |
| Model-load failures | worker `model_health` metrics plus dry-run `model_load_failures` injection |
| GPU/VRAM exhaustion | dry-run `gpu_vram_exhaustion` injection |
| Public ingress failure | health endpoint and dry-run public-ingress matcher |
| Disk/storage pressure | `/metrics` disk gauge; external threshold notification remains an operator rollout item |
| Payment/funding failures | account payment/receipt logs and Stripe dashboard correlation; external threshold notification remains an operator rollout item |

## Production Evidence Snapshots

Initial snapshot captured on 2026-09-26 after deploying core commit
`6456009dfecdf0cbc36af38cdc937e018ddd3931`:

```text
GET /health
{"nodes":1,"queue_depth":0,"status":"ok","version":"6456009"}
```

`/v1/network/control-plane` reported healthy status, no alerts, one tracked
online/ready creator node, zero busy/offline nodes, zero queued/running jobs,
zero active or at-risk claims, and zero unbatched receipts.

`/metrics` reported the worker online gauge, output disk free bytes, job-state
gauges, and categorized historical failures.

The admin dry-run request
`/v1/network/alerts/dry-run?inject=model_load_failures,gpu_vram_exhaustion`
returned schema `network-alert-dry-run.v1`, `delivery.mode=dry_run`,
`sent=false`, and matched real `job_error_spike` plus injected
`model_load_failures` and `gpu_vram_exhaustion` conditions.

Refresh snapshot later on 2026-09-26:

```text
GET /health
{"nodes":1,"queue_depth":0,"status":"ok","version":"dev"}
```

`/v1/network/control-plane` reported `queue.queued=0`, `queue.running=0`,
`oldest_wait_seconds=0`, one tracked online/ready node, zero active claims,
zero at-risk claims, zero unbatched receipts, and `health.status=healthy` with
an empty alert list. `havnai-coordinator.service` and `havnai-backup.timer`
were active on the coordinator host, with the next backup timer run observed for
2026-09-27 03:29:36 EDT.

The later refresh did not re-read `/metrics` or alert dry-run because no
admin/node token was available in the local shell or readable coordinator-host
environment files. The protected backup environment file correctly denied
unprivileged reads. Use an operator shell with the appropriate token for the
final pre-launch `/metrics` and dry-run refresh.

## Remaining Launch Gaps

- Capture saved dashboard links or screenshots if Jira requires visual evidence
  rather than API/dashboard endpoint links.
- Run a live mixed-model worker drill so `havnai_worker_model_failures` and
  `havnai_worker_model_unhealthy` have real production samples rather than only
  dry-run alert injection.
- Refresh `/metrics` and `/v1/network/alerts/dry-run` from an operator shell
  with the admin/node token during final pre-launch checks.
- Wire external alert delivery for disk/storage and payment/funding thresholds,
  or record explicit launch waivers for those notification paths.
- HAVN-44 and HAVN-46 now have production/prod-like restore/media and
  coordinator rollback evidence. HAVN-45/HAVN-49 still need a private
  accepted-work restart drill with a funded account bearer token.

# HavnAI Network-First Architecture

## Product principle

The GPU network is the platform. Creator generation, workflow royalties, and
the marketplace are clients of one observable execution and settlement layer.
No product surface should maintain a separate idea of capacity, job state, or
ownership.

## Network control loop

1. Nodes authenticate, register identity, and advertise models, pipelines,
   job types, GPU memory, and utilization.
2. The coordinator records heartbeats and derives online capacity.
3. Clients submit jobs against currently available capacity.
4. A compatible node atomically claims the oldest eligible job.
5. Execution emits lifecycle events and result metadata.
6. Validation and settlement produce trust, payouts, and provenance.
7. Creator and marketplace experiences consume those same records.

## Stable public contract

`GET /api/v1/network/summary` is the first versioned network contract. It
provides coordinator health, online/offline node counts, aggregate capacity,
and queue pressure. Fields may be added to v1, but existing meanings and types
must not change. Breaking changes require `/api/v2`.

## Reliability invariants

- A queued job can be claimed by at most one node.
- A claim is persisted before it is returned to a node.
- A node only receives work matching its role, job types, models, and pipelines.
- Completion is accepted only from the node owning the running claim.
- Public capacity is derived from fresh heartbeats, never registration count.
- Payouts must link to a job, execution attempt, node, and output asset.

## Evolution roadmap

### Phase 1: trustworthy coordinator

- Atomic claims and idempotent completion.
- Versioned network summary and lifecycle contracts.
- Runtime-pinned CI for core and web.
- Lease expiry and automatic recovery for abandoned claims. Implemented with
  `HAVNAI_JOB_LEASE_SECONDS` (default 1800) and `HAVNAI_JOB_MAX_RETRIES`
  (default 3).

Nodes renew every active claim during their normal `/register` heartbeat. A
long-running executor may also call `POST /tasks/heartbeat` with `node_id` and
`task_id`; this endpoint requires the join token. Before dispatch, expired
claims are recovered in lease-expiration order. Work within its retry budget
returns to `queued`; exhausted work becomes `failed` with
`last_failure_reason=lease_expired`.

### Capability-weighted scheduling

Atomic claims use the `capability_weighted_v1` strategy. The coordinator first
filters for online, idle nodes advertising the required role, job type, model,
pipeline, and reference-face capability. It then calculates a deterministic
0–100 score from availability, GPU headroom, VRAM, historical trust, and
heartbeat freshness.

The highest-scoring compatible node receives a preference window controlled by
`HAVNAI_SCHEDULER_PREFERENCE_GRACE_SECONDS` (default 8). Once that window
expires, any compatible polling node may claim the job. This preserves low
latency and prevents a preferred but unresponsive node from starving work.
New operators begin with neutral trust and gain scoring influence as their
completed sample size approaches 20 attempts.

Every claim records `preferred_node_id`, `dispatch_score`, and
`dispatch_reason`, making routing decisions auditable from job details.

### Durable execution timeline

`GET /jobs/<job_id>/timeline` returns an ordered, append-only execution ledger.
Each event includes its sequence, stage, status, node and attempt attribution,
structured metadata, latency since the prior event, and elapsed time since the
first event.

The current stage vocabulary is additive and includes `QUEUED`, `ROUTED`,
`CLAIMED`, `LEASE_RENEWED`, `GENERATING`, `UPLOADING`, `RESULT_RECEIVED`,
`VALIDATED`, `SETTLED`, `REQUEUED`, `RETRY_EXHAUSTED`, `SUCCEEDED`, `FAILED`,
and `CANCELLED`. Node clients report coarse generation and upload progress via
the authenticated task heartbeat. Routine lease renewals are retained at most
once per minute per job/node to bound telemetry volume.

This ledger is the input to future execution receipts: output hashes, model
versions, validator attestations, and settlement proofs can be attached without
changing the job lifecycle contract.

### Proof of Creation receipts

Successful settled jobs with an output artifact receive an immutable
`proof-of-creation.v1` receipt. The canonical JSON payload binds creator and job
identity, a prompt/configuration digest, model identity, routed node and attempt,
the execution-timeline digest, validation, settlement, and the artifact's
SHA-256 hash.

- `GET /jobs/<job_id>/receipt` returns the public receipt.
- `GET /jobs/<job_id>/receipt/verify` re-hashes the persisted payload and the
  artifact currently on disk.
- `HAVNAI_RECEIPT_SIGNING_KEY` enables an HMAC-SHA256 coordinator signature.
  Without it, integrity remains verifiable but authenticity is reported as
  `unsigned`.

The signing key must be supplied through the deployment secret manager, must
not be committed or sent to nodes, and should be rotated only with an explicit
key-version strategy. HMAC proves coordinator authenticity only to holders of
the shared secret; Ed25519 is the public-verification path. A later layer can
anchor receipt Merkle roots on-chain without changing v1 records.

For public verification, prefer Ed25519:

```bash
openssl rand -base64 32
```

Store the result as `HAVNAI_RECEIPT_ED25519_PRIVATE_KEY`. The coordinator
derives a raw public key and a stable key ID (the first 16 hex characters of
the public-key SHA-256) and publishes it at `GET /api/v1/receipts/keys`.
Ed25519 takes precedence when both Ed25519 and legacy HMAC keys are configured.

Receipts embed the public key and key ID used at issuance, so old receipts
remain independently verifiable after rotation. Rotation should introduce the
new private key, retain previously published public keys in deployment records,
and never rewrite existing receipts. Browser clients hash the exact canonical
payload and verify the Ed25519 signature locally through Web Crypto.

### Merkle batching and anchoring

Receipt hashes are accumulated into immutable `receipt-merkle-batch.v1` trees.
Leaves use `SHA-256(0x00 || receipt_hash_bytes)` and internal nodes use
`SHA-256(0x01 || left || right)`, preventing leaf/internal-node ambiguity. An
unpaired final node is duplicated at each level.

`HAVNAI_MERKLE_BATCH_SIZE` controls automatic batch creation (default 100).
An authenticated operator can flush outstanding receipts early with
`POST /admin/receipts/batches`. Public clients can use:

- `GET /api/v1/receipts/batches` for roots and anchor-ready payloads.
- `GET /jobs/<job_id>/receipt/proof` for an inclusion proof.

The web client verifies inclusion locally. A batch starts in `ready` state;
this means the root is internally committed and ready to anchor, not that it is
already on-chain. Only a confirmed transaction should transition a batch to
`anchored` with its network and transaction hash. Roots and membership are
never rewritten after batch creation.

### Network Alpha Command Center

`GET /api/v1/network/control-plane` provides the privacy-safe operational view
used by the web Network page. It intentionally excludes wallets, prompts, and
artifact contents. The contract reports node readiness, queue age, 24-hour
p50/p95 queue and runtime latency, active claim lease risk, scheduling fallback,
and receipt-batch readiness.

Initial alerts warn after one minute of queue delay, become critical after five
minutes, and flag claims inside the smaller of two minutes or 20% of their
configured lease. The web client polls every 15 seconds and refreshes
immediately on job lifecycle SSE events.

### Phase 2: scalable control plane

- Extract registration, dispatch, telemetry, and settlement from `app.py`.
- Replace JSON node persistence with the primary database.
- Add database migrations and PostgreSQL support.
- Move event fan-out and queue notification to Redis or a durable broker.

### Phase 3: economic network

- Signed node identity and capability attestations.
- Verifiable execution receipts and validator quorum.
- Capacity pricing based on hardware, demand, latency, and trust.
- On-chain settlement batches with auditable off-chain ledger proofs.

### Phase 4: network-powered products

- Creator Studio chooses routes using price, speed, quality, and availability.
- Workflows retain versioned model and execution provenance.
- Marketplace assets inherit ownership and royalty lineage from settled jobs.
- Operators receive live demand forecasts and earnings optimization guidance.

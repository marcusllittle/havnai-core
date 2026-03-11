# Job-Centric Settlement Architecture

## Overview

Every generation request on HavnAI creates a **durable job ticket** that tracks the full lifecycle from credit reservation through execution, validation, and settlement. This separates *compute completion* from *user satisfaction*.

## How Job Tickets Work

When a user submits a generation request (`/submit-job` or `/submit-faceswap-job`):

1. **Credit check** — verify the user has enough credits
2. **Credit reservation** — deduct credits from balance (reservation, not final spend)
3. **Job enqueue** — create a row in the `jobs` table (existing)
4. **Settlement ticket** — create a row in `job_settlement` with `execution_status='queued'`

The job ticket stores: wallet, job type, model, estimated cost, reserved amount, execution status, quality status, settlement outcome, and marketplace eligibility.

## How Credit Reservation Works

Credits use a **reserve-then-settle** pattern:

| Event | Balance Effect | Ledger Entry |
|-------|---------------|--------------|
| **Reserve** (job submitted) | balance -= cost | `reserve` |
| **Spend** (valid completion) | no change (already deducted) | `spend` |
| **Release** (technical failure) | balance += cost | `release` |
| **Refund** (malformed output) | balance += cost | `refund` |

The `credits` table's `balance` and `total_spent` columns are updated atomically. `release_credits()` reverses a deduction. The `credit_ledger` table provides a full audit trail.

## How Settlement Works

When a node submits results (`/results`):

1. **Complete attempt** — mark the `job_attempts` row as finished
2. **Determine execution status** — `completed` (success) or `technical_failed`
3. **Validate output** — run minimum quality checks (file exists, valid image header, plausible dimensions)
4. **Settle** — based on execution + quality status:

| Execution | Quality | Outcome | Credits | Node Payout |
|-----------|---------|---------|---------|-------------|
| completed | valid | **spent** | Permanently spent | Full reward |
| completed | unchecked | **spent** | Permanently spent | Full reward |
| completed | malformed | **full_refund** | Returned to user | 10% consolation |
| completed | policy_blocked | **released** | Returned to user | None |
| technical_failed | any | **released** | Returned to user | None |

5. **Create payout** — insert into `node_payouts` if reward > 0

## How Malformed Output is Handled

Malformed output = technically present but unusable (e.g., corrupt image, wrong format, empty file, dimensions outside 64-8192px range).

- Credits are fully refunded to the user
- Node receives 10% consolation payout (they did compute work, even if output was bad)
- Settlement outcome is `full_refund` — traceable in the ledger
- The `quality_status` is set to `malformed`

## Why Valid-But-Ugly Output is Treated Differently

If the output is a valid, readable image with correct dimensions — the node completed the work correctly. The fact that the image is aesthetically disappointing is not a node failure.

- Credits are **spent** normally
- Node receives **full payout**
- `quality_status` = `valid`
- Future: hooks exist for optional reroll UX, but this is not an automatic refund case

This prevents gaming where users request refunds for subjective dissatisfaction while nodes bear the compute cost.

## How Payout Records Work

The `node_payouts` table records every payout:
- `node_id` — which node earned the payout
- `job_id` — which job it completed
- `reward_amount` — computed reward (model weight * pipeline factor * runtime factor)
- `reward_asset_type` — currently `simulated_hai` (no on-chain settlement yet)
- `status` — `completed` (future: `pending`, `on_chain`)
- `tx_hash` — null for now (future: on-chain tx reference)

## Marketplace Eligibility

- Standard image generation (`IMAGE_GEN`) outputs are **marketplace-eligible** by default
- Face swap (`FACE_SWAP`) outputs are **not marketplace-eligible** by default
- Eligibility requires: `quality_status = valid` AND `settlement_outcome = spent`
- The `marketplace_eligible` flag on `job_settlement` can be queried before listing

## Data Model

### job_settlement (1:1 with jobs)
Core settlement tracking per job — status, costs, quality, outcome.

### job_attempts (1:N per job)
Records each node claim/attempt with timing and error details.

### node_payouts (1 per settled job)
Payout records linking nodes to rewards.

### credit_ledger (audit trail)
Every credit-affecting event with wallet, amount, balance, job reference.

## API Endpoints

- `GET /jobs/<id>` — now includes `settlement` object in response
- `GET /settlement/<id>` — full settlement record + attempts
- `GET /settlement/<id>/attempts` — attempt history
- `GET /payouts/node/<node_id>` — payout history for a node
- `GET /credits/ledger?wallet=0x...` — credit audit trail

## Files Changed

- `server/settlement.py` — new: job ticket lifecycle, validation, settlement logic
- `server/credits.py` — added: `release_credits()`, `check_and_reserve_credits()`
- `server/app.py` — wired settlement into submit/results/detail endpoints + new API routes

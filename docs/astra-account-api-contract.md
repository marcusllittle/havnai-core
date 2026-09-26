# Astra account API contract

Work items: HAVN-56 and HAVN-57. Owner split: HavnAI core owns the
`/v2/astra/*` backend contract; Claude owns the Astra client integration.

These routes let Astra use normal HavnAI account sign-in. The wallet-era
`/astra/*` routes remain for compatibility, but new account play must use the
`/v2/astra/*` routes with a Clerk-backed bearer token. No wallet signature is
required for normal play, rewards, or spends.

## Authorization

Every route below requires:

```http
Authorization: Bearer <HavnAI account session token>
Cache-Control: private, no-store
Vary: Authorization
```

Core derives the immutable `acct_...` ID from the verified provider session. The
client must not send `account_id`, wallet, reward amount, or credit balance in
the request body. Owner tokens, wallet addresses, and legacy Astra session tokens
are not accepted on these routes.

## Session discovery

```http
GET /v2/astra/session
```

Response:

```json
{
  "mode": "account",
  "account_id": "acct_...",
  "auth": "bearer",
  "wallet_required": false,
  "endpoints": {
    "start_run": "/v2/astra/run/start",
    "reward": "/v2/astra/reward",
    "spend": "/v2/astra/spend",
    "stats": "/v2/astra/stats"
  }
}
```

Use this as Astra's account-mode capability check after sign-in or refresh.
Guest play can still run locally, but reward/spend UI should show sign-in
required when this endpoint returns `401`.

## Start run

```http
POST /v2/astra/run/start
Content-Type: application/json

{"map_id":"nebula-runway"}
```

Response `201`:

```json
{
  "mode": "account",
  "map_id": "nebula-runway",
  "run_token": "<opaque single-use token>",
  "started_at": 1790457688.5278685
}
```

The token is bound to the account and expires after the same TTL as the legacy
Astra run token. Store it only for the active run.

## Reward

```http
POST /v2/astra/reward
Content-Type: application/json

{
  "run_token": "<token from start_run>",
  "score": 100000,
  "grade": "S",
  "duration_s": 120,
  "map_id": "nebula-runway"
}
```

Successful response `200`:

```json
{
  "ok": true,
  "run_id": "acct_astra_...",
  "reward_units": 15000,
  "reward": 15.0,
  "scale": 1000,
  "daily_earned_units": 15000,
  "daily_cap_units": 50000,
  "bonuses": ["first_win_of_day"],
  "multiplier": 2.0,
  "ledger_entry_id": 123
}
```

Rejected response examples use the standard `/v2` error envelope:

```json
{"error":{"code":"run_too_short","message":"Run too short"},"request_id":"..."}
```

Important rejection codes:

- `missing_run_token`
- `unknown_run_token`
- `run_token_account_mismatch`
- `run_token_used`
- `run_token_expired`
- `score_too_low`
- `run_too_short`
- `cooldown`
- `daily_cap_reached`

Core computes the reward amount. The client-supplied `duration_s` is advisory;
core uses the server-side `started_at` timestamp. Reward units are written to
`account_credit_ledger` with operation `astra_reward`.

## Spend

```http
POST /v2/astra/spend
Idempotency-Key: astra-spend-<uuid>
Content-Type: application/json

{"action":"gacha_10"}
```

Successful response `200`:

```json
{
  "ok": true,
  "action": "gacha_10",
  "cost_units": 80000,
  "scale": 1000,
  "ledger_entry_id": 124,
  "balance": {
    "scale": 1000,
    "settled_units": 20000,
    "reserved_units": 0,
    "available_units": 20000,
    "debt_units": 0
  }
}
```

Retries with the same `Idempotency-Key` and same action replay the original
result instead of charging again. A changed action with the same key returns
`idempotency_conflict`.

Rejected spend codes:

- `idempotency_key_required`
- `invalid_action`
- `insufficient_credits`
- `idempotency_conflict`

Core writes successful spends to `account_credit_ledger` with operation
`astra_spend`. Failed insufficient-credit attempts are audited in
`account_astra_spends` as `failed` and are not replayed as success later.

## Stats

```http
GET /v2/astra/stats
```

Response:

```json
{
  "mode": "account",
  "account_id": "acct_...",
  "total_runs": 1,
  "best_score": 100000,
  "total_earned_units": 15000,
  "daily_earned_units": 15000,
  "daily_cap_units": 50000,
  "cooldown_remaining": 0,
  "scale": 1000
}
```

## Compatibility notes

- Existing wallet-era `/astra/*` routes remain intact for older clients.
- Account routes use separate `account_astra_*` tables. They do not write
  `acct_...` values into legacy wallet columns.
- Legacy wallet-earned Astra rewards are not silently moved. A future migration
  must use the explicit HAVN-18 wallet-link/import path.
- Astra client should treat account mode as the commercial-launch path. MetaMask
  is optional unless the user chooses wallet-link/import/blockchain features.

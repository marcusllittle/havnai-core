# Account credit payments (HAVN-21)

This branch implements account Checkout, provider reconciliation, account receipts,
and refund/dispute ledger adjustments. It is not a completed production rollout:
Clerk configuration, account pricing UI, approved terms/refund policy, and a real
Stripe sandbox acceptance run are still required before enabling Checkout.

## Boundary and configuration

The account payment path is separate from legacy wallet payments. It never uses a
wallet address or owner token to identify a buyer. POST `/v2/account/checkout`
requires a verified account bearer token, a stable `Idempotency-Key` (16–128
characters), and exactly `{ "package_id": "starter", "terms_version": "..." }`.
Core selects price, USD currency, credit units, and return URLs. Client-supplied
account IDs, credit quantities, prices, and return URLs are rejected.

Private coordinator environment, not web environment:

```dotenv
STRIPE_SECRET_KEY=<existing Stripe project's server key>
STRIPE_ACCOUNT_WEBHOOK_SECRET=<signing secret for the v2 webhook endpoint>
HAVNAI_CHECKOUT_ORIGIN=https://joinhavn.io
HAVNAI_CREDIT_TERMS_VERSION=<approved published policy revision>
HAVNAI_ACCOUNT_CHECKOUT_ENABLED=false
```

Only enable after the pricing/terms/refund UI and acceptance checks are complete.
Test checkout may use an exact localhost HTTP origin; live checkout requires
HTTPS. Test and live account purchases cannot share a database. Use a separate
staging coordinator/database and Clerk development application. Changing the
Stripe key to live does not convert test credit balances into paid balances.
Disabling checkout does not disable webhook processing or refund reconciliation.

Configure a direct-account Stripe webhook endpoint for
`/v2/payments/stripe/webhook` with the events listed in
`server/account_payments.py:EVENT_TYPES`. The raw request body and Stripe signature
must reach core unchanged. Connected-account events are rejected. Public package
availability is at GET `/v2/credit-packages`; account history is at GET
`/v2/account/purchases`, with an opaque `before` purchase-id cursor, and GET
`/v2/account/purchases/:id`. These private responses are not cacheable.

## Integrity rules

- Persist purchase, server-selected amount/quantity, accepted terms revision,
  and exact Checkout parameters **before** making the Stripe create request.
  Retry using the persisted purchase's Stripe idempotency key and exact parameters.
- An unresolved creation older than 23 hours is held for reconciliation, because
  Stripe can prune idempotency keys after 24 hours. Do not blindly create another
  session for that request.
- A success redirect does not fund anything. Verify webhook signatures and read
  current PaymentIntent, Charge, and Dispute state using the server API key.
  Match purchase metadata, payment ID, amount, currency, and live/test mode.
- A payment ID and Checkout session can bind to only one purchase globally.
  Funding, immutable receipt creation, any adjustment, and purchase status commit
  in one SQLite transaction. Event acknowledgement follows that transaction; if
  acknowledgement fails, replay rechecks state without duplicating credits.
- Current provider reads happen outside SQLite write transactions. A revision
  fence rejects stale concurrent reads with a retryable response. Out-of-order
  event payloads are lookup hints, never financial truth.
- Credits use integer units (1000 units = one credit). Successful partial refunds
  and formal active/lost disputes reduce the original purchase's retained grant,
  capped at the whole purchase. The retained portion rounds down to a unit.
  Won disputes restore only the grant still justified after successful refunds.
  Inquiries and prevented disputes do not revoke credits.
- Reversing spent credits may create debt. New spending remains blocked by the
  ledger's available-balance condition; releasing a job reservation does not
  recreate refunded funding. Subsequent paid funding offsets debt normally.
- Receipts and adjustment records are append-only. Adjustment records retain
  provider dispute IDs/statuses and the monetary snapshot used for the change.
  Public account APIs omit payment-method data and provider credentials.

The existing service grants no credit for failed, unpaid, cancelled, or expired
Checkout attempts. A cancellation redirect alone does not cancel a still-open
Stripe session. No automatic refund is promised or initiated by these routes.

## Recovery without creating another charge

Run with the coordinator's private environment loaded and the existing database
path. This command only retrieves provider state and reconciles local records:

```bash
python server/account_payments.py --database /path/to/coordinator.db --purchase pur_...
```

If Checkout's create response was lost before its session ID was saved, locate the
session in the Stripe dashboard by metadata `havnai_purchase=pur_...` and supply
`--session cs_...`. Recovery verifies the session's metadata, reference, currency,
amount, and mode before saving any binding or reconciling credits. A session ID
alone is never enough to grant credits. Back up the database before operational
recovery and verify the resulting account receipt and ledger entries afterward.

Alert on failed webhook delivery, prolonged pending purchases, binding mismatch,
or reconciliation errors. Stripe delivery retries and this recovery command are
available; scheduled reconciliation/operational alert configuration is still a
rollout task. Do not acknowledge failures manually without reconciliation.

## Evidence and remaining live gate

`tests/test_account_payments.py` uses real SQLite transactions, real Stripe webhook
HMAC verification, and Stripe SDK response objects. Provider network requests are
mocked; no charge is made. It covers duplicate funding, lost creation responses,
lost acknowledgements, failed/cancelled payments, wrong binding, cross-account
receipt access, refunds after spending, dispute wins, stale reconciliation,
rollback, receipt immutability, expiration, recovery, and mode isolation.

Production evidence still requires configured Clerk sign-in, an actual Stripe
sandbox purchase, verified webhook delivery, account funding/receipt visibility,
a real refund, and account-funded generation/publication. No test count is a
substitute for that complete flow.

References: [Stripe idempotency](https://docs.stripe.com/api/idempotent_requests),
[dispute states](https://docs.stripe.com/api/disputes/object),
[SDK resource conversion](https://github.com/stripe/stripe-python#working-with-api-resources).

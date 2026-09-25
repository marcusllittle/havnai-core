# HAVN-11 verification checkpoint — 2026-09-25

HAVN-11 remains incomplete. This checkpoint distinguishes executable regression
evidence from the live acceptance still required by the Jira story.

## Executed regression checks

Core commit `f9a3104`, Python 3.12 in the isolated WSL virtual environment:

- All 25 `tests/test_account*.py` files: **442 passed** in 43.46 seconds. This includes account
  identity/auth/lifecycle, wallet linking, ledger and credit imports, payment
  reconciliation, account jobs/music/video, marketplace, workflows, signed
  migration, receipt recovery and database restore tests.
- Earlier, at `9ac2454`, `test_backend_reliability.py`, `test_music_discover.py`,
  `test_music_discover_api.py`, `test_platform_v1.py`: **61 passed, 12 subtests
  passed**. These check legacy routing and ownership compatibility.
- SQLite temporary test databases were under `/dev/shm`; no production database
  or GPU service was used by these runs.

Web commit `d5939a4`: **452 tests in 75 files passed**, with successful TypeScript
checking and Next.js production build. Local guest Templates browser verification
at core `85335cc` confirmed a 200 catalog response through the web proxy, no
browser errors, and a 390-pixel document at a 390-pixel mobile viewport. See web
`docs/account-workflows.md` for its narrower scope.

The local preview database restore drill verified 71 tables, integrity, foreign
keys and equal backup/restored file hashes. A separate signed-import test used a
restored fixture copy and proved the source did not change. This does not prove
production database/media restore readiness.

## Acceptance still open

| Requirement | Evidence still needed |
| --- | --- |
| Conventional production credit funding | Sandbox account/key and local webhook listener are now configured privately. A provider read confirmed the expected account and `livemode=false`; the local catalog reports checkout available. Actual checkout/webhook/receipt/ledger acceptance is pending. Production configuration and approved published terms/pricing/refund policy remain open. |
| Refund correctness in service | Actual Stripe sandbox refund and repeated provider reconciliation now verified below. Production and missed-webhook acceptance remain open; regression-suite provider calls are mocked. |
| Generate/recover/publish/manage without MetaMask | Signed-in browser acceptance against real generation workers for supported media, including reload recovery and private library/publication checks. Unit/API tests alone are insufficient. |
| Optional wallet linkage | Real wallet recent-auth link/unlink/import acceptance; confirm ordinary account navigation never prompts. |
| Legacy association and launch | Signed selective migration now includes song likes/saves and workflows. Source uploads have an explicit account re-upload path because legacy labels do not establish ownership. Legacy Stripe credit provenance, audited reversal/compensation, production backup/restore and migration review remain rollout gates. Import execution remains disabled by default. |
| Production authentication | Clerk production configuration and lifecycle webhook delivery/revocation evidence. Earlier user-confirmed development Google sign-in is not production proof. |
| Operational delivery | Reviewed deployment, scheduler/alert activation and evidence collection. Timer templates are committed but have not been installed or enabled. |

Keep the feature branches and homepage revert intact. Do not close HAVN-11 on
test counts alone or turn this checkpoint into an assertion of live readiness.

## Scope and handoff audit

The HAVN-11 parent and HAVN-18/19/20/21/22/23 descriptions were read directly from
Jira on 2026-09-25. Their explicit deliverables include a safe documented legacy
association path and a clear HAVN-13 handoff. Web
`docs/havn-13-wallet-handoff.md` identifies the account/wallet boundary, relevant
components and remaining real-wallet checks. No Jira status or comment was changed.

Production funding is an explicit parent requirement and remains unproven.
Provider-mocked tests, development Google sign-in, and a successful production
build do not satisfy it. Broader automatic source-reference migration is not
promised: the documented re-upload route preserves the verified ownership boundary.
The operational gates above still apply before activating migration in production;
they are not evidence that a rollout has occurred.

## Local sandbox setup checkpoint

Core `3ca7cce` adds explicit test-only preview configuration; its 17 preview tests
pass. On 2026-09-25 the isolated loopback preview was restarted with that option,
using a private environment outside Git. The sandbox key's Stripe account matched
the CLI-authorized account, and Stripe's balance endpoint returned `livemode=false`.
The listener forwards only direct-account events to the v2 webhook endpoint.
GET `http://localhost:3100/api/v2/credit-packages` reports checkout available and
the explicit `sandbox-2026-09-25` policy. This proves configuration, not funding.
The development-only web policy is not commercial terms and returns 404 outside
development. The user is performing browser acceptance in their regular browser
because Google rejected the automation browser. No production service was changed.

## Actual Stripe sandbox purchase and refund

On 2026-09-25 at 16:04 local time, the user's signed-in browser purchased the
50-credit/$5 sandbox pack. Purchase `pur_df393363413a481dbc37a0472f72676a` reached
`paid` through the local webhook. Read-only database checks showed 50,000 available
units, zero reserved/debt units, one `fund` ledger entry and one immutable payment
receipt with the test policy revision. Two subsequent provider reconciliations
left all economic records unchanged: no duplicate funding or receipt.

At 16:06, an idempotent full sandbox refund (`re_3UJfMMFWBrjj49xV07gYUnEU`) returned
`succeeded` for 500 cents. Without manual ledger edits or recovery, a refund webhook
changed the purchase to `refunded`, balance to zero and added exactly one -50,000
unit payment adjustment. The original receipt remained intact. Two further provider
reconciliations retained zero balance and the same two ledger entries.

Concurrent notifications produced one 503 on purchase and two 503s on refund while
the sibling notifications returned 200 and committed the correct result. The code
has a retryable revision fence for concurrent provider reads; the listener did not
capture the 503 response bodies, so that cause is not conclusively established by
these logs. Explicit failed-event redelivery remains to verify. Browser display of
the refunded receipt, failed-payment acceptance, missed-webhook recovery and real
account-funded worker generation remain open. No actual money moved, and this
sandbox result does not satisfy the required production payment path.

The user subsequently confirmed the refreshed account displayed zero available
credits and the refunded purchase. The three previously failed event IDs were
retrieved from Stripe and each delivered twice to the local HTTP webhook with a
fresh signature using the private listener secret. All six requests returned 200.
The older payment-success notification did not resurrect credits after refund;
the duplicate refund events added no deductions. Balance remained zero, with one
funding entry, one adjustment and the original receipt. This is an explicit local
redelivery test using actual provider event data, not evidence of Stripe's automatic
retry scheduling. Failed-payment and missed-webhook acceptance still remain open.

At 16:53 local time the user confirmed Stripe declined the test card. The
`payment_intent.payment_failed` notification returned HTTP 200. Purchase
`pur_ac02734d062849e4a2a1e938812fc760` remained pending/retryable, with no new paid
receipt or ledger entry. Available/settled/reserved balances remained zero and the
original refunded receipt was unchanged. This verifies actual sandbox decline
handling, rather than only a mocked provider failure. Missed-webhook recovery and
account-funded generation remain open.

At 16:57 local time, after deliberately stopping the local Stripe listener, the
user completed the same previously declined checkout with a successful test card.
A direct Stripe read confirmed Checkout `complete` / payment `paid` while the
local purchase remained `pending` and balance zero. A private SQLite backup was
taken before running the actual `server/account_payments.py --batch --limit 25`
CLI against the isolated preview. It recovered the purchase to `paid`, created
one 50,000-unit funding entry and receipt #2, and left the earlier refunded purchase
unchanged. A second identical CLI run produced no additional economic entries or
receipts. This verifies actual sandbox missed-webhook recovery and retry safety.
The webhook listener was then restarted with its unchanged signing secret and
confirmed ready. It remains a manually run local listener; no production scheduler
was enabled. The preview still has zero generation workers, so account-funded
generation/recovery/publication acceptance is not yet possible there.

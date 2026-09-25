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
| Refund correctness in service | Real provider refund and missed-webhook recovery, with account balance/receipt verification. Current provider calls in tests are mocked. |
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

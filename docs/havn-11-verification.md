# HAVN-11 verification checkpoint — 2026-09-25

HAVN-11 remains incomplete. This checkpoint distinguishes executable regression
evidence from the live acceptance still required by the Jira story.

## Executed regression checks

Core commit `9ac2454`, Python 3.12 in the isolated WSL virtual environment:

- All 24 `tests/test_account*.py` files: **431 passed**. This includes account
  identity/auth/lifecycle, wallet linking, ledger and credit imports, payment
  reconciliation, account jobs/music/video, marketplace, workflows, signed
  migration, receipt recovery and database restore tests.
- `test_backend_reliability.py`, `test_music_discover.py`,
  `test_music_discover_api.py`, `test_platform_v1.py`: **61 passed, 12 subtests
  passed**. These check legacy routing and ownership compatibility.
- SQLite temporary test databases were under `/dev/shm`; no production database
  or GPU service was used by these runs.

Web commit `b8be183`: **449 tests in 75 files passed**, with successful TypeScript
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
| Conventional production credit funding | Correct Stripe account, private test/live configuration, approved published terms/pricing/refund policy, real checkout and verified webhook/receipt/ledger evidence. Local repositories contain no Stripe setup; account/dashboard identification has been requested. |
| Refund correctness in service | Real provider refund and missed-webhook recovery, with account balance/receipt verification. Current provider calls in tests are mocked. |
| Generate/recover/publish/manage without MetaMask | Signed-in browser acceptance against real generation workers for supported media, including reload recovery and private library/publication checks. Unit/API tests alone are insufficient. |
| Optional wallet linkage | Real wallet recent-auth link/unlink/import acceptance; confirm ordinary account navigation never prompts. |
| Legacy association and launch | Remaining saved engagement/reference dependencies, legacy Stripe provenance, audited reversal/compensation, production backup/restore and migration review. Import execution remains disabled by default. |
| Production authentication | Clerk production configuration and lifecycle webhook delivery/revocation evidence. Earlier user-confirmed development Google sign-in is not production proof. |
| Operational delivery | Reviewed deployment, scheduler/alert activation and evidence collection. Timer templates are committed but have not been installed or enabled. |

Keep the feature branches and homepage revert intact. Do not close HAVN-11 on
test counts alone or turn this checkpoint into an assertion of live readiness.

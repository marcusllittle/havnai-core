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
| Conventional production credit funding | Actual sandbox purchase, webhook funding, immutable receipt, decline, refund and missed-webhook recovery are verified below. Production configuration, an actual production payment path, and approved published terms/pricing/refund policy remain open. |
| Refund correctness in service | Actual Stripe sandbox refund, duplicate delivery and repeated provider reconciliation are verified below. Production acceptance remains open; dispute and cancellation coverage still relies on mocked provider calls. |
| Generate/recover/publish/manage without MetaMask | Two real account-funded GPU images completed. The user confirmed Collection visibility, successful opening, the corrected watermark and no MetaMask prompts. Publication/management, generation recovery after reload and supported music/video browser flows still need evidence. Unit/API tests alone are insufficient. |
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

## First real account-funded GPU image

The dedicated `havn11-local-3060` worker now runs on the user's RTX 3060 using the
existing CUDA environment and D-drive models, with separate runtime/output paths
and loopback preview credentials. The existing production node/coordinator were
not restarted. Placeholder generation and startup preloading are disabled.

Job `job-97acd6b82a9c46d28436d1151e0c94a3` used `juggernautXL_ragnarokBy`, completed
on that worker, and captured its 1,000-unit (one credit) reservation. Logs show a
real Diffusers SDXL pipeline load and 48.7-second generation. The user confirmed
the image rendered well but reported the wrong watermark. Visual inspection found
the text fallback: this isolated core checkout lacked the logo and could not use
the developer-only sibling-web fallback. The exact existing web logo is now stored
at `static/HavnAI-logo.png` and included in distributed node bundles. All 11 bundle
tests pass, including logo byte preservation. Existing immutable generation
artifacts were not overwritten; new renders can use the packaged logo. Refresh
recovery and publication still need separate acceptance evidence.

The next browser-submitted image, `job-91bd7785b31b4f3cbb9460e22e7dbe8f`, completed
on the same worker at 17:48 with 45.7 seconds of generation. Direct inspection of
the saved output confirmed the packaged HavnAI logo replaced the text fallback.
Its one-credit reservation was captured once; the account had 48,000 settled units
and zero reserved units after the two successful images. The original job's private
API and artifact URL returned 401 without authentication; legacy result and static
output URLs returned 404. These checks do not substitute for another-account
browser access tests or the pending Collection refresh/publication acceptance.

The user subsequently confirmed that both images are in Collection. This verifies
their visibility in the signed-in account's collection, but does not by itself
establish a refresh, successful opening/download, absence of wallet prompts, or
publication. Those checks remain distinct from the completed generation and
credit-capture evidence above.

The user clarified that the requested opening, corrected-watermark and no-MetaMask
checks had already passed for the Collection images. Those browser checks are now
accepted; do not ask the user to repeat them. Download, publication/management and
recovery of an in-progress generation after reload remain separate checks.

## First real account-funded music job

Job `job-b7dba0f9d4954d538d89607faaf607a0` completed successfully using
`ace_step_1_5_turbo`: 30 seconds, instrumental, 8 inference steps. The isolated
preview's ledger shows its 1,000-unit reservation captured. The user confirmed
playback worked, while reporting lower perceived quality than their usual setup.
The ACE-Step adapter matches the main core checkout byte-for-byte; this does not
establish equivalent prompts, service configuration or generation quality.
No account music publication existed at this check. Publish/unpublish, anonymous
playback and retained private access after unpublishing remain to be verified.

## Live preview deletion checkpoint

On 2026-09-25 at 22:08 and 22:09 local time, the signed-in browser deleted image
`job-91bd7785b31b4f3cbb9460e22e7dbe8f` and song
`job-b7dba0f9d4954d538d89607faaf607a0`. Both DELETE requests returned HTTP 200.
Read-only inspection confirmed persisted deletion timestamps and exactly 30-day
recovery deadlines. Neither item was restored or physically purged at this check.
All three generation reservations remained captured at 1,000 units each.
This verifies live deletion writes, not browser refresh/no-resurrection or restore
acceptance. The purge scheduler remains uninstalled; fixture-based CLI tests are
not evidence of production activation.

The user then confirmed that restore worked and requested improvements to the
recovery page and sign-in/sign-out controls. Web `c746ad9` supplies the blue
recovery layout and a shared header position for sign-in/sign-out. Seventeen
targeted web tests, TypeScript, and the production build passed. This accepts the
user-reported local restoration check; it is not evidence of production purge,
cross-device recovery, or live sign-out/re-authentication acceptance.

## HAVN-32 policy page checkpoint

The web feature branch now implements `/terms/credits-v1`, `/refunds/credits-v1`
and `/support`, with links from pricing, account, receipt details and public
navigation/footer surfaces. See web `docs/credit-policy-release.md` for the exact
scope, signed-out desktop/mobile browser evidence, 28 passing targeted tests,
TypeScript and production build checks, and the required policy configuration.

Owner approval of the proposed manual refund-review wording and confirmation of
the existing support mailbox remain open. Normal public HTTPS probes to all three
joinhavn.io routes returned 404 on this check; they are not yet deployed. The local
catalog still reports the original explicit sandbox policy. No coordinator
configuration or production service was changed. Public deployment, allowlisted
running policy-config evidence, a live receipt bound to the approved revision,
and Jira evidence linkage remain unproven. HAVN-11 is not complete.

The owner subsequently reviewed the policy pages, approved their wording, and
confirmed `team@joinhavn.io` is the correct support address. This clears the copy
approval item above; it does not establish public deployment, active production
configuration, or actual support-mail delivery. Those remain separate evidence.

## Recovery pagination checkpoint

Replaced the recovery endpoint's fixed 100-row cutoff with owner-scoped keyset
pages. Timestamp ties use job IDs for stable ordering; restoring a page boundary
does not lose older entries. The web recovery page offers Older deletions and
Latest deletions, preserves the current cursor on retry, and cancels stale
requests when the account changes.

Validation: 30 targeted backend tests passed across recovery paging, lifecycle,
and purge operations; seven recovery UI tests, TypeScript, and the web production
build passed. Paging coverage includes 136 recoverable entries, equal timestamps,
restoration between pages, invalid cursors, and owner isolation. This is automated
fixture evidence; the running preview coordinator has not yet loaded this change.

Storage review also identified derived last-frame assets under ASSETS_DIR and
worker storage as remaining physical-retention boundaries. Existing purge handles
registered artifact paths under the configured outputs root only. Full storage
reclamation and production purge activation are not established by these tests.

## Combined regression checkpoint after retention and video deletion

At core `55945dd` and web `896c47b`, the combined account regression run passed:

- Core: `python -B -m pytest tests/test_account*.py tests/test_artifact*.py tests/test_worker_artifact_cleanup.py -q -x --tb=short`,
  using the isolated Python environment and `/dev/shm` temporary fixtures:
  **502 passed in 58.88 seconds**.
- Web: `npx vitest run`: **481 tests across 79 files passed**.
- Web TypeScript and production build passed at the video-deletion checkpoint.

This exercises account auth, ledger/payments, imports, marketplace, music/video,
recovery, purge and worker cleanup interactions under fixtures. It does not
substitute for production Clerk/Stripe setup, live wallet acceptance, the mixed
model worker run, or production operations drills required by HAVN-25.

The isolated coordinator and worker are running the updated lifecycle code;
live cleanup polling returned HTTP 200 with zero purged creations and preserved
all eight worker output files. Production was not restarted. Clerk production
creation remains rejected by the provider's plan-feature mismatch; the operator's
dashboard identifies the enabled paid feature as biometric sign-in. Disabling
that optional native sign-in method and retrying creation remain pending.

The operator completed the workstation hosts repair. On September 26, normal
Windows requests (without forced DNS) to `https://api.joinhavn.io/healthz`
returned `ok: true`, five successful concurrency attempts and no errors;
`https://api.joinhavn.io/models/list` also returned a models payload. This clears
the workstation DNS override blocker, but does not prove the separate canonical
worker heartbeat or production launch acceptance requirements.

# Account images and collections

HAVN-11 development implementation. Production acceptance and real GPU inference
remain required; passing mocked worker tests is not evidence of generated quality.

Account image and face-swap requests use `/v2/jobs`, with the authenticated account
as owner and an idempotency key persisted before submission. Inputs are private
`/v2/assets` uploads. No raw remote face URL or wallet identity is accepted as an
account input. The worker receives owned asset IDs and downloads them through its
authenticated asset transport. Ownership is checked again under the enqueue lock
before credits are reserved.

`GET /v2/jobs?collection=1&type=visual` excludes account-hidden jobs and supports
search, type/status filtering, limit/offset pagination, and newest/oldest ordering.
`PUT /v2/account/collection` accepts `job_ids` and boolean `hidden`. The whole batch
must belong to the account; failure changes nothing. Hiding is a durable account
preference, not job deletion, and costs no credits. Full job history remains intact.

## Saved faces / identity anchors

`GET /v2/account/identity-anchors` lists this account's references.
`PUT /v2/account/identity-anchors/<slug>` accepts exactly `asset_id` and
`display_name`. The image must already belong to this account. Slugs are lowercase
letters, digits, hyphens, or underscores, 1–64 characters. Names are 1–100 trimmed
characters. An identical PUT is idempotent; a conflicting existing slug returns
409 rather than silently replacing a face. Different accounts may use the same slug.

`DELETE` on the same path idempotently removes only that account's saved reference.
It does not delete the uploaded asset or affect previously queued/completed jobs.
Wallet-era `identity_anchors` remain separate; linking a wallet imports nothing.

An image prompt may contain one `[IDENTITY ANCHOR: slug]` tag. Core removes the tag
from the render prompt and resolves its private image asset. Only SDXL image face
conditioning supports this input; combining it with an explicit face or refinement
source is rejected. The resolved asset is frozen in the queued job. The saved
reference is rechecked under the enqueue lock, so deletion before enqueue fails
without a reservation. A retry of an already accepted request recovers the original
job even after the saved reference has been removed.

Web Create exposes Saved faces in image controls. Reads are on demand, uploads and
mutations use the account session, requests abort on account change/unmount, and a
lost save response can retry against the same uploaded asset. No MetaMask call is
part of this flow. Anchor CRUD itself costs no credits.

Coverage: `tests/test_account_jobs.py`, web `AccountIdentityAnchors.test.tsx`,
`AccountCreate.test.tsx`, `AccountCollection.test.tsx`, and `JobDetailsDrawer.test.tsx`.

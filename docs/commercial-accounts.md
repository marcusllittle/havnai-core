# HAVN-11: commercial account and ownership contract

Contract version: 1. Work item: HAVN-18, under [HAVN-11](https://havnai.atlassian.net/browse/HAVN-11).

Status: account authentication/lifecycle, wallet-link API, integer ledger, account
checkout/receipts, job/asset APIs, account music publication/libraries/playlists,
account Video Studio, image/face-swap generation, private saved face references,
durable video sequences/stitching, account marketplace trading, and account-owned visual collections are implemented on
the feature branch. Development Google sign-in has been verified. Production
rollout, content migration, workflow-registry ownership, reference-sheet video
support, and live paid acceptance remain.
Those require HAVN-19 through HAVN-23 and the evidence below.

## Decision and trust boundary

HavnAI owns an immutable `acct_<uuid>` account ID. A managed authentication
provider proves a user identity; its verified `(issuer, subject)` maps uniquely to
that account. Email, a connected wallet, request body fields, and environment
wallet defaults are never account identifiers or authorization evidence.

The provider adapter is separate from the account repository. Clerk development
sign-in is configured and verified; production configuration remains outstanding. It supports the
existing Next.js Pages Router and Python backend. No custom password database
is proposed. Changing providers must preserve account IDs through a separately
audited, authenticated identity migration; matching email addresses never merges
accounts automatically.

Core is the authorization authority. The web client obtains a short-lived provider
session token and forwards it as `Authorization: Bearer ...` through the same-origin
API. Core validates signature, configured issuer, intended audience, authorized
origin/party, expiry, not-before, token type, and subject using the provider SDK.
Keys come only from the configured issuer, never a URL supplied by a token.
Provider failures fail closed. Provider suspension/deletion events disable the
local account; privileged operations require a freshly authenticated session.
The adapter must bound revocation delay and document it before launch.

The auth adapter allows up to five seconds of issuer/server clock difference for
`iat` and `nbf`, but independently enforces strict `exp` expiry, the 120-second
maximum token lifetime, and the existing five-minute factor-verification limit
for privileged actions. A local regression reproduced a roughly two-second
backward clock correction after a token was minted; zero leeway had rejected
that valid session. Deterministic tests cover that correction, the five-second
boundary, rejection beyond it, strict expiry and unchanged recent-auth checks.
Bearer scheme casing is normalized before provider verification. Rejection logs
contain only provider reason enum names or a fixed local-policy message, never
tokens, keys or identity claims.

Account APIs do not accept the shared `HAVNAI_OWNER_TOKEN`, the studio access key,
music read tokens, a bare wallet address, or arbitrary `X-Account-ID` headers as
user authentication. Keep owner/node administration on its existing explicit
boundary. `pages/api/owner/[...path].ts` is not the commercial-account gateway.

Authenticated responses are `private, no-store`. Do not forward cookies or tokens
to artifact/CDN URLs. Do not put bearer tokens in URLs, localStorage, telemetry,
or error messages. Same-origin cookie-based auth handlers enforce CSRF protection;
bearer-only core handlers do not fall back to ambient cookies. CORS allows only
configured application origins.

## Durable identity records

| Record | Required data and constraints |
| --- | --- |
| accounts | Immutable ID; active/suspended status; creation time. Never use an email or wallet as the primary key. |
| account_identities | Account FK; verified issuer and subject; unique `(issuer, subject)`. No client-driven identity attachment. |
| wallet_links | Link ID, account FK, normalized EVM address, chain namespace `eip155`, verified_at/linked_at/unlinked_at, audit actor. One active account per address across EVM chains. Many wallets per account. |
| account_wallet_challenges | Cryptographically random nonce, account FK, authenticated session binding, wallet, configured origin and chain, purpose, exact message, issued/expiry/used times. |
| account_audit_events | Append-only actor account/session reference, operation, target link/resource, timestamp, correlation ID; no raw session token or signature. |
| ownership_migrations | Account, verified wallet, explicit selected scope, snapshot digest/version, confirmation challenge, status, before/after audit references, timestamps. |

No wallet is required or implicitly primary. If a user chooses a preferred payout
wallet later, it is a separate setting with explicit confirmation, not a source
of identity or automatic content ownership. Unlinking that wallet clears the
preference. Node reward destinations stay under their existing independent rules.

## Ownership rules and compatibility

Add account columns; never place `acct_...` strings in Ethereum wallet columns.
Old wallet fields remain provenance/compatibility data, not permission shortcuts.
New commercial writes require account ownership and may have no wallet at all.

| Existing surface | Account-first representation |
| --- | --- |
| jobs.wallet; generation/recovery history | `creator_account_id` immutable; `owner_account_id` current access owner. Job requests derive both from the session. |
| assets.owner; artifacts.job_id | Uploaded assets gain `owner_account_id`; derived artifacts inherit job access. Validate access on downloads, status, recovery, cancellation, and source-asset reuse. |
| music_publications.creator_wallet | `creator_account_id` plus current ownership/access check against the source artifact. Public creator profile uses a separate nonsecret public profile ID. |
| music_library_saves; music_publication_likes | Account FK in uniqueness key `(account_id, publication_id)`; migrate conflicting saves once. |
| music_playlists.owner_wallet | `owner_account_id`; private playlist reads and all edits are account-authorized. Items inherit playlist access. |
| gallery_listings.owner_wallet/seller_wallet | `owner_account_id` and `seller_account_id`, distinct from immutable creator provenance. Historical sales retain original parties. |
| gallery_sales; gallery_ownership_log | Account buyer/seller/from/to columns for new activity; immutable wallet-era rows remain intact. |
| workflow_registry.creator_wallet | Immutable creator and current owner account columns; verify account for updates/publication. |
| credits.wallet | Separate `account_credit_balances` and `account_credit_ledger`; explicit audited balance migration. |
| stripe_payments.wallet | Account FK on new payments; immutable original payment provenance. Existing pending wallet checkouts settle to the old ledger until explicitly migrated. |
| payment receipts; generation receipts | Account receipt access references; preserve immutable receipt payload/hash and blockchain commitments. Do not rewrite anchored history. |
| rewards; node_wallets; payout claims | Remain wallet/chain-based. A wallet link permits an account view, not claim authority or reassignment. |

Ownership is checked server-side per resource, not merely when listing a library.
If a row has account ownership, matching an old wallet column never grants access.
For legacy rows without an account owner, existing signed-wallet authorization can
continue temporarily. A supplied but invalid account token returns 401; never
silently retry as the shared owner or a legacy wallet.

An authenticated account sees its account-owned assets only, plus explicit legacy
previews. It must not acquire legacy assets merely by connecting/linking a wallet.
Wallet switching leaves the current account, credits, and library unchanged.
Logout or account change cancels pending mutations and clears all account-scoped
client caches. Wallet changes invalidate only wallet-scoped caches/challenges.

## Link and unlink protocol

1. The signed-in user explicitly selects **Link wallet**. Explain optional rewards,
   token transfers, and blockchain ownership features. Browsing, creating, buying
   credits, publishing, saves, and playlists do not request a wallet signature.
2. `POST /v2/account/wallet-challenges` requires a recent account authentication,
   wallet address, allowed chain ID, and purpose `wallet_link`. Core creates a
   random 32-byte nonce, five-minute expiry, and binds it to account and session.
3. Return the exact UTF-8 message and opaque challenge ID. The message contains the
   configured domain/origin, account ID, wallet address, chain ID, purpose, nonce,
   issued time and expiry. It explicitly states that linking moves no content or
   credits. Use EIP-191 personal signing for the initial EOA implementation.
4. `POST /v2/account/wallet-links` accepts only challenge ID and signature. Recover
   the signer from the stored message. Require exact account/session/purpose,
   matching recovered address, unused challenge, and unexpired time.
5. Consume the challenge, insert the verified link, and append the audit event in
   one transaction. A unique active-wallet constraint prevents two racing accounts
   from attaching the same wallet. A conflict is 409; no automatic reassignment.
6. A successful link does not query-and-rewrite assets or fund either ledger.

Initial support is EVM EOAs only. Unsupported contract wallets return a specific
error; EIP-1271 support requires chain verification, not pretending an EOA recovery
failure is a valid signature.

Unlink is voluntary and requires recent account reauthentication plus a separate
account/session-bound `wallet_unlink` signature challenge during this rollout.
Requiring both matches HAVN-22's wallet-operation boundary. An inaccessible wallet
uses a manual recovery process with identity verification and an audit trail; it
cannot strand ordinary account access. Unlink atomically marks the link inactive,
invalidates outstanding challenges for that link, and records the actor. It does
not delete or reassign content, balance, receipts, or the account. A later link to
a different account gives no rights over anything already migrated.

## Explicit legacy migration

The read-only first stage is available at `GET
/v2/account/wallet-links/:link_id/import-preview?limit=50&offset=0`. It requires
the signed-in account's active EVM wallet link and reads a consistent database
snapshot. Its current scope is generation history, dependent music publications,
selected playlists and available legacy credits; save, reference and workflow inventories
still need their own dependency checks. No resource or balance changes during
this preview. Publication summaries accompany the current page of jobs, with a
flag when the bounded 100-publication selection limit is exceeded.
It resolves current gallery ownership, excludes active listings, non-final jobs
and account-owned resources, and supplies bounded pagination with global counts.
It does not expose prompts, source paths or another account's identity. Legacy
credit amounts that exceed bounds or require rounding are marked for review.
The preview is informational. `POST /v2/account/wallet-links/:link_id/import-snapshots`
now persists an immutable, five-minute selection using an `Idempotency-Key` and
`{job_ids: [...], include_credits: boolean, publication_ids?: [...], playlist_ids?: [...]}`. It requires recent account
authentication, accepts at most 100 distinct eligible jobs, and rejects the whole
selection if any job is unavailable. Credits are separately opt-in. The digest
binds the account, current session, wallet link, exact selection, expiry, and
hashes of the selected job/listing/artifact and credit rows. Private prompts and
filesystem paths are not copied into the snapshot. Retrying the same key and
selection returns the original snapshot, never refreshed inventory; changed
selection returns 409. `GET /v2/account/import-snapshots/:id` is restricted to the
original account session and still-active link; expired snapshots return 409.
Retries of version 1/2 selections normalize missing optional fields as empty
selections for comparison only. They return the original snapshot and never
rewrite its digest, extend its expiry or add newly supported resource types.
Neither endpoint changes ownership or balances, and `transfer_authorized` remains
false. Fresh import proof, dependency revalidation under the execution write lock,
atomic transfer and rollback/compensation controls below are still required before
enabling execution. Snapshot hashes currently cover only the stated inventory;
reference and workflow dependencies remain pending.

`POST /v2/account/import-snapshots/:id/challenge` accepts only `{chain_id}` and
requires recent account authentication plus an allowlisted request Origin. It
rechecks the selected job/listing/artifact rows, current ownership, active link,
and any selected credit balance under a write lock before issuing a distinct
`legacy_import` message. Changed selections return 409; the user must prepare a
new snapshot. The message binds the snapshot digest, exact job IDs and credit
units, account/session/wallet/link, origin, network and expiry. Its nonce is
stable for retries of that snapshot and cannot be reused with a different
network or origin. Unselected jobs and balances never enlarge the selection.
`GET /v2/account/import-capabilities` reports whether execution is enabled.
`POST /v2/account/import-snapshots/:id/execute` now accepts only the challenge ID,
signature and chain ID. It requires recent account authentication, an allowlisted
request Origin, and a supported network. Both challenge issuance and execution
return 503 unless `HAVNAI_ACCOUNT_IMPORT_ENABLED=1`; receipt recovery remains
available when execution is disabled. The local preview runner defaults this
switch off and supports `--enable-imports` for its isolated loopback environment.
No frontend requests this signature yet. Production enablement still requires the
backup/restore, migration and rollout checks below.

The internal `account_import.execute` now verifies the stored EIP-191 message,
rechecks account/session/proof context under a write lock, consumes the nonce,
assigns selected job ownership, settles selected credits and writes an immutable
receipt plus audit event in one transaction. It preserves original wallet
attribution; importing a purchased job never makes its buyer the original creator.
Legacy job/gallery access is denied once account ownership is assigned. Signed
retries return the original receipt, including under concurrent execution.
Receipt failure rolls back proof consumption, ownership and both balances.
`GET /v2/account/import-receipts/:snapshot_id` recovers a completed import's
durable result using ordinary account authentication, without a new wallet
signature or recent-login requirement. It survives snapshot expiry, wallet
unlinking and a new session for the same account. Pending/unknown imports and
other accounts' receipts return 404. `GET /v2/account/import-receipts` returns
paginated summaries with resource counts and integer credit units. Both are
private/no-store and reject suspended accounts. These reads never execute or
repeat a transfer; they are the recovery source for a lost completion response.
Publication dependencies now require explicit `publication_ids` (at most 100)
alongside their selected jobs. Preparation rejects incomplete selections, other
wallets' publications, existing account owners and mismatched audio artifacts.
Version 2 snapshots and confirmation messages include those publication IDs;
title, audio, state, attribution and ownership are rechecked during execution.
Engagement counters and their update timestamp are deliberately excluded from
the content digest. The existing publication rows move to account ownership,
preserving public links, saved references, publication times and play/like counts.
An account creator profile is created when needed. Legacy wallet unpublish is
denied afterward; account listing, profile display, playback and unpublish are
covered by integration tests. Selected balances with legacy Stripe history still
require explicit payment/refund provenance handling before enabling those cases.
Reference assets, saves and audit-driven reversal remain migration
integration work, not completed behavior.

Version 3 snapshots support up to 100 explicit playlist IDs, independently of
jobs, publications and credits. The preview includes a separately counted playlist
page using the same limit/offset. Review records title, sharing setting, ordered
publication IDs and a digest of playlist metadata and membership. The signature
binds those playlist IDs. Execution transfers only playlist ownership, preserving
track references/order, sharing, descriptions and URLs; it does not transfer the
referenced songs unless they are separately selected. Account-owned or foreign
wallet playlists are rejected, and metadata/membership changes invalidate the
snapshot. Account management survives unlinking the wallet; legacy wallet
management is denied after import. Integration tests cover private/public reads,
account updates, other-account denial and unchanged publication ownership.
Legacy playlist metadata edits, deletion, item addition/removal and reordering
now hold a write transaction from the ownership check through the final mutation.
This closes an import race where a wallet could pass the old ownership check and
then edit an already-transferred account playlist. Separate-connection tests prove
the ownership write cannot interleave, and failure tests prove multirow edits
roll back instead of leaving partial membership changes.

Each transferred job now has immutable indexed provenance in
`account_import_job_transfers`, linked to its signed import receipt. Startup can
backfill these rows from existing receipts, but never from wallet links or an
inferred owner. Completed imported image creations may be published and resold
through the account marketplace using this provenance instead of a nonexistent
account-generation charge. An existing reserved/released account charge still
blocks listing. Tests exercise real image previews, private original downloads,
sale and resale, original creator attribution, denial of the seller's access
after sale, and denial of legacy static access after import.

The internal `account_ledger.import_legacy_in_transaction` settlement primitive
moves the exact, positive available legacy balance to account credits in a
caller-owned write transaction. It requires an active account wallet link and
rejects ambiguous wallet rows, changed balances, fractional units and overflow.
It sets the old balance to zero without floating-point subtraction and preserves
historical deposit/spend totals. An append-only `account_credit_imports` record
stores the negative legacy-unit delta and references the matching positive
account ledger entry. A global migration ID binds wallet, account and units;
retries return the original receipt without sweeping later legacy deposits.
Savepoints prevent a caught error from retaining half a transfer. Tests cover
concurrent retries, distinct imports, races with legacy spending, account limits,
receipt failures and outer transaction rollback. This primitive is not a public
transfer API: its caller still needs atomic proof consumption, resource transfer,
legacy payment/refund provenance handling and the rollout evidence below.

Migration is a separate **Import existing wallet content** action after linking.
Proof of the wallet alone is insufficient if a resource already belongs to an
account. Never trust the creator wallet as proof of current gallery ownership.

1. Build an account-authorized, paginated preview from the linked wallet. Include
   counts/IDs by type, eligible available credits, exclusions, and conflicts.
   Resolve gallery current ownership before determining asset eligibility.
2. Do not move on-chain tokens, reward destinations, pending sales, or anchored
   receipt payloads. Block import of assets with active transfers/listings until
   they are safely paused/resolved. Keep running jobs and pending payments in the
   old path; offer them in a later preview after settlement.
3. Persist a snapshot/version digest, account, wallet, selected scope, and short
   expiry. Show the actual effects before asking for explicit confirmation.
4. Obtain a fresh `legacy_import` signature bound to account/session/wallet and
   the exact preview digest. Link signatures cannot authorize import.
5. Under one database write transaction, recheck all resource owners, active link,
   versions, job/payment states, and balances. Reject any changed snapshot with
   409 and require a new preview; never expand the selection automatically.
6. Assign only eligible unowned-by-account rows. Move available legacy credits
   through paired migration-out/migration-in ledger entries with a unique migration
   ID. Never copy the balance while leaving legacy spend enabled. Record the old
   and new owner for each resource; make retries return the original result.
7. Keep original attribution/payment/receipt history. Audit-driven reversal is
   available only before dependent spending, publishing, or transfer. Otherwise
   use an explicit reviewed compensating action; unlink is never rollback.

Back up the database and verify a restore before migration rollout. Dry-run the
preview against a copy first. No automatic full-database backfill from wallet links.

`scripts/account_restore_drill.py --source /path/to/coordinator.sqlite3 --output
/private/existing-parent/new-drill-directory` uses SQLite's online backup API to
capture committed WAL data through a read-only source connection. It compresses
the snapshot, restores it to a separate file, compares SHA-256 digests and table
counts, and requires integrity and foreign-key checks to pass before writing
`report.json`. It refuses existing output directories; it never restores over a
running database. Run on a private Linux filesystem (directory mode 0700, files
0600); this report contains table counts but no row values or credentials.
Database copies still contain private application data and must remain private.

On 2026-09-25 the isolated local account-preview database passed this drill with
71 tables and matching snapshot/restore SHA-256
`110b7bff81551d3a6009e48cda8ec85966ec6ccac36f521b33e080fe0323cf98`.
Local evidence is in
`/home/marcus/.local/state/havnai/account-preview/restore-drill-20260925-1137/report.json`.
The import restore test additionally executes a real signed import and an
idempotent retry on a restored fixture database, verifies both balances and the
receipt, and proves source ownership, balances and unused challenge remain intact.
This is local database evidence only: production backup/restore, media-file
backup and external payment/provider reconciliation still require verification.

## Cross-repository API v2

Paths below define the target core API, proxied through `/api` on web. Account,
credit balance/ledger, wallet link/unlink, assets, jobs and capabilities routes are
registered on this branch. Payment, migration and account music routes remain
implementation work. The identity repository is called by the verified-session
adapter, never directly by a client-supplied principal.

| Method/path | Input / result | Authorization |
| --- | --- | --- |
| GET /v2/account | Immutable ID, status, optional public profile, linked wallets and capabilities | Valid session; provision by verified issuer/subject only |
| GET /v2/account/credits | Integer available/reserved/debt units, scale=1000 | Account |
| GET /v2/account/ledger | Cursor-paginated immutable ledger entries | Account |
| GET /v2/account/receipts | Cursor-paginated receipts with status and safe download links | Account |
| GET /v2/account/jobs | Cursor, state filters; only account-owned jobs | Account |
| POST /v2/jobs | Generation spec and Idempotency-Key; returns durable job ID | Account; atomically reserve credits |
| GET/PATCH /v2/jobs/:id | Status/recovery or supported operation | Resource owner |
| POST /v2/music/publications | Artifact ID/title/visibility; no wallet/nonce/amount | Source owner account |
| GET/POST/PATCH/DELETE /v2/music/library and /v2/music/playlists | Library/save/playlist operations; resource IDs | Account; private resource ACL |
| POST /v2/payments/checkout | Package ID and Idempotency-Key; hosted checkout URL | Account; server price and allowlisted return URL |
| POST /v2/payments/webhook | Raw provider event + provider signature | Provider verification only, no account token |
| POST /v2/account/wallet-challenges | Wallet, chain, link/unlink purpose | Recent account session |
| POST /v2/account/wallet-links | Challenge ID, EIP-191 signature | Same account/session |
| DELETE /v2/account/wallet-links/:id | Unlink challenge ID, signature | Same account/session and link owner |
| POST /v2/account/imports/preview | Active link ID, resource-type selection | Account |
| POST /v2/account/imports/:id/confirm | Preview digest, bound challenge/signature, Idempotency-Key | Same account/session and active link |

Public discover, playback, public artwork, published playlists, creator profiles,
and gallery browse remain anonymous GET/read flows. No page-mount signatures.
Playback/download of private originals is never covered by the public exception.

Responses use JSON error envelopes:
`{"error":{"code":"account_required","message":"Sign in to continue."},"request_id":"..."}`.
Use 401 for absent/invalid/expired sessions, 403 for suspended accounts or forbidden
operations, 404 for inaccessible private resources, 409 for stale migration,
wallet-already-linked or idempotency conflicts, 422 for invalid inputs, 429 for
rate limits, and 503 for unavailable auth/payment configuration. No HTML API errors.
Never expose another account's ID or email in wallet-conflict responses.

Idempotency is scoped to account + operation + key, with a canonical payload hash.
The same key and payload returns the original durable result; different payload
returns 409. Payment external IDs also have global uniqueness constraints. Client
retries cannot choose another account via body/query/header fields.

## Credit, payment, and receipt invariants (HAVN-20/21)

Represent credits as integer milli-credits (1000 units = 1 credit) and money in
integer minor currency units. Reject nonfinite, negative, fractional-unit or
overflowing input. Do not continue wallet-era floating-point accounting for new
commercial balances. Define conversion and reconcile old balances during migration.

The append-only ledger records funding, reserve, capture, release, refund,
chargeback, migration and administrative compensation with account, operation key,
resource/payment reference, unit delta, actor/source, reason and timestamp.
Balance and ledger writes are one transaction. A job's reserve/capture/release
state transition and durable creation/state change commit together. Repeated job
submission, completion or failure cannot charge/refund twice. Concurrent spends
use an atomic available-balance condition under the database transaction.

Marketplace settlement now has an internal `settle_sale_in_transaction` primitive.
It transfers integer units from the buyer's available balance to the seller and
records paired ledger entries plus an immutable `account_credit_sales` receipt.
The sale ID binds buyer, seller and price globally; identical retries return the
same receipt, while changed participants or price conflict. Generation reservations
remain unavailable to purchases. Both accounts must be active for a new sale.
Seller-limit failures undo the debit even if the caller catches the exception.
This preserves the existing marketplace's full-price seller credit rule; it does
not introduce cash payouts, fees, or wallet-to-account balance conversion.

The account marketplace mutation API validates ownership and price, allocates the
sale ID, and commits the ownership transfer in that same caller-owned write
transaction. `tests/test_account_ledger.py` covers receipt replay/conflicts,
concurrent purchases, reserved balances, seller failure and outer rollback.

`POST /v2/marketplace/listings` accepts job/artifact IDs, title, optional description
and category, and integer `price_units`. The job must belong to the signed-in
account, have succeeded, have captured its generation reservation, and be an
`IMAGE_GEN` output (the existing marketplace eligibility rule). The selected
image artifact must still exist. `POST /v2/marketplace/listings/:id/purchase`
requires `expected_price_units`. Both operations require an `Idempotency-Key`;
their account-scoped durable intent records return the original result on retry.
`DELETE /v2/marketplace/listings/:id` is an idempotent owner-only delist. A buyer
can relist using the listing creation route with a new key.

Sales transfer the job's account ownership, so artifact download and Collection
authorization follow the buyer immediately. Creator provenance and source-upload
ownership stay intact. Purchased jobs reappear in the buyer's Collection even if
previously hidden. Listing, sales and provenance rows use separate account
columns; wallet fields remain blank for account activity. Integer price columns
are authoritative; legacy REAL columns are compatibility mirrors only.
Account mutations are tested through authenticated HTTP and real database races
in `tests/test_account_marketplace.py`. The account-aware web marketplace is still
pending; these backend routes alone do not make the marketplace UI ready.

Guests can read `GET /v2/marketplace/listings` and `GET
/v2/marketplace/listings/:id`. Responses use an explicit field allowlist: listing
copy, model, integer price, creation time and preview URL. They omit prompts,
source assets, internal account IDs, job IDs and original-file URLs. Browse supports
literal search, category, stable sorting and bounded offset pagination. Only
active listings whose job owner matches an active account appear publicly.

`GET /v2/marketplace/listings/:id/preview` renders a JPEG at most 640 pixels per
side, with a visible preview label and no source metadata. It reads only image
artifacts inside the coordinator output directory and bounds source bytes/pixels.
The endpoint checks publication and ownership again after encoding. It returns
no-store/nosniff responses and closes on delisting, sale or account suspension.
Already downloaded previews cannot be recalled; private originals continue to
require current-owner authentication.

Authenticated `GET /v2/account/marketplace/listings` returns the current owner's
latest listing per job, including sold/delisted items and authenticated artifact
links. `GET /v2/account/marketplace/receipts` returns that account's purchase/sale
history with immutable ledger entry references. Both paginate, ignore supplied
account identifiers and remain private/no-store; sale history persists after
ownership changes.

Legacy gallery access now excludes any job with an account owner, even when an
old wallet listing, sale or ownership log still exists. This applies to browse
counts/results, details, downloads, wallet collections/history and mutations.
Listing inserts recheck account ownership in the write statement; purchases check
under their write lock. Historical wallet rows remain intact for migration audit.
The original wallet creator also cannot use the listing endpoint to reclaim an
asset already sold to another wallet. Backend reliability tests cover these
boundaries, including ownership changing between the initial check and insert.

Checkout persists an immutable server-priced purchase intent before contacting
Stripe, uses a stable provider idempotency key, and survives a crash between API
success and local persistence through reconciliation. Never use client-supplied
credit quantities, account IDs or success redirects as payment evidence.

Only verified successful paid events fund credits. Validate mode, provider account,
currency, amount, purchase intent and payment identity against local records.
Deduplicate by economic payment/funding identity as well as event ID, because
different events can describe one payment. Commit event handling, ledger entry,
receipt and payment state atomically. Unmatched events remain recoverable for
reconciliation; do not acknowledge them as already funded.

Failed/cancelled sessions fund zero credits. Handle asynchronous success/failure,
duplicate/out-of-order delivery, partial/full refunds and disputes explicitly.
Refund amounts aggregate against the original purchase; reverse only the matching
funding once. If refunded credits were already spent, record recoverable debt and
freeze new spending as needed. Do not clamp away the debt or allow a negative
spendable balance. New funding offsets debt before becoming spendable. Released
job reservations cannot recreate spendable funds while that debt remains.

Durable account receipts include purchase ID, provider reference, amount/currency,
credit units, terms/price version, timestamps and adjustments. They survive wallet
unlinking and provider retries. Payment receipts and on-chain artifact receipts
are distinct; owning one grants no authority over the other.

## Threat model and required proof

| Threat | Required control and test |
| --- | --- |
| Forged account or shared owner access | Only verified issuer/subject creates account; body/header wallet/account forgery and owner-token tests |
| Account confusion on email/wallet switch | Stable subject mapping; two-account, same-email and wallet-change isolation tests |
| Token replay/cross-origin token | SDK claim validation, short expiry, authorized origin, revoked/suspended account tests |
| Link signature replay or wrong session/account | Stored message binding, five-minute expiry, purpose separation, atomic nonce consumption; negative and race tests |
| Concurrent attachment to two accounts | Unique active address constraint + transaction; one winner, no partial audit/link |
| Link/unlink silently moves assets | Assert zero ownership/credit mutations; relink to another account preserves original owner |
| Legacy import race | Preview digest and fresh state check, current-owner checks; stale preview fails atomically |
| Private artifact/playlist leak | Per-object authorization on every read/write and download; guest/cross-account denial tests |
| Duplicate funding/charge | Economic-key uniqueness, atomic ledger transitions; duplicate webhook and job completion tests |
| Negative-balance race | Multiple real SQLite connections concurrently spend a bounded balance; at most affordable successes |
| Refund after spend or reordered events | Durable debt/holds, cumulative refund tracking; retry and ordering tests |
| Signature prompt storm | No wallet API calls during account navigation/generation/publication; browser assertions |

## Delivery and launch evidence

1. HAVN-18: this contract, isolated identity storage and challenge tests. No live
   data migration or runtime behavior change. Review schema before activation.
2. HAVN-19: choose/provision provider; configure Next.js and core verification;
   ship register/sign-in/account session and per-object ownership end to end.
3. HAVN-20: integer account ledger, atomic job reservation and legacy import,
   including concurrent/idempotency tests and reconciliation report.
4. HAVN-21: Stripe account checkout/webhooks/refunds/receipts, operational provider
   configuration, approved prices/terms/refund policy, test-mode then controlled
   production purchase evidence. Current package prices are not policy approval.
5. HAVN-22: route ordinary actions through account auth; retain explicit signing
   for link/unlink/import and blockchain/economy actions. Remove commercial
   dependence on shared studio owner credentials. Coordinate wallet UX HAVN-13.
6. HAVN-23: capture release commits, executed checks, restore drill, security
   review, live payment evidence and remaining limits. Do not close HAVN-11 while
   any acceptance item is only documented or mocked.

References: [Clerk session tokens](https://clerk.com/docs/guides/sessions/session-tokens),
[Clerk Python SDK](https://github.com/clerk/clerk-sdk-python).

## Evidence for this foundation change

`server/account_identity.py` implements the durable issuer/subject mapping,
account status checks, many-wallet links, stored EIP-191 challenges, one-time
consumption, unlink invalidation, and transactional append-only link audit. It is
now initialized by `app.py` with additive schema migrations on branch startup.
`VerifiedPrincipal` is an internal adapter input, not proof by itself. HTTP code
must never construct it from unverified request fields.

Executed `python -m pytest tests/test_account_identity.py -q`: **21 passed** using
Python 3.12, pytest 9.1.1 and eth-account 0.14.0 in an isolated test environment.
Tests use real EOA signatures and separate SQLite connections for concurrent
account creation, competing account links, and simultaneous nonce replay. They
also cover wrong account/session/purpose, altered signatures, expiry, suspension,
multiple wallets, unlink/relink, audit immutability and transactional rollback.

Later branch checks cover real RSA session signatures, audience/issuer/origin and
reauthentication validation, HTTP account isolation, integer ledger races/refunds,
and account job reservation/recovery/cancellation/completion. Job API tests exercise
the existing worker completion path and private artifact access; they use fixture
artifact bytes rather than running a GPU generation.

Private account jobs are excluded from legacy wallet history and the public job
feed. Old per-job/result routes and direct static artifact URLs cannot bypass
account authorization. The account artifact endpoint currently requires bearer
authentication; the web account-media route verifies the Clerk cookie session,
forwards a short-lived bearer token, and streams audio ranges with private/no-store
headers. It never forwards the browser cookie or a shared owner credential to core.

Provider token validation bounds session lifetime to 120 seconds. Verified Clerk
lifecycle delivery can revoke access earlier; see `docs/account-lifecycle.md` for
ordering, deletion tombstones, and the remaining live-delivery acceptance gate.
The feature branch has not been deployed to the production coordinator. Local
development accounts and isolated preview data have been used for verification;
production wallet links, assets, credits, and payments have not been migrated.

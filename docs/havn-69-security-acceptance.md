# HAVN-69 Security And Abuse-Control Acceptance

Date: 2026-09-27

Release branch: current `main` evidence port for HAVN-72 launch hardening

Core evidence branch: `codex/havn-69-security-main`

## Scope

This evidence packet covers the HavnAI platform security gate for public
surfaces, account-owned API isolation, lifecycle/session revocation, wallet
nonce replay handling, and marketplace account boundaries. It reuses existing
HAVN-20/HAVN-21 payment invariants, HAVN-14 content-isolation evidence, and
HAVN-57 reward validation instead of duplicating those implementation tracks.

Astra client/game work remains outside this ticket.

## Production Public-Surface Audit

Command:

```bash
python3 scripts/security_public_surface_audit.py \
  --web-base https://joinhavn.io \
  --api-base https://api.joinhavn.io
```

Result: passed.

Evidence captured at `2026-09-27T06:05:04Z`:

- Web pages `/`, `/create`, `/pricing`, `/support`,
  `/terms/credits-v1`, and `/refunds/credits-v1` returned HTTP 200.
- Public API routes `/health`, `/healthz`, `/models/list`,
  `/gallery/browse?limit=5`, and `/music/discover?limit=5` returned HTTP 200.
- The audit found no obvious secret-key, webhook-secret, private-key,
  internal-path, or SQLite path patterns in those anonymous responses.
- Anonymous protected routes `/v2/account`, `/v2/account/credits`,
  `/v2/astra/session`, and `/v2/astra/stats` returned HTTP 401 with
  `Cache-Control: private, no-store` and no secret/internal-path findings.

## Public Bundle / Static Asset Secret Audit

Command:

```bash
python3 scripts/security_bundle_audit.py \
  --web-base https://joinhavn.io
```

Coverage:

- Fetches selected public pages anonymously.
- Discovers same-origin linked Next.js/static assets from those pages.
- Scans bounded page and asset bodies for obvious Stripe/Clerk/GitHub secret
  keys, webhook secrets, private-key blocks, internal workstation paths, and
  SQLite database path leaks.
- This is a pattern audit for accidental public exposure; it does not replace
  credentialed two-account authorization tests.

Latest production result, 2026-09-27:

- `python3 -m py_compile scripts/security_bundle_audit.py` passed.
- `python3 scripts/security_bundle_audit.py --web-base https://joinhavn.io`
  passed.
- Report schema: `havn-69-public-bundle-secret-audit.v1`.
- Generated at `2026-09-27T11:20:14Z`.
- Scanned 46 anonymous page/static asset responses from current production.
- Findings: none.

## Account Boundary Regression Slice

Command:

```bash
uv run --with-requirements server/requirements.txt --with pytest --with pillow \
  python -m pytest \
  tests/test_account_auth.py \
  tests/test_account_identity.py \
  tests/test_account_lifecycle.py \
  tests/test_account_jobs.py::test_revoked_session_cannot_recover_or_submit_studio_jobs \
  tests/test_account_jobs.py::test_source_assets_and_worker_artifacts_are_private \
  tests/test_account_marketplace.py \
  tests/test_recovery_plan.py::CreditConversionNonceTests \
  -q
```

Result: `104 passed in 122.81s`.

Coverage represented:

- Account auth rejects missing, invalid, revoked, suspended, and wrong-session
  states.
- Wallet identity linking is account-bound and nonce/audit protected.
- Clerk lifecycle events revoke sessions and duplicate/mismatched replay is
  rejected.
- Revoked account sessions cannot recover or submit studio jobs.
- Source assets and worker artifacts remain private to the owner.
- Marketplace account purchase/listing/receipt boundaries remain covered.
- Legacy wallet credit conversion nonce behavior now accepts checksum-case
  wallet clients while doing case-insensitive nonce lookup/update; signed
  conversion succeeds once, replay is rejected, expired nonces report
  `nonce_expired`, and signature wallet mismatch is rejected.

## Fixed Finding: Legacy Credit-Convert Nonce Regression

The first broader security run included two legacy `/credits/convert` nonce
tests and failed:

```bash
uv run --with-requirements server/requirements.txt --with pytest --with pillow \
  python -m pytest \
  tests/test_account_auth.py \
  tests/test_account_identity.py \
  tests/test_account_lifecycle.py \
  tests/test_account_jobs.py::test_revoked_session_cannot_recover_or_submit_studio_jobs \
  tests/test_account_jobs.py::test_source_assets_and_worker_artifacts_are_private \
  tests/test_account_marketplace.py \
  tests/test_recovery_plan.py::CreditConversionNonceTests::test_nonce_signature_success_and_replay_rejected \
  tests/test_recovery_plan.py::CreditConversionNonceTests::test_nonce_expired_rejected \
  -q
```

Result: `100 passed`, `2 failed`.

Failures:

- `test_nonce_signature_success_and_replay_rejected`: expected successful
  signed credit conversion followed by replay rejection; first conversion
  returned HTTP 400.
- `test_nonce_expired_rejected`: expected `nonce_expired`; received
  `invalid_nonce`.

Root cause:

- `/wallet/nonce` stored the nonce wallet in lowercase, while clients and tests
  can send checksum-case wallet addresses back to `/credits/convert`.
- `/credits/convert` used exact-case `wallet_nonces` lookup/update, so a valid
  signed nonce was not found.

Fix:

- Preserve submitted wallet casing when issuing nonces.
- Rate-limit and credit-ledger operations still use lowercase wallet keys.
- Wallet nonce verification helpers and `/credits/convert` now perform
  case-insensitive nonce lookup/update using `lower(wallet)=?`.

Verification:

```bash
uv run --with-requirements server/requirements.txt --with pytest --with pillow \
  python -m pytest tests/test_recovery_plan.py::CreditConversionNonceTests -q
```

Result: `4 passed in 2.26s`.

Full security slice after fix: `104 passed in 122.81s`.

## Remaining HAVN-69 Gaps

- Run signed-in browser security checks with two real account sessions: direct
  API/media URL access, account switching, sign-out, revoked session, and cached
  page data behavior.
- Verify configured upload size/type limits and URL-fetch input restrictions in
  a live or production-equivalent browser flow.
- Run dependency/security scanning for core and web package sets and record
  critical/high findings or dated exceptions.

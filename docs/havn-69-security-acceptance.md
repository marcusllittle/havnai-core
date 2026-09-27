# HAVN-69 Security And Abuse-Control Acceptance

Date: 2026-09-27

Release branch: `feat/havn-11-commercial-accounts`

Core evidence branch: `codex/havn-69-security-acceptance`

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
  -q
```

Result: `100 passed in 131.06s`.

Coverage represented:

- Account auth rejects missing, invalid, revoked, suspended, and wrong-session
  states.
- Wallet identity linking is account-bound and nonce/audit protected.
- Clerk lifecycle events revoke sessions and duplicate/mismatched replay is
  rejected.
- Revoked account sessions cannot recover or submit studio jobs.
- Source assets and worker artifacts remain private to the owner.
- Marketplace account purchase/listing/receipt boundaries remain covered.

## Open Finding: Legacy Credit-Convert Nonce Regression

The broader security run included two legacy `/credits/convert` nonce tests and
failed:

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

Launch interpretation:

- The account-first commercial flow and account-owned wallet linking boundaries
  are not invalidated by this result.
- The legacy wallet credit-conversion endpoint should be triaged before any
  launch claim that legacy wallet credit conversion remains supported.
- If the endpoint is intentionally out of commercial launch scope, record an
  explicit release waiver/disablement decision and make sure public product copy
  does not advertise that path.

## Remaining HAVN-69 Gaps

- Run signed-in browser security checks with two real account sessions: direct
  API/media URL access, account switching, sign-out, revoked session, and cached
  page data behavior.
- Verify configured upload size/type limits and URL-fetch input restrictions in
  a live or production-equivalent browser flow.
- Run dependency/security scanning for core and web package sets and record
  critical/high findings or dated exceptions.
- Triage the legacy `/credits/convert` nonce regression above or explicitly
  waive/disable the legacy path for commercial launch.

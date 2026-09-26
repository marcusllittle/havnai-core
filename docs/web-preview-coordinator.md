# Coordinator integration for the current web preview

Branch: `fix/music-rewards-receipts`.

Includes the existing music APIs, marketplace purchase route, artifact receipts,
Merkle receipt batches, and node reward claim batches. Startup upgrades older
Astra reward tables with the nullable video_job_id column needed by receipts.

Test HAI requests persist in a manual-review queue. Submitting a request never
transfers tokens or grants credits. An administrator may explicitly grant app
credits during resolution; the grant and queue update commit atomically and
repeated resolution cannot grant the same credits twice. Resolution requires a
configured SERVER_JOIN_TOKEN. This route does not transfer HAI tokens.

Optional settings:
- HAVNAI_TESTER_DISTRIBUTION_ENABLED=0 disables requests (default enabled).
- HAVNAI_TESTER_DISTRIBUTION_ALLOWLIST accepts comma-separated wallet addresses.
- Requests default to 100 HAI; pending requests are deduplicated and resolved
  requests have a 24-hour cooldown measured from submission.

Receipt and reward ledger reads do not require signing or an RPC connection.
On-chain anchoring and claiming still require their contract/RPC configuration;
see contracts/README.md and the settings in receipt_anchors.py/payout_chain.py.

Music private reads now use an eight-hour read-only session from /music/session.
The frontend proves wallet ownership once, shares concurrent authorization,
stores the session in sessionStorage, and clears it on disconnect/account change.
The database stores only the token hash. Sessions cannot authorize purchases,
publishing, playlist changes, or other writes. Old per-read signatures remain
accepted for older clients. Deploy this coordinator before the session-enabled web.

Marketplace signatures must match the listing/job being changed; delisting now
always requires a signature. Non-finite amounts are rejected. Unexpected API
exceptions return JSON with a reference matching the server log; structured
context and stack traces are retained server-side.

Live read-only audit on September 17 at coordinator version 879cf57:
- Discover, gallery, marketplace, analytics, nodes/leaderboard, credit reference,
  payment packages, receipt batches, payout batches/claims, reward claimable,
  and Test HAI history returned JSON successfully through joinhavn.io/api.
- Reward claim_contract is null: chain claiming still needs deployment/configuration.
- /v1/network/summary and /v1/network/control-plane are absent. The web uses
  legacy summary data and reports advanced telemetry unavailable.
- Studio uses the protected /api/owner/v1/capabilities route, not /music/capabilities.
- No user wallet signatures, purchases, credit grants, or chain transactions were
  performed in the audit. These live mutations are not verified by read-only probes.

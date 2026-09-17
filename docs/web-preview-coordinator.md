# Coordinator integration for the current web preview

Branch: `fix/music-rewards-receipts`.

Includes the existing music APIs, marketplace purchase route, artifact receipts,
Merkle receipt batches, and node reward claim batches. Startup upgrades older
Astra reward tables with the nullable video_job_id column needed by receipts.

Test HAI requests now persist in a manual-review queue. No token transfer or
credit deposit occurs when a request is submitted or its metadata is resolved.
The administrator records fulfillment separately after a manual distribution.

Optional settings:
- HAVNAI_TESTER_DISTRIBUTION_ENABLED=0 disables requests (default enabled).
- HAVNAI_TESTER_DISTRIBUTION_ALLOWLIST accepts comma-separated wallet addresses.
- Requests default to 100 HAI; pending requests are deduplicated and resolved
  requests have a 24-hour cooldown measured from submission.

Receipt and reward ledger reads do not require signing or an RPC connection.
On-chain anchoring and claiming still require their contract/RPC configuration;
see contracts/README.md and the settings in receipt_anchors.py/payout_chain.py.

Validated 59 tests plus 10 subtests across receipt, Merkle, payout, platform,
music API, marketplace, and Test HAI request flows using isolated databases.

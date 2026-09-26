# HavnAI Mainnet Readiness Checklist

Work through this list before switching `CHAIN_RPC_URL` from Sepolia to mainnet.

## Secrets & Wallets

- [ ] Rotate any private keys that have ever been committed to git history
      (`git filter-repo` to scrub; re-generate wallet entirely)
- [ ] `HAVNAI_PAYER_KEY` is a dedicated hot wallet — no other funds on it
- [ ] Payer wallet holds only enough ETH for ~1 week of gas at current prices
- [ ] Treasury multi-sig holds the HAI reserve; payer topped up weekly via script
- [ ] All secrets in production are set via Vercel / Railway env vars, not in .env files
- [ ] `STRIPE_SECRET_KEY` is the live key, not the test key

## Smart Contract

- [ ] `HavnRewardDistributor.sol` audited or reviewed by a second engineer
- [ ] Deployed to Sepolia; all functions tested via `scripts/simulate_testnet.py`
- [ ] Ownership transferred to the team multi-sig (not a personal wallet)
- [ ] `batchDistribute` tested with 200-entry batch without OOG
- [ ] Source verified on Etherscan
- [ ] Deployed to mainnet; address set in `HAVNAI_DISTRIBUTOR_ADDRESS`

## Server Configuration

- [ ] `CHAIN_RPC_URL` points to mainnet RPC (not Sepolia)
- [ ] `HAI_TOKEN_ADDRESS` is the mainnet HAI contract
- [ ] `DATABASE_URL` points to a PostgreSQL instance (not SQLite)
- [ ] `REDIS_URL` set for production rate limiting
- [ ] `HAVNAI_ALERT_WEBHOOK_URL` wired to Slack/Discord for on-call
- [ ] `CORS_ORIGINS` locked to `https://joinhavn.io` (no wildcards)
- [ ] Gunicorn `--workers` tuned to 2× CPU cores
- [ ] Docker health check passing in staging

## Payout Worker

- [ ] Payout worker starts successfully with `payout_worker.start(get_db, log_event)` in app.py
- [ ] `PAYOUT_BATCH_SIZE` tuned so one batch fits in ~30s at current gas prices
- [ ] `PAYOUT_MIN_AMOUNT` set above dust threshold (suggest 0.1 HAI)
- [ ] Simulate full batch: 20 pending payouts → all confirmed with tx_hash
- [ ] Verify `get_payout_stats()` reports 0 pending after batch run
- [ ] Alert fires if payout queue grows beyond 500 pending (add to alerting.py)

## Economics & Limits

- [ ] `DAILY_EARN_CAP` in `astra_rewards.py` calibrated against token supply
- [ ] `REWARD_COOLDOWN_SECONDS` set to prevent farming (suggest 300s for mainnet)
- [ ] Node payout rates reviewed against expected compute volumes
- [ ] Total supply / circulating / reserve tracked in a spreadsheet
- [ ] HAI/USD peg or float policy documented for the team

## Frontend

- [ ] `havnai-web` `NEXT_PUBLIC_HAVNAI_API_BASE` set to `https://api.joinhavn.io`
- [ ] CSP `connect-src` includes only HTTPS origins (no HTTP)
- [ ] Vercel preview deployments use staging API key, not production
- [ ] `.env.local` is in `.gitignore` and not present in git history

## Astra Valkyries

- [ ] `VITE_HAVNAI_PROXY_TARGET` set correctly for production builds
- [ ] `astraReward()` tested end-to-end on Sepolia with real MetaMask wallet
- [ ] Leaderboard shows mainnet wallet addresses (not test addresses)
- [ ] Daily reward cap visible in-game and matches server-side cap

## Observability

- [ ] `/health` endpoint returns `db_ok: true` and correct `active_nodes`
- [ ] Logs shipping to a log aggregator (Datadog, Papertrail, etc.)
- [ ] Uptime monitor pinging `/health` every 60s with alert on 3 consecutive failures
- [ ] On-call rotation established for production incidents

## Rollback Plan

- [ ] Database snapshot taken immediately before mainnet switch
- [ ] `CHAIN_RPC_URL` can be blanked to drop back to simulation mode without restart
- [ ] `HavnRewardDistributor.withdraw()` tested — can recover HAI if contract needs replacement
- [ ] Incident runbook written and shared with the team

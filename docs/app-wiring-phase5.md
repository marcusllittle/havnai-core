# Phase 5 — app.py wiring guide

This supplements `docs/app-wiring.md`. Add these lines to `server/app.py`
after the existing Phase 2/3 startup calls.

## 1. Import at the top of app.py

```python
from server import chain          # EVM integration (no-op when CHAIN_RPC_URL unset)
from server import payout_worker  # On-chain payout daemon
```

## 2. Start payout worker after dependency injection

Add this alongside the stale_job_recovery and health module starts:

```python
# Phase 5 — on-chain payout worker
payout_worker.start(get_db, log_event)
```

## 3. Update /blockchain/verify route (optional)

If `blockchain.py` has a `verify_wallet` endpoint, replace the stub
response with a real signature check:

```python
# Before (stub):
return {"verified": False, "message": "not implemented"}

# After:
signature = request.json.get("signature", "")
message   = request.json.get("message", "")
verified  = chain.verify_eip191_signature(wallet, message, signature)
return {"wallet": wallet, "verified": verified}
```

## 4. New env vars

| Variable | Required for | Default |
|---|---|---|
| `CHAIN_RPC_URL` | Live on-chain transfers | — (simulation mode) |
| `HAI_TOKEN_ADDRESS` | Live on-chain transfers | — |
| `HAVNAI_PAYER_KEY` | Live on-chain transfers | — |
| `PAYOUT_INTERVAL_SECONDS` | Payout worker tuning | `120` |
| `PAYOUT_BATCH_SIZE` | Payout worker tuning | `20` |
| `PAYOUT_MIN_AMOUNT` | Skip dust payouts | `0.01` |

All are optional. Omitting them keeps the server in simulation mode with
no breaking changes to existing behaviour.

## 5. Test end-to-end

```bash
# Start the server, then:
python scripts/simulate_testnet.py --base-url http://localhost:5001
```

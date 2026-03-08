# Settlement Testing Guide

## Prerequisites

- Start the coordinator: `cd server && python app.py`
- Ensure `HAVNAI_CREDITS_ENABLED=true` in env
- Have a test wallet with credits deposited
- Have at least one node registered and online

## 1. Successful Image Job

```bash
# Submit a job
curl -X POST http://localhost:5050/submit-job \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","model":"auto","wallet":"0xYOUR_WALLET"}'
# Note the job_id

# Wait for completion, then check settlement
curl http://localhost:5050/jobs/JOB_ID
# Expected: settlement.execution_status = "settled"
#           settlement.quality_status = "valid"
#           settlement.settlement_outcome = "spent"
#           settlement.marketplace_eligible = true

# Check settlement detail
curl http://localhost:5050/settlement/JOB_ID
# Expected: full record with attempts array

# Check credit ledger
curl "http://localhost:5050/credits/ledger?wallet=0xYOUR_WALLET"
# Expected: "reserve" entry (negative) then "spend" entry
```

## 2. Technical Failure Path

```bash
# Submit a job to a model with no online nodes (or kill node mid-job)
# The coordinator will mark it as failed when the node disconnects or times out

# If manually testing, submit results with failure status:
curl -X POST http://localhost:5050/results \
  -H "Content-Type: application/json" \
  -d '{"node_id":"test-node","task_id":"JOB_ID","status":"error","metrics":{"error":"GPU OOM"}}'

# Check settlement
curl http://localhost:5050/settlement/JOB_ID
# Expected: execution_status = "technical_failed"
#           settlement_outcome = "released"
#           spent_amount = 0.0

# Check credit ledger
curl "http://localhost:5050/credits/ledger?wallet=0xYOUR_WALLET"
# Expected: "reserve" entry then "release" entry (credits returned)
```

## 3. Malformed Output Path

To test malformed output, submit a result with an image that fails validation:

```bash
# Create a tiny invalid "image" file
echo "not an image" | base64 > /tmp/bad_img.txt

# Submit results with invalid image data
curl -X POST http://localhost:5050/results \
  -H "Content-Type: application/json" \
  -d '{"node_id":"test-node","task_id":"JOB_ID","status":"success","metrics":{},"image_b64":"bm90IGFuIGltYWdl"}'

# Check settlement
curl http://localhost:5050/settlement/JOB_ID
# Expected: quality_status = "malformed"
#           settlement_outcome = "full_refund"
#           (node gets 10% consolation payout)

# Check credit ledger
curl "http://localhost:5050/credits/ledger?wallet=0xYOUR_WALLET"
# Expected: "reserve" entry then "refund" entry
```

## 4. Face Swap Path

```bash
# Submit a face swap job
curl -X POST http://localhost:5050/submit-faceswap-job \
  -H "Content-Type: application/json" \
  -d '{"prompt":"portrait","model":"epicrealismxl_vxviicrystalclear","wallet":"0xYOUR_WALLET","base_image_url":"https://example.com/face.jpg","face_source_url":"https://example.com/source.jpg"}'

# After completion, check settlement
curl http://localhost:5050/settlement/JOB_ID
# Expected: job_type = "FACE_SWAP"
#           marketplace_eligible = 0 (false)
#           settlement_outcome = "spent" (if valid)
```

## 5. Duplicate Settlement Prevention

```bash
# Submit the same result twice for the same job
curl -X POST http://localhost:5050/results \
  -H "Content-Type: application/json" \
  -d '{"node_id":"test-node","task_id":"JOB_ID","status":"success","metrics":{}}'

# Second submission should:
# - Be rejected by job_helpers.complete_job() (job no longer "running")
# - settlement.settle_job() returns "already_settled" if called again
# - No double credit spend or double payout

# Verify: check node_payouts
curl "http://localhost:5050/payouts/node/test-node"
# Expected: only one payout entry for this job_id
```

## 6. Retry Behavior

If a job fails and is re-queued (e.g., server restart resets running→queued):

- The `job_settlement.attempt_count` increments
- A new `job_attempts` row is created for the retry
- The original reservation is still in effect
- On successful retry, settlement proceeds normally
- `complete_job_if_queued()` in job_helpers handles late completions

## Verifying Data Integrity

```bash
# Check that all settled jobs have matching ledger entries
sqlite3 server/havnai.db "
  SELECT js.job_id, js.settlement_outcome, js.reserved_amount,
         GROUP_CONCAT(cl.event_type || ':' || cl.amount) as ledger
  FROM job_settlement js
  LEFT JOIN credit_ledger cl ON cl.job_id = js.job_id
  GROUP BY js.job_id
  ORDER BY js.created_at DESC
  LIMIT 20;
"
```

# Live Demo Runbook

Everything needed to stand up HavnAI Core and Astra Valkyries in front of an
audience, plus the numbers and answers to back it up. Written for a ~30 minute
technical talk.

Every figure quoted here was produced by running this stack, not estimated.

---

## 1. Pre-flight (do this before you leave)

Three terminals. Run them in this order — the game's proxy expects the
coordinator to already be listening.

### Terminal 1 — seed and boot the coordinator

```bash
cd havnai-core
python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt

# Writes ~a day of history into db/demo.db. Never touches db/ledger.db.
.venv/bin/python scripts/seed_demo.py --reset --db db/demo.db

cd server
HAVNAI_DB_PATH=../db/demo.db HAVNAI_NODES_PATH=../nodes.demo.json \
  SERVER_PORT=5001 ../.venv/bin/python app.py
```

Two things that will bite you if you improvise:

- **Run `app.py` from inside `server/`.** The modules import flat
  (`import safety`), so `python -m server.app` from the repo root fails.
- **Seed while the server is stopped.** `--reset` deletes the database file;
  doing that under a live server leaves it holding a dead handle and every
  endpoint returns 500.
- **Pass `HAVNAI_NODES_PATH`.** The seeder writes its node registry to
  `nodes.demo.json` and the coordinator defaults to the tracked `nodes.json`, so
  without this the dashboard's per-node HAI column reads zero. Both demo files
  are gitignored, so nothing you run here dirties the working tree.

### Terminal 2 — hold the nodes online

```bash
cd havnai-core
.venv/bin/python scripts/seed_demo.py --skip-seed --serve
```

The coordinator drops a node after **120 seconds** without a heartbeat. Without
this loop your node count decays to zero partway through the talk. Leave it
running.

### Terminal 3 — the game

```bash
cd Astra-Valkyries/rhythm-jet-squadron
npm ci
npm run dev          # http://localhost:5173
```

### 60-second verification

```bash
curl -s localhost:5001/analytics/overview
curl -s localhost:5173/api/astra/leaderboard
```

Expect `online_nodes: 4`, `success_rate: 95.3`, and a populated leaderboard. If
the leaderboard is `[]`, terminal 1 is pointed at the wrong database.

---

## 2. What the seeded network looks like

| Metric | Value |
|---|---|
| Jobs in the last 24h | 340 (324 success / 16 failed) |
| Success rate | 95.3% |
| $HAI distributed | 331.109681 |
| Models exercised | 25 |
| Creator wallets | 6 |
| Nodes online | 4 |
| Game runs on the leaderboard | 56 |

**These are real reward computations.** The seeder imports
`server/rewards.compute_reward` and `astra_rewards._interpolate_reward` and calls
them with the production manifest weights. The job history is synthetic; the
arithmetic on top of it is your engine's. Say that plainly if asked — it lands
better than letting someone discover it.

---

## 3. Demo sequence

Roughly 12 minutes of screen time. Ordered so each step motivates the next.

### Beat 1 — the grid is real (2 min)

Open **http://localhost:5001/dashboard**.

Point at: 4 nodes online, mixed GPU classes, live utilization, 331 HAI
distributed. Let the 10-second refresh tick once while you talk so they see it
move.

> You will see an orange **"Sync partial (/logs unavailable)"** line. That is
> pre-existing: `/logs` requires an admin token and `dashboard.js` never sends
> one, so it 401s in production too. Ignore it, or get ahead of it with "the
> event log needs an admin token, the telemetry doesn't."

### Beat 2 — weighted routing (3 min)

Show `server/manifests/registry.json`, then the routing call in
`server/app.py`:

```python
return random.choices(names, weights=weights, k=1)[0]
```

The point worth making: `model="auto"` is not "pick the best model." It is a
weighted lottery, and the weights are the product's opinion about which models
deserve traffic. 25 models, 182 total IMAGE_GEN weight.

### Beat 3 — rewards follow compute (3 min)

Hit the ledger:

```bash
curl -s localhost:5001/jobs/recent?limit=5 | jq '.jobs[] | {model, weight, reward}'
```

Then the formula from `server/rewards.py`:

```
reward = base × (weight/10) × pipeline_factor × runtime_factor × success_factor
```

Real spread from this seed:

| Job | Runtime | $HAI |
|---|---|---|
| SD1.5 image | 3.0s | 0.050000 |
| SDXL image | 9.0s | 0.168750 |
| LTX video | 120.0s | 5.625000 |

**112× between a fast image and a long video.** Have this ready — see §5 Q3,
because someone will do that division out loud.

### Beat 4 — the game is a client of the network (3 min)

Open **http://localhost:5173**, run a round, land on `/shmup-results`, then
`/leaderboard`.

The framing that makes this more than a side project: the game does not have its
own economy. It calls `/astra/run`, `/astra/reward`, `/astra/leaderboard` on the
same coordinator that serves image jobs. Score maps to credits through
`_interpolate_reward` — a 5,000-point run pays 2.0, a 100,000-point run caps at
15.0 — and those credits spend on generation. It is one balance sheet.

### Beat 5 — close on the registry (1 min)

Land where the room lives: 25 models including checkpoints no mainstream
platform will host, with the CSAM filter in §5 Q1 sitting in front of all of
them. That combination is the pitch.

---

## 4. If something dies on stage

| Symptom | Cause | Fix |
|---|---|---|
| Node count falls to 0 | Terminal 2 stopped | Restart it; recovers in one beat |
| Every endpoint 500s | Database deleted under a live server | Restart terminal 1 |
| Game HavnAI calls 404 | Coordinator not running, or wrong port | Check terminal 1 on 5001 |
| Leaderboard empty | Server on `ledger.db`, not `demo.db` | Re-check `HAVNAI_DB_PATH` |
| `ModuleNotFoundError: safety` | Launched from the repo root | `cd server` first |

**Hard fallback:** screenshots of the dashboard, leaderboard, and game title
screen. Have them in a folder on the desktop. If the laptop refuses to
cooperate, talk over the stills — the architecture story does not depend on
anything being live.

---

## 5. Technical FAQ

Answers for the questions this room will actually ask, with the honest version
where the honest version is a stronger answer than a dodge.

### Q1. "You host adult models. What about CSAM?"

**Lead with this. Do not wait to be asked.**

`server/safety.py` runs on every prompt before a job is enqueued, in two layers:

1. An unconditional block on explicit minor terms — `child`, `toddler`,
   `underage`, `loli`, `shota`, `barely legal`, school references. Context does
   not matter; the job never reaches a node.
2. A contextual block on the *intersection* of sexual language and ambiguous
   youth language (`girl`, `boy`, `young`, `youthful`). Either alone passes.
   Together they return `ambiguous_age_in_sexual_context`.

The design point worth stating out loud: the filter does not block adult
content, because adult content is the product. It blocks the intersection that
matters legally. That is a deliberate line, and drawing it deliberately is the
differentiator against platforms that either ban everything or filter nothing.

Enforced at `server/app.py:2917` and `:3612`, plus the task executor path.

### Q2. "What stops a node returning garbage and claiming the reward?"

Be straight here. `settlement.validate_output` checks artifact **integrity**:
the file exists, is non-trivial in size, and is a recognizable image format.
Failures are marked `QUALITY_MALFORMED` and do not settle.

What it does **not** do is prove the node ran the model it claimed. There is no
proof-of-inference and no re-execution sampling today. If someone pushes, say
so — the roadmap answer is redundant execution on a sampled percentage of jobs
with output comparison. Claiming cryptographic verification you do not have is
how you lose the room.

### Q3. "Runtime multiplies the reward. Can't operators just run slow?"

**Yes, currently.** This is the sharpest question available and you should own
it before someone else frames it as a gotcha.

`runtime_factor = max(1.0, runtime_sec / baseline_runtime)` is unbounded. Stall
an LTX job to 600 seconds and it pays **28.125 HAI — 562× a fast SD1.5 image.**

It is visible in this very demo: node `lyra` runs video and took **253 of 331
HAI, 76% of all rewards**, from 133 of 324 jobs.

The fix is a cap — `min(runtime_factor, ceiling)` — plus per-model expected
runtime bands, so unusually slow work stops earning more and starts looking like
what it is. Framing to use: the current formula pays for *elapsed time*, and it
should pay for *work done*. That is a known gap with a known fix, not a
surprise.

### Q4. "How is this different from renting GPUs?"

Rented GPUs are stateless. HavnAI routes, prices, and settles: weighted routing
decides *which* model runs, the reward engine prices the job against model
weight and pipeline cost, and settlement tracks per-wallet accounting. An
operator plugs in a consumer card and receives work matched to what that card
can hold — the manifest carries VRAM-aware constraints, which is why a 12GB 3060
gets AnimateDiff at 512×512 and 24 frames while the A5000 takes LTX.

### Q5. "Is $HAI live? Can I withdraw?"

No, and your own dashboard says so — *"Weights and rewards are simulated in
Alpha and may change. No payouts are active yet."* Rewards accrue in SQLite
against wallet addresses. Pluggable reward backends are Q4 on the roadmap. Do
not let enthusiasm in the room upgrade this to a claim you have not shipped.

### Q6. "Why weighted routing instead of always picking the best model?"

Two reasons. Load distribution — concentrating traffic on one checkpoint wastes
the rest of the grid and the VRAM diversity it represents. And weights are the
economic lever: they set both routing probability and reward multiplier, so
raising a model's weight simultaneously sends it more work and pays operators
more for it.

The honest caveat: weights are hand-set today. The benchmark pipeline that would
derive them from rubric scores is documented in the README as roadmap and is not
implemented. `docs/economics/weights_manifest.json` exists for that future and
is dormant.

### Q7. "How big is this?"

7,705 lines in `server/app.py`, 113 registered routes, 25 models, a bundled node
client, a desktop operator app, and a game client on 95 merged PRs. Private
alpha — the GPU grid runs internally and public node join is not open.

---

## 6. Claims to avoid

Discipline here is what separates a credible alpha from a pitch deck. Each of
these is false as written:

- ❌ "Fully decentralized" — the coordinator is a single Flask service.
- ❌ "Verified compute" / "proof of inference" — see Q2.
- ❌ "Live token economy" — simulated, no payouts.
- ❌ "Benchmark-driven weights" — roadmap, not code.
- ❌ "Fully autonomous quality scoring" — format validation only.

What you *can* say without qualification: it routes, it prices, it settles, it
survives node churn, it filters prompts before they reach a GPU, and a shipping
game runs on it as a client.

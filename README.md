# HavnAI Core

![Stage 6](https://img.shields.io/badge/Stage-6-blue)

Coordinator for the HavnAI network: weighted model routing, dynamic $HAI rewards, and GPU node orchestration.

---

## Overview

HavnAI Core (`havnai-core`) is the coordinator service that:

- Accepts image/video generation jobs from clients via HTTP (`/submit-job`).
- Selects a model using **weighted routing** driven by a manifest registry.
- Distributes jobs to GPU nodes that connect over the public node API.
- Computes **dynamic $HAI rewards** per job based on model weight and runtime.
- Exposes network stats (nodes, jobs, rewards) for dashboards and landing pages.

Core concepts:

- **Weighted model routing** – `model="auto"` uses `random.choices` with manifest weights.
- **Dynamic rewards** – higher-weight models earn more $HAI per unit of compute.
- **Node heartbeats** – nodes periodically call `/register` with GPU + uptime info.
- **Model registry** – `server/manifests/registry.json` defines the active model set.

> Quote from existing history: “Integrate /api/models/stats as live endpoint.”

---

## Architecture

```mermaid
flowchart LR
  Client[Client / dApp / Wallet] --> API[HavnAI Core API (Flask)]
  API --> Intake[Job Intake /submit-job]
  Intake --> Router[Weighted Model Router]
  Router --> Manifest[(Model Manifest registry.json)]
  Router --> Queue[(Jobs SQLite)]

  Nodes[GPU Nodes (havnai-node)] -->|Heartbeat + capabilities| Reg[/register/]
  Reg --> NodesState[(nodes.json + node_wallets)]
  Nodes -->|Poll tasks| Tasks[/tasks/creator/]
  Tasks --> Queue
  Nodes -->|Submit results + metrics| Results[/results/]
  Results --> Rewards[(rewards SQLite)]

  API --> Stats[/api/models/stats/]
  Stats --> Dashboard[Landing page / dashboard]
```

Key components:

- **API Layer (`server/app.py`)**
  - `POST /submit-job` – enqueue public jobs (wallet + model + prompt).
  - `GET /jobs/*` / `GET /rewards` – job and reward inspection.
  - `GET /api/models/stats` – lightweight network stats for the landing page.
- **Model Manifest**
  - JSON manifest at `server/manifests/registry.json`.
  - Enriched into in-memory `MANIFEST_MODELS` and `MODEL_WEIGHTS`.
- **Node Registry**
  - `POST /register` – node heartbeats (GPU, uptime, pipelines, models).
  - `POST /link-wallet` – bind node IDs to wallet addresses.
  - `nodes.json` persisted on disk for the coordinator process.
- **Job + Reward Engine**
  - Jobs stored in SQLite `jobs` table.
  - Rewards stored in `rewards` table with per-wallet accounting.

---

## Setup (Docker)

This repo is a standard Python/Flask application with a bundled node client. A minimal Docker setup might look like:

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/havnai-core.git
   cd havnai-core
   ```

2. **Environment configuration**

   Create `.env` with coordinator settings (example):

   ```bash
   cat > .env << 'EOF'
   PORT=8080
   HAVNAI_MANIFEST_DIR=/app/server/manifests
   HAVNAI_MANIFEST_FILE=/app/server/manifests/registry.json
   HAVNAI_OUTPUTS_DIR=/app/static/outputs
   REWARD_BASE_HAI=0.05
   REWARD_BASELINE_RUNTIME=8.0
   REWARD_SDXL_FACTOR=1.5
   REWARD_SD15_FACTOR=1.0
   REWARD_ANIME_FACTOR=0.7
   SERVER_JOIN_TOKEN=changeme
   EOF
   ```

3. **Example `Dockerfile`**

   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app
   COPY server/ server/
   COPY static/ static/
   COPY shared/ shared/
   COPY VERSION .

   RUN pip install --no-cache-dir flask flask-cors

   ENV PORT=8080
   EXPOSE 8080

   CMD ["python", "-m", "server.app"]
   ```

4. **Example `docker-compose.yml`**

   ```yaml
   version: "3.9"

   services:
     havnai-core:
       build: .
       ports:
         - "8080:8080"
       env_file:
         - .env
       volumes:
         - ./server/manifests:/app/server/manifests
         - ./static/outputs:/app/static/outputs
   ```

5. **Run**

   ```bash
   docker compose up --build
   ```

6. **Health checks**

   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/api/models/stats
   ```

---

## API Endpoints

Only the most relevant coordinator endpoints are covered here. See `server/app.py` for the full list.

### `POST /submit-job`

Public job intake used by wallets / dApps to submit creator jobs.

- **Request body (current schema)**

  ```json
  {
    "wallet": "0x0123456789abcdef0123456789abcdef01234567",
    "model": "auto",
    "prompt": "Describe HavnAI in one paragraph.",
    "negative_prompt": "blurry, low quality"
  }
  ```

  - `wallet` – required; must be a 42-char EVM address, validated via regex.
  - `model` – optional; set `"auto"` to enable weighted routing across IMAGE_GEN models.
  - `prompt` – text prompt for image/video generation.
  - `negative_prompt` – optional negative prompt, stored in the job payload.

- **Response (current implementation)**

  ```json
  {
    "status": "queued",
    "job_id": "job-01ab23cd45ef"
  }
  ```

Internally this:

1. Validates the wallet format.
2. Resolves the model config (or picks one via weighted routing when `model="auto"`).
3. Serializes the job payload (with special handling for AnimateDiff).
4. Enqueues a row into the `jobs` table with status `queued`.

### `GET /jobs/recent`

Recent creator jobs with reward info and optional output URLs.

- **Response (shape)**

  ```json
  {
    "jobs": [
      {
        "job_id": "job-...",
        "wallet": "0x...",
        "model": "epicrealismXL_vxviiCrystalclear",
        "task_type": "IMAGE_GEN",
        "status": "SUCCESS",
        "weight": 25.0,
        "reward": 0.123456,
        "submitted_at": "2025-01-05T12:00:00Z",
        "completed_at": "2025-01-05T12:00:08Z",
        "image_url": "/static/outputs/job-....png",
        "output_path": "/app/static/outputs/job-....png"
      }
    ],
    "summary": {
      "queued_jobs": 0,
      "active_jobs": 0,
      "total_distributed": 1.234567,
      "jobs_completed_today": 12
    }
  }
  ```

### `GET /api/models/stats`

Lightweight stats endpoint for public dashboards.

- **Response**

  ```json
  {
    "active_nodes": 3,
    "jobs_completed_24h": 1842,
    "success_rate": 99.2,
    "top_model": "juggernautxl_ragnarokby"
  }
  ```

- `active_nodes` – nodes with a heartbeat (`/register`) within the last `ONLINE_THRESHOLD` seconds.
- `jobs_completed_24h` – count of completed creator jobs over the last 24 hours.
- `success_rate` – percentage of finished jobs with status `completed` or `success`.
- `top_model` – model with the most successful completions (based on in-memory `MODEL_STATS`).

There is also a legacy alias `GET /models/stats` that returns the same payload.

---

## Node Heartbeats

Nodes run the bundled client (`client/client.py`) and connect to the coordinator using a simple heartbeat/polling protocol.

### `POST /register` (heartbeat)

Sent periodically by each node with GPU stats and capabilities.

- **Payload (current implementation)**

  ```json
  {
    "node_id": "node-01",
    "os": "Linux",
    "gpu": "NVIDIA RTX 4090",
    "gpu_stats": {
      "gpu_name": "NVIDIA RTX 4090",
      "memory_total_mb": 24564,
      "memory_used_mb": 4096,
      "utilization": 72
    },
    "start_time": 1736070000.0,
    "uptime": 86400.0,
    "role": "creator",
    "version": "stage7",
    "node_name": "my-havnai-node",
    "models": ["juggernautXL_ragnarokBy", "epicrealismXL_vxviiCrystalclear"],
    "pipelines": ["sdxl", "sd15"]
  }
  ```

- The coordinator:

  - Creates or updates an entry in the in-memory `NODES` dict and persists `nodes.json`.
  - Tracks `utilization` and `avg_utilization` from GPU stats over time.
  - Updates `last_seen` and `last_seen_unix`, used by `/api/models/stats` to compute `active_nodes`.

### `GET /tasks/creator`

Nodes poll for work:

- Query param: `node_id=<NODE_NAME>`.
- Response contains a list of tasks the node should execute, filtered by:
  - Node role (`creator`).
  - Supported models and pipelines.

### `POST /results`

Nodes submit job results and metrics:

- Includes:
  - `metrics.inference_time_ms` (used by the reward engine).
  - Output image/video URLs (where applicable).
  - Job status (`success` or `failed`).

The coordinator computes rewards and updates job state accordingly.

---

## Model Registry

The model registry is driven by a JSON manifest at `server/manifests/registry.json`.

### Example manifest entry

```json
{
  "models": [
    {
      "name": "juggernautXL_ragnarokBy",
      "pipeline": "sdxl",
      "path": "/mnt/d/havnai-storage/models/creator/juggernautXL_ragnarokBy.safetensors",
      "type": "checkpoint",
      "tags": ["realism", "flash", "studio_light", "smooth_skin"],
      "task_type": "IMAGE_GEN",
      "weight": 20,
      "reward_weight": 20,
      "strengths": "Elite realism and very strong flash/daylight skin performance.",
      "weaknesses": "Backgrounds and hands can occasionally look too stiff or polished."
    }
  ]
}
```

On startup, `load_manifest()`:

- Normalizes model names to lowercase keys (`juggernautxl_ragnarokby`).
- Populates `MANIFEST_MODELS` with metadata.
- Populates `MODEL_WEIGHTS` with the `weight` field (fallback default `10.0`).
- Initializes `MODEL_STATS` entries for each model.

---

## Weighted Routing Code Snippet

When the user passes `"model": "auto"` (or leaves it blank), the coordinator uses weighted routing based on manifest weights:

```python
def choose_model_for_auto() -> str:
    load_manifest()
    candidates = [
        meta
        for meta in MANIFEST_MODELS.values()
        if (meta.get("task_type") or CREATOR_TASK_TYPE).upper() == CREATOR_TASK_TYPE
    ]
    if not candidates:
        raise RuntimeError("no_creator_models")

    names = [meta["name"] for meta in candidates]
    weights = [resolve_weight(meta["name"].lower(), meta.get("reward_weight", 10.0)) for meta in candidates]
    return random.choices(names, weights=weights, k=1)[0]
```

This mirrors the logic in `/submit-job`:

- **Source**: `server/app.py` inside `submit_job`.
- **Weights**: `resolve_weight` returns the weight from `MODEL_WEIGHTS` with a default value.

---

## Reward Engine Formula

Rewards are computed by `compute_reward()` in `server/app.py`. The effective formula is:

```text
reward_hai =
  base_reward *
  weight_factor *
  compute_cost_factor *
  runtime_factor *
  success_factor
```

Where:

- `base_reward` – from `REWARD_CONFIG["base_reward"]` (env var `REWARD_BASE_HAI`, default `0.05`).
- `model_weight` – derived from `MODEL_WEIGHTS[model_name]` (manifest weight).
- `weight_factor` – `model_weight / 10.0`.
- `compute_cost_factor` – based on pipeline:
  - `"sdxl"` → `REWARD_SDXL_FACTOR` (default `1.5`).
  - `"sd15"` → `REWARD_SD15_FACTOR` (default `1.0`).
  - `"anime"` / `"cartoon"` → `REWARD_ANIME_FACTOR` (default `0.7`).
  - Others → `1.0`.
- `runtime_factor` – `max(1.0, runtime_sec / baseline_runtime)`, with `baseline_runtime` from `REWARD_BASELINE_RUNTIME`.
- `success_factor` – `1.0` when job status is `success`, otherwise `0.0`.

The result is rounded to 6 decimal places and stored along with a `reward_factors` object for later inspection via job detail APIs.

---

## Benchmark Workflow (Planned)

As of this version, **benchmarking and tier assignment are not implemented in code**. Weights are static values defined in the manifest. The intended workflow (roadmap) is:

1. **Define benchmark prompts**
   - Curated sets per domain (e.g., realism, motion, hands, text, etc.).
2. **Run benchmarks**
   - Execute identical prompt suites across all candidate models.
3. **Score with a rubric**
   - Use human evaluation or LLM-as-judge with a consistent scoring rubric.
4. **Map average scores → tiers**
   - Tier models as `S`, `A`, `B`, or `E` based on average rubric scores.
5. **Update manifest weights**
   - Translate tiers into target weights / reward multipliers and update `registry.json`.
6. **Deploy and monitor**
   - Use `/api/models/stats`, `/jobs/recent`, and node telemetry to validate that routing and rewards match expectations.

### Example Rubric Table (Conceptual)

| Criterion        | Description                        | Score (1–5) |
|-----------------|------------------------------------|------------|
| Correctness     | Factual accuracy / task completion | 1–5        |
| Coherence       | Logical structure and clarity      | 1–5        |
| Safety          | Harmful / disallowed content       | 1–5        |
| Visual quality  | Sharpness, composition, artifacts  | 1–5        |

### Example Tier Mapping (Conceptual)

| Average Score | Tier | Suggested Weight | Notes               |
|---------------|------|------------------|---------------------|
| 4.5–5.0       | S    | 20–25            | Best-in-class       |
| 4.0–4.49      | A    | 10–15            | Strong default      |
| 3.0–3.99      | B    | 5–8              | Acceptable baseline |
| < 3.0         | E    | 1–3              | Experimental        |

Implementing this fully will require external tooling or a new internal benchmarking service, plus automation to rewrite `registry.json`.

---

## Contributing

Contributions are welcome. To work on HavnAI Core:

1. **Fork & branch**

   ```bash
   git checkout -b feature/your-feature
   ```

2. **Install dependencies**

   From the repo root:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r server/requirements.txt  # if present
   ```

3. **Run the server**

   ```bash
   export FLASK_APP=server.app
   flask run --port 8080
   ```

4. **Testing & style**

   - Add unit tests for new behavior (if a test harness exists in this repo).
   - Keep code consistent with existing style in `server/app.py` and `client/client.py`.

5. **Open a PR**

   - Describe the change and affected endpoints.
   - Include before/after behavior for APIs.

---

## Roadmap (2025)

**Q1 2025**

- Harden `/submit-job` public intake with rate limits (already implemented) and richer validation.
- Finalize model manifest for primary creator workloads.
- Keep `/api/models/stats` wired as a live endpoint for landing pages.

**Q2 2025**

- Introduce richer `/api/models/stats` payloads (per-model performance summaries).
- Add node reliability scores and integrate them into job dispatch.
- Improve node registration/heartbeat with latency and queue-depth hints.

**Q3 2025**

- Build a benchmark pipeline (scheduled runs + rubric scoring).
- Use benchmark results to update manifest weights programmatically.
- Add admin tooling to inspect model tiers, node health, and routing decisions.

**Q4 2025**

- Pluggable reward backends (on-chain, custodial, or custom).
- Multi-tenant routing policies and per-tenant manifests.
- Security/audit work for mainnet-scale deployment.

---

HavnAI Core is evolving quickly; treat this README as the canonical reference for the current coordinator behavior. For anything unclear, consult `server/app.py` and `client/client.py` or open an issue. 

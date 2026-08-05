# Run a HavnAI node

A node is a machine that takes generation jobs from the coordinator and returns
results. This document is the complete path from a bare machine to a node
serving image, face swap and video work — including what the installer does,
how to confirm it worked, and what to do when it didn't.

- [What you need](#what-you-need)
- [Install](#install)
- [Verify](#verify)
- [Model weights](#model-weights)
- [Run the node](#run-the-node)
- [What the installer actually does](#what-the-installer-actually-does)
- [Capability reference](#capability-reference)
- [Troubleshooting](#troubleshooting)
- [Upgrading and uninstalling](#upgrading-and-uninstalling)
- [For coordinator operators](#for-coordinator-operators)

---

## What you need

| | Minimum | Recommended |
| --- | --- | --- |
| OS | 64-bit Linux (Ubuntu 22.04+, Debian 12+, RHEL/Rocky 9) or macOS 12+ | Ubuntu 22.04 LTS |
| Python | 3.10 | 3.11 |
| GPU | none (worker only) | NVIDIA, 12 GB+ VRAM, driver 535+ with CUDA 12.x |
| Disk | 25 GB free | 150 GB+ if you serve many checkpoints |
| Network | outbound HTTPS | 100 Mbit+ down for the initial weight pull |

Windows is not supported directly. Use WSL2 with an Ubuntu distribution and
follow the Linux path.

**On GPUs and roles.** A machine with no usable NVIDIA GPU joins as a *worker*
and is never assigned creator jobs. A GPU machine joins as a *creator* and can
serve image, face swap and video work. The installer detects this and you can
override it with `--creator` / `--no-creator`.

**On VRAM.** Image generation runs comfortably at 8 GB. Face swap and video
want 12 GB or more; below that the node will take those jobs and then fail them
on out-of-memory, which helps nobody. The preflight check warns you when VRAM is
marginal.

---

## Install

### Option A — desktop app

If you would rather not use a terminal, the desktop app installs and manages the
node for you: preflight checks, config, model downloads with progress, and
start/stop. See [`desktop/README.md`](../desktop/README.md) for builds and
prerequisites.

### Option B — one command

```bash
curl -fsSL https://api.joinhavn.io/installers/install-node.sh | bash -s -- \
    --server https://api.joinhavn.io \
    --token  YOUR_JOIN_TOKEN \
    --wallet 0xYourWalletAddress
```

The installer prints eight numbered steps and finishes by running the preflight
check. It exits non-zero if anything blocking remains, so it is safe to use in a
provisioning script.

**Flags**

| Flag | Meaning |
| --- | --- |
| `--server URL` | Coordinator base URL. Default `https://api.joinhavn.io`. |
| `--token TOKEN` | Join token from the grid operator. Required if the coordinator enforces one. |
| `--wallet 0x…` | Address credited for completed work. Set it, or you work for free. |
| `--name NAME` | Node name shown on the dashboard. Defaults to the hostname. |
| `--home DIR` | Install location. Default `~/.havnai`. |
| `--python BIN` | Interpreter used to build the venv. |
| `--creator` / `--no-creator` | Force the role instead of auto-detecting. |
| `--start` | Enable and start the service when the install finishes. |
| `--skip-models` | Install the runtime but download no weights. |
| `--skip-deps` | Reuse the existing venv without reinstalling packages. |

Re-running the installer is safe. It stages the new runtime beside the live one
and only swaps it in after verifying it, keeping the previous version in
`~/.havnai/previous` for rollback.

### Reviewing the script first

Piping a script into `bash` means trusting the host. To read it first:

```bash
curl -fsSL https://api.joinhavn.io/installers/install-node.sh -o install-node.sh
less install-node.sh
bash install-node.sh --server https://api.joinhavn.io --token YOUR_TOKEN
```

---

## Verify

The single most useful command on a node:

```bash
~/.havnai/bin/havnai-doctor
```

It reports what the machine can actually serve, grouped by capability, and gives
a remedy for every failure:

```
Image generation — READY
----------------------------------------------------------
[  OK  ] Image dependencies: 4 packages present
[  OK  ] CUDA available to PyTorch: NVIDIA GeForce RTX 4090 (23.6 GB VRAM)
[  OK  ] Model weights: 6/22 manifest models in ~/.havnai/models/creator

Face swap — NOT READY
----------------------------------------------------------
[ FAIL ] Face swap dependencies: missing: insightface
         → Reinstall dependencies: ~/.havnai/venv/bin/pip install -r ~/.havnai/requirements-node.txt
```

A capability is only ready when nothing gating it — and nothing in the core —
has failed. Warnings never block. `havnai-doctor --json` emits the same report
as machine-readable JSON; `--offline` skips the checks that contact the
coordinator.

**Run this before you start the node.** A node that registers while a capability
is broken will accept those jobs and fail them.

---

## Model weights

Weights are separate from code and are the bulk of the download. Each model in
the coordinator manifest declares where it comes from:

| Source | Meaning | Who pays the bandwidth |
| --- | --- | --- |
| `hf` | Public weights on the Hugging Face CDN | Hugging Face |
| `coordinator` | Weights the grid hosts, gated by your join token | The grid |
| `operator` | Restricted licence — you supply the file | You |

Fetch everything the manifest wants:

```bash
~/.havnai/bin/havnai-fetch-models --face-assets
```

Useful variants:

```bash
havnai-fetch-models --dry-run                 # show the plan, download nothing
havnai-fetch-models --only juggernautXL_ragnarokBy
havnai-fetch-models --pipelines sdxl          # skip video weights
havnai-fetch-models --json                    # JSON lines, for tooling
```

Downloads resume where they left off and are checksummed before being moved into
place, so an interrupted transfer never leaves a truncated checkpoint that would
load as corrupt weights.

Models reported as `manual` carry a restricted licence. The tool names the exact
filename and directory; obtain the file yourself and drop it in:

```
~/.havnai/models/creator/<filename>.safetensors
```

### Face swap assets

Face swap additionally needs the InstantID adapter and the antelopev2 face
analysis pack. `--face-assets` fetches both. The adapter also downloads
automatically on the first face swap job; antelopev2 does not reliably, which is
why the flag exists.

---

## Run the node

**Linux (systemd)**

```bash
systemctl --user start havnai-node      # start
systemctl --user enable havnai-node     # start at boot
systemctl --user status havnai-node     # state
journalctl --user -u havnai-node -f     # live logs
systemctl --user stop havnai-node       # stop
```

If the node should keep running after you log out:

```bash
sudo loginctl enable-linger "$USER"
```

**macOS (launchd)**

```bash
launchctl load -w ~/Library/LaunchAgents/com.havnai.node.plist    # start
launchctl unload -w ~/Library/LaunchAgents/com.havnai.node.plist  # stop
tail -f ~/.havnai/logs/launchd.log                                # logs
```

**Directly, any OS** — useful when debugging, since errors go to your terminal:

```bash
~/.havnai/bin/havnai-node
```

The node's own log is always at `~/.havnai/logs/node.log`.

---

## What the installer actually does

No hidden steps. In order:

1. **Checks prerequisites.** Finds a Python 3.10+ interpreter (installing one via
   `apt`/`dnf`/`brew` if absent), verifies `curl` and `tar`, and detects whether
   a GPU is present to pick the role.
2. **Downloads the runtime bundle** from `/client/bundle.tar.gz` into a staging
   directory, then verifies it carries every capability's modules before
   trusting it.
3. **Installs the Python environment** — a virtualenv at `~/.havnai/venv` with
   the dependencies from the bundle. This is the slow step; PyTorch alone is
   several gigabytes.
4. **Activates the runtime** by moving the staged tree to `~/.havnai/current`,
   keeping the previous one at `~/.havnai/previous`.
5. **Writes configuration** to `~/.havnai/.env` (mode `0600`, since it holds your
   join token) and installs the `havnai-node`, `havnai-doctor` and
   `havnai-fetch-models` commands into `~/.havnai/bin`.
6. **Fetches model weights**, unless `--skip-models` or the node is worker-only.
7. **Installs the service** — a systemd user unit or a launchd agent. Not started
   unless you pass `--start`.
8. **Verifies** by running the preflight check, and exits non-zero if anything
   blocking remains.

### Layout on disk

```
~/.havnai/
  .env                  configuration (0600 — contains your join token)
  VERSION               installed client version
  bin/                  havnai-node, havnai-doctor, havnai-fetch-models
  current/              the runtime: client/, engines/, shared/, common/
  previous/             prior runtime, kept for rollback
  venv/                 Python environment
  models/creator/       checkpoints
  loras/                LoRA weights
  instantid/            InstantID adapter + antelopev2
  outputs/              generated artifacts
  logs/node.log         node log
```

### Configuration keys

| Key | Purpose |
| --- | --- |
| `SERVER_URL` | Coordinator base URL |
| `JOIN_TOKEN` | Authenticates the node and gates coordinator-hosted weights |
| `WALLET` | Address credited for completed work |
| `NODE_NAME` | Name shown on the dashboard |
| `CREATOR_MODE` | `true` to serve creator jobs |
| `HAVNAI_MODEL_DIR` | Override the checkpoint directory (e.g. a second drive) |
| `HAVNAI_OUTPUTS_DIR` | Override where artifacts are written |
| `HAVNAI_LTX_VIDEO_PYTHON` | Interpreter for an isolated video environment |

Edit `~/.havnai/.env` and restart the node for changes to apply.

---

## Capability reference

What each capability needs, and what breaks without it.

### Image generation (SDXL / SD1.5)

- Packages: `torch`, `diffusers`, `transformers`, `accelerate`
- Weights: at least one checkpoint in `~/.havnai/models/creator`
- GPU: 8 GB VRAM is workable

### Face swap (InstantID)

- Packages: everything above, plus `insightface`, `opencv-python-headless`, `onnxruntime`
- Runtime modules: `pipeline_stable_diffusion_xl_instantid.py`,
  `pipeline_stable_diffusion_xl_instantid_inpaint.py`, and the `ip_adapter` package
- Weights: an SDXL checkpoint, the InstantID adapter, and antelopev2
- GPU: 12 GB VRAM recommended

### Video generation (LTX / LTX2 / AnimateDiff / WanGP)

- Runtime modules: the `engines` package
- Weights: depend on the pipeline; LTX-Video pulls from Hugging Face, WanGP is
  operator-supplied
- GPU: 12 GB VRAM minimum, 24 GB for longer clips
- Optional: set `HAVNAI_LTX_VIDEO_PYTHON` to run video in an isolated
  environment when its dependencies conflict with the main venv

The runtime modules matter. If they are missing, the node still starts and still
registers — it simply fails every job of that type on import. `havnai-doctor`
checks for them explicitly.

---

## Troubleshooting

### "Face swap modules: missing 6 file(s)" or "Video engine modules: missing…"

Your runtime is incomplete — this is what a pre-bundle install looks like.
Re-run the installer; it will fetch the full bundle.

### "Core dependencies: missing: …"

```bash
~/.havnai/venv/bin/pip install -r ~/.havnai/current/client/requirements-node.txt
```

If that fails to build a package, your Python is probably too new for one of the
pinned wheels. Install with an explicit interpreter:
`--python python3.11`.

### "CUDA available to PyTorch: torch.cuda.is_available() is False"

You have a CPU build of PyTorch. Replace it with a CUDA build matching your
driver:

```bash
~/.havnai/venv/bin/pip install --force-reinstall torch \
    --index-url https://download.pytorch.org/whl/cu124
```

Confirm the driver itself works first with `nvidia-smi`.

### "no manifest models found"

Weights were never downloaded, or they went to a different directory. Check the
plan and the resolved path:

```bash
havnai-fetch-models --dry-run
```

If you set `HAVNAI_MODEL_DIR`, make sure it matches where the files actually are.

### "coordinator rejected the join token"

`JOIN_TOKEN` in `~/.havnai/.env` is wrong or expired. Get a current one from the
grid operator. This error is not retried, because retrying a rejected token only
delays the real message.

### "Coordinator reachable: … unreachable"

Outbound HTTPS is blocked, or `SERVER_URL` is wrong. Test directly:

```bash
curl -v https://api.joinhavn.io/models/list
```

Note the API host may differ from the website host.

### Node starts, then exits immediately

```bash
journalctl --user -u havnai-node -n 50 --no-pager
```

Then run it in the foreground to see the error directly:

```bash
~/.havnai/bin/havnai-node
```

### Out-of-memory during video or face swap jobs

The GPU is too small for that workload. Either restrict the node to image work,
or lower concurrency. Serving jobs you cannot finish costs you reputation and
the requester their time.

### Disk fills up

Checkpoints are 2–7 GB each. Move them to a larger drive and point the node at
it:

```bash
echo 'HAVNAI_MODEL_DIR=/mnt/big/havnai/models' >> ~/.havnai/.env
systemctl --user restart havnai-node
```

---

## Upgrading and uninstalling

**Upgrade** — re-run the installer. The previous runtime is preserved:

```bash
curl -fsSL https://api.joinhavn.io/installers/install-node.sh | bash -s -- \
    --server https://api.joinhavn.io --skip-models
systemctl --user restart havnai-node
```

**Roll back** to the previous runtime:

```bash
systemctl --user stop havnai-node
rm -rf ~/.havnai/current && mv ~/.havnai/previous ~/.havnai/current
systemctl --user start havnai-node
```

**Uninstall**:

```bash
systemctl --user disable --now havnai-node
rm -f ~/.config/systemd/user/havnai-node.service
systemctl --user daemon-reload
rm -rf ~/.havnai          # also deletes downloaded weights
```

---

## For coordinator operators

### Adding a model to the manifest

Each entry in `server/manifests/registry.json` carries a `source` block telling
nodes how to obtain the weights.

Public weights on Hugging Face — preferred, since the CDN carries the bandwidth:

```json
"source": {
  "kind": "hf",
  "repo_id": "Lightricks/LTX-Video",
  "filename": "ltx-video-2b-v0.9.5.safetensors",
  "revision": "main",
  "sha256": "",
  "license": "see upstream repo"
}
```

Weights you host yourself, served through the token-gated endpoint:

```json
"source": {
  "kind": "coordinator",
  "filename": "juggernautXL_ragnarokBy.safetensors",
  "sha256": "…",
  "size_bytes": 6938040714
}
```

Weights you may not redistribute:

```json
"source": {
  "kind": "operator",
  "filename": "restricted-model.safetensors",
  "license": "research / owner only",
  "notes": "Obtain from the upstream project and place in your creator models directory."
}
```

Populate `sha256` wherever you can. Without it a download is accepted on
presence alone; with it, a corrupted transfer is caught and retried instead of
silently producing bad output.

An entry with no `source` block is inferred: models with a storage `path` are
treated as `coordinator`, everything else as `operator`. The manifest's internal
paths are never exposed to nodes.

### Serving self-hosted weights

`/models/download/<name>` streams artifacts to authenticated nodes with HTTP
Range support. Point it at your storage:

```bash
HAVNAI_MODEL_STORAGE_DIR=/mnt/storage/models/creator
```

The endpoint refuses to serve anything resolving outside that root, so a
mistaken manifest path fails loudly rather than exposing the filesystem.

### The runtime bundle

`/client/bundle.tar.gz` serves everything a node imports at runtime, built from
the repo and cached until the sources change. `/client/bundle/manifest` returns
per-file digests.

If you add a module the client imports, add it to `BUNDLE_FILES` or
`BUNDLE_DIRS` in `server/node_bundle.py` — and to the list in
`tests/test_node_bundle.py`, which fails when a capability's module stops being
packaged. That test is the guard against shipping partial installs again.

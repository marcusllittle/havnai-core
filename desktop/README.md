# HavnAI Node — desktop app

A control panel for operators who would rather not live in a terminal. It
installs the node, tells them what their machine can actually serve, downloads
model weights with visible progress, and starts and stops the node.

The app is deliberately thin. Every decision it displays is made by the Python
runtime — `client.doctor` decides what is healthy, `client.fetch_models` decides
what to download — so a desktop operator and an SSH operator can never get
different answers about the same machine.

## What it does

| Tab | Purpose |
| --- | --- |
| **Health** | Runs `client.doctor --json` and shows, per capability (image, face swap, video), whether the node can serve it and exactly what is blocking it. |
| **Setup** | Edits the node's `.env` and installs or repairs the runtime, streaming progress live. On Windows this is a native app-driven install under `%USERPROFILE%\.havnai`; on Linux/macOS it runs the coordinator shell installer. |
| **Models** | Shows the download plan and fetches missing weights with per-model progress. Distinguishes Hugging Face, coordinator-hosted and operator-supplied models. |
| **Activity** | Tails the node log. |

## Building

Requires the [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/)
for your platform, plus a Rust toolchain (1.77+).

On Debian/Ubuntu:

```bash
sudo apt-get install -y libwebkit2gtk-4.1-dev libsoup-3.0-dev libgtk-3-dev \
    libayatana-appindicator3-dev librsvg2-dev patchelf
```

Then, from `desktop/src-tauri`:

```bash
cargo build              # debug binary at target/debug/havnai-node-desktop
cargo test               # unit tests
cargo clippy --all-targets
```

To produce installers (`.deb`, `.AppImage`, `.dmg`, `.exe`):

```bash
cargo install tauri-cli --version '^2'
cargo tauri build
```

The frontend is plain HTML, CSS and JavaScript in `desktop/src`, loaded through
Tauri's `withGlobalTauri` bridge. There is no bundler and no npm install step —
`cargo build` is the whole build.

## Architecture

```
desktop/src/            frontend (no framework, no build step)
desktop/src-tauri/
  src/lib.rs            Tauri commands: install, doctor, models, service control
  src/main.rs           entry point
  tauri.conf.json       window, bundle and CSP configuration
```

Long-running work (installs, multi-gigabyte downloads) runs as a child process
whose stdout and stderr are streamed to the UI as `install-output` and
`models-output` events, so the window never freezes and the operator can see
progress as it happens.

## Windows operator flow

The Windows app is intended for non-technical operators:

1. Install Python 3.10+ from python.org and enable **Add python.exe to PATH**.
2. Open HavnAI Node.
3. Enter the coordinator URL, join token, wallet and node name.
4. Click **Install node**.
5. When preflight passes, click **Start node**.

The app creates `%USERPROFILE%\.havnai`, downloads the runtime bundle from
`/client/bundle.tar.gz`, creates the Python virtual environment, installs node
dependencies, writes `.env`, and creates `.cmd` launchers for the node, doctor
and model fetcher.

## Notes

- The app never invents state. If no node is installed, Health says so and
  points at the Setup tab rather than reporting a false diagnosis.
- The join token is written to `~/.havnai/.env` with `0600` permissions.
- Windows installs are native and do not require WSL2.

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

1. Install Python 3.12 from python.org (3.10 to 3.13 work) and enable
   **Add python.exe to PATH**.
2. Open HavnAI Node.
3. Enter the coordinator URL, join token, wallet and node name.
4. Click **Install node**.
5. When preflight passes, click **Start node**.

The app creates `%USERPROFILE%\.havnai`, downloads the runtime bundle from
`/client/bundle.tar.gz`, creates the Python virtual environment, installs node
dependencies, writes `.env`, and creates `.cmd` launchers for the node, doctor
and model fetcher.

Two things differ from the Linux dependency install, both because of what PyPI
publishes for Windows:

- **torch** comes from PyTorch's CUDA index when `nvidia-smi` finds a GPU. The
  PyPI Windows wheel is CPU-only, so a plain `pip install torch` would leave a
  GPU node unable to serve jobs.
- **triton**, **xformers** and **insightface** are left out of the required
  set. triton has no Windows wheels, xformers would replace the CUDA torch with
  its own pinned build, and insightface builds from source. insightface is
  attempted separately afterwards; if it fails (no C++ Build Tools), face swap
  is reported unavailable by the Health tab and everything else still works.

To exercise the real install end to end on a Windows machine (slow - it
downloads CUDA torch):

```bat
set HAVNAI_E2E_HOME=C:\path	o\empty\dir
cargo test windows_install_end_to_end -- --ignored --nocapture
```

## Notes

- The app never invents state. If no node is installed, Health says so and
  points at the Setup tab rather than reporting a false diagnosis.
- The join token is written to `~/.havnai/.env` with `0600` permissions.
- Windows installs are native and do not require WSL2.

## Releasing

Bump `version` in `src-tauri/tauri.conf.json` and `src-tauri/Cargo.toml`, merge,
then push a matching tag:

```bash
git tag desktop-v0.2.0 && git push havnai-core desktop-v0.2.0
```

`.github/workflows/desktop-release.yml` builds every platform and publishes the
installers to a GitHub release. The website's download panel picks the newest
`desktop-v*` release that has installers attached.

### macOS release gate

Public releases require a paid Apple Developer Program team with a valid
[Developer ID Application certificate](https://developer.apple.com/developer-id/)
and access to Apple's notary service. Use the repository's GitHub Actions secrets
UI; never commit or paste a certificate, private key or password in a ticket,
chat, command history or build log.

| GitHub Actions secret | Expected value |
| --- | --- |
| `APPLE_CERTIFICATE` | Base64-encoded, password-protected PKCS#12 export containing the Developer ID Application certificate and its private key |
| `APPLE_CERTIFICATE_PASSWORD` | Password used to export that PKCS#12 |
| `APPLE_SIGNING_IDENTITY` | Full matching Keychain identity, `Developer ID Application: … (TEAMID)` |
| `APPLE_ID` | Apple Account email allowed to submit for the team |
| `APPLE_PASSWORD` | Apple Account [app-specific password](https://support.apple.com/en-us/102654), not the account password |
| `APPLE_TEAM_ID` | The ten-character Apple Developer team ID for the certificate |

The Mac runners use Tauri CLI 2.12.5. It signs the app and DMG but only
offers automatic notarization for the inner app. The build step gets the
signing identity, notary credentials are passed only to the preflight and
DMG verification steps, and CI
[notarizes only the outer DMG](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution).
Check the operator's Apple team and expiry of the existing certificate
before requesting new credentials.

1. Run **Desktop app release** from Actions against the agreed ref with
   **notarize_macos** enabled. This uploads test artifacts only, even when
   a tag was selected. A run without that option may produce unsigned
   internal test artifacts; never share those as public installers.
2. Check both Mac jobs for verification of the Developer ID signature, the
   embedded app architecture, Apple `Accepted`, stapler validation, and
   Gatekeeper disk image assessment. If Apple declines, retrieve the notary
   log by the recorded submission ID with `xcrun notarytool log` on a
   trusted Mac. Never copy credential flags into logs.
3. On a clean Apple Silicon Mac and a clean Intel Mac (or matching clean
   VMs), download the matching artifact in a browser, open the DMG, drag
   HavnAI Node to Applications and open normally with Gatekeeper enabled.
   Do not strip quarantine or bypass Gatekeeper. Record the macOS version,
   architecture, filename and result. An Intel cross-build on an ARM runner
   does not count as an Intel launch test.
4. Once approved, use a **new** version and matching tag for the public
   release. Both notarized DMGs are required before assets and checksums
   are published. Repeat the normal download/open check against the published
   **Run a Node** links; validate the links select the matching architecture.

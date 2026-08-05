#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# HavnAI node installer
#
# Installs the complete node runtime: interpreter deps, every runtime module
# (image, face swap and video), model weights, and a supervised service.
#
# The install is staged. The new runtime is unpacked beside the live one and
# only swapped in once it verifies, so a failed or interrupted install leaves
# the previous working node untouched.
#
#   curl -fsSL https://api.joinhavn.io/installers/install-node.sh | bash -s -- \
#       --server https://api.joinhavn.io --token <TOKEN> --wallet 0x...
# ---------------------------------------------------------------------------

SERVER_URL="https://api.joinhavn.io"
JOIN_TOKEN=""
WALLET=""
NODE_NAME="$(hostname)"
CREATOR_MODE=""
HAVNAI_HOME="${HAVNAI_HOME:-$HOME/.havnai}"
AUTO_START="false"
SKIP_MODELS="false"
SKIP_DEPS="false"
PYTHON_BIN=""

print_usage() {
  cat <<'USAGE'
Usage: install-node.sh [options]

  --server URL     Coordinator base URL (default: https://api.joinhavn.io)
  --token TOKEN    Join token issued by the coordinator
  --wallet 0x...   EVM wallet address credited for completed work
  --name NAME      Node name (default: hostname)
  --home DIR       Install location (default: ~/.havnai)
  --python BIN     Python interpreter to build the venv with
  --creator        Force creator mode on (default: auto-detect from GPU)
  --no-creator     Force creator mode off (worker only)
  --start          Enable and start the service when the install finishes
  --skip-models    Install the runtime but do not download model weights
  --skip-deps      Reuse the existing venv without reinstalling packages
  -h, --help       Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)      SERVER_URL="$2"; shift 2 ;;
    --token)       JOIN_TOKEN="$2"; shift 2 ;;
    --wallet)      WALLET="$2"; shift 2 ;;
    --name)        NODE_NAME="$2"; shift 2 ;;
    --home)        HAVNAI_HOME="$2"; shift 2 ;;
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --creator)     CREATOR_MODE="true"; shift ;;
    --no-creator)  CREATOR_MODE="false"; shift ;;
    --start)       AUTO_START="true"; shift ;;
    --skip-models) SKIP_MODELS="true"; shift ;;
    --skip-deps)   SKIP_DEPS="true"; shift ;;
    -h|--help)     print_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
  esac
done

SERVER_URL="${SERVER_URL%/}"

VENV_PATH="$HAVNAI_HOME/venv"
BIN_DIR="$HAVNAI_HOME/bin"
RUNTIME_DIR="$HAVNAI_HOME/current"
STAGING_DIR="$HAVNAI_HOME/.staging"
ENV_FILE="$HAVNAI_HOME/.env"
SERVICE_PATH="$HOME/.config/systemd/user/havnai-node.service"
LAUNCHD_PATH="$HOME/Library/LaunchAgents/com.havnai.node.plist"

TOTAL_STEPS=8
STEP=0

step() {
  STEP=$((STEP + 1))
  printf '\n\033[36m[%d/%d]\033[0m %s\n' "$STEP" "$TOTAL_STEPS" "$1"
}
info()  { printf '      %s\n' "$1"; }
warn()  { printf '\033[33m      warning: %s\033[0m\n' "$1"; }
fail()  { printf '\033[31m      error: %s\033[0m\n' "$1" >&2; exit 1; }

cleanup() { rm -rf "$STAGING_DIR"; }
trap cleanup EXIT

mkdir -p "$HAVNAI_HOME" "$BIN_DIR" "$HAVNAI_HOME/logs" "$HAVNAI_HOME/models/creator" \
         "$HAVNAI_HOME/outputs" "$HAVNAI_HOME/loras"

# ---------------------------------------------------------------------------
step "Checking prerequisites"
# ---------------------------------------------------------------------------

OS_NAME="$(uname -s)"

command -v curl >/dev/null 2>&1 || fail "curl is required but not installed."
command -v tar  >/dev/null 2>&1 || fail "tar is required but not installed."

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$candidate"; break; fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "$OS_NAME" == "Linux" && -f /etc/os-release ]]; then
    . /etc/os-release
    case "${ID:-}" in
      ubuntu|debian)
        info "installing python3 via apt-get"
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip curl ;;
      rhel|centos|fedora|rocky|almalinux)
        info "installing python3 via dnf"
        sudo dnf install -y python3 python3-pip curl ;;
      *) fail "python3 not found; install Python 3.10+ and re-run." ;;
    esac
    PYTHON_BIN="python3"
  elif [[ "$OS_NAME" == "Darwin" ]]; then
    command -v brew >/dev/null 2>&1 || fail "python3 not found; install Homebrew or Python 3.10+."
    info "installing python via brew"
    brew install python
    PYTHON_BIN="python3"
  else
    fail "python3 not found and automatic installation is unsupported here."
  fi
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
  fail "Python $PY_VERSION is too old; 3.10 or newer is required."
fi
info "python $PY_VERSION at $(command -v "$PYTHON_BIN")"

if [[ -z "$CREATOR_MODE" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CREATOR_MODE="true"
    info "GPU detected - enabling creator mode"
  else
    CREATOR_MODE="false"
    warn "no NVIDIA GPU detected - joining as a worker (no creator jobs)"
  fi
fi

# ---------------------------------------------------------------------------
step "Downloading node runtime"
# ---------------------------------------------------------------------------

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
BUNDLE_TGZ="$STAGING_DIR/runtime.tar.gz"

if ! curl -fsSL --retry 3 --retry-delay 2 "$SERVER_URL/client/bundle.tar.gz" -o "$BUNDLE_TGZ"; then
  fail "could not download the runtime bundle from $SERVER_URL/client/bundle.tar.gz"
fi

BUNDLE_BYTES="$(wc -c < "$BUNDLE_TGZ" | tr -d ' ')"
[[ "$BUNDLE_BYTES" -gt 1024 ]] || fail "runtime bundle looks truncated ($BUNDLE_BYTES bytes)."
info "runtime bundle: $BUNDLE_BYTES bytes"

mkdir -p "$STAGING_DIR/runtime"
tar -xzf "$BUNDLE_TGZ" -C "$STAGING_DIR/runtime" || fail "runtime bundle could not be extracted."

# The bundle is only useful if it carries every capability's modules. Verify
# before we replace a working install with it.
for required in \
  "client/client.py" \
  "client/pipeline_stable_diffusion_xl_instantid.py" \
  "client/ip_adapter/resampler.py" \
  "engines/ltx_video/runner.py" \
  "engines/animatediff/animatediff_runner.py"; do
  [[ -f "$STAGING_DIR/runtime/$required" ]] || fail "runtime bundle is incomplete (missing $required)."
done
info "runtime verified: $(find "$STAGING_DIR/runtime" -name '*.py' | wc -l | tr -d ' ') modules"

curl -fsSL "$SERVER_URL/client/version" -o "$STAGING_DIR/VERSION" 2>/dev/null || echo "unknown" > "$STAGING_DIR/VERSION"

# ---------------------------------------------------------------------------
step "Installing Python environment"
# ---------------------------------------------------------------------------

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH" || fail "could not create the virtualenv at $VENV_PATH"
  info "created virtualenv"
else
  info "reusing virtualenv at $VENV_PATH"
fi

VENV_PY="$VENV_PATH/bin/python"
[[ -x "$VENV_PY" ]] || fail "virtualenv is broken; remove $VENV_PATH and re-run."

if [[ "$SKIP_DEPS" == "true" ]]; then
  info "skipping dependency install (--skip-deps)"
else
  "$VENV_PY" -m pip install --upgrade pip wheel >/dev/null 2>&1 || warn "pip self-upgrade failed; continuing"
  REQ_FILE="$STAGING_DIR/runtime/client/requirements-node.txt"
  [[ -f "$REQ_FILE" ]] || fail "requirements file missing from the runtime bundle."
  info "installing dependencies (this can take several minutes)"
  "$VENV_PY" -m pip install --no-cache-dir -r "$REQ_FILE" || fail "dependency installation failed."
fi

# ---------------------------------------------------------------------------
step "Activating runtime"
# ---------------------------------------------------------------------------

# Swap the verified runtime into place, keeping one generation for rollback.
if [[ -d "$RUNTIME_DIR" ]]; then
  rm -rf "$HAVNAI_HOME/previous"
  mv "$RUNTIME_DIR" "$HAVNAI_HOME/previous"
fi
mv "$STAGING_DIR/runtime" "$RUNTIME_DIR"
mv "$STAGING_DIR/VERSION" "$HAVNAI_HOME/VERSION" 2>/dev/null || true
info "runtime installed at $RUNTIME_DIR"

# Legacy layout: older services launch ~/.havnai/havnai_client.py directly.
ln -sf "$RUNTIME_DIR/client/client.py" "$HAVNAI_HOME/havnai_client.py"

# ---------------------------------------------------------------------------
step "Writing configuration"
# ---------------------------------------------------------------------------

write_env_key() {
  local key="$1" value="$2"
  if [[ -f "$ENV_FILE" ]] && grep -q "^${key}=" "$ENV_FILE"; then
    local tmp="$ENV_FILE.tmp"
    grep -v "^${key}=" "$ENV_FILE" > "$tmp" || true
    mv "$tmp" "$ENV_FILE"
  fi
  echo "${key}=${value}" >> "$ENV_FILE"
}

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

write_env_key "SERVER_URL"      "$SERVER_URL"
write_env_key "HAVNAI_SERVER_URL" "$SERVER_URL"
write_env_key "NODE_NAME"       "$NODE_NAME"
write_env_key "CREATOR_MODE"    "$CREATOR_MODE"
write_env_key "HAVNAI_HOME"     "$HAVNAI_HOME"
write_env_key "HAVNAI_OUTPUTS_DIR" "$HAVNAI_HOME/outputs"
[[ -n "$JOIN_TOKEN" ]] && write_env_key "JOIN_TOKEN" "$JOIN_TOKEN"
[[ -n "$JOIN_TOKEN" ]] && write_env_key "HAVNAI_NODE_TOKEN" "$JOIN_TOKEN"

if [[ -n "$WALLET" ]]; then
  write_env_key "WALLET" "$WALLET"
elif ! grep -q "^WALLET=" "$ENV_FILE" 2>/dev/null; then
  write_env_key "WALLET" "0xYOUR_WALLET_ADDRESS"
  warn "no wallet set - edit WALLET in $ENV_FILE to get credited for work"
fi

info "configuration written to $ENV_FILE"

# Wrapper: runs the node from the runtime root so `-m client.client` resolves.
NODE_RUNNER="$BIN_DIR/havnai-node"
cat > "$NODE_RUNNER" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
export HAVNAI_HOME="$HAVNAI_HOME"
set -a
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
set +a
cd "$RUNTIME_DIR"
exec "$VENV_PY" -m client.client "\$@"
RUNNER
chmod +x "$NODE_RUNNER"

# Companion commands so operators can diagnose without remembering paths.
for tool in doctor:client.doctor fetch-models:client.fetch_models; do
  name="${tool%%:*}"; module="${tool#*:}"
  cat > "$BIN_DIR/havnai-$name" <<TOOL
#!/usr/bin/env bash
set -euo pipefail
export HAVNAI_HOME="$HAVNAI_HOME"
set -a
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
set +a
cd "$RUNTIME_DIR"
exec "$VENV_PY" -m $module "\$@"
TOOL
  chmod +x "$BIN_DIR/havnai-$name"
done
info "commands installed: havnai-node, havnai-doctor, havnai-fetch-models"

# ---------------------------------------------------------------------------
step "Fetching model weights"
# ---------------------------------------------------------------------------

if [[ "$SKIP_MODELS" == "true" ]]; then
  info "skipped (--skip-models); run 'havnai-fetch-models' later"
elif [[ "$CREATOR_MODE" != "true" ]]; then
  info "worker mode - no model weights required"
else
  if ! "$BIN_DIR/havnai-fetch-models" --face-assets; then
    warn "some models could not be downloaded; run 'havnai-fetch-models' to retry"
  fi
fi

# ---------------------------------------------------------------------------
step "Installing service"
# ---------------------------------------------------------------------------

if [[ "$OS_NAME" == "Linux" ]]; then
  mkdir -p "$(dirname "$SERVICE_PATH")"
  cat > "$SERVICE_PATH" <<SERVICE
[Unit]
Description=HavnAI GPU Node Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$RUNTIME_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$NODE_RUNNER
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
KillMode=mixed
LimitNOFILE=65536

[Install]
WantedBy=default.target
SERVICE
  systemctl --user daemon-reload >/dev/null 2>&1 || warn "systemctl --user unavailable; start the node manually"
  START_CMD="systemctl --user start havnai-node"
  STATUS_CMD="journalctl --user -u havnai-node -f"
  if [[ "$AUTO_START" == "true" ]]; then
    systemctl --user enable --now havnai-node >/dev/null 2>&1 && info "service started" \
      || warn "could not start the service automatically"
  else
    info "service installed (not started)"
  fi
else
  mkdir -p "$(dirname "$LAUNCHD_PATH")"
  cat > "$LAUNCHD_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.havnai.node</string>
  <key>ProgramArguments</key><array><string>$NODE_RUNNER</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HAVNAI_HOME/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$HAVNAI_HOME/logs/launchd.err</string>
</dict>
</plist>
PLIST
  START_CMD="launchctl load -w $LAUNCHD_PATH"
  STATUS_CMD="tail -f $HAVNAI_HOME/logs/launchd.log"
  if [[ "$AUTO_START" == "true" ]]; then
    launchctl load -w "$LAUNCHD_PATH" >/dev/null 2>&1 && info "service started" \
      || warn "could not load the launch agent automatically"
  else
    info "launch agent installed (not started)"
  fi
fi

# ---------------------------------------------------------------------------
step "Verifying install"
# ---------------------------------------------------------------------------

set +e
"$BIN_DIR/havnai-doctor"
DOCTOR_STATUS=$?
set -e

cat <<SUMMARY

HavnAI node installed at $HAVNAI_HOME

  Start the node      $START_CMD
  Watch the logs      $STATUS_CMD
  Re-check health     $BIN_DIR/havnai-doctor
  Download models     $BIN_DIR/havnai-fetch-models
  Configuration       $ENV_FILE

SUMMARY

if [[ $DOCTOR_STATUS -ne 0 ]]; then
  cat <<'WARNING'
The preflight check reported blocking issues (listed above). Resolve them and
re-run havnai-doctor before starting the node, or it will accept jobs it cannot
complete.

WARNING
  exit 1
fi

echo "Preflight passed. Your node is ready to serve jobs."

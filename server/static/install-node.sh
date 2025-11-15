#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# HavnAI node installer
# ---------------------------------------------------------------------------

SERVER_URL=""
JOIN_TOKEN=""
WALLET=""
CREATOR_MODE="false"
NODE_NAME="$(hostname)"

print_usage() {
  cat <<USAGE
Usage: bash install-node.sh [--server URL] [--token TOKEN] [--wallet 0x...] [--creator]

Options:
  --server   Base URL of the HavnAI coordinator (e.g. http://192.168.1.10:5001)
  --token    Optional join token required by the coordinator
  --wallet   Optional wallet address to pre-populate ~/.havnai/.env
  --creator  Enable creator mode (set CREATOR_MODE=true)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)
      SERVER_URL="$2"
      shift 2
      ;;
    --token)
      JOIN_TOKEN="$2"
      shift 2
      ;;
    --wallet)
      WALLET="$2"
      shift 2
      ;;
    --creator)
      CREATOR_MODE="true"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ -z "$SERVER_URL" ]]; then
  SERVER_URL="http://localhost:5001"
  echo "[WARN] --server not provided, defaulting to $SERVER_URL"
fi

HAVNAI_HOME="${HOME}/.havnai"
VENV_PATH="$HAVNAI_HOME/venv"
BIN_DIR="$HAVNAI_HOME/bin"
CLIENT_PATH="$HAVNAI_HOME/havnai_client.py"
REQUIREMENTS_PATH="$HAVNAI_HOME/requirements-node.txt"
SERVICE_PATH="$HOME/.config/systemd/user/havnai-node.service"
LAUNCHD_PATH="$HOME/Library/LaunchAgents/com.havnai.node.plist"
DOWNLOAD_CACHE_DIR="$HAVNAI_HOME/downloads"

mkdir -p "$HAVNAI_HOME" "$HAVNAI_HOME/models" "$HAVNAI_HOME/models/creator" "$HAVNAI_HOME/logs" "$DOWNLOAD_CACHE_DIR" "$BIN_DIR"

# Detect platform
OS_NAME="$(uname -s)"
PACKAGE_MANAGER=""
INSTALL_PYTHON=""

if [[ "$OS_NAME" == "Linux" ]]; then
  if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    case "$ID" in
      ubuntu|debian)
        PACKAGE_MANAGER="apt-get"
        INSTALL_PYTHON="sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip curl"
        ;;
      rhel|centos|fedora|rocky|almalinux)
        PACKAGE_MANAGER="dnf"
        INSTALL_PYTHON="sudo dnf install -y python3 python3-venv python3-pip curl"
        ;;
    esac
  fi
elif [[ "$OS_NAME" == "Darwin" ]]; then
  PACKAGE_MANAGER="brew"
  INSTALL_PYTHON="brew install python"
fi

if ! command -v python3 >/dev/null 2>&1; then
  if [[ -n "$INSTALL_PYTHON" ]]; then
    echo "[INFO] Installing python3 and dependencies..."
    eval "$INSTALL_PYTHON"
  else
    echo "[ERROR] python3 not found and automatic installation unsupported on this platform" >&2
    exit 1
  fi
fi

# Create virtualenv and install requirements
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip
curl -fsSL "$SERVER_URL/client/requirements" -o "$REQUIREMENTS_PATH"
pip install --no-cache-dir -r "$REQUIREMENTS_PATH"

# Download client modules + version
curl -fsSL "$SERVER_URL/client/download" -o "$CLIENT_PATH"
curl -fsSL "$SERVER_URL/client/registry.py" -o "$HAVNAI_HOME/registry.py"
chmod +x "$CLIENT_PATH"
curl -fsSL "$SERVER_URL/client/version" -o "$HAVNAI_HOME/VERSION"

# GPU detection to enable creator mode automatically (portable, no nested heredoc in if)
echo "[INFO] Checking for GPU support..."
GPU_AVAILABLE="false"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_AVAILABLE="true"
else
  if python3 - <<'PY' 2>/dev/null | grep -q "True"; then GPU_AVAILABLE="true"; fi
import sys
try:
    import torch
    print(torch.cuda.is_available())
except Exception:
    print(False)
PY
fi
if [[ "$GPU_AVAILABLE" == "true" ]]; then
  echo "[INFO] GPU detected — enabling creator mode."
  CREATOR_MODE="true"
else
  echo "[INFO] No GPU detected — joining as worker only."
fi

# Write .env
ENV_FILE="$HAVNAI_HOME/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat <<ENVEOF > "$ENV_FILE"
SERVER_URL=$SERVER_URL
WALLET=${WALLET:-0xYOUR_WALLET_ADDRESS}
CREATOR_MODE=$CREATOR_MODE
NODE_NAME=$NODE_NAME
JOIN_TOKEN=$JOIN_TOKEN
ENVEOF
else
  tmp="$ENV_FILE.tmp"
  grep -v '^SERVER_URL=' "$ENV_FILE" | grep -v '^JOIN_TOKEN=' | grep -v '^CREATOR_MODE=' | grep -v '^NODE_NAME=' > "$tmp" || true
  echo "SERVER_URL=$SERVER_URL" >> "$tmp"
  echo "JOIN_TOKEN=$JOIN_TOKEN" >> "$tmp"
  echo "CREATOR_MODE=$CREATOR_MODE" >> "$tmp"
  echo "NODE_NAME=$NODE_NAME" >> "$tmp"
  if [[ -n "$WALLET" ]]; then
    grep -v '^WALLET=' "$tmp" > "$tmp.2" || true
    mv "$tmp.2" "$tmp"
    echo "WALLET=$WALLET" >> "$tmp"
  fi
  mv "$tmp" "$ENV_FILE"
fi

# Ensure CREATOR_MODE key is up to date (idempotent refresh)
sed -i '/^CREATOR_MODE=/d' "$ENV_FILE" 2>/dev/null || true
echo "CREATOR_MODE=$CREATOR_MODE" >> "$ENV_FILE"

# Runner script
NODE_RUNNER="$BIN_DIR/havnai-node"
cat <<'RUNNER' > "$NODE_RUNNER"
#!/usr/bin/env bash
set -e
export HAVNAI_HOME="$HOME/.havnai"
source "$HAVNAI_HOME/venv/bin/activate"
python "$HAVNAI_HOME/havnai_client.py"
RUNNER
chmod +x "$NODE_RUNNER"

# Systemd unit (Linux)
if [[ "$OS_NAME" == "Linux" ]]; then
  mkdir -p "$(dirname "$SERVICE_PATH")"
  cat <<SERVICE > "$SERVICE_PATH"
[Unit]
Description=HavnAI Node
After=network-online.target

[Service]
Type=simple
Environment=HAVNAI_HOME=%h/.havnai
EnvironmentFile=%h/.havnai/.env
ExecStart=%h/.havnai/bin/havnai-node
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
  ENABLE_CMD="systemctl --user daemon-reload && systemctl --user enable --now havnai-node"
  STATUS_CMD="journalctl --user -u havnai-node -f"
else
  # launchd (macOS)
  mkdir -p "$(dirname "$LAUNCHD_PATH")"
  cat <<PLIST > "$LAUNCHD_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.havnai.node</string>
  <key>ProgramArguments</key>
  <array>
    <string>$NODE_RUNNER</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HAVNAI_HOME</key>
    <string>$HAVNAI_HOME</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HAVNAI_HOME/logs/launchd.log</string>
  <key>StandardErrorPath</key>
  <string>$HAVNAI_HOME/logs/launchd.err</string>
</dict>
</plist>
PLIST
  ENABLE_CMD="launchctl load -w $LAUNCHD_PATH"
  STATUS_CMD="log stream --predicate 'process == \"havnai-node\"'"
fi

deactivate || true

# Download all manifest creator models on GPU nodes
if [[ "$CREATOR_MODE" == "true" ]]; then
  echo "[INFO] Ensuring creator manifest models exist locally..."
  mkdir -p "$HAVNAI_HOME/models/creator"
  manifest_json=""
  if command -v curl >/dev/null 2>&1; then
    manifest_json=$(curl -fsSL "$SERVER_URL/models/list" || true)
  fi
  if [[ -z "$manifest_json" ]]; then
    echo "[WARN] Unable to fetch manifest; skipping auto-copy."
  else
    mapfile -t MODEL_MAP < <(printf '%s\n' "$manifest_json" | python3 <<'PY' 2>/dev/null
import json,sys
try:
    data=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for entry in data.get("models", []):
    name=(entry.get("name") or "").strip()
    path=(entry.get("path") or "").strip()
    if name and path:
        print(f"{name}|{path}")
PY
)
    for pair in "${MODEL_MAP[@]}"; do
      name=${pair%%|*}
      path=${pair#*|}
      [[ -z "$name" || -z "$path" ]] && continue
      local_path="$HAVNAI_HOME/models/creator/${name}.safetensors"
      if [[ -f "$local_path" ]]; then
        echo "[INFO] $name already present."
        continue
      fi
      if [[ -f "$path" ]]; then
        if command -v rsync >/dev/null 2>&1; then
          echo "[INFO] Copying $name from $path"
          rsync -av "$path" "$local_path" || echo "[WARN] Failed to copy $name"
        else
          cp "$path" "$local_path" || echo "[WARN] Failed to copy $name"
        fi
      else
        echo "[WARN] Source not found for $name ($path); please copy manually."
      fi
    done
  fi
fi

AUTO_START="false"
for arg in "$@"; do
  if [[ "$arg" == "--start" ]]; then AUTO_START="true"; fi
done

# Optionally start the service automatically if requested
if [[ "$OS_NAME" == "Linux" && "$AUTO_START" == "true" ]]; then
  systemctl --user daemon-reload || true
  systemctl --user enable --now havnai-node || true
fi

cat <<SUMMARY

HavnAI node installed at $HAVNAI_HOME

Safe connect options:
- Start via systemd (Linux):
    systemctl --user start havnai-node
- Or run directly (any OS):
    source "$VENV_PATH/bin/activate" && python "$CLIENT_PATH"

To disconnect manually:
    curl -X POST "$SERVER_URL/disconnect" -H 'Content-Type: application/json' -d '{"node_id":"$NODE_NAME"}'

To tail logs (Linux systemd):
    $STATUS_CMD

Creator mode: set CREATOR_MODE=true in $ENV_FILE and place models in $HAVNAI_HOME/models/creator
SUMMARY

#!/usr/bin/env bash
set -euo pipefail

# HavnAI Node installer (user space)
# - Creates ~/.havnai
# - Sets up Python venv in ~/.havnai/venv
# - Downloads the client from the coordinator and installs deps
# - Installs systemd --user service to run the node

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  -s <server_url>     Coordinator base URL (e.g. http://192.168.1.10:5001)
  -t <join_token>     Join token required by the server
  -w <wallet>         EVM wallet address (0x...)
  -n <node_name>      Node name (defaults to hostname)
  -c <creator_mode>   true/false (default: true)
  -H <home_dir>       HAVNAI_HOME (default: ~/.havnai)
EOF
}

SERVER_URL=""
JOIN_TOKEN=""
WALLET=""
NODE_NAME="$(hostname)"
CREATOR_MODE="true"
HAVNAI_HOME="${HAVNAI_HOME:-$HOME/.havnai}"

while getopts ":s:t:w:n:c:H:h" opt; do
  case "$opt" in
    s) SERVER_URL="$OPTARG" ;;
    t) JOIN_TOKEN="$OPTARG" ;;
    w) WALLET="$OPTARG" ;;
    n) NODE_NAME="$OPTARG" ;;
    c) CREATOR_MODE="$OPTARG" ;;
    H) HAVNAI_HOME="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; usage; exit 2 ;;
    \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SERVER_URL" ]]; then
  echo "-s <server_url> is required" >&2; usage; exit 2
fi

mkdir -p "$HAVNAI_HOME/bin" "$HAVNAI_HOME/logs"

echo "[1/6] Creating venv…"
python3 -m venv "$HAVNAI_HOME/venv"
source "$HAVNAI_HOME/venv/bin/activate"
python -m pip install -U pip wheel

echo "[2/6] Fetching client + requirements from server…"
curl -fsSL "$SERVER_URL/client/download" -o "$HAVNAI_HOME/havnai_client.py"
curl -fsSL "$SERVER_URL/client/requirements" -o "$HAVNAI_HOME/requirements-node.txt"

echo "[3/6] Installing Python dependencies…"
python -m pip install -r "$HAVNAI_HOME/requirements-node.txt"
python -m pip install "diffusers>=0.30" "transformers>=4.36" "accelerate>=0.24" pillow scipy safetensors || true

echo "[4/6] Writing environment file…"
ENV_FILE="$HAVNAI_HOME/.env"
{
  echo "SERVER_URL=${SERVER_URL}"
  echo "JOIN_TOKEN=${JOIN_TOKEN}"
  echo "WALLET=${WALLET:-0xYOUR_WALLET_ADDRESS}"
  echo "NODE_NAME=${NODE_NAME}"
  echo "CREATOR_MODE=${CREATOR_MODE}"
} > "$ENV_FILE"

echo "[5/6] Installing wrapper and systemd unit…"
WRAP="$HAVNAI_HOME/bin/havnai-node"
cat > "$WRAP" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
export HAVNAI_HOME="${HAVNAI_HOME:-$HOME/.havnai}"
source "$HAVNAI_HOME/venv/bin/activate"
exec python "$HAVNAI_HOME/havnai_client.py"
WRAP
chmod +x "$WRAP"

mkdir -p "$HOME/.config/systemd/user"
UNIT="$HOME/.config/systemd/user/havnai-node.service"
cat > "$UNIT" <<UNIT
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
UNIT

systemctl --user daemon-reload
systemctl --user enable --now havnai-node.service

echo "[6/6] Done. Logs: journalctl --user -u havnai-node -f"

cat <<'NOTE'

WAN I2V (optional, for video jobs):
  - Ensure ffmpeg is installed and available on PATH.
    On Debian/Ubuntu, for example:
        sudo apt-get update && sudo apt-get install -y ffmpeg
  - Download any WAN I2V .safetensors checkpoints to the exact paths
    referenced by your coordinator's manifest entry.
    A common layout is:
        /mnt/d/havnai-storage/models/video/wan-i2v/wan_2.2_lightning.safetensors
  - Keep CREATOR_MODE=true in ~/.havnai/.env if you want this node to
    accept WAN video generation tasks.

NOTE

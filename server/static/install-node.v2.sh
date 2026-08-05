#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Deprecated. install-node.sh is now the single installer and delivers the
# complete runtime (image, face swap and video) plus model weights.
#
# This shim keeps older bookmarks and copy-pasted commands working by mapping
# the v2 getopts flags onto the current installer.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/install-node.sh"

ARGS=()
while getopts ":s:t:w:n:c:H:h" opt; do
  case "$opt" in
    s) ARGS+=(--server "$OPTARG") ;;
    t) ARGS+=(--token "$OPTARG") ;;
    w) ARGS+=(--wallet "$OPTARG") ;;
    n) ARGS+=(--name "$OPTARG") ;;
    c) [[ "$OPTARG" == "true" ]] && ARGS+=(--creator) || ARGS+=(--no-creator) ;;
    H) ARGS+=(--home "$OPTARG") ;;
    h) ARGS+=(--help) ;;
    :)  echo "Missing argument for -$OPTARG" >&2; exit 2 ;;
    \?) echo "Unknown option -$OPTARG" >&2; exit 2 ;;
  esac
done

echo "note: install-node.v2.sh is deprecated; use install-node.sh" >&2

if [[ -x "$INSTALLER" || -f "$INSTALLER" ]]; then
  exec bash "$INSTALLER" "${ARGS[@]}"
fi

# Fetched standalone (piped from curl): pull the current installer instead.
SERVER_URL=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--server" ]]; then SERVER_URL="${ARGS[$((i + 1))]}"; fi
done
SERVER_URL="${SERVER_URL:-https://api.joinhavn.io}"

exec bash -c "curl -fsSL '${SERVER_URL%/}/installers/install-node.sh' | bash -s -- $(printf '%q ' "${ARGS[@]}")"

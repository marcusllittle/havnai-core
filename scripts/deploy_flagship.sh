#!/usr/bin/env bash
set -euo pipefail

BRANCH="feat/joinhavn-flagship-redesign"
PI_HOST="${PI_HOST:-marcus@100.122.73.117}"
GPU_HOST="${GPU_HOST:-localhost}"
SHA="${1:-$(git rev-parse HEAD)}"

test "$(git branch --show-current)" = "$BRANCH" || {
  echo "Refusing deploy from any branch except $BRANCH" >&2
  exit 1
}
git cat-file -e "${SHA}^{commit}"
git merge-base --is-ancestor "$SHA" "$BRANCH" || {
  echo "Refusing deploy: $SHA is not on $BRANCH" >&2
  exit 1
}

archive="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$archive"' EXIT
git archive --format=tar.gz --output="$archive" "$SHA"

deploy_pi() {
  scp "$archive" "$PI_HOST:/tmp/havnai-$SHA.tar.gz"
  ssh "$PI_HOST" "SHA='$SHA' bash -s" <<'REMOTE'
set -euo pipefail
release="/opt/havnai/releases/$SHA"
previous="$(readlink -f /opt/havnai/current 2>/dev/null || true)"
printf '%s' "$previous" | sudo tee "/tmp/havnai-$SHA.previous" >/dev/null
sudo mkdir -p "$release" /var/lib/havnai/backups
sudo tar -xzf "/tmp/havnai-$SHA.tar.gz" -C "$release"
printf '%s\n' "$SHA" | sudo tee "$release/RELEASE_SHA" >/dev/null
sudo chown -R havnai:havnai "$release"
sudo -u havnai /opt/havnai/venv/bin/pip install -r "$release/server/requirements.txt"
sudo -u havnai HAVNAI_DB_PATH=/var/lib/havnai/ledger.db /opt/havnai/venv/bin/python "$release/scripts/backup_coordinator.py"
sudo ln -sfn "$release" /opt/havnai/current
sudo systemctl restart havnai-coordinator.service
for _ in $(seq 1 20); do curl -fsS http://127.0.0.1:5001/healthz && exit 0; sleep 2; done
if [ -n "$previous" ]; then sudo ln -sfn "$previous" /opt/havnai/current; fi
sudo systemctl restart havnai-coordinator.service
exit 1
REMOTE
  curl -fsS https://api.joinhavn.io/healthz >/dev/null
}

rollback_pi() {
  ssh "$PI_HOST" "SHA='$SHA' bash -s" <<'REMOTE'
set -euo pipefail
marker="/tmp/havnai-$SHA.previous"
previous="$(sudo cat "$marker" 2>/dev/null || true)"
if [ -n "$previous" ] && [ -d "$previous" ]; then
  sudo ln -sfn "$previous" /opt/havnai/current
  sudo systemctl restart havnai-coordinator.service
fi
REMOTE
}

deploy_gpu() {
  if [ "$GPU_HOST" = "localhost" ]; then
    target="$HOME/.havnai/releases/$SHA"
    previous="$(readlink -f "$HOME/.havnai/current" 2>/dev/null || true)"
    mkdir -p "$target"
    tar -xzf "$archive" -C "$target"
    printf '%s\n' "$SHA" > "$target/RELEASE_SHA"
    ln -sfn "$target" "$HOME/.havnai/current"
    systemctl --user restart havnai-node.service
    if ! wait_for_local_node; then
      if [ -n "$previous" ] && [ -d "$previous" ]; then
        ln -sfn "$previous" "$HOME/.havnai/current"
        systemctl --user restart havnai-node.service
      fi
      return 1
    fi
  else
    scp "$archive" "$GPU_HOST:/tmp/havnai-$SHA.tar.gz"
    ssh "$GPU_HOST" "SHA='$SHA' bash -s" <<'REMOTE'
set -euo pipefail
target="$HOME/.havnai/releases/$SHA"
previous="$(readlink -f "$HOME/.havnai/current" 2>/dev/null || true)"
mkdir -p "$target"
tar -xzf "/tmp/havnai-$SHA.tar.gz" -C "$target"
printf '%s\n' "$SHA" > "$target/RELEASE_SHA"
ln -sfn "$target" "$HOME/.havnai/current"
systemctl --user restart havnai-node.service
for _ in $(seq 1 30); do
  if systemctl --user is-active --quiet havnai-node.service \
    && journalctl --user -u havnai-node.service --since '-60 seconds' --no-pager \
      | grep -q 'Wallet linked with coordinator'; then
    exit 0
  fi
  sleep 2
done
if [ -n "$previous" ] && [ -d "$previous" ]; then
  ln -sfn "$previous" "$HOME/.havnai/current"
  systemctl --user restart havnai-node.service
fi
exit 1
REMOTE
  fi
}

wait_for_local_node() {
  for _ in $(seq 1 30); do
    if systemctl --user is-active --quiet havnai-node.service \
      && journalctl --user -u havnai-node.service --since '-60 seconds' --no-pager \
        | grep -q 'Wallet linked with coordinator'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

deploy_pi
if ! deploy_gpu; then
  rollback_pi
  echo "GPU deploy failed; both hosts were rolled back" >&2
  exit 1
fi
ssh "$PI_HOST" "sudo rm -f '/tmp/havnai-$SHA.previous'"
echo "deployed $SHA to coordinator and GPU node"

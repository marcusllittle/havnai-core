#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-main}"
PI_HOST="${PI_HOST:-marcus@192.168.4.105}"
GPU_HOST="${GPU_HOST:-localhost}"
COORDINATOR_REPO="${COORDINATOR_REPO:-/home/marcus/Downloads/source-code/havnai-core}"
COORDINATOR_DB_PATH="${COORDINATOR_DB_PATH:-/home/marcus/Downloads/source-code/havnai-core/db/ledger.db}"
COORDINATOR_BACKUP_DIR="${COORDINATOR_BACKUP_DIR:-/home/marcus/Downloads/source-code/havnai-core/db/backups}"
SHA="${1:-$(git rev-parse HEAD)}"

test "$(git branch --show-current)" = "$BRANCH" || {
  echo "Refusing deploy from any branch except $BRANCH" >&2
  exit 1
}
git cat-file -e "${SHA}^{commit}"
git merge-base --is-ancestor "$SHA" "refs/heads/$BRANCH" || {
  echo "Refusing deploy: $SHA is not on $BRANCH" >&2
  exit 1
}

archive="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$archive"' EXIT
git archive --format=tar.gz --output="$archive" "$SHA"

deploy_pi() {
  ssh "$PI_HOST" "SHA='$SHA' BRANCH='$BRANCH' COORDINATOR_REPO='$COORDINATOR_REPO' COORDINATOR_DB_PATH='$COORDINATOR_DB_PATH' COORDINATOR_BACKUP_DIR='$COORDINATOR_BACKUP_DIR' bash -s" <<'REMOTE'
set -euo pipefail
repo="$COORDINATOR_REPO"
cd "$repo"
previous="$(git rev-parse HEAD)"
printf '%s' "$previous" | sudo tee "/tmp/havnai-$SHA.previous" >/dev/null
nodes_backup="/tmp/havnai-$SHA.nodes.json"
if [ -f nodes.json ]; then cp nodes.json "$nodes_backup"; fi
git fetch origin "$BRANCH"
git switch "$BRANCH"
git checkout -- nodes.json 2>/dev/null || true
git merge --ff-only "$SHA"
if [ -f "$nodes_backup" ]; then cp "$nodes_backup" nodes.json; fi
.venv/bin/python -m pip install -r server/requirements.txt
HAVNAI_DB_PATH="$COORDINATOR_DB_PATH" HAVNAI_BACKUP_DIR="$COORDINATOR_BACKUP_DIR" .venv/bin/python scripts/backup_coordinator.py
sudo systemctl restart havnai-coordinator.service
for _ in $(seq 1 20); do curl -fsS http://127.0.0.1:5001/healthz && exit 0; sleep 2; done
if [ -n "$previous" ]; then
  if [ -f "$nodes_backup" ]; then cp "$nodes_backup" nodes.json; fi
  git switch --detach "$previous"
fi
sudo systemctl restart havnai-coordinator.service
exit 1
REMOTE
  curl -fsS https://api.joinhavn.io/healthz >/dev/null
}

rollback_pi() {
  ssh "$PI_HOST" "SHA='$SHA' COORDINATOR_REPO='$COORDINATOR_REPO' bash -s" <<'REMOTE'
set -euo pipefail
marker="/tmp/havnai-$SHA.previous"
previous="$(sudo cat "$marker" 2>/dev/null || true)"
if [ -n "$previous" ]; then
  cd "$COORDINATOR_REPO"
  nodes_backup="/tmp/havnai-$SHA.nodes.json"
  if [ -f nodes.json ]; then cp nodes.json "$nodes_backup"; fi
  git checkout -- nodes.json 2>/dev/null || true
  git switch --detach "$previous"
  if [ -f "$nodes_backup" ]; then cp "$nodes_backup" nodes.json; fi
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

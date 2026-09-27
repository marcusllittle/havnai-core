#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-main}"
PI_HOST="${PI_HOST:-marcus@192.168.4.105}"
COORDINATOR_REPO="${COORDINATOR_REPO:-/home/marcus/Downloads/source-code/havnai-core}"
COORDINATOR_DB_PATH="${COORDINATOR_DB_PATH:-/home/marcus/Downloads/source-code/havnai-core/db/ledger.db}"
COORDINATOR_BACKUP_DIR="${COORDINATOR_BACKUP_DIR:-/home/marcus/Downloads/source-code/havnai-core/db/backups}"
SHA="${1:-$(git rev-parse HEAD)}"

test "$(git branch --show-current)" = "$BRANCH" || {
  echo "Refusing coordinator deploy from any branch except $BRANCH" >&2
  exit 1
}
git cat-file -e "${SHA}^{commit}"
git merge-base --is-ancestor "$SHA" "refs/heads/$BRANCH" || {
  echo "Refusing coordinator deploy: $SHA is not on $BRANCH" >&2
  exit 1
}

archive="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$archive"' EXIT
git archive --format=tar.gz --output="$archive" "$SHA"

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
for _ in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:5001/healthz; then
    sudo rm -f "/tmp/havnai-$SHA.previous"
    exit 0
  fi
  sleep 2
done
if [ -n "$previous" ]; then
  if [ -f "$nodes_backup" ]; then cp "$nodes_backup" nodes.json; fi
  git switch --detach "$previous"
fi
sudo systemctl restart havnai-coordinator.service
exit 1
REMOTE

curl -fsS https://api.joinhavn.io/healthz >/dev/null
echo "deployed coordinator $SHA"

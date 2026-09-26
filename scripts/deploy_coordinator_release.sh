#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-main}"
PI_HOST="${PI_HOST:-marcus@100.122.73.117}"
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
for _ in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:5001/healthz; then
    sudo rm -f "/tmp/havnai-$SHA.previous"
    exit 0
  fi
  sleep 2
done
if [ -n "$previous" ] && [ -d "$previous" ]; then
  sudo ln -sfn "$previous" /opt/havnai/current
fi
sudo systemctl restart havnai-coordinator.service
exit 1
REMOTE

curl -fsS https://api.joinhavn.io/healthz >/dev/null
echo "deployed coordinator $SHA"

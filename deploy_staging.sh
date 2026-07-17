#!/usr/bin/env bash
# Deploy NextMove to the STAGING environment on the VPS.
# Fails fast: if the test suite doesn't pass, the staging service is never restarted.
set -euo pipefail

STAGING_DIR="${NEXTMOVE_STAGING_DIR:-/opt/nextmove-staging}"
BRANCH="${NEXTMOVE_STAGING_BRANCH:-v5-matching}"
SERVICE_NAME="${NEXTMOVE_STAGING_SERVICE:-nextmove-staging}"
VENV_DIR="$STAGING_DIR/.venv"

echo "==> Deploying branch '$BRANCH' to $STAGING_DIR"

cd "$STAGING_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -r requirements.txt
pip install -q pytest-cov httpx

echo "==> Running test suite before touching the running service"
python3 -m pytest
echo "==> Tests passed."

echo "==> Restarting $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sleep 2
sudo systemctl is-active --quiet "$SERVICE_NAME" && echo "==> $SERVICE_NAME is active" || {
  echo "!! $SERVICE_NAME failed to start — check: journalctl -u $SERVICE_NAME -n 50"
  exit 1
}

echo "==> Staging deploy complete."

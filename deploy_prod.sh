#!/usr/bin/env bash
# Deploy NextMove to PRODUCTION on the VPS (http://54.38.26.33/nextmove-v5/).
# Fails fast: if the test suite doesn't pass, the prod service is never restarted.
set -euo pipefail

PROD_DIR="${NEXTMOVE_PROD_DIR:-/home/olive/NEXTMOVE-V5}"
BRANCH="${NEXTMOVE_PROD_BRANCH:-v5-matching}"
SERVICE_NAME="${NEXTMOVE_PROD_SERVICE:-nextmove-v5-matching-api}"
VENV_DIR="$PROD_DIR/venv"

echo "==> Deploying branch '$BRANCH' to $PROD_DIR (PRODUCTION)"

cd "$PROD_DIR"
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

echo "==> Production deploy complete."

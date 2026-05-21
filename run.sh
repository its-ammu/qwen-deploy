#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${API_KEY:?Set API_KEY in .env or environment}"
: "${PORT:=7860}"
: "${HOST:=0.0.0.0}"

export PORT HOST

if [[ "${MOCK_INFERENCE:-false}" != "true" ]]; then
  echo "Starting Qwen3-Omni server on ${HOST}:${PORT}"
  exec gunicorn \
    --bind "${HOST}:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --graceful-timeout 120 \
    "app:app"
else
  echo "Starting in MOCK_INFERENCE mode on ${HOST}:${PORT}"
  exec python app.py
fi

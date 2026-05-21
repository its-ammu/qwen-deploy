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

echo "Starting Qwen3-Omni server on ${HOST:-0.0.0.0}:${PORT:-7860}"
exec python app.py

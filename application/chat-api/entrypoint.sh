#!/usr/bin/env bash
set -euo pipefail

NGROK_DOMAIN="${NGROK_DOMAIN:-alleen-wieldier-dionne.ngrok-free.dev}"
NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN:-}"
NGROK_PORT="${NGROK_PORT:-8000}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_PORT="${UVICORN_PORT:-8000}"

# Accept full URL or bare domain for convenience.
NGROK_DOMAIN="${NGROK_DOMAIN#https://}"
NGROK_DOMAIN="${NGROK_DOMAIN#http://}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok binary not found; ensure it is installed in the image." >&2
  exit 1
fi

if [[ -n "${NGROK_AUTHTOKEN}" ]]; then
  ngrok config add-authtoken "${NGROK_AUTHTOKEN}"
fi

# Run ngrok and uvicorn together; shut down both if either exits.
ngrok http --log=stdout --domain="${NGROK_DOMAIN}" "${NGROK_PORT}" &
NGROK_PID=$!

uvicorn main:app --host "${UVICORN_HOST}" --port "${UVICORN_PORT}" &
UVICORN_PID=$!

wait -n
EXIT_CODE=$?
kill "$NGROK_PID" "$UVICORN_PID" 2>/dev/null || true
wait || true
exit "$EXIT_CODE"

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import fastapi, httpx, pydantic, requests, uvicorn" 2>/dev/null; then
  .venv/bin/pip install -r requirements.txt
fi
if [[ ! -d node_modules ]]; then
  if [[ -n "${NPM_REGISTRY:-}" ]]; then
    npm install --registry="$NPM_REGISTRY"
  elif ! npm install; then
    echo "Default npm registry failed; retrying with the configured fallback mirror."
    npm install --registry=https://registry.npmmirror.com
  fi
fi

npm run build

cleanup() {
  kill "${PROXY_PID:-}" "${HOST_PID:-}" 2>/dev/null || true
  wait "${PROXY_PID:-}" "${HOST_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/uvicorn proxy:app --host 127.0.0.1 --port 8000 &
PROXY_PID=$!
node --import tsx serve.ts &
HOST_PID=$!

echo
echo "Genie Agent Mode visualization UI: http://localhost:8080"
echo "The UI forces enable_viz=true on every Genie Agent Mode request."
echo

(
  for _ in {1..360}; do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1 &&
       curl -fsS http://localhost:8080 >/dev/null 2>&1; then
      python3 -c 'import webbrowser; webbrowser.open("http://localhost:8080")'
      exit 0
    fi
    sleep 1
  done
) &

wait "$PROXY_PID" "$HOST_PID"

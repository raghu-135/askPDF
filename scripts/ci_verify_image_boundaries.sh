#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAIN_COMPOSE=(
  docker compose --env-file .env.ci
  -f docker-compose.yml
  -f docker-compose.ci.yml
)

rag_image="$("${MAIN_COMPOSE[@]}" images rag-service --quiet 2>/dev/null | head -1)"
langgraph_image="$("${MAIN_COMPOSE[@]}" images langgraph-runtime --quiet 2>/dev/null | head -1)"
if [ -z "$rag_image" ] || [ -z "$langgraph_image" ]; then
  echo "Compose images missing after build (rag-service or langgraph-runtime)" >&2
  exit 1
fi

docker run --rm "$rag_image" python -c 'import importlib.util; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm "$langgraph_image" python -c 'import importlib.util, langgraph, runtime_protocol, langgraph_runtime; assert importlib.util.find_spec("app") is None'
docker run --rm askpdf-hermes-runtime python -c 'import importlib.util, hermes_runtime, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("app") is None'

docker run --rm askpdf-control-plane-tests python -c 'import importlib.util, app, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm askpdf-langgraph-runtime-tests python -c 'import importlib.util, langgraph, langgraph_runtime, runtime_protocol; assert importlib.util.find_spec("app") is None'
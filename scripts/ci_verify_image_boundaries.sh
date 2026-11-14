#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAIN_COMPOSE=(
  docker compose --env-file .env.ci
  -f docker-compose.yml
  -f docker-compose.ci.yml
)

project_name="$(basename "$ROOT" | tr '[:upper:]' '[:lower:]')"
rag_image="${project_name}-rag-service"
langgraph_image="askpdf-langgraph-runtime"

for candidate in "$rag_image" "$langgraph_image"; do
  if ! docker image inspect "$candidate" >/dev/null 2>&1; then
    echo "Expected image '$candidate' missing after build" >&2
    docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}' | head -20
    exit 1
  fi
done

docker run --rm "$rag_image" python -c 'import importlib.util; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm "$langgraph_image" python -c 'import importlib.util, langgraph, runtime_protocol, langgraph_runtime; assert importlib.util.find_spec("app") is None'
docker run --rm askpdf-hermes-runtime python -c 'import importlib.util, hermes_runtime, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("app") is None'

docker run --rm askpdf-control-plane-tests python -c 'import importlib.util, app, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm askpdf-langgraph-runtime-tests python -c 'import importlib.util, langgraph, langgraph_runtime, runtime_protocol; assert importlib.util.find_spec("app") is None'
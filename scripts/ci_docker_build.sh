#!/usr/bin/env bash
# Build images for GitHub Actions with a shared BuildKit cache directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

CACHE_DIR="${ASKPDF_BUILDX_CACHE_DIR:-/tmp/.buildx-cache}"
CACHE_NEW="${CACHE_DIR}-new"
mkdir -p "$CACHE_DIR"

cache_from=()
if [ -n "$(ls -A "$CACHE_DIR" 2>/dev/null || true)" ]; then
  cache_from=(--cache-from "type=local,src=$CACHE_DIR")
fi

retry() {
  local attempt=1
  local delay=45
  while [ "$attempt" -le 4 ]; do
    if "$@"; then
      return 0
    fi
    echo "Command failed (attempt ${attempt}/4), retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay + 15))
  done
  return 1
}

build_with_cache() {
  local tag="$1"
  shift
  retry docker buildx build \
    "${cache_from[@]}" \
    --cache-to "type=local,dest=$CACHE_NEW,mode=max" \
    --load \
    -t "$tag" \
    "$@" \
    .
  if [ -d "$CACHE_NEW" ]; then
    rm -rf "$CACHE_DIR"
    mv "$CACHE_NEW" "$CACHE_DIR"
    CACHE_NEW="${CACHE_DIR}-new"
    cache_from=(--cache-from "type=local,src=$CACHE_DIR")
  fi
}

MAIN_COMPOSE=(
  docker compose --env-file .env.ci
  -f docker-compose.yml
  -f docker-compose.ci.yml
)

"${MAIN_COMPOSE[@]}" config --quiet
# CI does not need frontend or browser-capture images (avoids lscr.io rate limits).
retry "${MAIN_COMPOSE[@]}" build rag-service langgraph-runtime db-migrate

build_with_cache askpdf-control-plane-tests \
  --build-arg DOWNLOAD_LOCAL_MODELS=false \
  --build-arg WARM_DOCLING_MODELS=false \
  -f rag_service/Dockerfile.test

build_with_cache askpdf-langgraph-runtime-tests \
  -f langgraph_runtime/Dockerfile.test

build_with_cache askpdf-hermes-runtime \
  -f hermes_runtime/Dockerfile

TEST_COMPOSE=(
  docker compose --env-file .env.ci
  -p askpdf-test
  -f docker-compose.test.yml
  -f docker-compose.test.ci.yml
)
retry "${TEST_COMPOSE[@]}" build test-runner

RUNTIME_COMPOSE=(
  docker compose --env-file .env.ci
  -p askpdf-runtime-prefetch
  -f docker-compose.runtime-integration.yml
)
retry "${RUNTIME_COMPOSE[@]}" build rag-service langgraph-runtime

echo "Docker image build complete (cache at $CACHE_DIR)"
#!/usr/bin/env bash
# Build all images required for GitHub Actions CI (single authoritative build).
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"

CACHE_DIR="${ASKPDF_BUILDX_CACHE_DIR:-/tmp/.buildx-cache}"
CACHE_NEW="${CACHE_DIR}-new"
mkdir -p "$CACHE_DIR"

cache_from=()
if [ -n "$(ls -A "$CACHE_DIR" 2>/dev/null || true)" ]; then
  cache_from=(--cache-from "type=local,src=$CACHE_DIR")
fi

build_with_cache() {
  local tag="$1"
  shift
  docker buildx build \
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

should_build_main_optional() {
  local var="$1"
  case "${!var:-auto}" in
    0|false|no) return 1 ;;
    1|true|yes) return 0 ;;
    *) return 0 ;;
  esac
}

cp .env.ci .env
main_compose config --quiet

CORE_SERVICES=(langgraph-runtime rag-service db-migrate)
if should_build_main_optional ASKPDF_CI_BUILD_FRONTEND; then
  CORE_SERVICES+=(frontend)
fi
if should_build_main_optional ASKPDF_CI_BUILD_BROWSER_CAPTURE; then
  CORE_SERVICES+=(browser-capture)
fi

echo "Building main compose services: ${CORE_SERVICES[*]}"
main_compose build "${CORE_SERVICES[@]}"

rag_id="$(main_compose images rag-service --quiet | head -1)"
langgraph_id="$(main_compose images langgraph-runtime --quiet | head -1)"
docker tag "$rag_id" askpdf-runtime-control-plane
docker tag "$rag_id" askpdf-control-plane
docker tag "$langgraph_id" askpdf-langgraph-runtime

echo "Building test-runner image"
test_compose build test-runner
test_id="$(test_compose images test-runner --quiet | head -1)"
docker tag "$test_id" askpdf-test-runner:ci
docker tag "$test_id" askpdf-control-plane-tests

build_with_cache askpdf-langgraph-runtime-tests \
  -f langgraph_runtime/Dockerfile.test

build_with_cache askpdf-hermes-runtime \
  -f hermes_runtime/Dockerfile

echo "Building runtime-integration stack (includes hermes-upstream once)"
runtime_compose askpdf-ci-prefetch build \
  rag-service langgraph-runtime db-migrate runtime-db-migrate \
  fake-llm hermes-upstream hermes hermes-runtime test-runner runtime-test-runner

tag_from_runtime() {
  local service="$1"
  local tag="$2"
  local id
  id="$(runtime_compose askpdf-ci-prefetch images "$service" --quiet | head -1)"
  docker tag "$id" "$tag"
}

tag_from_runtime rag-service askpdf-runtime-control-plane
tag_from_runtime langgraph-runtime askpdf-langgraph-runtime
tag_from_runtime fake-llm askpdf-fake-llm:ci
tag_from_runtime hermes askpdf-hermes-gateway:ci
tag_from_runtime hermes-runtime askpdf-hermes-runtime:ci
tag_from_runtime hermes-upstream "$HERMES_UPSTREAM_IMAGE"

runtime_test_id="$(runtime_compose askpdf-ci-prefetch images runtime-test-runner --quiet | head -1)"
docker tag "$runtime_test_id" askpdf-langgraph-runtime-tests

echo "CI image build complete"
for tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "expected image missing after build: $tag" >&2
    exit 1
  fi
done
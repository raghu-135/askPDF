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

image_id_for_service() {
  local compose_fn=(main_compose)
  local service="$1"
  local id
  id="$("${compose_fn[@]}" images "$service" --quiet 2>/dev/null | head -1)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  id="$(docker images --quiet --filter "label=com.docker.compose.service=$service" | head -1)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  echo "could not resolve image id for service $service" >&2
  return 1
}

rag_id="$(image_id_for_service rag-service)"
langgraph_id="$(image_id_for_service langgraph-runtime)"
docker tag "$rag_id" askpdf-runtime-control-plane
docker tag "$rag_id" askpdf-control-plane
docker tag "$langgraph_id" askpdf-langgraph-runtime

runtime_image_id_for_service() {
  local project="$1"
  local service="$2"
  local id
  id="$(runtime_compose "$project" images "$service" --quiet 2>/dev/null | head -1)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  id="$(docker images --quiet --filter "label=com.docker.compose.project=$project" --filter "label=com.docker.compose.service=$service" | head -1)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  echo "could not resolve image id for $project/$service" >&2
  return 1
}

echo "Building test-runner image"
test_compose build test-runner
test_id="$(test_compose images test-runner --quiet | head -1)"
if [ -z "$test_id" ]; then
  test_id="$(docker images --quiet --filter "label=com.docker.compose.project=askpdf-test" --filter "label=com.docker.compose.service=test-runner" | head -1)"
fi
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
  id="$(runtime_image_id_for_service askpdf-ci-prefetch "$service")"
  docker tag "$id" "$tag"
}

tag_from_runtime rag-service askpdf-runtime-control-plane
tag_from_runtime langgraph-runtime askpdf-langgraph-runtime
tag_from_runtime fake-llm askpdf-fake-llm:ci
tag_from_runtime hermes askpdf-hermes-gateway:ci
tag_from_runtime hermes-runtime askpdf-hermes-runtime:ci
tag_from_runtime hermes-upstream "$HERMES_UPSTREAM_IMAGE"

runtime_test_id="$(runtime_image_id_for_service askpdf-ci-prefetch runtime-test-runner)"
docker tag "$runtime_test_id" askpdf-langgraph-runtime-tests

echo "CI image build complete"
for tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "expected image missing after build: $tag" >&2
    exit 1
  fi
done
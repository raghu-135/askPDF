#!/usr/bin/env bash
# Build all images required for GitHub Actions CI (single authoritative build).
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"
# shellcheck source=ci_buildkit_cache.sh
source "$(dirname "$0")/ci_buildkit_cache.sh"

CACHE_DIR="${ASKPDF_BUILDX_CACHE_DIR:-/tmp/.buildx-cache}"
CACHE_NEW="${CACHE_DIR}-new"
mkdir -p "$CACHE_DIR"

ci_log_buildkit_cache_status

build_with_cache() {
  local tag="$1"
  shift
  ci_refresh_buildkit_cache_args "$CACHE_NEW"
  docker buildx build \
    "${CI_BUILDKIT_CACHE_FROM[@]}" \
    "${CI_BUILDKIT_CACHE_TO[@]}" \
    --load \
    --provenance=false \
    --sbom=false \
    -t "$tag" \
    "$@" \
    .
  if [ -d "$CACHE_NEW" ]; then
    rm -rf "$CACHE_DIR"
    mv "$CACHE_NEW" "$CACHE_DIR"
    CACHE_NEW="${CACHE_DIR}-new"
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

CORE_SERVICES=(langgraph-runtime rag-service)
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
docker tag askpdf-hermes-runtime askpdf-hermes-runtime:ci

# Integration-only services (gateway depends on hermes-upstream additional_contexts).
RUNTIME_CI_BUILD_SERVICES=(fake-llm hermes-upstream hermes)
echo "Building runtime integration services (no duplicate rag/langgraph/test rebuild): ${RUNTIME_CI_BUILD_SERVICES[*]}"
runtime_compose askpdf-ci-prefetch build "${RUNTIME_CI_BUILD_SERVICES[@]}"

tag_runtime_compose_image() {
  local service="$1"
  local tag="$2"
  if docker image inspect "$tag" >/dev/null 2>&1; then
    return 0
  fi
  local id
  id="$(runtime_compose askpdf-ci-prefetch images "$service" --quiet 2>/dev/null | head -1)"
  if [ -z "$id" ]; then
    id="$(docker images --quiet --filter "label=com.docker.compose.project=askpdf-ci-prefetch" --filter "label=com.docker.compose.service=$service" | head -1)"
  fi
  if [ -z "$id" ]; then
    echo "could not resolve runtime image for $service" >&2
    return 1
  fi
  docker tag "$id" "$tag"
}

tag_runtime_compose_image fake-llm askpdf-fake-llm:ci
tag_runtime_compose_image hermes askpdf-hermes-gateway:ci
tag_runtime_compose_image hermes-upstream "$HERMES_UPSTREAM_IMAGE"

docker tag "$langgraph_id" askpdf-langgraph-runtime

echo "CI image build complete"
for tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "expected image missing after build: $tag" >&2
    exit 1
  fi
done
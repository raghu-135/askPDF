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

OPTIONAL_MAIN_SERVICES=()
if should_build_main_optional ASKPDF_CI_BUILD_FRONTEND; then
  OPTIONAL_MAIN_SERVICES+=(frontend)
fi
if should_build_main_optional ASKPDF_CI_BUILD_BROWSER_CAPTURE; then
  OPTIONAL_MAIN_SERVICES+=(browser-capture)
fi
if [ "${#OPTIONAL_MAIN_SERVICES[@]}" -gt 0 ]; then
  echo "Building optional main compose services: ${OPTIONAL_MAIN_SERVICES[*]}"
  main_compose build "${OPTIONAL_MAIN_SERVICES[@]}"
fi

echo "Building control-plane images (shared registry cache)"
build_with_cache askpdf-langgraph-runtime -f langgraph_runtime/Dockerfile
build_with_cache askpdf-runtime-control-plane \
  -f rag_service/Dockerfile \
  --build-arg "DOWNLOAD_LOCAL_MODELS=false"
docker tag askpdf-runtime-control-plane askpdf-control-plane

build_with_cache askpdf-test-runner:ci \
  -f rag_service/Dockerfile.test \
  --build-arg "DOWNLOAD_LOCAL_MODELS=false" \
  --build-arg "WARM_DOCLING_MODELS=false"
docker tag askpdf-test-runner:ci askpdf-control-plane-tests

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

echo "CI image build complete"
for tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "expected image missing after build: $tag" >&2
    exit 1
  fi
done
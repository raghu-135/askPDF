#!/usr/bin/env bash
# Shared helpers for GitHub Actions Docker CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"
export COMPOSE_PROFILES="${ASKPDF_CI_COMPOSE_PROFILES:-}"
export ASKPDF_BUILDKIT_CACHE_REF="${ASKPDF_BUILDKIT_CACHE_REF:-}"

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.override.yml -f docker-compose.integration.yml)
HERMES_UPSTREAM_IMAGE="askpdf-hermes-upstream:${HERMES_UPSTREAM_REVISION:-2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19}"

# Images that downstream jobs must load (local tags).
ASKPDF_CI_IMAGE_TAGS=(
  askpdf-runtime-control-plane
  askpdf-langgraph-runtime
  askpdf-hermes-runtime
  askpdf-control-plane-tests
  askpdf-langgraph-runtime-tests
  askpdf-test-runner:ci
  askpdf-fake-llm:ci
  askpdf-hermes-gateway:ci
  askpdf-hermes-runtime:ci
  "$HERMES_UPSTREAM_IMAGE"
)

compose_env() {
  docker compose --env-file .env.ci "${COMPOSE_FILES[@]}" "$@"
}

main_compose() {
  COMPOSE_PROFILES="${ASKPDF_CI_MAIN_PROFILES:-}" compose_env "$@"
}

test_compose() {
  COMPOSE_PROFILES="${ASKPDF_CI_TEST_PROFILES:-test}" compose_env -p askpdf-test "$@"
}

runtime_compose() {
  local project="${1:-askpdf-ci-prefetch}"
  shift
  COMPOSE_PROFILES="${ASKPDF_CI_RUNTIME_PROFILES:-integration}" \
    compose_env -p "$project" "$@"
}
#!/usr/bin/env bash
# Shared helpers for GitHub Actions Docker CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"

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
  docker compose --env-file .env.ci "$@"
}

main_compose() {
  compose_env -f docker-compose.yml -f docker-compose.ci.yml "$@"
}

test_compose() {
  compose_env -p askpdf-test -f docker-compose.test.yml -f docker-compose.test.ci.yml "$@"
}

runtime_compose() {
  local project="${1:-askpdf-ci-prefetch}"
  shift
  compose_env -p "$project" -f docker-compose.runtime-integration.yml -f docker-compose.runtime-integration.ci.yml "$@"
}
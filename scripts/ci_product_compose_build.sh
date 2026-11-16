#!/usr/bin/env bash
# Build product (hermes) and real-hermes gateway images; verify upstream Hermes venv exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"

cp -f .env.ci .env

resolve_built_service_image() {
  local project="$1"
  local service="$2"
  local id=""
  id="$(docker compose -p "$project" images "$service" --quiet 2>/dev/null | head -1 || true)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  id="$(docker images --quiet "${project}-${service}:latest" 2>/dev/null | head -1 || true)"
  if [ -n "$id" ]; then
    echo "$id"
    return 0
  fi
  id="$(
    docker images --format '{{.Repository}} {{.ID}}' \
      | awk -v prefix="${project}-${service}" '$1 == prefix { print $2; exit }'
  )"
  echo "$id"
}

verify_hermes_venv() {
  local image_id="$1"
  local label="$2"
  if [ -z "$image_id" ]; then
    echo "missing image id for $label" >&2
    return 1
  fi
  echo "Verifying /opt/hermes/.venv/bin/python in $label ($image_id)"
  docker run --rm "$image_id" test -x /opt/hermes/.venv/bin/python
  docker run --rm "$image_id" /opt/hermes/.venv/bin/python -c "import pathlib; assert pathlib.Path('/opt/hermes/.venv/bin/python').is_file()"
}

PRODUCT_PROJECT="${ASKPDF_CI_PRODUCT_PROJECT:-askpdf-ci-product}"
REAL_PROJECT="${ASKPDF_CI_REAL_HERMES_PROJECT:-askpdf-ci-real-hermes}"

echo "Building hermes profile compose images (upstream gateway chain + app stack)"
HERMES_BUILD_SERVICES=(
  env-secrets
  hermes-upstream
  hermes-config-init
  hermes
  hermes-runtime
  rag-service
  langgraph-runtime
  frontend
  browser-capture
)
COMPOSE_PROFILES=hermes main_compose -p "$PRODUCT_PROJECT" build "${HERMES_BUILD_SERVICES[@]}"

hermes_id="$(resolve_built_service_image "$PRODUCT_PROJECT" hermes)"
verify_hermes_venv "$hermes_id" "product hermes gateway"

echo "Building real-hermes gateway (pinned upstream smoke)"
COMPOSE_PROFILES=real-hermes main_compose -p "$REAL_PROJECT" build hermes-upstream-real hermes-real
real_id="$(resolve_built_service_image "$REAL_PROJECT" hermes-real)"
verify_hermes_venv "$real_id" "real-hermes gateway"

echo "Product compose build verification complete"
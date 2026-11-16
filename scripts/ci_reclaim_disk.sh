#!/usr/bin/env bash
set -euo pipefail
COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.override.yml)
docker compose "${COMPOSE_FILES[@]}" -p askpdf-test --profile test down --volumes --remove-orphans 2>/dev/null || true
for project in askpdf-runtime-integration-test askpdf-hermes-ci askpdf-ci-prefetch; do
  COMPOSE_PROFILES=integration docker compose "${COMPOSE_FILES[@]}" -p "$project" down --volumes --remove-orphans 2>/dev/null || true
done
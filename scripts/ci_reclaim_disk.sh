#!/usr/bin/env bash
# Free CI disk without invalidating the BuildKit layer cache (no builder prune).
set -euo pipefail

docker compose -p askpdf-test -f docker-compose.test.yml down --volumes --remove-orphans 2>/dev/null || true
docker compose -p askpdf-runtime-collection \
  -f docker-compose.runtime-integration.yml \
  -f docker-compose.runtime-integration.ci.yml \
  down --volumes --remove-orphans 2>/dev/null || true
docker compose -p askpdf-ci-prefetch \
  -f docker-compose.runtime-integration.yml \
  -f docker-compose.runtime-integration.ci.yml \
  down --volumes --remove-orphans 2>/dev/null || true

docker volume prune -f || true
docker image prune -f || true
df -h
#!/usr/bin/env bash
# BuildKit registry cache helpers for GitHub Actions CI.
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ci_common.sh"

ci_buildkit_cache_ref() {
  local registry="${ASKPDF_CI_REGISTRY:-ghcr.io}"
  local owner_lc repo_lc
  owner_lc="$(echo "${GITHUB_REPOSITORY_OWNER:-local}" | tr '[:upper:]' '[:lower:]')"
  repo_lc="$(echo "${GITHUB_REPOSITORY:-local/askpdf}" | tr '[:upper:]' '[:lower:]')"
  repo_lc="${repo_lc#*/}"
  local branch="${ASKPDF_BUILDKIT_CACHE_BRANCH:-main}"
  printf '%s/%s/%s/ci/buildkit-cache:%s' "$registry" "$owner_lc" "$repo_lc" "$branch"
}

ci_export_buildkit_cache_env() {
  export ASKPDF_BUILDKIT_CACHE_REF="${ASKPDF_BUILDKIT_CACHE_REF:-$(ci_buildkit_cache_ref)}"
}

ci_log_buildkit_cache_status() {
  ci_export_buildkit_cache_env
  echo "BuildKit registry cache ref: ${ASKPDF_BUILDKIT_CACHE_REF}"
  if [ -n "${ASKPDF_BUILDX_CACHE_DIR:-}" ] && [ -d "${ASKPDF_BUILDX_CACHE_DIR}" ]; then
    echo "Legacy GHA buildx dir size: $(du -sh "${ASKPDF_BUILDX_CACHE_DIR}" 2>/dev/null | awk '{print $1}')"
  fi
  if command -v docker >/dev/null 2>&1; then
    docker buildx version
    docker buildx inspect --bootstrap 2>/dev/null | head -5 || true
  fi
}

# Populates CI_BUILDKIT_CACHE_FROM / CI_BUILDKIT_CACHE_TO for docker buildx build.
CI_BUILDKIT_CACHE_FROM=()
CI_BUILDKIT_CACHE_TO=()

ci_refresh_buildkit_cache_args() {
  local local_dir="${ASKPDF_BUILDX_CACHE_DIR:-}"
  local local_new="${1:-}"
  ci_export_buildkit_cache_env
  CI_BUILDKIT_CACHE_FROM=(--cache-from "type=registry,ref=${ASKPDF_BUILDKIT_CACHE_REF}")
  CI_BUILDKIT_CACHE_TO=(
    --cache-to "type=registry,ref=${ASKPDF_BUILDKIT_CACHE_REF},mode=max,compression=zstd,ignore-error=true"
  )
  if [ -n "$local_dir" ] && [ -n "$(ls -A "$local_dir" 2>/dev/null || true)" ]; then
    CI_BUILDKIT_CACHE_FROM+=(--cache-from "type=local,src=${local_dir}")
  fi
  if [ -n "$local_new" ]; then
    CI_BUILDKIT_CACHE_TO+=(--cache-to "type=local,dest=${local_new},mode=max")
  fi
}
#!/usr/bin/env bash
# Upload shared benchmark PDFs into a running askPDF stack and wait for indexing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export ASKPDF_INCLUDE_HOLDOUT="${ASKPDF_INCLUDE_HOLDOUT:-0}"
export ASKPDF_BENCHMARK_THREAD_ID="${ASKPDF_BENCHMARK_THREAD_ID:-}"

if [[ -z "${ASKPDF_ADMIN_TOKEN:-}" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
  ASKPDF_ADMIN_TOKEN="$(grep -E '^ASKPDF_ADMIN_TOKEN=' "${REPO_ROOT}/.env" | head -1 | cut -d= -f2- | tr -d '\"' | tr -d "'" || true)"
  export ASKPDF_ADMIN_TOKEN
fi

exec python3 "${REPO_ROOT}/scripts/upload_benchmark_fixtures.py"
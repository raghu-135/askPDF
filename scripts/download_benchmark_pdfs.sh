#!/usr/bin/env bash
# Download pinned arXiv benchmark PDFs into git-ignored .benchmark-pdf-cache (verified sha256).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${REPO_ROOT}/scripts/benchmark_papers.py" "$@"
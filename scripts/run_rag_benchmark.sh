#!/usr/bin/env bash
# On-demand RAG benchmark (not part of default CI). Requires a running dev stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export ASKPDF_BENCHMARK_MODEL="${ASKPDF_BENCHMARK_MODEL:-openai/gpt-4.1-mini}"
export ASKPDF_BENCHMARK_JUDGE_MODEL="${ASKPDF_BENCHMARK_JUDGE_MODEL:-google/gemini-2.5-flash}"

python3 "${REPO_ROOT}/scripts/benchmark_papers.py"
exec python3 benchmarks/run_benchmark.py --judge "$@"
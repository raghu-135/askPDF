#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export ASKPDF_BENCHMARK_MODEL="${ASKPDF_BENCHMARK_MODEL:-inclusionai/ling-3.2-flash:free}"
export ASKPDF_EVAL_WORKFLOW="${ASKPDF_EVAL_WORKFLOW:-router_rag_agent}"
export ASKPDF_EVAL_MIN_INTERVAL_S="${ASKPDF_EVAL_MIN_INTERVAL_S:-3.5}"

API="${ASKPDF_BENCHMARK_API:-${ASKPDF_API_URL:-http://127.0.0.1:8000/api}}"
export ASKPDF_API_URL="${API}"

if ! curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "askPDF API not reachable; start the stack first (scripts/cloud-agent-setup.sh stack)." >&2
  exit 1
fi

python3 -c "import sys; sys.path.insert(0, '${REPO_ROOT}/scripts'); from benchmark_papers import paper_path; paper_path('1706.03762')"
mkdir -p "${REPO_ROOT}/evals/promptfoo/output"

if [[ -n "${ASKPDF_EVAL_REUSE_THREAD:-}" ]]; then
  export ASKPDF_EVAL_THREAD_ID="${ASKPDF_EVAL_REUSE_THREAD}"
else
export ASKPDF_EVAL_THREAD_ID="$(
  python3 -c "
import os, sys, time
from pathlib import Path
REPO = Path('${REPO_ROOT}')
sys.path.insert(0, str(REPO / 'scripts'))
from benchmark_config import load_benchmark_config, thread_settings_baseline
from benchmark_papers import fixture_path, indexing_timeout_seconds
from live_stack_client import create_thread, live_client, upload_pdf, wait_indexed

api = os.environ['ASKPDF_API_URL']
wf = os.environ['ASKPDF_EVAL_WORKFLOW']
baseline = thread_settings_baseline(load_benchmark_config())
with live_client(api_base=api) as client:
    tid = create_thread(client, f'promptfoo-poc-{int(time.time())}', workflow_id=wf, settings_patch=baseline)
    path = fixture_path('1706.03762.pdf')
    fh = upload_pdf(client, tid, path)
    wait_indexed(client, thread_id=tid, file_hash=fh, timeout=indexing_timeout_seconds('1706.03762.pdf'))
    print(tid)
"
)"
fi

echo "promptfoo POC thread=${ASKPDF_EVAL_THREAD_ID} workflow=${ASKPDF_EVAL_WORKFLOW} model=${ASKPDF_BENCHMARK_MODEL}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  while IFS= read -r line; do
    case "$line" in
      OPENROUTER_API_KEY=*|OPENAI_API_KEY=*|ASKPDF_ADMIN_TOKEN=*)
        export "$line"
        ;;
    esac
  done <"${REPO_ROOT}/.env"
fi

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY (or OPENAI_API_KEY in .env) required for llm-rubric judge." >&2
  exit 1
fi

cd "${REPO_ROOT}/evals/promptfoo"
exec npx --yes promptfoo@latest eval -c promptfooconfig.poc.yaml "$@"
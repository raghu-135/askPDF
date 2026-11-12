#!/usr/bin/env bash
# Minimal browser + API smoke proof for Cloud Agent VMs (requires stack up).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ART_DIR="${ASKPDF_E2E_ARTIFACTS:-/opt/cursor/artifacts}"
mkdir -p "$ART_DIR"
BASE_URL="${ASKPDF_E2E_BASE_URL:-http://127.0.0.1:3000}"
API_URL="${ASKPDF_E2E_API_URL:-http://127.0.0.1:8000}"
MODEL="${ASKPDF_E2E_LLM_MODEL:-openai/gpt-4.1-nano}"
PDF="${ASKPDF_E2E_PDF:-$REPO_ROOT/rag_service/tests/01030000000000.pdf}"

set -a
# shellcheck disable=SC1091
source .env
set +a

ADMIN="${ASKPDF_ADMIN_TOKEN:?ASKPDF_ADMIN_TOKEN required in .env}"
AUTH=( -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" )

playwright_screenshot() {
  local out="$1"
  local path="${2:-/}"
  (cd frontend && ASKPDF_E2E_BASE_URL="$BASE_URL" node --input-type=module -e "
    import { chromium } from '@playwright/test';
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.goto('${BASE_URL}${path}', { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: '${ART_DIR}/${out}', fullPage: false });
    await browser.close();
  ")
}

echo "[e2e] app screenshot..."
playwright_screenshot "ui-home.png" "/"

project_id=$(curl -sf "${API_URL}/api/projects" "${AUTH[@]}" \
  -d '{"name":"cloud-agent-proof","description":"vm"}' \
  -X POST | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

thread_id=$(curl -sf "${API_URL}/api/projects/${project_id}/threads" "${AUTH[@]}" \
  -d '{"title":"proof"}' -X POST | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

upload=$(curl -sf "${API_URL}/api/projects/${project_id}/files" \
  -H "Authorization: Bearer ${ADMIN}" \
  -F "file=@${PDF}")
file_id=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$upload")

curl -sf "${API_URL}/api/threads/${thread_id}/files" "${AUTH[@]}" \
  -d "{\"file_id\":\"${file_id}\"}" -X POST >/dev/null

echo "[e2e] langgraph chat (cited answer)..."
lg=$(curl -sf "${API_URL}/api/threads/${thread_id}/messages" "${AUTH[@]}" -d "{
  \"content\": \"What is this document about? Cite the PDF.\",
  \"runtime_framework\": \"langgraph\",
  \"llm_model\": \"${MODEL}\"
}" -X POST)
python3 -c "import json,sys; m=json.load(sys.stdin); print('langgraph_answer_len', len(m.get('content',''))); print(m.get('content','')[:240])" <<<"$lg"

playwright_screenshot "langgraph-thread.png" "/"

echo "[e2e] hermes chat..."
hermes=$(curl -sf "${API_URL}/api/threads/${thread_id}/messages" "${AUTH[@]}" -d "{
  \"content\": \"Summarize the PDF in one sentence with a citation.\",
  \"runtime_framework\": \"hermes\",
  \"llm_model\": \"${MODEL}\"
}" -X POST)
python3 -c "import json,sys; m=json.load(sys.stdin); print('hermes_answer_len', len(m.get('content',''))); print(m.get('content','')[:240])" <<<"$hermes"

echo "[e2e] deep research → awaiting_approval (langgraph)..."
task=$(curl -sf "${API_URL}/api/threads/${thread_id}/agent-tasks" "${AUTH[@]}" -d "{
  \"objective\": \"List at least 10 limitations of the methodology in the attached PDF with citations.\",
  \"runtime_framework\": \"langgraph\",
  \"llm_model\": \"${MODEL}\",
  \"file_ids\": [\"${file_id}\"]
}" -X POST)
task_id=$(python3 -c "import json,sys; print(json.load(sys.stdin)['id'])" <<<"$task")

for _ in $(seq 1 90); do
  st=$(curl -sf "${API_URL}/api/agent-tasks/${task_id}" -H "Authorization: Bearer ${ADMIN}" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  if [[ "$st" == "awaiting_approval" ]]; then echo "awaiting_approval"; break; fi
  if [[ "$st" == "failed" || "$st" == "cancelled" || "$st" == "completed" ]]; then echo "terminal:$st"; break; fi
  sleep 2
done

playwright_screenshot "deep-research-review.png" "/"

curl -sf "${API_URL}/api/agent-tasks/${task_id}/cancel" "${AUTH[@]}" -d '{}' -X POST >/dev/null || true
for _ in $(seq 1 60); do
  st=$(curl -sf "${API_URL}/api/agent-tasks/${task_id}" -H "Authorization: Bearer ${ADMIN}" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  if [[ "$st" == "cancelled" ]]; then echo "cancelled"; break; fi
  sleep 2
done

playwright_screenshot "deep-research-cancelled.png" "/"

echo "[e2e] artifacts in ${ART_DIR}"

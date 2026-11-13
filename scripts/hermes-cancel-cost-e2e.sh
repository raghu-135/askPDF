#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ART_DIR="${ASKPDF_E2E_ARTIFACTS:-/opt/cursor/artifacts}"
mkdir -p "$ART_DIR"
API_URL="${ASKPDF_E2E_API_URL:-http://127.0.0.1:8000}"
BASE_URL="${ASKPDF_E2E_BASE_URL:-http://127.0.0.1:3000}"
MODEL="${ASKPDF_E2E_LLM_MODEL:-openai/gpt-4.1-nano}"
UNIQ="${ASKPDF_E2E_UNIQ:-$(date +%s)}"

set -a
# shellcheck disable=SC1091
source .env
set +a
ADMIN="${ASKPDF_ADMIN_TOKEN:?ASKPDF_ADMIN_TOKEN required}"
CONTEXT="${HERMES_MODEL_CONTEXT_LENGTH:-64000}"

playwright_screenshot() {
  local out="$1"
  local path="$2"
  (cd frontend && ASKPDF_E2E_BASE_URL="$BASE_URL" node --input-type=module -e "
    import { chromium } from '@playwright/test';
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.goto('${BASE_URL}${path}', { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForTimeout(4000);
    await page.screenshot({ path: '${ART_DIR}/${out}', fullPage: false });
    await browser.close();
  ")
}

create_thread() {
  local name="$1"
  local project_id
  project_id=$(curl -sf "${API_URL}/api/projects" \
    -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
    -d "{\"name\":\"${name}\",\"description\":\"cancel/cost e2e\",\"embedding_model\":\"BAAI/bge-m3\"}" -X POST \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  curl -sf "${API_URL}/api/projects/${project_id}/threads" \
    -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
    -d '{"name":"hermes-e2e"}' -X POST \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])"
}

create_and_start_hermes_task() {
  local thread_id="$1"
  local objective="$2"
  local key="$3"
  local created
  local body
  body=$(OBJECTIVE="$objective" MODEL="$MODEL" CONTEXT="$CONTEXT" python3 -c 'import json, os; print(json.dumps({"definition_id":"hermes_rag_agent","objective":os.environ["OBJECTIVE"],"llm_model":os.environ["MODEL"],"context_window":int(os.environ["CONTEXT"]),"web_search_mode":"off"}))')
  created=$(curl -sf "${API_URL}/api/threads/${thread_id}/agent-tasks" \
    -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
    -H "Idempotency-Key: ${key}" \
    -d "$body" \
    -X POST)
  local task_id version
  task_id=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['id'])" <<<"$created")
  version=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['version'])" <<<"$created")
  curl -sf "${API_URL}/api/agent-tasks/${task_id}/commands/start?thread_id=${thread_id}" \
    -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
    -H "Idempotency-Key: ${key}-start" \
    -d "{\"expected_version\": ${version}}" -X POST >/dev/null
  echo "$task_id"
}

wait_task_terminal() {
  local thread_id="$1"
  local task_id="$2"
  local out_file="$3"
  for _ in $(seq 1 120); do
    local body
    body=$(curl -sf "${API_URL}/api/agent-tasks/${task_id}?thread_id=${thread_id}" \
      -H "Authorization: Bearer ${ADMIN}")
    local st
    st=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['status'])" <<<"$body")
    if [[ "$st" == "completed" || "$st" == "cancelled" || "$st" == "failed" || "$st" == "awaiting_approval" ]]; then
      echo "$body" > "$out_file"
      python3 -c "import json,sys; t=json.load(sys.stdin)['task']; print(t['status'], t.get('terminal_reason'), (t.get('active_run') or {}).get('id',''))" <<<"$body"
      return 0
    fi
    sleep 1
  done
  return 1
}

accept_incomplete_result_if_needed() {
  local thread_id="$1"
  local task_id="$2"
  local body
  body=$(curl -sf "${API_URL}/api/agent-tasks/${task_id}?thread_id=${thread_id}" \
    -H "Authorization: Bearer ${ADMIN}")
  local st ver run_id interrupt_id
  st=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['status'])" <<<"$body")
  if [[ "$st" != "awaiting_approval" ]]; then
    return 0
  fi
  read -r ver run_id interrupt_id < <(python3 -c "import json,sys; t=json.load(sys.stdin)['task']; i=(t.get('active_run') or {}).get('pending_interrupt') or {}; print(t['version'], (t.get('active_run') or {}).get('id',''), i.get('interrupt_id',''))" <<<"$body")
  curl -sf "${API_URL}/api/agent-tasks/${task_id}/result-review/responses?thread_id=${thread_id}" \
    -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
    -H "Idempotency-Key: hermes-cost-accept-${UNIQ}" \
    -d "{\"run_id\":\"${run_id}\",\"interrupt_id\":\"${interrupt_id}\",\"decision\":\"accept\",\"expected_version\": ${ver}}" \
    -X POST >/dev/null
  wait_task_terminal "$thread_id" "$task_id" "$ART_DIR/hermes-cost-task.json" >/dev/null
}

echo "[e2e] Hermes short task for provider cost trace..."
thread_id=$(create_thread "hermes-cost-${UNIQ}")
cost_task_id=$(create_and_start_hermes_task "$thread_id" "Reply with exactly PROOF." "hermes-cost-${UNIQ}")
read -r cost_status _ cost_run_id < <(wait_task_terminal "$thread_id" "$cost_task_id" "$ART_DIR/hermes-cost-task.json")
accept_incomplete_result_if_needed "$thread_id" "$cost_task_id"
if [[ -z "$cost_run_id" ]]; then
  cost_run_id=$(python3 -c "import json; t=json.load(open('${ART_DIR}/hermes-cost-task.json')); print((t.get('task') or {}).get('primary_run_id') or (t.get('task') or {}).get('active_run_id') or '')")
fi

if [[ -n "$cost_run_id" ]]; then
  payload=$(curl -sf "${API_URL}/api/agent-runs/${cost_run_id}?thread_id=${thread_id}" -H "Authorization: Bearer ${ADMIN}")
  python3 - "$payload" "$ART_DIR/hermes-cost-trace.json" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
run = data.get("agent_run") if isinstance(data.get("agent_run"), dict) else data
debug = run.get("debug") or {}
trace = debug.get("trace") or {}
metrics = trace.get("metrics") or {}
models = debug.get("models") or trace.get("models") or []
model_costs = []
for row in models:
    usage = (row.get("payload") or row).get("usage") or {}
    for key in ("cost", "provider_cost_usd"):
        if usage.get(key) is not None:
            model_costs.append(float(usage[key]))
            break
out = {
    "run_id": run.get("id"),
    "status": run.get("status"),
    "llm_provider_cost_usd": metrics.get("llm_provider_cost_usd"),
    "model_costs": model_costs,
    "model_count": len(models),
}
print(json.dumps(out, indent=2))
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
PY
  playwright_screenshot "hermes-debug-trace-provider-cost.png" "/threads/${thread_id}/runs/${cost_run_id}/debug"
fi

echo "[e2e] cancel during hermes run..."
cancel_thread=$(create_thread "hermes-cancel-${UNIQ}")
cancel_task_id=$(create_and_start_hermes_task "$cancel_thread" "List ten imaginary risks and stop before finishing." "hermes-cancel-${UNIQ}")
sleep 2
ver=$(curl -sf "${API_URL}/api/agent-tasks/${cancel_task_id}?thread_id=${cancel_thread}" -H "Authorization: Bearer ${ADMIN}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['task']['version'])")
curl -sf "${API_URL}/api/agent-tasks/${cancel_task_id}/commands/cancel?thread_id=${cancel_thread}" \
  -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
  -H "Idempotency-Key: hermes-cancel-${UNIQ}" \
  -d "{\"expected_version\": ${ver}}" -X POST >/dev/null
read -r cancel_status cancel_reason _ < <(wait_task_terminal "$cancel_thread" "$cancel_task_id" "$ART_DIR/hermes-cancel-task.json")
echo "cancel_terminal=${cancel_status} reason=${cancel_reason}"

playwright_screenshot "hermes-e2e-home.png" "/"
echo "[e2e] artifacts in ${ART_DIR}"

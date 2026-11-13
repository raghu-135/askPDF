#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ART_DIR="${ASKPDF_E2E_ARTIFACTS:-/opt/cursor/artifacts}"
mkdir -p "$ART_DIR"
API_URL="${ASKPDF_E2E_API_URL:-http://127.0.0.1:8000}"
MODEL="${ASKPDF_E2E_LLM_MODEL:-openai/gpt-4.1-nano}"
PDF="${ASKPDF_E2E_PDF:-/tmp/attention.pdf}"
LOG="$ART_DIR/verification-leftovers-e2e.jsonl"

set -a
# shellcheck disable=SC1091
source .env
set +a
ADMIN="${ASKPDF_ADMIN_TOKEN:?ASKPDF_ADMIN_TOKEN required}"

log_line() {
  python3 -c "import json,sys; print(json.dumps(sys.argv[1:], ensure_ascii=False))" "$@" >>"$LOG"
}

api() {
  local method="$1"
  local path="$2"
  shift 2
  curl -sf -X "$method" "${API_URL}${path}" \
    -H "Authorization: Bearer ${ADMIN}" \
    -H "Content-Type: application/json" \
    "$@"
}

create_thread_with_pdf() {
  local project_id
  project_id=$(api POST /api/projects -d '{"name":"verification-leftovers","description":"e2e","embedding_model":"BAAI/bge-m3"}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  local thread_id
  thread_id=$(api POST "/api/projects/${project_id}/threads" -d '{"name":"leftovers"}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  local upload
  upload=$(curl -sf -X POST "${API_URL}/api/threads/${thread_id}/files/upload" \
    -H "Authorization: Bearer ${ADMIN}" \
    -F "file=@${PDF}")
  local file_hash
  file_hash=$(python3 -c "import json,sys; print(json.load(sys.stdin)['file_hash'])" <<<"$upload")
  for _ in $(seq 1 180); do
  local status
    status=$(curl -sf "${API_URL}/api/threads/${thread_id}/files/${file_hash}/status" \
      -H "Authorization: Bearer ${ADMIN}" \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" || echo "")
    if [[ "$status" == "ready" ]]; then
      break
    fi
    sleep 3
  done
  echo "$thread_id"
}

set_workflow() {
  local thread_id="$1"
  local workflow="$2"
  api PUT "/api/threads/${thread_id}/settings" -d "{\"agent_workflow\":{\"workflow_id\":\"${workflow}\"}}"
}

run_chat() {
  local thread_id="$1"
  local question="$2"
  local started
  started=$(date +%s%3N)
  local payload
  payload=$(THREAD_ID="$thread_id" QUESTION="$question" MODEL="$MODEL" python3 -c 'import json,os; print(json.dumps({"thread_id":os.environ["THREAD_ID"],"question":os.environ["QUESTION"],"llm_model":os.environ["MODEL"],"use_web_search":False,"use_reranker":True,"context_window":64000}))')
  local body
  body=$(api POST "/api/threads/${thread_id}/chat" -d "$payload")
  local elapsed=$(( $(date +%s%3N) - started ))
  local answer
  answer=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('answer') or json.load(sys.stdin).get('final_answer') or '')" <<<"$body" 2>/dev/null || true)
  if [[ -z "$answer" ]]; then
    answer=$(python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('assistant_message') or {}).get('content',''))" <<<"$body")
  fi
  printf '%s\n%s\n' "$elapsed" "$answer"
}

THREAD_ID=$(create_thread_with_pdf)
log_line "thread_id" "$THREAD_ID" "model" "$MODEL"

for workflow in router_rag_agent orchestrator_worker_rag_agent; do
  set_workflow "$THREAD_ID" "$workflow"
  mapfile -t OOS < <(run_chat "$THREAD_ID" "What is the population of Tokyo?")
  log_line "out_of_scope" "$workflow" "${OOS[0]}" "${OOS[1]}"
  mapfile -t INS < <(run_chat "$THREAD_ID" "For the Transformer (base model) in Table 2, what is the EN-FR training cost in FLOPs? Reply with the value only.")
  log_line "table2_enfr" "$workflow" "${INS[0]}" "${INS[1]}"
done

HERMES_BODY=$(OBJECTIVE="Say hello in one short sentence." MODEL="$MODEL" python3 -c 'import json,os; print(json.dumps({"definition_id":"hermes_rag_agent","objective":os.environ["OBJECTIVE"],"llm_model":os.environ["MODEL"],"context_window":32000,"web_search_mode":"off"}))')
TASK=$(curl -sf -X POST "${API_URL}/api/threads/${THREAD_ID}/agent-tasks" \
  -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
  -H "Idempotency-Key: verify-cost-${RANDOM}" \
  -d "$HERMES_BODY")
TASK_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['id'])" <<<"$TASK")
VERSION=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['version'])" <<<"$TASK")
curl -sf -X POST "${API_URL}/api/agent-tasks/${TASK_ID}/commands/start?thread_id=${THREAD_ID}" \
  -H "Authorization: Bearer ${ADMIN}" -H "Content-Type: application/json" \
  -H "Idempotency-Key: verify-cost-start-${RANDOM}" \
  -d "{\"expected_version\": ${VERSION}}" >/dev/null
for _ in $(seq 1 90); do
  TASK_BODY=$(curl -sf "${API_URL}/api/agent-tasks/${TASK_ID}?thread_id=${THREAD_ID}" -H "Authorization: Bearer ${ADMIN}")
  STATUS=$(python3 -c "import json,sys; print(json.load(sys.stdin)['task']['status'])" <<<"$TASK_BODY")
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then
    echo "$TASK_BODY" >"$ART_DIR/hermes-cost-task.json"
    python3 -c "import json,sys; t=json.load(sys.stdin)['task']; m=((t.get('active_run') or {}).get('trace') or {}).get('metrics') or {}; print(m.get('llm_provider_cost_usd'))" <<<"$TASK_BODY" | tee "$ART_DIR/hermes-provider-cost.txt"
    break
  fi
  sleep 2
done

echo "Wrote $LOG"

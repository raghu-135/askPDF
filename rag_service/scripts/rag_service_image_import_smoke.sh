#!/usr/bin/env bash
# Build the production rag-service image and verify projection imports work without langgraph_runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${ASKPDF_RAG_IMAGE_SMOKE_TAG:-askpdf-rag-service-smoke:local}"

echo "Building rag-service image: ${IMAGE}"
docker build -f "${ROOT}/rag_service/Dockerfile" -t "${IMAGE}" "${ROOT}"

ENV_FILE_ARGS=()
if [ -f "${ROOT}/.env" ]; then
  ENV_FILE_ARGS=(--env-file "${ROOT}/.env")
elif [ -f "${ROOT}/.env.example" ]; then
  ENV_FILE_ARGS=(--env-file "${ROOT}/.env.example")
fi

# Dummy values satisfy module-level config reads during import; no DB/network is used.
SMOKE_ENV=(
  -e DATABASE_URL=postgresql+asyncpg://smoke:smoke@127.0.0.1:5432/smoke
  -e TEST_DATABASE_URL=postgresql+asyncpg://smoke:smoke@127.0.0.1:5432/smoke_test
  -e POSTGRES_POOL_SIZE=1
  -e POSTGRES_MAX_OVERFLOW=0
  -e DEFAULT_TOKEN_BUDGET=8192
  -e REPLANS_LIMIT=10
  -e LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
  -e LOCAL_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
  -e EMBEDDING_DEVICE=cpu
  -e RERANKER_DEVICE=cpu
)

echo "Running projection import smoke inside image (network isolated)..."
docker run --rm --network none "${ENV_FILE_ARGS[@]}" "${SMOKE_ENV[@]}" "${IMAGE}" python -c "
from app.services.agent_task_runtime_projection import _filter_interrupt_gaps, _scrub_review_warnings

gaps = ['Document index is not ready for this thread.']
filtered = _filter_interrupt_gaps(
    gaps,
    'Mara Quill (zephyr.pdf, page 1)',
    {'gaps': gaps},
)
assert filtered == [], filtered

warnings = [{
    'code': 'evidence_critic_issues',
    'details': {'issues': ['document index not ready', 'Unique finding']},
}]
scrubbed = _scrub_review_warnings(warnings, gaps)
issues = scrubbed[0]['details']['issues']
assert issues == ['Unique finding'], issues
print('rag_service_projection_import_smoke: ok')
"

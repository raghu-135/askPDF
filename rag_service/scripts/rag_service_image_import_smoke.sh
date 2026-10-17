#!/usr/bin/env bash
# Build the production rag-service image and verify projection imports work without langgraph_runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${ASKPDF_RAG_IMAGE_SMOKE_TAG:-askpdf-rag-service-smoke:local}"

echo "Building rag-service image: ${IMAGE}"
docker build -f "${ROOT}/rag_service/Dockerfile" -t "${IMAGE}" "${ROOT}"

echo "Running projection import smoke inside image..."
docker run --rm "${IMAGE}" python -c "
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

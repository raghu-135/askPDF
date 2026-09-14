from types import SimpleNamespace

import pytest

from app.services import document_projection_service
from app.tools.context import ToolInvocationContext
from app.tools.contracts import SearchKnowledgeRequest
from app.tools import retrieval_knowledge


@pytest.mark.asyncio
async def test_readiness_does_not_consider_an_unrelated_published_manifest(monkeypatch):
    canonical = SimpleNamespace(generation="current-generation", status="completed")
    manifest = SimpleNamespace(
        status="completed",
        vector_status="completed",
        published_at=object(),
        superseded_at=None,
        generation="current-generation",
        expected_chunk_count=1,
        expected_source_ids=["current-source"],
        vector_count=0,
        manifest_id="current-manifest",
    )

    class Repo:
        async def get(self, _file_hash):
            return canonical

        async def get_ready_manifest(self, *_args, **_kwargs):
            raise AssertionError("readiness must not select a fallback manifest")

    class VectorDb:
        async def has_file_indexed_chunks(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(document_projection_service, "get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr(document_projection_service, "ensure_retrieval_projection", lambda **_kwargs: _projection(manifest))
    monkeypatch.setattr("app.db.vector.get_vector_db", lambda: VectorDb())

    readiness = await document_projection_service.evaluate_retrieval_readiness("file-a", "model-a")

    assert readiness["ready"] is False
    assert "fallback_ready" not in readiness


async def _projection(manifest):
    return manifest, [{"chunk_id": "chunk-a"}], {"repair_source_version": "repair-a"}


def test_canonical_schema_version_invalidates_extraction_fingerprint(monkeypatch):
    try:
        from app.services import document_conversion_service
    except ImportError as exc:
        pytest.skip(f"Docling runtime is unavailable in this host environment: {exc}")
    original = document_conversion_service.current_extraction_fingerprint(b"fixture-pdf")
    monkeypatch.setattr(document_conversion_service, "CANONICAL_SCHEMA_VERSION", "docling-canonical-v-next")
    changed = document_conversion_service.current_extraction_fingerprint(b"fixture-pdf")
    assert changed != original


@pytest.mark.asyncio
async def test_search_enqueues_repair_and_reports_indexing_in_progress(monkeypatch):
    queued = []

    async def readiness(_file_hash, _model):
        return {"ready": False, "reason": "manifest_incomplete", "repair_source_version": "repair-a"}

    async def enqueue(**kwargs):
        queued.append(kwargs)

    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-a": {}}

    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)
    monkeypatch.setattr("app.services.embedding_materialization_service.ensure_embedding_job", enqueue)

    result = await retrieval_knowledge.search_knowledge(
        SearchKnowledgeRequest(query="what is this?"),
        ToolInvocationContext(thread_id="thread-a", embedding_model="model-a"),
        services=Services(),
    )

    assert result.ok is True
    assert "indexing_in_progress" in result.warnings
    assert result.artifacts["repair_scheduled"] is True
    assert queued == [{
        "resource_type": "document",
        "resource_id": "file-a",
        "scope_id": "thread-a",
        "embedding_model": "model-a",
        "source_version": "repair-a",
        "requeue_completed": True,
    }]

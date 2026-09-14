from types import SimpleNamespace

import pytest

from app.services import document_projection_service
from app.tools.context import ToolInvocationContext
from app.tools.contracts import SearchKnowledgeRequest
from app.tools import retrieval_knowledge


@pytest.mark.asyncio
async def test_readiness_does_not_consider_an_unrelated_published_manifest(monkeypatch):
    canonical = SimpleNamespace(
        file_hash="file-a",
        generation="current-generation",
        extraction_fingerprint="extract-a",
        docling_version="unknown",
        status="completed",
        source_metadata_json={
            "_file_hash": "file-a",
            "_extraction_contract_fingerprint": document_projection_service._current_extraction_contract_fingerprint(),
            "_extraction_pipeline_version": "docling-pdf-v2",
        },
        document_json={
            "schema_version": "docling-canonical-v1",
            "docling": {},
            "elements": [],
            "sections": [],
        },
    )
    config = SimpleNamespace(fingerprint="tokenizer-a", identity="tokenizer-a", revision=None, effective_input_limit=512)
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
        chunking_fingerprint=document_projection_service.retrieval_chunking_fingerprint(canonical, "model-a", config.fingerprint),
    )

    class Repo:
        async def get(self, _file_hash):
            return canonical

        async def get_manifest(self, *_args, **_kwargs):
            return manifest

    class VectorDb:
        async def has_file_indexed_chunks(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(document_projection_service, "get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr(document_projection_service, "resolve_embedding_tokenizer", lambda _model: (config, None))
    monkeypatch.setattr("app.db.vector.get_vector_db", lambda: VectorDb())

    readiness = await document_projection_service.evaluate_retrieval_readiness("file-a", "model-a")

    assert readiness["ready"] is False
    assert "fallback_ready" not in readiness
    assert readiness["reason"] == "manifest_incomplete"


@pytest.mark.asyncio
async def test_readiness_uses_persisted_manifest_without_materializing(monkeypatch):
    canonical = SimpleNamespace(
        file_hash="file-a",
        generation="current-generation",
        extraction_fingerprint="extract-a",
        docling_version="unknown",
        status="completed",
        source_metadata_json={
            "_file_hash": "file-a",
            "_extraction_contract_fingerprint": document_projection_service._current_extraction_contract_fingerprint(),
            "_extraction_pipeline_version": "docling-pdf-v2",
        },
        document_json={"schema_version": "docling-canonical-v1", "docling": {}, "elements": [], "sections": []},
    )
    config = SimpleNamespace(fingerprint="tokenizer-a", identity="tokenizer-a", revision=None, effective_input_limit=512)
    expected = ["source-a"]
    manifest = SimpleNamespace(
        status="completed",
        vector_status="completed",
        published_at=object(),
        superseded_at=None,
        generation=canonical.generation,
        expected_chunk_count=1,
        expected_source_ids=expected,
        vector_count=1,
        manifest_id="manifest-a",
        chunking_fingerprint=document_projection_service.retrieval_chunking_fingerprint(canonical, "model-a", config.fingerprint),
        file_hash="file-a",
    )

    class Repo:
        async def get(self, _file_hash):
            return canonical

        async def get_manifest(self, *_args, **_kwargs):
            return manifest

    class VectorDb:
        async def has_file_indexed_chunks(self, *_args, **_kwargs):
            return True

    async def fail_materialization(**_kwargs):
        raise AssertionError("readiness must not materialize a projection")

    monkeypatch.setattr(document_projection_service, "get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr(document_projection_service, "resolve_embedding_tokenizer", lambda _model: (config, None))
    monkeypatch.setattr(document_projection_service, "ensure_retrieval_projection", fail_materialization)
    monkeypatch.setattr("app.db.vector.get_vector_db", lambda: VectorDb())

    readiness = await document_projection_service.evaluate_retrieval_readiness("file-a", "model-a")

    assert readiness["ready"] is True
    assert readiness["manifest"].manifest_id == "manifest-a"


@pytest.mark.asyncio
async def test_stale_reading_cache_is_rejected_before_serving(monkeypatch):
    canonical = SimpleNamespace(
        file_hash="file-a",
        generation="generation-current",
        extraction_fingerprint="extract-current",
        docling_version="unknown",
        status="completed",
        source_metadata_json={
            "_file_hash": "file-a",
            "_extraction_contract_fingerprint": document_projection_service._current_extraction_contract_fingerprint(),
            "_extraction_pipeline_version": "docling-pdf-v2",
        },
        document_json={"schema_version": "docling-canonical-v1", "docling": {}, "elements": [], "sections": []},
    )
    monkeypatch.setattr(document_projection_service, "get_canonical_document_repo", lambda: SimpleNamespace(get=lambda _hash: _get_value(canonical)))
    monkeypatch.setattr(
        document_projection_service,
        "get_file_parsed_sentences",
        lambda _hash: _get_value({"version": "2.0", "generation": "generation-old", "extraction_fingerprint": "extract-old", "sentences": []}),
    )

    freshness = await document_projection_service.evaluate_document_freshness("file-a", require_reading=True)

    assert freshness["canonical_ready"] is True
    assert freshness["reading_ready"] is False
    assert freshness["reason"] == "reading_stale"


async def _get_value(value):
    return value


def test_canonical_schema_version_invalidates_extraction_fingerprint(monkeypatch):
    try:
        from app.services import document_conversion_service
    except ImportError as exc:
        pytest.skip(f"Docling runtime is unavailable in this host environment: {exc}")
    original = document_conversion_service.current_extraction_fingerprint(b"fixture-pdf")
    monkeypatch.setattr(document_conversion_service, "CANONICAL_SCHEMA_VERSION", "docling-canonical-v-next")
    changed = document_conversion_service.current_extraction_fingerprint(b"fixture-pdf")
    assert changed != original


def test_page_normalization_expands_compact_ranges():
    assert retrieval_knowledge._normalise_pages("12-14") == [12, 13, 14]
    assert retrieval_knowledge._normalise_pages(["2", "4-5"]) == [2, 4, 5]


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


@pytest.mark.asyncio
async def test_ready_empty_search_reports_no_relevant_content(monkeypatch):
    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-a": {}}

        async def embed(self, _model, _query):
            return [0.1]

        async def rerank(self, _query, raw):
            return raw

        def vector_db(self):
            class VectorDb:
                async def search_knowledge_sources(self, **_kwargs):
                    return []
            return VectorDb()

    manifest = SimpleNamespace(file_hash="file-a", generation="generation-a", manifest_id="manifest-a")
    async def readiness(_file_hash, _model):
        return {"ready": True, "manifest": manifest, "metadata": {}, "repair_source_version": "repair-a"}

    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)
    result = await retrieval_knowledge.search_knowledge(
        SearchKnowledgeRequest(query="missing page", filters={"pages": [99]}),
        ToolInvocationContext(thread_id="thread-a", embedding_model="model-a"),
        services=Services(),
    )

    assert result.ok is True
    assert result.warnings == ["no_relevant_content"]
    assert "indexing_in_progress" not in result.warnings


@pytest.mark.asyncio
async def test_document_discovery_uses_file_identity_and_title(monkeypatch):
    class CanonicalRepo:
        async def get(self, _file_hash):
            return SimpleNamespace(
                source_metadata_json={"original_title": "Canonical title.pdf"},
                document_json={"filename": "fallback.pdf"},
            )

    class File:
        file_name = "Displayed title.pdf"

    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-a": {}}

        async def embed(self, _model, _query):
            return [0.1]

        async def rerank(self, _query, raw):
            return raw

        def vector_db(self):
            class VectorDb:
                async def search_knowledge_sources(self, **_kwargs):
                    return [{
                        "file_hash": "file-a",
                        "score": 0.9,
                        "text": "context",
                        "metadata": {"section_id": "section-a", "pages": "12-14", "source_element_ids": ["element-a"]},
                    }]
            return VectorDb()

    monkeypatch.setattr(retrieval_knowledge, "get_canonical_document_repo", lambda: CanonicalRepo())
    monkeypatch.setattr("app.db.get_file", lambda _file_hash: _get_file(File()))
    async def readiness(_file_hash, _model):
        return {"ready": True, "manifest": SimpleNamespace(file_hash="file-a", generation="g", manifest_id="m"), "metadata": {}}
    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)

    result = await retrieval_knowledge.search_knowledge(
        SearchKnowledgeRequest(query="context", level="document"),
        ToolInvocationContext(thread_id="thread-a", embedding_model="model-a"),
        services=Services(),
    )

    assert result.sources[0]["source_id"] == "file-a"
    assert result.sources[0]["title"] == "Displayed title.pdf"
    assert result.sources[0]["pages"] == [12, 13, 14]


async def _get_file(value):
    return value

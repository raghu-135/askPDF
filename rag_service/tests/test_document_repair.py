from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import document_projection_service
from app.services import embedding_materialization_service
from app.services.document_pipeline import CANONICAL_SCHEMA_VERSION, EXTRACTION_PIPELINE_VERSION
from app.tools.context import ToolInvocationContext
from app.tools.contracts import SearchKnowledgeRequest
from app.tools import retrieval_knowledge


@pytest.fixture(autouse=True)
def _test_sentence_pipeline_identity(monkeypatch):
    monkeypatch.setattr(
        document_projection_service,
        "sentence_pipeline_identity",
        lambda: "en_core_web_sm:test-double",
    )


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
            "_extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
        },
        document_json={
            "schema_version": CANONICAL_SCHEMA_VERSION,
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
        extraction_fingerprint=canonical.extraction_fingerprint,
        source_version=document_projection_service.retrieval_source_version(
            canonical.file_hash, canonical, "model-a",
            document_projection_service.retrieval_chunking_fingerprint(canonical, "model-a", config.fingerprint),
        ),
        is_current=False,
    )

    class Repo:
        async def get(self, _file_hash):
            return canonical

        async def get_manifest(self, *_args, **_kwargs):
            return manifest

        async def get_chunks(self, *_args, **_kwargs):
            return []

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
async def test_missing_tokenizer_fails_before_repair_scheduling(monkeypatch):
    from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError

    def fail_tokenizer(_model):
        raise EmbeddingTokenizerUnavailableError("configure tokenizer")

    async def fail_if_called(**_kwargs):
        raise AssertionError("tokenizer failure must not schedule an embedding job")

    monkeypatch.setattr(
        document_projection_service,
        "resolve_embedding_tokenizer",
        fail_tokenizer,
    )
    monkeypatch.setattr(
        "app.services.embedding_materialization_service.ensure_embedding_job",
        fail_if_called,
    )

    with pytest.raises(EmbeddingTokenizerUnavailableError, match="configure tokenizer"):
        await document_projection_service.evaluate_retrieval_readiness(
            "file-a", "model-a", thread_id="thread-a"
        )


def test_retrieval_source_version_changes_with_manifest_inputs():
    canonical = SimpleNamespace(
        generation="generation-a",
        extraction_fingerprint="extraction-a",
    )
    first = document_projection_service.retrieval_source_version(
        "file-a", canonical, "model-a", "chunking-a"
    )
    assert first == document_projection_service.retrieval_source_version(
        "file-a", canonical, "model-a", "chunking-a"
    )
    assert first != document_projection_service.retrieval_source_version(
        "file-a", canonical, "model-a", "chunking-b"
    )


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
            "_extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
        },
        document_json={"schema_version": CANONICAL_SCHEMA_VERSION, "docling": {}, "elements": [], "sections": []},
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
        expected_chunk_ids=["chunk-a"],
        expected_source_ids=expected,
        vector_count=1,
        manifest_id="manifest-a",
        embedding_model="model-a",
        chunking_fingerprint=document_projection_service.retrieval_chunking_fingerprint(canonical, "model-a", config.fingerprint),
        file_hash="file-a",
        extraction_fingerprint=canonical.extraction_fingerprint,
        source_version=document_projection_service.retrieval_source_version(
            canonical.file_hash, canonical, "model-a",
            document_projection_service.retrieval_chunking_fingerprint(canonical, "model-a", config.fingerprint),
        ),
        is_current=True,
    )

    class Repo:
        async def get(self, _file_hash):
            return canonical

        async def get_manifest(self, *_args, **_kwargs):
            return manifest

        async def get_chunks(self, *_args, **_kwargs):
            return [SimpleNamespace(
                chunk_id="chunk-a",
                source_id="source-a",
                manifest_id="manifest-a",
                file_hash="file-a",
                embedding_model="model-a",
                generation=canonical.generation,
            )]

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
            "_extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
        },
        document_json={"schema_version": CANONICAL_SCHEMA_VERSION, "docling": {}, "elements": [], "sections": []},
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


def test_manifest_validation_rejects_equal_count_rows_from_another_manifest():
    manifest = SimpleNamespace(
        manifest_id="new-manifest",
        file_hash="file-a",
        generation="generation-a",
        extraction_fingerprint="extraction-a",
        chunking_fingerprint="chunking-a",
        source_version="source-a",
        expected_chunk_count=2,
        expected_chunk_ids=["new-1", "new-2"],
        expected_source_ids=["src-new-1", "src-new-2"],
    )
    rows = [
        SimpleNamespace(chunk_id="old-1", source_id="src-old-1", manifest_id="old-manifest", file_hash="file-a", embedding_model="model-a"),
        SimpleNamespace(chunk_id="old-2", source_id="src-old-2", manifest_id="old-manifest", file_hash="file-a", embedding_model="model-a"),
    ]
    canonical = SimpleNamespace(file_hash="file-a", generation="generation-a", extraction_fingerprint="extraction-a")
    assert document_projection_service._manifest_rows_complete(manifest, rows, canonical, "model-a") is False


def test_canonical_schema_version_invalidates_extraction_fingerprint(monkeypatch):
    try:
        from app.services import document_conversion_service
    except ImportError as exc:
        pytest.skip(f"Docling runtime is unavailable in this host environment: {exc}")
    monkeypatch.setattr(
        document_conversion_service,
        "sentence_pipeline_identity",
        lambda: "en_core_web_sm:test-double",
    )
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

    async def readiness(_file_hash, _model, **_kwargs):
        return {"ready": False, "canonical_ready": True, "reason": "manifest_incomplete", "source_version": "repair-a", "repair_source_version": "repair-a"}

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
async def test_document_embedding_waits_for_conversion_without_charging_retry(monkeypatch):
    job = SimpleNamespace(
        id="job-1",
        resource_type="document",
        resource_id="file-a",
        scope_id="thread-a",
        embedding_model="model-a",
        source_version="conversion-version",
    )
    pending = document_projection_service.DocumentConversionPendingError("file-a")
    freshness = {"canonical_ready": False, "source_version": None}
    monkeypatch.setattr(document_projection_service, "evaluate_document_freshness", AsyncMock(return_value=freshness))
    monkeypatch.setattr(document_projection_service, "ensure_retrieval_projection", AsyncMock(side_effect=pending))
    deferred = AsyncMock()
    monkeypatch.setattr(embedding_materialization_service, "defer_document_embedding_job", deferred)
    await embedding_materialization_service.process_embedding_job(job)
    deferred.assert_awaited_once_with(job, reason="waiting for canonical document conversion")


@pytest.mark.asyncio
async def test_document_embedding_refreshes_stale_thread_version_without_indexing(monkeypatch):
    job = SimpleNamespace(
        id="job-1",
        resource_type="document",
        resource_id="file-a",
        scope_id="thread-a",
        embedding_model="model-a",
        source_version="old-version",
    )
    monkeypatch.setattr(
        document_projection_service,
        "evaluate_document_freshness",
        AsyncMock(return_value={"canonical_ready": True, "source_version": "new-version"}),
    )
    deferred = AsyncMock()
    monkeypatch.setattr(embedding_materialization_service, "defer_document_embedding_job", deferred)
    await embedding_materialization_service.process_embedding_job(job)
    deferred.assert_awaited_once_with(
        job,
        source_version="new-version",
        reason="document retrieval target refreshed",
    )


@pytest.mark.asyncio
async def test_ready_empty_search_reports_no_relevant_content(monkeypatch):
    captured = {}
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
                    captured.update(_kwargs)
                    return []
            return VectorDb()

    manifest = SimpleNamespace(file_hash="file-a", generation="generation-a", manifest_id="manifest-a")
    async def readiness(_file_hash, _model, **_kwargs):
        return {"ready": True, "canonical_ready": True, "manifest": manifest, "metadata": {}, "source_version": "ready-a", "repair_source_version": "ready-a"}

    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)
    result = await retrieval_knowledge.search_knowledge(
        SearchKnowledgeRequest(query="Which section refers to page 10?"),
        ToolInvocationContext(thread_id="thread-a", embedding_model="model-a"),
        services=Services(),
    )

    assert result.ok is True
    assert result.warnings == ["no_relevant_content"]
    assert "indexing_in_progress" not in result.warnings
    assert result.artifacts["readiness"] == "ready"
    assert captured["pages"] == []


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
                        "metadata": {
                            "section_id": "section-a",
                            "pages": "12-14",
                            "source_element_ids": ["element-a"],
                            "generation": "g",
                            "manifest_id": "m",
                            "extraction_fingerprint": "x",
                        },
                    }]
            return VectorDb()

    monkeypatch.setattr(retrieval_knowledge, "get_canonical_document_repo", lambda: CanonicalRepo())
    monkeypatch.setattr("app.db.get_file", lambda _file_hash: _get_file(File()))
    async def readiness(_file_hash, _model, **_kwargs):
        return {"ready": True, "canonical_ready": True, "manifest": SimpleNamespace(file_hash="file-a", generation="g", manifest_id="m"), "metadata": {}, "source_version": "ready-a"}
    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)

    result = await retrieval_knowledge.search_knowledge(
        SearchKnowledgeRequest(query="context", level="document"),
        ToolInvocationContext(thread_id="thread-a", embedding_model="model-a"),
        services=Services(),
    )

    assert result.sources[0]["source_id"] == "file-a"
    assert result.sources[0]["title"] == "Displayed title.pdf"
    assert result.sources[0]["pages"] == [12, 13, 14]


@pytest.mark.asyncio
async def test_reconcile_enqueues_missing_thread_version_pointer(monkeypatch):
    captured = {}
    queued = []

    async def readiness(_file_hash, _model, **kwargs):
        captured["thread_id"] = kwargs.get("thread_id")
        if kwargs.get("thread_id"):
            return {
                "ready": False,
                "canonical_ready": True,
                "source_version": "version-a",
                "reason": "thread_version_missing",
            }
        return {"ready": True, "canonical_ready": True, "source_version": "version-a"}

    async def enqueue(**kwargs):
        queued.append(kwargs)

    monkeypatch.setattr(embedding_materialization_service, "require_embedding_model_ready", AsyncMock())
    monkeypatch.setattr(
        "app.db.get_effective_thread_files",
        AsyncMock(return_value=[SimpleNamespace(file_hash="file-a", file_name="paper.pdf")]),
    )
    monkeypatch.setattr(document_projection_service, "evaluate_retrieval_readiness", readiness)
    monkeypatch.setattr(embedding_materialization_service, "ensure_embedding_job", enqueue)
    monkeypatch.setattr(
        embedding_materialization_service,
        "get_vector_db",
        lambda: SimpleNamespace(has_chat_memory_indexed=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr("app.db.get_thread_turns", AsyncMock(return_value=[]))
    monkeypatch.setattr(embedding_materialization_service, "async_session_maker", _empty_sessionmaker())

    counts = await embedding_materialization_service.reconcile_thread_embedding_targets("thread-a", "model-a")

    assert captured["thread_id"] == "thread-a"
    assert queued == [{
        "resource_type": "document",
        "resource_id": "file-a",
        "scope_id": "thread-a",
        "embedding_model": "model-a",
        "source_version": "version-a",
        "requeue_completed": True,
    }]
    assert counts["documents"] == 1


def _empty_sessionmaker():
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return self

        async def execute(self, *_args, **_kwargs):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

    def maker():
        return Session()

    return maker


@pytest.mark.asyncio
async def test_background_index_enqueues_embedding_job_instead_of_indexing(monkeypatch):
    from app.services import file_processing_service

    queued = []
    indexed = []

    monkeypatch.setattr(file_processing_service, "_enqueue_pdf_conversion", AsyncMock())
    monkeypatch.setattr(file_processing_service, "update_indexing_status", AsyncMock())
    monkeypatch.setattr(
        document_projection_service,
        "evaluate_retrieval_readiness",
        AsyncMock(return_value={"ready": False, "canonical_ready": True, "source_version": "version-a"}),
    )
    monkeypatch.setattr(
        "app.services.embedding_materialization_service.ensure_embedding_job",
        AsyncMock(side_effect=lambda **kwargs: queued.append(kwargs)),
    )
    monkeypatch.setattr(
        file_processing_service,
        "index_document_for_thread",
        AsyncMock(side_effect=lambda **kwargs: indexed.append(kwargs) or {"status": "success"}),
    )

    await file_processing_service._background_index("file-a", "thread-a", "model-a", "paper.pdf", "")

    assert indexed == []
    assert queued == [{
        "resource_type": "document",
        "resource_id": "file-a",
        "scope_id": "thread-a",
        "embedding_model": "model-a",
        "source_version": "version-a",
        "requeue_completed": True,
    }]


@pytest.mark.asyncio
async def test_queue_file_processing_enqueues_browser_capture_conversion(monkeypatch):
    from app.db import FileSourceType
    from app.services import file_processing_service

    enqueued = []
    monkeypatch.setattr(file_processing_service, "create_or_get_file", AsyncMock())
    monkeypatch.setattr(file_processing_service, "add_file_to_thread", AsyncMock())
    monkeypatch.setattr(
        file_processing_service,
        "_enqueue_pdf_conversion",
        AsyncMock(side_effect=lambda file_hash, file_name: enqueued.append((file_hash, file_name))),
    )
    monkeypatch.setattr(file_processing_service, "get_file_status", AsyncMock(return_value={"parsing": {"status": "pending"}, "indexing": {"status": "pending"}}))
    monkeypatch.setattr(file_processing_service, "update_indexing_status", AsyncMock())
    monkeypatch.setattr(file_processing_service, "update_parsing_status", AsyncMock())
    monkeypatch.setattr(file_processing_service, "get_file_parsed_sentences", AsyncMock(return_value=None))
    monkeypatch.setattr(file_processing_service, "get_canonical_document_repo", lambda: SimpleNamespace(get=AsyncMock(return_value=None)))
    monkeypatch.setattr(document_projection_service, "evaluate_document_freshness", AsyncMock(return_value={"canonical_ready": False, "reading_ready": False}))
    monkeypatch.setattr("app.db.get_scoped_indexing_status", lambda *_args, **_kwargs: {"status": "pending"})

    await file_processing_service.queue_file_processing(
        background_tasks=SimpleNamespace(add_task=lambda *_args, **_kwargs: None),
        thread=SimpleNamespace(id="thread-a", embedding_model="model-a"),
        file_hash="capture-a",
        file_name="Captured page",
        source_type=FileSourceType.BROWSER.value,
    )

    assert enqueued == [("capture-a", "Captured page")]


@pytest.mark.asyncio
async def test_ready_manifest_lookup_requires_source_version():
    from app.db.repositories.canonical_document_repo import CanonicalDocumentRepository

    repo = CanonicalDocumentRepository()
    with pytest.raises(ValueError, match="nonempty source_version"):
        await repo.get_ready_manifest("file-a", "model-a", "generation-a", "chunk-a", "")


async def _get_file(value):
    return value

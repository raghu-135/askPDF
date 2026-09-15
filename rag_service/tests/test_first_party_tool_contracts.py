from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.agent import external_research_tools
from app.agent.tool_contract import ToolWarningCode, normalize_tool_result
from app.tools.context import ToolInvocationContext
from app.tools.contracts import DocumentSearchRequest, InspectDocumentRequest, ReadContextRequest, SearchKnowledgeRequest, TimelineRequest
from app.tools.retrieval_conversation import search_thread_conversation_history as neutral_history
from app.tools.retrieval_knowledge import (
    _context_cursor,
    _context_cursor_payload,
    _source_from_chunk,
    inspect_document,
    read_context,
    search_knowledge as neutral_knowledge,
)
from app.tools.retrieval_timeline import search_thread_events as neutral_events
from app.tools.thread_shape import invoke_thread_shape
from app.tools.thread_shape import ThreadShapeRequest
from app.services.document_pipeline import TokenCounter
from app.services.document_projection_service import DocumentConversionPendingError


def _config(**overrides):
    configurable = {
        "agent_run_id": "run-1",
        "thread_id": "thread-1",
        "embedding_model": "embed-1",
        "caller_node": "test_node",
        "route": "document",
        "use_reranker": False,
        "web_search_index": False,
    }
    configurable.update(overrides)
    return {"configurable": configurable}


def _assert_contract(payload, *, tool_name: str, caller_node: str = "test_node", artifact_keys=(), warning=None):
    assert isinstance(payload["content"], str)
    assert payload["trace"]["tool_name"] == tool_name
    assert payload["trace"]["agent_run_id"] == "run-1"
    assert payload["trace"]["thread_id"] == "thread-1"
    assert payload["trace"]["caller_node"] == caller_node
    assert isinstance(payload["metrics"]["elapsed_ms"], (int, float))
    assert payload["metrics"]["result_chars"] == len(payload["content"])
    assert payload["metrics"]["warning_count"] == len(payload["warnings"])
    for key in artifact_keys:
        assert key in payload["artifacts"]
    if warning:
        assert warning in payload["warnings"]


def _context(**overrides):
    values = _config(**overrides)["configurable"]
    values["run_id"] = values.get("agent_run_id")
    values["caller_node_type"] = values.get("caller_node")
    return ToolInvocationContext.from_mapping(values)


def _patch_ready_manifest(monkeypatch):
    async def ready(file_hash, _embedding_model, **_kwargs):
        return {
            "ready": True,
            "canonical_ready": True,
            "source_version": f"ready-{file_hash}",
            "manifest": SimpleNamespace(
                file_hash=file_hash,
                manifest_id=f"manifest-{file_hash}",
                generation="generation-1",
                source_version=f"ready-{file_hash}",
                extraction_fingerprint="extract-1",
                chunking_fingerprint="chunk-1",
            ),
        }

    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_retrieval_readiness",
        ready,
    )


@pytest.mark.asyncio
async def test_get_thread_shape_returns_tool_contract(monkeypatch):
    import app.db as db_module

    monkeypatch.setattr(
        db_module,
        "get_thread_shape",
        AsyncMock(
            return_value={
                "total_qa_pairs": 2,
                "avg_qa_chars": 120,
                "total_qa_chars": 240,
                "documents": {
                    "file-1": {
                        "file_name": "paper.pdf",
                        "source_type": "pdf",
                        "chunk_count": 3,
                        "total_chars": 1200,
                        "indexing_status": "completed",
                    }
                },
            }
        ),
    )

    raw = await invoke_thread_shape(ThreadShapeRequest(), _context())
    payload = normalize_tool_result(raw.to_json(), tool_name="get_thread_shape")

    assert payload["ok"] is True
    _assert_contract(payload, tool_name="get_thread_shape", artifact_keys=("thread_shape",))
    assert payload["artifacts"]["thread_shape"]["total_qa_pairs"] == 2


@pytest.mark.asyncio
async def test_search_knowledge_returns_sources_and_artifacts_contract(monkeypatch):
    _patch_ready_manifest(monkeypatch)
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(
            return_value=[{
                "file_hash": "file-1", "chunk_id": 1, "source_id": "src-1", "score": 0.9,
                "text": "seed", "manifest_id": "manifest-file-1", "generation": "generation-1",
                "metadata": {"extraction_fingerprint": "extract-1", "chunking_fingerprint": "chunk-1"},
            }]
        ),
    )
    class Services:
        async def embed(self, _model, _query): return [0.1, 0.2, 0.3]
        def vector_db(self): return fake_db
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf", "source_type": "pdf"}}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_knowledge(
        SearchKnowledgeRequest(query="diffusion", max_results=5),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_knowledge")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_knowledge",
        caller_node="retrieval_worker",
        artifact_keys=("document_sources", "matches"),
    )
    assert fake_db.search_knowledge_sources.call_args.kwargs["embedding_model"] == "embed-1"
    assert fake_db.search_knowledge_sources.call_args.kwargs["filters"]["manifest_ids"] == ["manifest-file-1"]
    assert payload["artifacts"]["matches"][0]["source_id"] == "src-1"


@pytest.mark.asyncio
async def test_search_knowledge_chunk_level_returns_body_text_over_structural_context(monkeypatch):
    _patch_ready_manifest(monkeypatch)
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(
            return_value=[{
                "file_hash": "file-1",
                "chunk_id": 1,
                "source_id": "src-1",
                "score": 0.9,
                "text": "Document section: Introduction\\nstructural context",
                "manifest_id": "manifest-file-1",
                "generation": "generation-1",
                "metadata": {
                    "body_text": "The paper evaluates evidence-grounded research artifacts.",
                    "pages": [3],
                    "extraction_fingerprint": "extract-1",
                    "chunking_fingerprint": "chunk-1",
                },
            }]
        ),
    )

    class Services:
        async def embed(self, _model, _query): return [0.1, 0.2, 0.3]
        def vector_db(self): return fake_db
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf"}}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_knowledge(
        SearchKnowledgeRequest(query="evidence", level="chunk", max_results=1),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_knowledge")

    assert payload["ok"] is True
    assert "The paper evaluates evidence-grounded research artifacts." in payload["content"]
    assert "Document section: Introduction" not in payload["content"]


@pytest.mark.asyncio
async def test_search_knowledge_enforces_document_ownership(monkeypatch):
    _patch_ready_manifest(monkeypatch)
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(return_value=[{
            "file_hash": "owned", "chunk_id": 0, "source_id": "src-owned", "text": "seed",
            "manifest_id": "manifest-owned", "generation": "generation-1",
            "metadata": {"extraction_fingerprint": "extract-1", "chunking_fingerprint": "chunk-1"},
        }]),
    )
    class Services:
        async def embed(self, _model, _query): return [0.1, 0.2]
        def vector_db(self): return fake_db
        async def document_lookup(self, _thread_id): return {"owned": {"file_name": "paper.pdf", "source_type": "pdf"}}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_knowledge(
        SearchKnowledgeRequest(query="focused", document_id="owned", max_results=5),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_knowledge")
    assert payload["ok"] is True
    _assert_contract(payload, tool_name="search_knowledge", caller_node="retrieval_worker", artifact_keys=("document_sources",))
    assert fake_db.search_knowledge_sources.call_args.kwargs["file_hash"] == "owned"

    unowned = await neutral_knowledge(
        SearchKnowledgeRequest(query="focused", document_id="not-owned"),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    unowned_payload = normalize_tool_result(unowned.to_json(), tool_name="search_knowledge")
    assert unowned_payload["ok"] is False
    assert unowned_payload["error"]["code"] == "document_scope_forbidden"


@pytest.mark.asyncio
async def test_search_knowledge_does_not_query_unpublished_materialization(monkeypatch):
    readiness = AsyncMock(return_value={"ready": False, "canonical_ready": True, "reason": "manifest_incomplete", "source_version": "repair-file-1"})
    monkeypatch.setattr("app.services.document_projection_service.evaluate_retrieval_readiness", readiness)
    monkeypatch.setattr("app.services.embedding_materialization_service.ensure_embedding_job", AsyncMock())
    embed = AsyncMock(return_value=[0.1, 0.2])
    search = AsyncMock(return_value=[])

    class Services:
        async def embed(self, model, query): return await embed(model, query)
        def vector_db(self): return SimpleNamespace(search_knowledge_sources=search)
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf"}}

    raw = await neutral_knowledge(
        SearchKnowledgeRequest(query="unpublished", max_results=1),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_knowledge")

    assert payload["ok"] is True
    assert "missing_document_vectors" in payload["warnings"]
    embed.assert_not_awaited()
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_context_expands_descendant_sections_and_uses_large_budget(monkeypatch):
    body = " ".join(f"word-{index}" for index in range(700))
    chunk = SimpleNamespace(
        source_id="src-chunk",
        chunk_id="chunk-1",
        manifest_id="manifest-file-1",
        chunk_order=0,
        file_hash="file-1",
        embedding_model="embed-1",
        generation="generation-1",
        section_id="child",
        table_id=None,
        body_text=body,
        contextualized_text=body,
        page_start=1,
        page_end=1,
        sentence_ids=["sentence-1"],
        source_element_ids=["element-1"],
        metadata_json={
                "pages": [1],
                "generation": "generation-1",
                "extraction_fingerprint": "extract-1",
                "chunking_fingerprint": "chunk-1",
                "source_version": "ready-file-1",
        },
    )
    canonical = SimpleNamespace(
        status="completed", generation="generation-1", extraction_fingerprint="extract-1"
    )
    repo = SimpleNamespace(
        get=AsyncMock(return_value=canonical),
        get_chunks=AsyncMock(side_effect=[[chunk], [chunk]]),
        get_chunks_by_source_id=AsyncMock(return_value=[]),
        get_sections=AsyncMock(return_value=[
            SimpleNamespace(section_id="root", parent_section_id=None),
            SimpleNamespace(section_id="child", parent_section_id="root"),
        ]),
        get_descendant_section_ids=AsyncMock(return_value=["root", "child"]),
        get_manifest_by_id=AsyncMock(return_value=SimpleNamespace(
            manifest_id="manifest-file-1",
            source_version="ready-file-1",
        )),
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)

    async def fresh_document(*_args, **_kwargs):
        return {
            "canonical_ready": True,
            "manifest_ready": True,
            "ready": True,
            "source_version": "ready-file-1",
            "manifest": SimpleNamespace(manifest_id="manifest-file-1", source_version="ready-file-1"),
        }

    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_retrieval_readiness",
        fresh_document,
    )
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (
                SimpleNamespace(fingerprint="tokenizer-1"),
            TokenCounter(
                count=lambda value: len(str(value).split()),
                split=lambda value, limit: [" ".join(str(value).split()[index:index + limit]) for index in range(0, len(str(value).split()), limit)],
            ),
        ),
    )

    class Services:
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf"}}

    raw = await read_context(
        ReadContextRequest(source_id="root", expansion="section", token_budget=800),
        _context(), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="read_context")

    assert payload["ok"] is True
    assert payload["artifacts"]["token_count"] == 700
    assert payload["artifacts"]["truncated"] is False
    assert repo.get_chunks.await_args.kwargs["section_ids"] == {"root", "child"}


@pytest.mark.asyncio
async def test_read_context_reserves_hit_before_section_expansion(monkeypatch):
    def make_chunk(order, source_id, text):
        return SimpleNamespace(
            source_id=source_id,
            chunk_id=f"chunk-{order}",
            manifest_id="manifest-file-1",
            file_hash="file-1",
            embedding_model="embed-1",
            section_id="section-1",
            table_id=None,
            chunk_order=order,
            body_text=text,
            contextualized_text=text,
            page_start=1,
            page_end=1,
            sentence_ids=[f"sentence-{order}"],
            source_element_ids=[f"element-{order}"],
            metadata_json={
                "pages": [1],
                "generation": "generation-1",
                "extraction_fingerprint": "extract-1",
                "chunking_fingerprint": "chunk-1",
                "source_version": "ready-file-1",
            },
        )

    chunks = [
        make_chunk(0, "src-before", "before evidence"),
        make_chunk(1, "src-hit", "requested evidence"),
        make_chunk(2, "src-after", "after evidence"),
    ]
    repo = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(
            status="completed", generation="generation-1", extraction_fingerprint="extract-1"
        )),
        get_chunks_by_source_id=AsyncMock(return_value=[chunks[1]]),
        get_chunks=AsyncMock(return_value=chunks),
        get_sections=AsyncMock(return_value=[]),
        get_descendant_section_ids=AsyncMock(return_value=["section-1"]),
        get_manifest_by_id=AsyncMock(return_value=SimpleNamespace(
            manifest_id="manifest-file-1",
            source_version="ready-file-1",
        )),
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)
    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_retrieval_readiness",
        AsyncMock(return_value={
            "canonical_ready": True,
            "manifest_ready": True,
            "ready": True,
            "source_version": "ready-file-1",
            "manifest": SimpleNamespace(manifest_id="manifest-file-1", source_version="ready-file-1"),
        }),
    )
    monkeypatch.setattr(
        "app.services.embedding_materialization_service.get_document_embedding_job",
        AsyncMock(return_value=SimpleNamespace(status="completed", source_version="ready-file-1")),
    )
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (
                SimpleNamespace(fingerprint="tokenizer-1"),
            TokenCounter(
                count=lambda value: len(str(value).split()),
                split=lambda value, limit: [" ".join(str(value).split()[index:index + limit]) for index in range(0, len(str(value).split()), limit)],
            ),
        ),
    )

    class Services:
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf"}}

    raw = await read_context(
        ReadContextRequest(source_id="src-hit", expansion="section", token_budget=20),
        _context(), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="read_context")

    assert payload["ok"] is True
    assert payload["content"].startswith("requested evidence")
    assert payload["artifacts"]["document_sources"][0]["evidence_role"] == "evidence"
    assert all(source["manifest_id"] == "manifest-file-1" for source in payload["artifacts"]["document_sources"])


def test_search_knowledge_rejects_path_and_url_identifiers():
    for value in ("../secret.pdf", "/tmp/file", "https://example.com/file"):
        with pytest.raises(ValidationError):
            SearchKnowledgeRequest(query="q", document_id=value)


def test_citation_requires_generation_manifest_and_extraction_identity():
    with pytest.raises(ValueError, match="citation metadata is incomplete"):
        _source_from_chunk({
            "file_hash": "file-1",
            "source_id": "src-1",
            "manifest_id": "manifest-file-1",
            "text": "evidence",
            "metadata": {"extraction_fingerprint": "extract-1"},
        })


def test_vector_citation_preserves_validated_provenance_metadata():
    source = _source_from_chunk({
        "file_hash": "file-1",
        "source_id": "src-1",
        "generation": "generation-1",
        "manifest_id": "manifest-file-1",
        "text": "evidence",
        "metadata": {
            "body_text": "evidence",
            "generation": "generation-1",
            "manifest_id": "manifest-file-1",
            "extraction_fingerprint": "extract-1",
            "chunking_fingerprint": "chunk-1",
            "source_element_ids": ["element-1"],
        },
    })
    assert source["generation"] == "generation-1"
    assert source["manifest_id"] == "manifest-file-1"
    assert source["extraction_fingerprint"] == "extract-1"
    assert source["chunking_fingerprint"] == "chunk-1"
    assert source["source_element_ids"] == ["element-1"]


def test_context_cursor_binds_budget_and_tokenizer_version():
    cursor = _context_cursor(
        manifest_id="manifest-1",
        generation="generation-1",
        source_version="source-1",
        anchor_chunk_id="chunk-1",
        expansion="section",
        token_budget=512,
        tokenizer_fingerprint="tokenizer-1",
        segment_index=3,
    )
    payload = _context_cursor_payload(cursor)
    assert payload["token_budget"] == 512
    assert payload["tokenizer_fingerprint"] == "tokenizer-1"
    assert payload["segment_index"] == 3


@pytest.mark.asyncio
async def test_inspect_document_keeps_outline_out_of_document_sources(monkeypatch):
    repo = SimpleNamespace(
        get_sections=AsyncMock(return_value=[
            SimpleNamespace(
                section_id="section-1",
                parent_section_id=None,
                title="Intro",
                level=1,
                heading_path=["Intro"],
                page_start=1,
                page_end=1,
            )
        ]),
        get_elements=AsyncMock(return_value=[SimpleNamespace(element_id="table-1", element_type="table")]),
        get_descendant_section_ids=AsyncMock(return_value=["section-1"]),
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)
    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_document_freshness",
        AsyncMock(return_value={
            "canonical_ready": True,
            "canonical": SimpleNamespace(generation="generation-1", status="completed"),
        }),
    )

    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-1": {"file_name": "paper.pdf"}}

    raw = await inspect_document(
        InspectDocumentRequest(document_id="file-1"),
        _context(),
        services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="inspect_document")
    document_sources = []
    from app.agent.tool_contract import collect_tool_sources
    collect_tool_sources(raw.to_json(), document_sources, [], [])

    assert payload["ok"] is True
    assert "document_sources" not in payload["artifacts"]
    assert payload["artifacts"]["outline"][0]["section_id"] == "section-1"
    assert document_sources == []


@pytest.mark.asyncio
async def test_read_context_does_not_enqueue_null_source_version(monkeypatch):
    queued = []

    async def enqueue(**kwargs):
        queued.append(kwargs)

    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_retrieval_readiness",
        AsyncMock(return_value={"ready": False, "canonical_ready": False, "source_version": None, "manifest": None}),
    )
    monkeypatch.setattr(
        "app.services.document_projection_service.ensure_retrieval_projection",
        AsyncMock(side_effect=DocumentConversionPendingError("file-1")),
    )
    monkeypatch.setattr("app.services.embedding_materialization_service.ensure_embedding_job", enqueue)
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (SimpleNamespace(fingerprint="tokenizer-1"), TokenCounter(count=lambda value: 1, split=lambda value, limit: [value])),
    )

    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-1": {"file_name": "paper.pdf"}}

    raw = await read_context(
        ReadContextRequest(source_id="src-hit"),
        _context(),
        services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="read_context")

    assert payload["ok"] is True
    assert "indexing_in_progress" in payload["warnings"]
    assert queued == []


@pytest.mark.asyncio
async def test_read_context_uses_selected_chunk_manifest_version(monkeypatch):
    def make_chunk(file_hash, manifest_id, source_id):
        return SimpleNamespace(
            source_id=source_id,
            chunk_id=f"{file_hash}-chunk",
            manifest_id=manifest_id,
            file_hash=file_hash,
            embedding_model="embed-1",
            section_id="section-1",
            table_id=None,
            chunk_order=0,
            body_text="selected evidence",
            contextualized_text="selected evidence",
            page_start=1,
            page_end=1,
            sentence_ids=["sentence-1"],
            source_element_ids=["element-1"],
            metadata_json={
                "pages": [1],
                "generation": "generation-1",
                "extraction_fingerprint": "extract-1",
                "chunking_fingerprint": "chunk-1",
                "source_version": "version-selected",
            },
        )

    selected = make_chunk("file-1", "manifest-file-1", "src-hit")
    other = make_chunk("file-2", "manifest-file-2", "src-other")

    async def get_chunks(file_hash, *_args, **_kwargs):
        return [selected] if file_hash == "file-1" else [other]

    async def get_chunks_by_source_id(source_id, *_args, **kwargs):
        if kwargs.get("file_hash") == "file-1" and source_id == "src-hit":
            return [selected]
        return []

    async def get_manifest_by_id(manifest_id):
        versions = {
            "manifest-file-1": "version-selected",
            "manifest-file-2": "version-other",
        }
        return SimpleNamespace(manifest_id=manifest_id, source_version=versions[manifest_id])

    repo = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(
            status="completed", generation="generation-1", extraction_fingerprint="extract-1"
        )),
        get_chunks=get_chunks,
        get_chunks_by_source_id=get_chunks_by_source_id,
        get_sections=AsyncMock(return_value=[]),
        get_descendant_section_ids=AsyncMock(return_value=[]),
        get_manifest_by_id=get_manifest_by_id,
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)

    async def readiness(file_hash, _model, **_kwargs):
        if file_hash == "file-1":
            return {
                "ready": True,
                "canonical_ready": True,
                "source_version": "version-selected",
                "manifest": SimpleNamespace(manifest_id="manifest-file-1", source_version="version-selected"),
            }
        return {
            "ready": True,
            "canonical_ready": True,
            "source_version": "version-other",
            "manifest": SimpleNamespace(manifest_id="manifest-file-2", source_version="version-other"),
        }

    monkeypatch.setattr("app.services.document_projection_service.evaluate_retrieval_readiness", readiness)
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (
            SimpleNamespace(fingerprint="tokenizer-1"),
            TokenCounter(count=lambda value: len(str(value).split()), split=lambda value, limit: [value]),
        ),
    )

    class Services:
        async def document_lookup(self, _thread_id):
            return {"file-1": {"file_name": "one.pdf"}, "file-2": {"file_name": "two.pdf"}}

    raw = await read_context(
        ReadContextRequest(source_id="src-hit"),
        _context(),
        services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="read_context")
    cursor = _context_cursor_payload(payload["artifacts"]["continuation"]) if payload["artifacts"].get("continuation") else {}

    assert payload["ok"] is True
    assert payload["content"] == "selected evidence"
    assert all(source["manifest_id"] == "manifest-file-1" for source in payload["artifacts"]["document_sources"])
    if cursor:
        assert cursor["source_version"] == "version-selected"
        assert cursor["manifest_id"] == "manifest-file-1"


@pytest.mark.asyncio
async def test_search_thread_conversation_history_returns_used_chat_ids_contract(monkeypatch):
    class Services:
        async def embed(self, _model, _query): return [0.4, 0.5]
        async def semantic_history(self, **_kwargs): return ("Q: earlier\nA: useful memory", ["turn-1:assistant"], [])
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_history(
        DocumentSearchRequest(query="earlier discussion", max_results=3),
        _context(caller_node="thread_conversation_history_worker", route="thread_conversation_history"),
        services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_thread_conversation_history")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_thread_conversation_history",
        caller_node="thread_conversation_history_worker",
        artifact_keys=("used_chat_ids",),
    )
    assert payload["artifacts"]["used_chat_ids"] == ["turn-1:assistant"]


@pytest.mark.asyncio
async def test_search_thread_events_returns_timeline_artifacts_contract(monkeypatch):
    fake_db = SimpleNamespace(
        search_chat_memory=AsyncMock(
            return_value=[
                {
                    "text": "Q: earlier\nA: memory",
                    "message_id": "turn-1:assistant",
                    "message_created_at": "2026-06-25T19:10:00Z",
                    "score": 0.7,
                }
            ]
        ),
        search_web_chunks=AsyncMock(return_value=[]),
    )

    class Services:
        def vector_db(self): return fake_db
        async def embed(self, _model, _query): return [0.1, 0.2]
        async def document_lookup(self, _thread_id): return {}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_events(
        TimelineRequest(query="timeline", sources="conversation", order="oldest", max_results=5),
        _context(caller_node="thread_events_worker", route="thread_events"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_thread_events")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_thread_events",
        caller_node="thread_events_worker",
        artifact_keys=("timeline_events", "evidence_segments"),
    )
    assert payload["artifacts"]["timeline_events"][0]["message_id"] == "turn-1:assistant"
    assert payload["artifacts"]["evidence_segments"][0]["source_id"] == "conversation:turn-1:assistant"
    assert fake_db.search_chat_memory.call_args.kwargs["embedding_model"] == "embed-1"


@pytest.mark.asyncio
async def test_search_web_returns_web_source_contract(monkeypatch):
    from app.mcp import tool_adapter

    class FakeClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "search_web", "description": "Web search", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "live_web_recon", "com.askpdf/contract-version": "1"}}]}
            assert method == "tools/call"
            assert params["name"] == "search_web"
            return {
                "content": [{"type": "text", "text": "fresh web evidence"}],
                "structuredContent": {
                    "ok": True,
                    "content": "fresh web evidence",
                    "sources": [{"url": "https://example.com", "title": "Example", "text": "fresh web evidence"}],
                        "artifacts": {
                            "web_sources": [{"url": "https://example.com", "title": "Example", "text": "fresh web evidence"}],
                            "evidence_segments": [{"source_id": "web:https://example.com/"}],
                        },
                        "warnings": [],
                        "error": None,
                        "metrics": {"elapsed_ms": 1.0, "result_chars": 18, "warning_count": 0},
                    "trace": {"tool_name": "search_web", "agent_run_id": "run-1", "thread_id": "thread-1", "caller_node": "web_worker"},
                },
                "isError": False,
            }

    monkeypatch.setattr(tool_adapter, "get_mcp_client", lambda: FakeClient())

    raw = await external_research_tools.search_web.ainvoke(
        {"query": "latest diffusion"},
        config=_config(caller_node="web_worker", route="web", use_web_search=True),
    )
    payload = normalize_tool_result(raw, tool_name="search_web")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_web",
        caller_node="web_worker",
        artifact_keys=("web_sources", "evidence_segments"),
    )
    assert payload["sources"] == payload["artifacts"]["web_sources"]
    assert payload["artifacts"]["evidence_segments"][0]["source_id"] == "web:https://example.com/"


@pytest.mark.asyncio
async def test_warning_paths_still_return_valid_tool_contracts(monkeypatch):
    monkeypatch.setattr("app.mcp.server.persist_tool_audit", AsyncMock())
    raw = await external_research_tools.search_web.ainvoke(
        {"query": "latest diffusion"},
        config=_config(caller_node="web_worker", route="web", use_web_search=False),
    )
    payload = normalize_tool_result(raw, tool_name="search_web")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_web",
        caller_node="web_worker",
        warning=ToolWarningCode.WEB_SEARCH_DISABLED,
    )

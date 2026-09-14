from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.agent import external_research_tools
from app.agent.tool_contract import ToolWarningCode, normalize_tool_result
from app.tools.context import ToolInvocationContext
from app.tools.contracts import DocumentSearchRequest, ReadContextRequest, SearchKnowledgeRequest, TimelineRequest
from app.tools.retrieval_conversation import search_thread_conversation_history as neutral_history
from app.tools.retrieval_knowledge import read_context, search_knowledge as neutral_knowledge
from app.tools.retrieval_timeline import search_thread_events as neutral_events
from app.tools.thread_shape import invoke_thread_shape
from app.tools.thread_shape import ThreadShapeRequest
from app.services.document_pipeline import TokenCounter


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
    async def ready(file_hash, _embedding_model):
        return {
            "ready": True,
            "manifest": SimpleNamespace(
                file_hash=file_hash,
                manifest_id=f"manifest-{file_hash}",
                generation="generation-1",
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
            return_value=[{"file_hash": "file-1", "chunk_id": 1, "score": 0.9, "text": "seed"}]
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
    assert payload["artifacts"]["matches"][0]["source_id"] == "1"


@pytest.mark.asyncio
async def test_search_knowledge_chunk_level_returns_body_text_over_structural_context(monkeypatch):
    _patch_ready_manifest(monkeypatch)
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(
            return_value=[{
                "file_hash": "file-1",
                "chunk_id": 1,
                "score": 0.9,
                "text": "Document section: Introduction\\nstructural context",
                "metadata": {"body_text": "The paper evaluates evidence-grounded research artifacts.", "pages": [3]},
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
        search_knowledge_sources=AsyncMock(return_value=[{"file_hash": "owned", "chunk_id": 0, "text": "seed"}]),
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
    readiness = AsyncMock(return_value={"ready": False, "reason": "manifest_incomplete"})
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
        metadata_json={"pages": [1], "generation": "generation-1"},
    )
    canonical = SimpleNamespace(status="completed", generation="generation-1")
    repo = SimpleNamespace(
        get=AsyncMock(return_value=canonical),
        get_chunks=AsyncMock(side_effect=[[chunk], [chunk]]),
        get_chunks_by_source_id=AsyncMock(return_value=[]),
        get_sections=AsyncMock(return_value=[
            SimpleNamespace(section_id="root", parent_section_id=None),
            SimpleNamespace(section_id="child", parent_section_id="root"),
        ]),
        get_descendant_section_ids=AsyncMock(return_value=["root", "child"]),
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)

    async def fresh_document(*_args, **_kwargs):
        return {
            "canonical_ready": True,
            "manifest_ready": True,
            "ready": True,
            "manifest": SimpleNamespace(manifest_id="manifest-file-1"),
        }

    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_document_freshness",
        fresh_document,
    )
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (
            SimpleNamespace(),
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
            metadata_json={"pages": [1]},
        )

    chunks = [
        make_chunk(0, "src-before", "before evidence"),
        make_chunk(1, "src-hit", "requested evidence"),
        make_chunk(2, "src-after", "after evidence"),
    ]
    repo = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status="completed", generation="generation-1")),
        get_chunks_by_source_id=AsyncMock(return_value=[chunks[1]]),
        get_chunks=AsyncMock(return_value=chunks),
        get_sections=AsyncMock(return_value=[]),
        get_descendant_section_ids=AsyncMock(return_value=["section-1"]),
    )
    monkeypatch.setattr("app.tools.retrieval_knowledge.get_canonical_document_repo", lambda: repo)
    monkeypatch.setattr(
        "app.services.document_projection_service.evaluate_document_freshness",
        AsyncMock(return_value={
            "canonical_ready": True,
            "manifest_ready": True,
            "manifest": SimpleNamespace(manifest_id="manifest-file-1"),
        }),
    )
    monkeypatch.setattr(
        "app.tools.retrieval_knowledge.resolve_embedding_tokenizer",
        lambda _model: (
            SimpleNamespace(),
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

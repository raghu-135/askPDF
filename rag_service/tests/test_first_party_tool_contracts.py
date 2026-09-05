from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.agent import external_research_tools
from app.agent.tool_contract import ToolWarningCode, normalize_tool_result
from app.tools.context import ToolInvocationContext
from app.tools.contracts import DocumentSearchRequest, FocusedDocumentSearchRequest, TimelineRequest
from app.tools.retrieval_conversation import search_thread_conversation_history as neutral_history
from app.tools.retrieval_documents import search_document_by_id as neutral_document_by_id
from app.tools.retrieval_documents import search_documents as neutral_documents
from app.tools.retrieval_timeline import search_thread_events as neutral_events
from app.tools.thread_shape import invoke_thread_shape
from app.tools.thread_shape import ThreadShapeRequest


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
async def test_search_documents_returns_sources_and_artifacts_contract(monkeypatch):
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(
            return_value=[{"file_hash": "file-1", "chunk_id": 1, "score": 0.9, "text": "seed"}]
        ),
        get_knowledge_source_chunks_by_ids=AsyncMock(
            return_value=[
                {
                    "file_hash": "file-1",
                    "chunk_id": 1,
                    "text": "Document evidence",
                    "score": 0.9,
                    "metadata": {"pages": "2", "page_start": 2, "page_end": 2},
                }
            ]
        ),
        search_web_chunks=AsyncMock(return_value=[{
            "url": "https://example.com/cached",
            "title": "Cached result",
            "text": "Previously fetched web evidence",
            "score": 0.8,
            "web_search_performed_at": "2026-08-01T12:00:00Z",
        }]),
    )
    class Services:
        async def embed(self, _model, _query): return [0.1, 0.2, 0.3]
        def vector_db(self): return fake_db
        async def document_lookup(self, _thread_id): return {"file-1": {"file_name": "paper.pdf", "source_type": "pdf"}}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_documents(
        DocumentSearchRequest(query="diffusion", max_results=5),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_documents")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_documents",
        caller_node="retrieval_worker",
        artifact_keys=("document_sources", "web_sources", "evidence_segments"),
    )
    assert payload["sources"] == [
        *payload["artifacts"]["document_sources"],
        *payload["artifacts"]["web_sources"],
    ]
    assert fake_db.search_knowledge_sources.call_args.kwargs["embedding_model"] == "embed-1"
    assert fake_db.get_knowledge_source_chunks_by_ids.call_args.kwargs["embedding_model"] == "embed-1"
    assert fake_db.search_web_chunks.call_args.kwargs["embedding_model"] == "embed-1"
    assert payload["artifacts"]["evidence_segments"][0]["source_id"] == "doc:file-1:1"
    assert payload["artifacts"]["evidence_segments"][0]["content"] == "Document evidence"
    cached = payload["artifacts"]["evidence_segments"][1]
    assert cached["source_id"] == "web:https://example.com/cached"
    assert cached["content"] == "Previously fetched web evidence"
    assert payload["artifacts"]["web_sources"][0]["web_search_performed_at"] == "2026-08-01T12:00:00Z"


@pytest.mark.asyncio
async def test_search_document_by_id_enforces_ownership_and_returns_bounded_sources(monkeypatch):
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(return_value=[{"file_hash": "owned", "chunk_id": 0, "text": "seed"}]),
        get_knowledge_source_chunks_by_ids=AsyncMock(return_value=[{
            "file_hash": "owned", "chunk_id": 0, "text": "Focused evidence",
            "metadata": {"page_start": 1, "page_end": 1},
        }]),
    )
    class Services:
        async def embed(self, _model, _query): return [0.1, 0.2]
        def vector_db(self): return fake_db
        async def document_lookup(self, _thread_id): return {"owned": {"file_name": "paper.pdf", "source_type": "pdf"}}
        async def rerank(self, _query, chunks): return chunks

    raw = await neutral_document_by_id(
        FocusedDocumentSearchRequest(query="focused", file_hash="owned", max_results=5),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    payload = normalize_tool_result(raw.to_json(), tool_name="search_document_by_id")
    assert payload["ok"] is True
    _assert_contract(payload, tool_name="search_document_by_id", caller_node="retrieval_worker", artifact_keys=("document_sources",))
    assert fake_db.search_knowledge_sources.call_args.kwargs["file_hashes"] == ["owned"]
    assert len(fake_db.get_knowledge_source_chunks_by_ids.call_args.kwargs["chunk_ids"]) <= 21

    unowned = await neutral_document_by_id(
        FocusedDocumentSearchRequest(query="focused", file_hash="not-owned"),
        _context(caller_node="retrieval_worker"), services=Services(),
    )
    unowned_payload = normalize_tool_result(unowned.to_json(), tool_name="search_document_by_id")
    assert ToolWarningCode.NO_THREAD_DOCUMENTS in unowned_payload["warnings"]


def test_search_document_by_id_rejects_path_and_url_identifiers():
    for value in ("../secret.pdf", "/tmp/file", "https://example.com/file"):
        with pytest.raises(ValidationError):
            FocusedDocumentSearchRequest(query="q", file_hash=value)


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
async def test_warning_paths_still_return_valid_tool_contracts():
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

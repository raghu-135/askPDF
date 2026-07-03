from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent import external_research_tools
from app.agent.tool_contract import ToolWarningCode, normalize_tool_result
from app.rag import agent_tools


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

    raw = await agent_tools.get_thread_shape.ainvoke({}, config=_config())
    payload = normalize_tool_result(raw, tool_name="get_thread_shape")

    assert payload["ok"] is True
    _assert_contract(payload, tool_name="get_thread_shape", artifact_keys=("thread_shape",))
    assert payload["artifacts"]["thread_shape"]["total_qa_pairs"] == 2


@pytest.mark.asyncio
async def test_search_documents_returns_sources_and_artifacts_contract(monkeypatch):
    class FakeEmbeddingModel:
        async def aembed_query(self, query):
            return [0.1, 0.2, 0.3]

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
        search_web_chunks=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(agent_tools, "get_embedding_model", lambda _name: FakeEmbeddingModel())
    monkeypatch.setattr(agent_tools, "get_vector_db", lambda: fake_db)
    monkeypatch.setattr(
        agent_tools,
        "get_document_metadata_lookup",
        AsyncMock(return_value={"file-1": {"file_name": "paper.pdf", "source_type": "pdf"}}),
    )

    raw = await agent_tools.search_documents.ainvoke(
        {"query": "diffusion", "max_results": 5},
        config=_config(caller_node="retrieval_worker"),
    )
    payload = normalize_tool_result(raw, tool_name="search_documents")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_documents",
        caller_node="retrieval_worker",
        artifact_keys=("document_sources", "web_sources"),
    )
    assert payload["sources"] == payload["artifacts"]["document_sources"]
    assert payload["__document_sources__"] == payload["artifacts"]["document_sources"]


@pytest.mark.asyncio
async def test_search_conversation_history_returns_used_chat_ids_contract(monkeypatch):
    class FakeEmbeddingModel:
        async def aembed_query(self, query):
            return [0.4, 0.5]

    monkeypatch.setattr(agent_tools, "get_embedding_model", lambda _name: FakeEmbeddingModel())
    monkeypatch.setattr(
        agent_tools,
        "fetch_semantic_history",
        AsyncMock(return_value=("Q: earlier\nA: useful memory", ["turn-1:assistant"])),
    )

    raw = await agent_tools.search_conversation_history.ainvoke(
        {"query": "earlier discussion", "max_results": 3},
        config=_config(caller_node="memory_worker", route="memory"),
    )
    payload = normalize_tool_result(raw, tool_name="search_conversation_history")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_conversation_history",
        caller_node="memory_worker",
        artifact_keys=("used_chat_ids",),
    )
    assert payload["artifacts"]["used_chat_ids"] == ["turn-1:assistant"]
    assert payload["__used_chat_ids__"] == ["turn-1:assistant"]


@pytest.mark.asyncio
async def test_search_thread_timeline_returns_timeline_artifacts_contract(monkeypatch):
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

    class FakeEmbeddingModel:
        async def aembed_query(self, query):
            return [0.1, 0.2]

    monkeypatch.setattr(agent_tools, "get_vector_db", lambda: fake_db)
    monkeypatch.setattr(agent_tools, "get_embedding_model", lambda _name: FakeEmbeddingModel())
    monkeypatch.setattr(
        agent_tools,
        "get_document_metadata_lookup",
        AsyncMock(return_value={}),
    )

    raw = await agent_tools.search_thread_timeline.ainvoke(
        {"query": "timeline", "sources": "conversation", "order": "oldest", "max_results": 5},
        config=_config(caller_node="timeline_worker", route="timeline"),
    )
    payload = normalize_tool_result(raw, tool_name="search_thread_timeline")

    assert payload["ok"] is True
    _assert_contract(
        payload,
        tool_name="search_thread_timeline",
        caller_node="timeline_worker",
        artifact_keys=("timeline_events",),
    )
    assert payload["artifacts"]["timeline_events"][0]["message_id"] == "turn-1:assistant"
    assert payload["__timeline_events__"] == payload["artifacts"]["timeline_events"]


@pytest.mark.asyncio
async def test_search_web_returns_web_source_contract(monkeypatch):
    monkeypatch.setattr(
        external_research_tools,
        "_run_web_search",
        AsyncMock(
            return_value={
                "texts": ["fresh web evidence"],
                "urls": ["https://example.com"],
                "titles": ["Example"],
            }
        ),
    )

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
        artifact_keys=("web_sources",),
    )
    assert payload["sources"] == payload["artifacts"]["web_sources"]
    assert payload["__web_sources__"] == payload["artifacts"]["web_sources"]


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

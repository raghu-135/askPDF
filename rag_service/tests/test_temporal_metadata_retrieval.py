from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

import pytest

from app.agent import external_research_tools
from app.db.vector.adapter import WeaviateAdapter
from app.rag import indexer
from app.rag.retrieval import fetch_semantic_history, group_document_chunks


class CapturingAdapter(WeaviateAdapter):
    def __init__(self):
        self.collection_manager = AsyncMock()
        self.collection_manager.validate_vectors_for_model.return_value = True
        self.collection_manager.get_collection.return_value = object()
        self.points = None

    async def _insert_many_model_aware(self, collection, points):
        self.points = points
        return len(points)


@pytest.mark.asyncio
async def test_document_vector_properties_include_page_metadata_not_thread_temporal_metadata():
    adapter = CapturingAdapter()

    count = await adapter.index_pdf_chunks(
        thread_id="thread-1",
        embedding_model_name="embed-1",
        file_hash="file-1",
        texts=["benefits text"],
        embeddings=[[0.1, 0.2]],
        metadatas=[
            {
                "document_available_in_thread_at": "2026-06-25T19:00:00Z",
                "document_indexed_at": "2026-06-25T19:01:00Z",
                "page_start": 3,
                "page_end": 4,
                "pages": "3-4",
                "timeline_event_at": "2026-06-25T19:00:00Z",
                "timeline_event_type": "document_available_in_thread",
            }
        ],
    )

    assert count == 1
    props = adapter.points[0]["properties"]
    assert "document_available_in_thread_at" not in props
    assert "document_indexed_at" not in props
    assert "timeline_event_at" not in props
    assert "timeline_event_type" not in props
    assert props["page_start"] == 3
    assert props["page_end"] == 4
    assert props["pages"] == "3-4"
    assert "document_available_in_thread_at" not in props["metadata_json"]
    assert "document_indexed_at" not in props["metadata_json"]
    assert "timeline_event_at" not in props["metadata_json"]
    assert "timeline_event_type" not in props["metadata_json"]


@pytest.mark.asyncio
async def test_chat_and_web_vectors_store_event_specific_timestamp_only():
    adapter = CapturingAdapter()

    await adapter.index_chat_memory(
        thread_id="thread-1",
        message_id="msg-1",
        question="Q?",
        answer="A.",
        texts=["Q: Q?\nA: A."],
        embeddings=[[0.1, 0.2]],
        embedding_model_name="embed-1",
        message_created_at="2026-06-25T19:10:00Z",
    )
    chat_props = adapter.points[0]["properties"]
    assert chat_props["message_created_at"] == "2026-06-25T19:10:00Z"
    assert "timeline_event_at" not in chat_props
    assert "timeline_event_type" not in chat_props

    await adapter.index_web_search_chunks(
        thread_id="thread-1",
        query="query",
        texts=["snippet"],
        embeddings=[[0.1, 0.2]],
        embedding_model_name="embed-1",
        urls=["https://example.com"],
        titles=["Example"],
        web_search_performed_at="2026-06-25T19:15:00Z",
    )
    web_props = adapter.points[0]["properties"]
    assert web_props["web_search_performed_at"] == "2026-06-25T19:15:00Z"
    assert "timeline_event_at" not in web_props
    assert "timeline_event_type" not in web_props


@pytest.mark.asyncio
async def test_chat_and_web_search_results_derive_timeline_metadata():
    adapter = CapturingAdapter()
    collection = SimpleNamespace(query=SimpleNamespace(near_vector=MagicMock()))
    adapter.collection_manager.get_collection.return_value = collection

    collection.query.near_vector.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                properties={
                    "text": "memory",
                    "message_id": "msg-1",
                    "message_created_at": "2026-06-25T19:10:00Z",
                },
                metadata=None,
            )
        ]
    )
    chat_results = await adapter.search_chat_memory(
        thread_id="thread-1",
        query_vector=[0.1, 0.2],
        embedding_model_name="embed-1",
    )
    assert chat_results[0]["message_created_at"] == "2026-06-25T19:10:00Z"
    assert chat_results[0]["timeline_event_at"] == "2026-06-25T19:10:00Z"
    assert chat_results[0]["timeline_event_type"] == "message_created"

    collection.query.near_vector.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                properties={
                    "text": "snippet",
                    "url": "https://example.com",
                    "web_search_performed_at": "2026-06-25T19:15:00Z",
                },
                metadata=None,
            )
        ]
    )
    web_results = await adapter.search_web_chunks(
        thread_id="thread-1",
        query_vector=[0.1, 0.2],
        embedding_model_name="embed-1",
    )
    assert web_results[0]["web_search_performed_at"] == "2026-06-25T19:15:00Z"
    assert web_results[0]["timeline_event_at"] == "2026-06-25T19:15:00Z"
    assert web_results[0]["timeline_event_type"] == "web_search_performed"


def test_document_grouping_uses_thread_availability_and_page_labels():
    context, sources = group_document_chunks(
        [
            {
                "text": "Page three text",
                "file_hash": "file-1",
                "chunk_id": 7,
                "page_start": 3,
                "page_end": 4,
                "pages": "3-4",
            }
        ],
        {
            "file-1": {
                "file_name": "benefits.pdf",
                "document_available_in_thread_at": "2026-06-25T19:00:00Z",
            }
        },
    )

    assert "[Source: PDF: benefits.pdf, pages 3-4]" in context
    assert sources[0]["document_available_in_thread_at"] == "2026-06-25T19:00:00Z"
    assert sources[0]["timeline_event_at"] == "2026-06-25T19:00:00Z"
    assert sources[0]["timeline_event_type"] == "document_available_in_thread"
    assert sources[0]["page_start"] == 3
    assert sources[0]["page_end"] == 4
    assert sources[0]["pages"] == "3-4"


@pytest.mark.asyncio
async def test_document_indexing_keeps_thread_availability_out_of_shared_chunk_metadata(monkeypatch):
    fake_db = SimpleNamespace(
        has_file_indexed=AsyncMock(return_value=False),
        collection_manager=SimpleNamespace(validate_vectors_for_model=AsyncMock(return_value=True)),
        index_pdf_chunks=AsyncMock(return_value=1),
    )
    global_file_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    thread_file_added_at = datetime(2026, 6, 25, 19, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(indexer, "get_vector_db", lambda: fake_db)
    monkeypatch.setattr(
        indexer,
        "get_thread_file_association",
        AsyncMock(return_value={"added_at": thread_file_added_at}),
    )
    monkeypatch.setattr(
        indexer,
        "get_file",
        AsyncMock(
            return_value=SimpleNamespace(
                file_name="benefits.pdf",
                source_type="pdf",
                created_at=global_file_created_at,
            )
        ),
    )
    monkeypatch.setattr("app.db.update_indexing_status", AsyncMock())
    monkeypatch.setattr(indexer, "upsert_document_in_stats", AsyncMock())
    monkeypatch.setattr(
        indexer,
        "get_chunks_with_metadata",
        AsyncMock(return_value=[{"text": "Chunk text", "metadata": {"pages": "3", "page_start": 3, "page_end": 3}}]),
    )
    monkeypatch.setattr(indexer, "generate_embeddings", AsyncMock(return_value=[[0.1, 0.2]]))

    result = await indexer.index_document_for_thread(
        thread_id="thread-1",
        file_hash="file-1",
        embedding_model_name="embed-1",
    )

    assert result["status"] == "success"
    metadata = fake_db.index_pdf_chunks.call_args.kwargs["metadatas"][0]
    assert "document_available_in_thread_at" not in metadata
    assert "timeline_event_at" not in metadata
    assert "timeline_event_type" not in metadata
    assert metadata["pages"] == "3"


def test_document_grouping_ignores_stale_vector_temporal_metadata():
    context, sources = group_document_chunks(
        [
            {
                "text": "Shared chunk text",
                "file_hash": "file-1",
                "metadata": {
                    "document_available_in_thread_at": "2026-01-01T00:00:00Z",
                    "timeline_event_at": "2026-01-01T00:00:00Z",
                    "timeline_event_type": "document_available_in_thread",
                    "pages": "5",
                },
            }
        ],
        {
            "file-1": {
                "file_name": "shared.pdf",
                "document_available_in_thread_at": "2026-06-25T19:00:00Z",
            }
        },
    )

    assert "[Source: PDF: shared.pdf, pages 5]" in context
    assert sources[0]["document_available_in_thread_at"] == "2026-06-25T19:00:00Z"
    assert sources[0]["timeline_event_at"] == "2026-06-25T19:00:00Z"


@pytest.mark.asyncio
async def test_semantic_history_includes_message_time_when_present(monkeypatch):
    fake_db = SimpleNamespace(
        search_chat_memory=AsyncMock(
            return_value=[
                {
                    "text": "Q: earlier\nA: answer",
                    "message_id": "msg-1",
                    "message_created_at": "2026-06-25T19:10:00Z",
                }
            ]
        )
    )
    monkeypatch.setattr("app.rag.retrieval.get_vector_db", lambda: fake_db)

    history, used_ids = await fetch_semantic_history(
        thread_id="thread-1",
        query_vector=[0.1],
        query_text=None,
        limit=5,
        embedding_model_name="embed-1",
    )

    assert "Earlier exchange at 2026-06-25T19:10:00Z:" in history
    assert used_ids == ["msg-1"]


def test_web_context_exposes_search_performed_time():
    payload = external_research_tools._format_web_context(
        texts=["snippet"],
        urls=["https://example.com"],
        titles=["Example"],
        web_search_performed_at="2026-06-25T19:15:00Z",
    )

    assert "Web result from search performed at 2026-06-25T19:15:00Z" in payload["content"]
    assert payload["__web_sources__"][0]["web_search_performed_at"] == "2026-06-25T19:15:00Z"
    assert payload["__web_sources__"][0]["timeline_event_at"] == "2026-06-25T19:15:00Z"
    assert payload["__web_sources__"][0]["timeline_event_type"] == "web_search_performed"

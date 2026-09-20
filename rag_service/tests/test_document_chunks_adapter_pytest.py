"""Unit tests for WeaviateAdapter document chunk inspection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.vector.adapter import WeaviateAdapter


@pytest.mark.asyncio
async def test_get_document_chunks_for_file_returns_sorted_page():
    adapter = WeaviateAdapter.__new__(WeaviateAdapter)
    adapter.collection_manager = SimpleNamespace(
        get_collection=AsyncMock(return_value=MagicMock()),
    )

    collection = await adapter.collection_manager.get_collection("Document", "BAAI/bge-m3")
    collection.aggregate.over_all = MagicMock(return_value=SimpleNamespace(total_count=2))
    collection.query.fetch_objects = MagicMock(
        return_value=SimpleNamespace(
            objects=[
                SimpleNamespace(
                    properties={
                        "text": "second",
                        "file_hash": "fh1",
                        "chunk_id": 1,
                        "metadata_json": '{"chunking_fingerprint":"fp"}',
                        "source_id": "src-1",
                    }
                ),
                SimpleNamespace(
                    properties={
                        "text": "first",
                        "file_hash": "fh1",
                        "chunk_id": 0,
                        "metadata_json": "{}",
                        "source_id": "src-0",
                    }
                ),
            ]
        )
    )

    result = await adapter.get_document_chunks_for_file(
        "fh1",
        "BAAI/bge-m3",
        limit=50,
        offset=0,
    )

    assert result["total_count"] == 2
    assert result["limit"] == 50
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == [0, 1]
    assert result["chunks"][1]["metadata"]["chunking_fingerprint"] == "fp"

"""Tests for thread embedding projection endpoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


class TestEmbeddingProjectionApi:
    @pytest.mark.asyncio
    async def test_embeddings_projection_endpoint_returns_points(self, async_api_client):
        client = async_api_client
        thread_response = await client.post("/api/threads", json={"name": "Embedding projection thread"})
        assert thread_response.status_code == 200
        thread_id = thread_response.json()["id"]

        mock_points = [
            {
                "id": "src-0",
                "vector": [1.0, 0.0, 0.0],
                "chunk_id": 0,
                "file_hash": "fh1",
                "text": "hello",
                "page_start": 1,
                "page_end": 1,
                "pages": "1",
                "source_kind": "pdf",
                "table_id": None,
                "section_id": None,
                "title": "Doc",
                "metadata": {},
            }
        ]
        mock_context = SimpleNamespace(
            embedding_model="BAAI/bge-m3",
            thread=SimpleNamespace(id=thread_id),
            project=SimpleNamespace(id="project-1"),
        )
        mock_file = SimpleNamespace(file_hash="fh1", file_name="Doc.pdf")

        with (
            patch(
                "app.api.threads.require_thread_embedding_ready",
                new=AsyncMock(return_value=mock_context),
            ),
            patch(
                "app.api.threads.get_effective_thread_files",
                new=AsyncMock(return_value=[mock_file]),
            ),
            patch("app.api.threads.get_vector_db") as get_vector_db,
        ):
            get_vector_db.return_value.get_thread_vector_points = AsyncMock(return_value=mock_points)
            response = await client.get(f"/api/threads/{thread_id}/embeddings-projection?limit=50")

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["thread_id"] == thread_id
        assert payload["embedding_model"] == "BAAI/bge-m3"
        assert payload["point_count"] == 1
        point = payload["points"][0]
        assert point["id"] == "src-0"
        assert point["file_hash"] == "fh1"
        assert point["file_name"] == "Doc.pdf"
        assert all(key in point for key in ("x", "y", "z"))
        assert "edges" in payload
        assert isinstance(payload["edges"], list)

        await client.delete(f"/api/threads/{thread_id}")

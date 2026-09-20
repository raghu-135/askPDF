"""Tests for read-only document chunk inspection endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


class TestFileChunksApi:
    @pytest_asyncio.fixture
    async def client(self, async_api_client):
        yield async_api_client

    @pytest.mark.asyncio
    async def test_thread_file_chunks_returns_vector_payload(self, client):
        thread_response = await client.post("/api/threads", json={"name": "Chunk inspect thread"})
        assert thread_response.status_code == 200
        thread_id = thread_response.json()["id"]
        file_hash = "abc123deadbeef"

        mock_payload = {
            "file_hash": file_hash,
            "embedding_model": "BAAI/bge-m3",
            "total_count": 2,
            "limit": 100,
            "offset": 0,
            "chunks": [
                {
                    "chunk_id": 0,
                    "source_id": "src-0",
                    "text": "First chunk body",
                    "page_start": 1,
                    "page_end": 1,
                    "metadata": {"chunking_fingerprint": "fp-1"},
                },
                {
                    "chunk_id": 1,
                    "source_id": "src-1",
                    "text": "Second chunk body",
                    "metadata": {},
                },
            ],
        }

        with patch("app.api.files.get_vector_db") as get_vector_db:
            get_vector_db.return_value.get_document_chunks_for_file = AsyncMock(return_value=mock_payload)
            response = await client.get(f"/api/threads/{thread_id}/files/{file_hash}/chunks")

        assert response.status_code == 404

        with (
            patch("app.api.files.is_file_accessible_to_thread", AsyncMock(return_value=True)),
            patch("app.api.files.get_vector_db") as get_vector_db,
        ):
            get_vector_db.return_value.get_document_chunks_for_file = AsyncMock(return_value=mock_payload)
            response = await client.get(f"/api/threads/{thread_id}/files/{file_hash}/chunks")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total_count"] == 2
        assert body["chunks"][0]["text"] == "First chunk body"
        assert body["chunks"][0]["metadata"]["chunking_fingerprint"] == "fp-1"
        get_vector_db.return_value.get_document_chunks_for_file.assert_awaited_once_with(
            file_hash,
            "BAAI/bge-m3",
            limit=100,
            offset=0,
        )

        await client.delete(f"/api/threads/{thread_id}")

    @pytest.mark.asyncio
    async def test_thread_file_chunks_rejects_invalid_limit(self, client):
        thread_response = await client.post("/api/threads", json={"name": "Chunk limit thread"})
        thread_id = thread_response.json()["id"]

        with patch("app.api.files.is_file_accessible_to_thread", AsyncMock(return_value=True)):
            response = await client.get(
                f"/api/threads/{thread_id}/files/file-hash/chunks",
                params={"limit": 0},
            )

        assert response.status_code == 400
        await client.delete(f"/api/threads/{thread_id}")

    @pytest.mark.asyncio
    async def test_project_file_chunks_requires_project_membership(self, client):
        project_response = await client.post(
            "/api/projects",
            json={"name": "Chunk project", "embedding_model": "BAAI/bge-m3"},
        )
        assert project_response.status_code == 200
        project_id = project_response.json()["id"]
        file_hash = "project-file-hash"

        response = await client.get(f"/api/projects/{project_id}/files/{file_hash}/chunks")
        assert response.status_code == 404

        await client.delete(f"/api/projects/{project_id}")

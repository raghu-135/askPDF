"""
test_api_integration_pytest.py - Integration tests for API endpoints using new database.

These tests verify that API endpoints work correctly with the PostgreSQL database,
covering the main CRUD operations through the HTTP layer.
"""

from pathlib import Path

import pytest
import pytest_asyncio

from app.db import OperationResultStatus, ProcessStatus, update_indexing_status
from app.time_utils import iso_utc_z, utc_now


class TestAPIIntegration:
    """Test API endpoints with PostgreSQL database."""

    @pytest_asyncio.fixture
    async def client(self, async_api_client):
        """Keep existing test signature while using the shared async API client fixture."""
        yield async_api_client

    @pytest.mark.asyncio
    async def test_create_thread_endpoint(self, client):
        """POST /api/threads with PostgreSQL."""
        response = await client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Thread"
        assert data["embedding_model"] == "BAAI/bge-m3"

        # Cleanup
        await client.delete(f"/api/threads/{data['id']}")

    @pytest.mark.asyncio
    async def test_upload_parse_sentences_status_and_download(self, client, monkeypatch):
        """Exercise the real upload/attachment/parse/status/sentences HTTP flow.

        Vector indexing is replaced at its external boundary so this test is
        deterministic and does not require an embedding server or Weaviate.
        The API, database attachment, background orchestration, content store,
        and bundled-PDF parsing still execute for real.
        """
        thread_response = await client.post("/api/threads", json={"name": "PDF integration thread"})
        assert thread_response.status_code == 200
        thread_id = thread_response.json()["id"]

        async def fake_index_document_for_thread(
            *, thread_id, file_hash, embedding_model, **_kwargs
        ):
            await update_indexing_status(
                file_hash=file_hash,
                status=ProcessStatus.COMPLETED.value,
                embedding_model=embedding_model,
                thread_id=thread_id,
                finished_at=iso_utc_z(utc_now()),
                chunk_count=1,
                total_chars=1,
            )
            return {"status": OperationResultStatus.SUCCESS.value}

        monkeypatch.setattr(
            "app.services.file_processing_service.index_document_for_thread",
            fake_index_document_for_thread,
        )

        pdf_path = Path(__file__).with_name("01030000000000.pdf")
        upload_response = await client.post(
            f"/api/threads/{thread_id}/files/upload",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        )
        assert upload_response.status_code == 200, upload_response.text
        upload = upload_response.json()
        file_hash = upload["file_hash"]
        assert upload["sentences"] is None
        assert file_hash

        status_response = await client.get(f"/api/threads/{thread_id}/files/{file_hash}/status")
        assert status_response.status_code == 200, status_response.text
        status = status_response.json()
        assert status["file_hash"] == file_hash
        assert status["indexing"]["status"] == ProcessStatus.COMPLETED.value
        assert status["parsing"]["status"] == ProcessStatus.COMPLETED.value

        sentences_response = await client.get(
            f"/api/threads/{thread_id}/files/{file_hash}/sentences"
        )
        assert sentences_response.status_code == 200, sentences_response.text
        sentences = sentences_response.json()
        assert isinstance(sentences["sentences"], list)
        assert sentences["sentences"]
        assert all(item.get("text", "").strip() for item in sentences["sentences"])

        download_response = await client.get(
            f"/api/threads/{thread_id}/files/{file_hash}/download"
        )
        assert download_response.status_code == 200
        assert download_response.headers["content-type"].startswith("application/pdf")
        assert download_response.content == pdf_path.read_bytes()

        files_response = await client.get(f"/api/threads/{thread_id}/files")
        assert files_response.status_code == 200
        assert any(item["file_hash"] == file_hash for item in files_response.json()["files"])

        await client.delete(f"/api/threads/{thread_id}")


# Note: Additional async HTTP tests are skipped due to pytest-asyncio fixture
# isolation issues when running in a suite. Use test_api_endpoints_pytest.py
# for comprehensive API testing with sync TestClient.

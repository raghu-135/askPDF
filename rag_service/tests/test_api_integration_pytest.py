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
        embedding_model = thread_response.json()["embedding_model"]

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
            "app.rag.indexer.index_document_for_thread",
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

        # Conversion and embedding are durable in-process queues. Drain both
        # explicitly after the request-owned enqueue.
        from app.services.file_processing_service import _background_index
        from app.services.embedding_materialization_service import drain_embedding_jobs
        from app.workers.document_conversion_worker import drain_conversion_jobs
        assert await drain_conversion_jobs() == 1
        await _background_index(
            file_hash,
            thread_id,
            embedding_model,
            pdf_path.name,
            "",
            {},
        )
        assert await drain_embedding_jobs() >= 1

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

    @pytest.mark.asyncio
    async def test_project_pdf_upload_endpoint(self, client, monkeypatch):
        """POST /api/projects/{id}/files/upload accepts multipart PDF like thread upload."""
        project_response = await client.post(
            "/api/projects",
            json={"name": "Upload project", "embedding_model": "BAAI/bge-m3"},
        )
        assert project_response.status_code == 200
        project_id = project_response.json()["id"]
        embedding_model = project_response.json()["embedding_model"]

        async def fake_index_document_for_thread(**_kwargs):
            await update_indexing_status(
                file_hash=_kwargs["file_hash"],
                status=ProcessStatus.COMPLETED.value,
                embedding_model=embedding_model,
                finished_at=iso_utc_z(utc_now()),
                chunk_count=1,
                total_chars=1,
            )
            return {"status": OperationResultStatus.SUCCESS.value}

        monkeypatch.setattr(
            "app.rag.indexer.index_document_for_thread",
            fake_index_document_for_thread,
        )

        async def noop_queue_project(*_args, **_kwargs):
            return None

        monkeypatch.setattr(
            "app.api.files.queue_project_file_processing",
            noop_queue_project,
        )

        pdf_path = Path(__file__).with_name("01030000000000.pdf")
        upload_response = await client.post(
            f"/api/projects/{project_id}/files/upload",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        )
        assert upload_response.status_code == 200, upload_response.text
        upload = upload_response.json()
        assert upload["file_hash"]
        assert upload["sentences"] is None

        await client.delete(f"/api/projects/{project_id}")

    @pytest.mark.asyncio
    async def test_multipart_on_json_file_promote_endpoint_rejected(self, client):
        """Multipart to POST .../files (JSON body) must not 500 with UnicodeDecodeError."""
        project_response = await client.post(
            "/api/projects",
            json={"name": "Bad upload", "embedding_model": "BAAI/bge-m3"},
        )
        project_id = project_response.json()["id"]
        pdf_path = Path(__file__).with_name("01030000000000.pdf")
        bad_response = await client.post(
            f"/api/projects/{project_id}/files",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        )
        assert bad_response.status_code in {400, 422}, bad_response.text
        assert bad_response.status_code != 500
        assert "UnicodeDecodeError" not in bad_response.text

        await client.delete(f"/api/projects/{project_id}")


# Note: Additional async HTTP tests are skipped due to pytest-asyncio fixture
# isolation issues when running in a suite. Use test_api_endpoints_pytest.py
# for comprehensive API testing with sync TestClient.

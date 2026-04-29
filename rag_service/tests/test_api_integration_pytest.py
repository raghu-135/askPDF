"""
test_api_integration_pytest.py - Integration tests for API endpoints using new database.

These tests verify that API endpoints work correctly with the PostgreSQL database,
covering the main CRUD operations through the HTTP layer.
"""

import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from httpx import AsyncClient, ASGITransport
    from fastapi import FastAPI
    from app.db.models_sqlmodel import Thread, File, Message
    from app.db.connection_sqlmodel import init_db, get_session
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestAPIIntegration:
    """Test API endpoints with PostgreSQL database."""

    @pytest_asyncio.fixture
    async def app(self):
        """Create FastAPI app instance for testing."""
        # This will need to be adapted to the actual app structure
        # For now, we'll skip this test
        pytest.skip("FastAPI app structure not yet available for testing")

    @pytest_asyncio.fixture
    async def client(self, app):
        """Create async HTTP client."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_create_thread_endpoint(self, client):
        """POST /threads with PostgreSQL."""
        response = await client.post(
            "/threads",
            json={
                "name": "Test Thread",
                "embed_model": "BAAI/bge-m3"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Thread"
        assert data["embed_model"] == "BAAI/bge-m3"

    @pytest.mark.asyncio
    async def test_list_threads_endpoint(self, client):
        """GET /threads with counts."""
        response = await client.get("/threads")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Each thread should have message_count and file_count
        if len(data) > 0:
            assert "message_count" in data[0]
            assert "file_count" in data[0]

    @pytest.mark.asyncio
    async def test_upload_file_endpoint(self, client):
        """POST /files with file processing."""
        # This would test file upload and processing
        pytest.skip("File upload test requires multipart form data handling")

    @pytest.mark.asyncio
    async def test_add_file_to_thread_endpoint(self, client, sample_thread, sample_file):
        """POST /threads/{id}/files."""
        response = await client.post(
            f"/threads/{sample_thread.id}/files",
            json={"file_hash": sample_file.file_hash}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or data is not None

    @pytest.mark.asyncio
    async def test_create_message_endpoint(self, client, sample_thread):
        """POST /threads/{id}/messages."""
        response = await client.post(
            f"/threads/{sample_thread.id}/messages",
            json={
                "role": "user",
                "content": "Hello, world!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["role"] == "user"
        assert data["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_get_thread_messages_endpoint(self, client, sample_thread):
        """GET /threads/{id}/messages."""
        response = await client.get(
            f"/threads/{sample_thread.id}/messages"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_delete_thread_endpoint(self, client, sample_thread):
        """DELETE /threads/{id}."""
        response = await client.delete(f"/threads/{sample_thread.id}")
        
        assert response.status_code == 200 or response.status_code == 204

    @pytest.mark.asyncio
    async def test_get_thread_shape_endpoint(self, client, sample_thread):
        """GET /threads/{id}/shape."""
        response = await client.get(f"/threads/{sample_thread.id}/shape")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_qa_pairs" in data
        assert "total_qa_chars" in data
        assert "documents" in data

    @pytest.mark.asyncio
    async def test_update_thread_settings_endpoint(self, client, sample_thread):
        """PUT /threads/{id}/settings."""
        response = await client.put(
            f"/threads/{sample_thread.id}/settings",
            json={"max_iterations": 20, "token_budget": 16384}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["max_iterations"] == 20

    @pytest.mark.asyncio
    async def test_get_file_status_endpoint(self, client, sample_file):
        """GET /files/{hash}/status."""
        response = await client.get(f"/files/{sample_file.file_hash}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "parsing" in data or "indexing" in data

    @pytest.mark.asyncio
    async def test_update_file_status_endpoint(self, client, sample_file):
        """PUT /files/{hash}/status."""
        response = await client.put(
            f"/files/{sample_file.file_hash}/status",
            json={
                "parsing": {"status": "completed"},
                "indexing": {"status": "running"}
            }
        )
        
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_thread_annotations_endpoint(self, client, sample_thread, sample_file):
        """GET /threads/{id}/files/{hash}/annotations."""
        response = await client.get(
            f"/threads/{sample_thread.id}/files/{sample_file.file_hash}/annotations"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "annotations" in data or data is None

    @pytest.mark.asyncio
    async def test_upsert_annotations_endpoint(self, client, sample_thread, sample_file):
        """PUT /threads/{id}/files/{hash}/annotations."""
        annotations = [
            {"page": 1, "text": "Test", "bbox": [0, 0, 100, 20]}
        ]
        response = await client.put(
            f"/threads/{sample_thread.id}/files/{sample_file.file_hash}/annotations",
            json={"annotations": annotations}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["annotations"]) == 1


# Note: These tests are placeholders that will need to be adapted
# to the actual API structure once the migration is complete.
# The tests use httpx for async HTTP requests and will need
# the actual FastAPI app to be properly initialized.

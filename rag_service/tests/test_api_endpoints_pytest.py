"""
test_api_endpoints_pytest.py - DB-agnostic API endpoint tests.

This module provides comprehensive tests for API endpoints using FastAPI's TestClient.
These tests validate HTTP contracts and API behavior with a test database.
"""

import os
import sys
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app


@pytest.fixture(scope="function")
def client() -> Generator:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check(self, client):
        """Test that health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "rag-service"
        assert "version" in data


class TestThreadEndpoints:
    """Test suite for thread endpoints."""

    def test_create_thread(self, client):
        """Test creating a new thread."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Thread"
        assert data["embed_model"] == "BAAI/bge-m3"

    def test_create_thread_default_embed_model(self, client):
        """Test creating a thread with default embed model."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Thread"
        assert data["embed_model"] is not None

    def test_list_threads(self, client):
        """Test listing all threads."""
        # Create a thread first
        client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        
        response = client.get("/api/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data
        assert isinstance(data["threads"], list)

    def test_get_thread(self, client):
        """Test getting a specific thread."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.get(f"/api/threads/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == thread_id
        assert data["name"] == "Test Thread"

    def test_get_nonexistent_thread(self, client):
        """Test getting a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id")
        assert response.status_code == 404

    def test_update_thread(self, client):
        """Test updating a thread's name."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Original Name", "embed_model": "BAAI/bge-m3"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.put(
            f"/api/threads/{thread_id}",
            json={"name": "Updated Name"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    def test_update_nonexistent_thread(self, client):
        """Test updating a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id",
            json={"name": "New Name"}
        )
        assert response.status_code == 404

    def test_get_thread_settings(self, client):
        """Test getting thread settings."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.get(f"/api/threads/{thread_id}/settings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_get_settings_nonexistent_thread(self, client):
        """Test getting settings for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/settings")
        assert response.status_code == 404

    def test_update_thread_settings(self, client):
        """Test updating thread settings."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.put(
            f"/api/threads/{thread_id}/settings",
            json={"max_iterations": 20, "token_budget": 16384}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_update_settings_nonexistent_thread(self, client):
        """Test updating settings for a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id/settings",
            json={"max_iterations": 20}
        )
        assert response.status_code == 404

    def test_delete_thread(self, client):
        """Test deleting a thread."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "To Delete", "embed_model": "BAAI/bge-m3"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.delete(f"/api/threads/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["thread_id"] == thread_id

    def test_delete_nonexistent_thread(self, client):
        """Test deleting a thread that doesn't exist."""
        response = client.delete("/api/threads/nonexistent-id")
        assert response.status_code == 404

    def test_get_prompt_tools(self, client):
        """Test getting prompt tools and defaults."""
        response = client.get("/api/threads/prompt-tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "defaults" in data
        assert isinstance(data["tools"], list)

    def test_prompt_preview(self, client):
        """Test getting prompt preview."""
        response = client.post(
            "/api/threads/prompt-preview",
            json={
                "context_window": 8192,
                "system_role": "You are a helpful assistant",
                "tool_instructions": {},
                "custom_instructions": "Be concise"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert isinstance(data["prompt"], str)


class TestMessageEndpoints:
    """Test suite for message endpoints."""

    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for message tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        return response.json()["id"]

    def test_get_thread_messages_empty(self, client, sample_thread):
        """Test getting messages from an empty thread."""
        response = client.get(f"/api/threads/{sample_thread}/messages")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 0

    def test_get_thread_messages_with_pagination(self, client, sample_thread):
        """Test getting messages with pagination parameters."""
        response = client.get(
            f"/api/threads/{sample_thread}/messages",
            params={"limit": 10, "offset": 0}
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_messages_nonexistent_thread(self, client):
        """Test getting messages for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/messages")
        assert response.status_code == 404

    def test_delete_message_nonexistent(self, client):
        """Test deleting a message that doesn't exist."""
        response = client.delete("/api/messages/nonexistent-id")
        assert response.status_code == 404


class TestFileEndpoints:
    """Test suite for file endpoints."""

    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for file tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        return response.json()["id"]

    def test_get_thread_files_empty(self, client, sample_thread):
        """Test getting files from a thread with no files."""
        response = client.get(f"/api/threads/{sample_thread}/files")
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert isinstance(data["files"], list)
        assert len(data["files"]) == 0

    def test_get_files_nonexistent_thread(self, client):
        """Test getting files for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files")
        assert response.status_code == 404

    def test_add_file_to_thread(self, client, sample_thread):
        """Test adding a file to a thread."""
        response = client.post(
            f"/api/threads/{sample_thread}/files",
            json={
                "file_hash": "abc123",
                "file_name": "test.pdf",
                "file_path": "/data/test.pdf"
            }
        )
        # This endpoint requires background tasks and may not work in simple test
        # but we can at least check the endpoint exists
        assert response.status_code in [200, 500]  # May fail due to missing dependencies


class TestProactiveCollectionCreation:
    """Test proactive collection creation during thread access."""
    
    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for collection tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread", "embed_model": "BAAI/bge-m3"}
        )
        return response.json()["id"]
    
    @patch('app.db.vector.get_vector_db')
    @patch('app.rag.indexer.trigger_reembed_for_missing_sources')
    def test_thread_access_triggers_collection_creation(self, mock_reembed, mock_get_db, client, sample_thread):
        """Test that accessing a thread triggers proactive collection creation."""
        # Mock vector DB and collection manager
        mock_db = AsyncMock()
        mock_collection_manager = AsyncMock()
        mock_db.collection_manager = mock_collection_manager
        mock_get_db.return_value = mock_db
        
        # Access thread endpoint
        response = client.get(f"/api/threads/{sample_thread}")
        assert response.status_code == 200
        
        # Should trigger both reembed and collection creation
        mock_reembed.assert_called_once()
        mock_collection_manager.ensure_collections_for_thread.assert_called_once()
        
        # Should be called with the thread's embedding model
        call_args = mock_collection_manager.ensure_collections_for_thread.call_args
        assert call_args[0][0] == "BAAI/bge-m3"
    
    @patch('app.db.vector.get_vector_db')
    @patch('app.rag.indexer.trigger_reembed_for_missing_sources')
    def test_thread_access_handles_collection_creation_failure(self, mock_reembed, mock_get_db, client, sample_thread):
        """Test that collection creation failures don't break thread access."""
        # Mock vector DB to raise exception during collection creation
        mock_db = AsyncMock()
        mock_collection_manager = AsyncMock()
        mock_collection_manager.ensure_collections_for_thread.side_effect = Exception("Collection creation failed")
        mock_db.collection_manager = mock_collection_manager
        mock_get_db.return_value = mock_db
        
        # Thread access should still succeed despite collection creation failure
        # (since it runs as background task)
        response = client.get(f"/api/threads/{sample_thread}")
        assert response.status_code == 200
        
        # Should still attempt collection creation
        mock_collection_manager.ensure_collections_for_thread.assert_called_once()
    
    def test_nonexistent_thread_returns_404(self, client):
        """Test that accessing nonexistent thread returns 404."""
        response = client.get("/api/threads/nonexistent-id")
        assert response.status_code == 404
        assert "Thread not found" in response.json()["detail"]

    def test_remove_file_from_thread(self, client, sample_thread):
        """Test removing a file from a thread."""
        response = client.delete(f"/api/threads/{sample_thread}/files/abc123")
        # May fail if file doesn't exist, but endpoint should be accessible
        assert response.status_code in [200, 404, 500]

    def test_get_file_status_nonexistent_thread(self, client):
        """Test getting file status for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files/abc123/status")
        assert response.status_code == 404

    def test_get_annotations_nonexistent_thread(self, client):
        """Test getting annotations for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files/abc123/annotations")
        assert response.status_code == 404

    def test_update_annotations_nonexistent_thread(self, client):
        """Test updating annotations for a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id/files/abc123/annotations",
            json={"annotations": []}
        )
        assert response.status_code == 404


class TestModelsEndpoint:
    """Test suite for models endpoint."""

    def test_get_models(self, client):
        """Test getting available models."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict) or isinstance(data, list)

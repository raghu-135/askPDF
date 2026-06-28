"""
test_api_integration_pytest.py - Integration tests for API endpoints using new database.

These tests verify that API endpoints work correctly with the PostgreSQL database,
covering the main CRUD operations through the HTTP layer.
"""

import pytest
import pytest_asyncio


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

        # Cleanup
        await client.delete(f"/api/threads/{data['id']}")


# Note: Additional async HTTP tests are skipped due to pytest-asyncio fixture
# isolation issues when running in a suite. Use test_api_endpoints_pytest.py
# for comprehensive API testing with sync TestClient.

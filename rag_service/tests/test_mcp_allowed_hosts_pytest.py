"""MCP streamable HTTP must accept docker bridge Host headers that include ports."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.mcp.server import get_http_app


@pytest.mark.asyncio
async def test_hermes_offline_mcp_accepts_host_docker_internal_with_port():
    mcp_app = get_http_app(require_execution_token=False)
    async with mcp_app.router.lifespan_context(mcp_app):
        transport = ASGITransport(app=mcp_app)
        async with AsyncClient(transport=transport, base_url="http://host.docker.internal:8000") as client:
            response = await client.post(
                "/",
                headers={
                    "accept": "application/json, text/event-stream",
                    "host": "host.docker.internal:8000",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "askpdf-first-party"

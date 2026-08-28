import pytest
import asyncio
from httpx import ASGITransport, AsyncClient

from app.mcp.transport import InProcessMCPClient


@pytest.mark.asyncio
async def test_mcp_initialize_and_tools_list():
    client = InProcessMCPClient()
    initialized = await client.request("initialize", {"protocolVersion": "2025-06-18"})
    assert initialized["serverInfo"]["name"] == "askpdf-first-party"

    listed = await client.request("tools/list")
    names = {item["name"] for item in listed["tools"]}
    assert {"wikipedia", "get_thread_shape"}.issubset(names)


@pytest.mark.asyncio
async def test_hermes_mcp_catalog_is_filtered_and_uses_transport_context(monkeypatch):
    from app.mcp import server as server_module
    from app.mcp.server import get_http_app
    from app.mcp.transport import LoopbackHTTPMCPClient
    from app.mcp.execution_context_token import TOKEN_HEADER, issue_execution_context_token
    from app.tools.context import ToolInvocationContext
    from app.agent.tool_contract import ToolResult

    monkeypatch.setenv("HERMES_MCP_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]

    async def handler(request, context):
        assert context.run_id == "run-1"
        return ToolResult(content="thread shape", sources=[{"thread_id": context.thread_id}])

    monkeypatch.setitem(
        server_module.MCP_TOOL_DEFINITIONS,
        "get_thread_shape",
        definition.__class__(definition.name, definition.request_model, handler, definition.registry_contract_id, definition.contract_version, definition.server_name),
    )

    mcp_app = get_http_app(
        allowed_tools=frozenset({"get_thread_shape", "search_documents"}),
        require_execution_token=True,
    )
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", embedding_model="embed", context_window=8192),
        task_id="task-1", allowed_tools=["get_thread_shape"],
    )
    async with mcp_app.router.lifespan_context(mcp_app):
        async with AsyncClient(
            transport=ASGITransport(app=mcp_app),
            base_url="http://localhost",
            headers={TOKEN_HEADER: token},
        ) as http_client:
            client = LoopbackHTTPMCPClient("http://localhost/", http_client=http_client)
            listed = await client.request("tools/list")
            assert {tool["name"] for tool in listed["tools"]} == {"get_thread_shape", "search_documents"}
            assert all("_askpdf_context_token" not in tool["inputSchema"].get("properties", {}) for tool in listed["tools"])
            accepted = await client.request("tools/call", {"name": "get_thread_shape", "arguments": {}})
            assert accepted["isError"] is False
            assert accepted["structuredContent"]["result_count"] == 1
        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://localhost") as http_client:
            client = LoopbackHTTPMCPClient("http://localhost/", http_client=http_client)
            rejected = await client.request("tools/call", {"name": "get_thread_shape", "arguments": {}})
            assert rejected["isError"] is True
            assert "execution context is required" in rejected["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_rejects_unknown_tool():
    client = InProcessMCPClient()
    with pytest.raises(RuntimeError, match="Unknown tool"):
        await client.request("tools/call", {"name": "does_not_exist", "arguments": {}})


@pytest.mark.asyncio
async def test_transports_reject_unsupported_methods_consistently():
    from app.mcp.transport import LoopbackHTTPMCPClient
    from app.mcp.server import get_http_app

    with pytest.raises(RuntimeError, match="Unsupported MCP method"):
        await InProcessMCPClient().request("resources/list")

    mcp_app = get_http_app()
    async with mcp_app.router.lifespan_context(mcp_app):
        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://localhost") as http_client:
            with pytest.raises(RuntimeError, match="Unsupported MCP method"):
                await LoopbackHTTPMCPClient("http://localhost/", http_client=http_client).request("resources/list")


@pytest.mark.asyncio
async def test_sdk_mcp_request_id_is_distinct_from_tool_call_id(monkeypatch):
    from app.mcp import server as server_module
    from app.mcp.transport import InProcessMCPClient

    captured = {}
    async def handler(request, context):
        captured.update({"request": request, "context": context})
        from app.agent.tool_contract import ToolResult
        return ToolResult(content="thread shape")

    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]
    monkeypatch.setitem(
        server_module.MCP_TOOL_DEFINITIONS,
        "get_thread_shape",
        definition.__class__(definition.name, definition.request_model, handler, definition.registry_contract_id, definition.contract_version, definition.server_name),
    )
    client = InProcessMCPClient()
    await client.request("tools/call", {
        "name": "get_thread_shape", "arguments": {},
        "_meta": {"com.askpdf/runtime-context": {"tool_call_id": "call-1", "thread_id": "thread-1", "caller_node": "context_loader", "caller_node_type": "context_loader"}},
    })
    assert captured["context"].tool_call_id == "call-1"
    assert captured["context"].mcp_request_id
    assert captured["context"].mcp_request_id != captured["context"].tool_call_id


@pytest.mark.asyncio
async def test_direct_mcp_call_does_not_require_framework_caller(monkeypatch):
    from app.mcp import server as server_module
    from app.agent.tool_contract import ToolResult

    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]

    async def handler(request, context):
        assert context.thread_id == "thread-1"
        assert context.caller_node is None
        return ToolResult(content="direct MCP result")

    monkeypatch.setitem(
        server_module.MCP_TOOL_DEFINITIONS,
        "get_thread_shape",
        definition.__class__(
            definition.name,
            definition.request_model,
            handler,
            definition.registry_contract_id,
            definition.contract_version,
            definition.server_name,
        ),
    )
    result = await InProcessMCPClient().request(
        "tools/call",
        {
            "name": "get_thread_shape",
            "arguments": {},
            "_meta": {
                "com.askpdf/runtime-context": {
                    "thread_id": "thread-1",
                }
            },
        },
    )
    assert result["isError"] is False
    assert "direct MCP result" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_cancellation_allows_ephemeral_correlation_ids(monkeypatch):
    from app.mcp import server as server_module

    async def missing_run(run_id):
        raise ValueError(f"Agent run {run_id!r} does not exist")

    monkeypatch.setattr(server_module, "run_cancel_requested", missing_run)

    assert await server_module._mcp_run_cancel_requested("curator-correlation-id") is False


@pytest.mark.asyncio
async def test_mcp_cancellation_preserves_real_run_errors(monkeypatch):
    from app.mcp import server as server_module

    async def orphaned_run(_run_id):
        raise ValueError("Agent run 'run-1' has no owning task")

    monkeypatch.setattr(server_module, "run_cancel_requested", orphaned_run)

    with pytest.raises(ValueError, match="has no owning task"):
        await server_module._mcp_run_cancel_requested("run-1")


@pytest.mark.asyncio
async def test_internal_http_endpoint_preserves_mcp_protocol():
    from main import app
    from main import MCP_HTTP_APP

    async with MCP_HTTP_APP.router.lifespan_context(MCP_HTTP_APP):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.post(
                "/internal/mcp/",
                headers={"accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {}},
            )
    assert response.status_code == 200
    assert response.json()["id"] == 9
    assert {item["name"] for item in response.json()["result"]["tools"]} >= {"wikipedia", "get_thread_shape"}


@pytest.mark.asyncio
async def test_internal_http_endpoint_can_restart_its_lifespan():
    from app.mcp.server import get_http_app

    mcp_app = get_http_app()
    for _ in range(2):
        async with mcp_app.router.lifespan_context(mcp_app):
            async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://localhost") as client:
                response = await client.post(
                    "/",
                    headers={"accept": "application/json, text/event-stream"},
                    json={"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}},
                )
                assert response.status_code == 200


@pytest.mark.asyncio
async def test_loopback_client_initializes_lists_and_calls_over_streamable_http():
    from main import app
    from app.mcp.server import get_http_app
    from app.mcp.transport import LoopbackHTTPMCPClient

    mcp_app = get_http_app()
    async with mcp_app.router.lifespan_context(mcp_app):
        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://localhost") as http_client:
            client = LoopbackHTTPMCPClient("http://localhost/", http_client=http_client)
            listed = await client.request("tools/list")
            assert any(item["name"] == "get_thread_shape" for item in listed["tools"])
            result = await client.request("tools/call", {
                "name": "get_thread_shape",
                "arguments": {},
                "_meta": {"com.askpdf/runtime-context": {"thread_id": "thread-1", "run_id": "run-1", "tool_call_id": "call-1", "mcp_request_id": "mcp-http-1"}},
            })
            assert result["structuredContent"]["trace"]["tool_call_id"] == "call-1"
            assert result["structuredContent"]["trace"]["mcp_request_id"] == "mcp-http-1"


@pytest.mark.asyncio
async def test_in_process_mcp_timeout_is_enforced(monkeypatch):
    from app.mcp import server as server_module
    from app.agent.tool_contract import ToolResult

    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]

    async def slow_handler(request, context):
        await asyncio.sleep(1)
        return ToolResult(content="late")

    monkeypatch.setitem(server_module.MCP_TOOL_DEFINITIONS, "get_thread_shape", definition.__class__(definition.name, definition.request_model, slow_handler, definition.registry_contract_id, definition.contract_version, definition.server_name))
    monkeypatch.setenv("MCP_REQUEST_TIMEOUT_SECONDS", "0.1")
    with pytest.raises(TimeoutError, match="MCP request timed out"):
        await InProcessMCPClient().request("tools/call", {"name": "get_thread_shape", "arguments": {}})

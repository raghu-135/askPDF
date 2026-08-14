import asyncio
import json

import pytest


def test_mcp_mode_replaces_thread_shape_without_changing_tool_name(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_TOOL_MODE", "mcp")
    monkeypatch.setenv("MCP_TOOLS", "get_thread_shape")

    from app.mcp.langchain_adapter import create_thread_shape_tool

    thread_shape = create_thread_shape_tool()
    assert thread_shape.name == "get_thread_shape"
    assert "config" not in thread_shape.args_schema.model_fields


def test_mcp_wrapper_uses_authoritative_description():
    from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
    from app.mcp.langchain_adapter import create_mcp_langchain_tool

    for name in ("get_thread_shape", "search_documents", "search_web", "wikipedia", "wikidata"):
        tool = create_mcp_langchain_tool(name)
        assert tool.description == TOOL_FRIENDLY_CONFIG[name]["description"]


async def test_thread_shape_mcp_wrapper_preserves_thread_context(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_TOOL_MODE", "mcp")
    monkeypatch.setenv("MCP_TOOLS", "get_thread_shape")

    from app.mcp import langchain_adapter

    calls = []

    class FakeClient:
        async def request(self, method, params):
            calls.append((method, params))
            if method == "tools/list":
                return {"tools": [{"name": "get_thread_shape", "description": "Thread shape", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "thread_shape", "com.askpdf/contract-version": "1"}}]}
            return {"content": [{"type": "text", "text": "[THREAD SHAPE]\\n1. paper.pdf | 12 pages"}], "structuredContent": {"ok": True, "content": "[THREAD SHAPE]\\n1. paper.pdf | 12 pages", "sources": [], "artifacts": {}, "warnings": [], "metrics": {}, "trace": {}}, "isError": False}

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: FakeClient())
    thread_shape = langchain_adapter.create_thread_shape_tool()
    result = await thread_shape.ainvoke(
        {},
        config={"configurable": {"app_thread_id": "thread-123", "run_id": "run-456"}},
    )

    assert "12 pages" in result
    call = next(params for method, params in calls if method == "tools/call")
    assert call["name"] == "get_thread_shape"
    assert call["_meta"]["com.askpdf/runtime-context"]["thread_id"] == "thread-123"
    assert call["_meta"]["com.askpdf/runtime-context"]["run_id"] == "run-456"


async def test_mcp_request_id_is_unique_per_call(monkeypatch):
    from app.mcp import langchain_adapter

    request_ids = []

    class FakeClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            assert method == "tools/call"
            request_ids.append(
                params["_meta"]["com.askpdf/runtime-context"]["mcp_request_id"]
            )
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"ok": True, "content": "ok", "sources": [], "artifacts": {}, "warnings": [], "metrics": {}, "trace": {}},
                "isError": False,
            }

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: FakeClient())
    await langchain_adapter.call_mcp_tool("wikipedia", {"query": "one"})
    await langchain_adapter.call_mcp_tool("wikipedia", {"query": "two"})

    assert len(request_ids) == 2
    assert request_ids[0].startswith("mcp-")
    assert request_ids[0] != request_ids[1]


async def test_mcp_wrapper_preserves_error_and_artifact_envelope(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_TOOL_MODE", "mcp")
    monkeypatch.setenv("MCP_TOOLS", "wikipedia")

    from app.mcp import langchain_adapter

    class FakeClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            assert method == "tools/call"
            assert params["name"] == "wikipedia"
            return {
                "content": [{"type": "text", "text": "Wikipedia unavailable"}],
                "structuredContent": {
                    "ok": False,
                    "content": "Wikipedia unavailable",
                    "sources": [],
                    "warnings": ["wikipedia_lookup_failed"],
                    "error": {"code": "wikipedia_lookup_failed"},
                    "artifacts": {"source": "wikipedia"},
                    "metrics": {}, "trace": {},
                },
                "isError": True,
            }

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: FakeClient())
    result = await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ayurveda"})

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "wikipedia_lookup_failed"
    assert payload["artifacts"] == {"source": "wikipedia"}


async def test_mcp_adapter_never_invokes_legacy_tool(monkeypatch):
    from app.mcp import langchain_adapter

    class FakeClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            assert method == "tools/call"
            assert params["name"] == "wikipedia"
            return {
                "content": [{"type": "text", "text": "MCP result"}],
                "structuredContent": {"ok": True, "content": "MCP result", "sources": [], "artifacts": {}, "warnings": [], "metrics": {}, "trace": {}},
                "isError": False,
            }

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: FakeClient())
    tool = langchain_adapter.create_mcp_langchain_tool("wikipedia")
    assert await tool.ainvoke({"query": "Ada Lovelace"})


@pytest.mark.asyncio
async def test_cancellation_checker_cancels_in_flight_mcp_call(monkeypatch):
    from app.mcp import langchain_adapter

    cancelled = False

    class SlowClient:
        async def request(self, method, params):
            nonlocal cancelled
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled = True
                raise

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: SlowClient())
    checks = 0

    async def checker():
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(asyncio.CancelledError):
        await langchain_adapter.call_mcp_tool("wikipedia", {"query": "cancel"}, {"configurable": {"cancellation_checker": checker}})
    assert cancelled is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError("offline"), TimeoutError("slow")])
async def test_mcp_transport_failures_raise_typed_error(monkeypatch, failure):
    from app.mcp import langchain_adapter
    from app.mcp.errors import MCPUnavailableError

    class BrokenClient:
        async def request(self, method, params):
            raise failure

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: BrokenClient())
    with pytest.raises(MCPUnavailableError) as caught:
        await langchain_adapter.call_mcp_tool(
            "wikipedia",
            {"query": "Ada"},
            {"configurable": {"run_id": "run-1", "tool_call_id": "call-1", "thread_id": "thread-1"}},
        )
    assert caught.value.tool_name == "wikipedia"
    assert caught.value.run_id == "run-1"
    assert caught.value.tool_call_id == "call-1"
    assert caught.value.category in {"connection", "timeout"}


@pytest.mark.asyncio
async def test_mcp_domain_error_result_remains_structured(monkeypatch):
    from app.mcp import langchain_adapter

    class DomainFailureClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            return {"content": [{"type": "text", "text": "provider failed"}], "structuredContent": {"ok": False, "content": "provider failed", "warnings": ["provider_failed"], "error": {"code": "provider_failed", "message": "provider failed", "type": "ProviderError", "retryable": True}, "sources": [], "artifacts": {}, "metrics": {}, "trace": {}}, "isError": True}

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: DomainFailureClient())
    result = json.loads(await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "provider_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("structured_ok", "is_error"),
    [(False, False), (True, True)],
)
async def test_mcp_rejects_contradictory_success_error_envelopes(
    monkeypatch,
    structured_ok,
    is_error,
):
    from app.mcp import langchain_adapter
    from app.mcp.errors import MCPUnavailableError

    class ContradictoryClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            return {
                "content": [{"type": "text", "text": "contradictory result"}],
                "structuredContent": {
                    "ok": structured_ok,
                    "content": "contradictory result",
                    "sources": [],
                    "artifacts": {},
                    "warnings": [],
                    "metrics": {},
                    "trace": {},
                },
                "isError": is_error,
            }

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: ContradictoryClient())
    with pytest.raises(MCPUnavailableError) as caught:
        await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"})
    assert caught.value.category == "protocol"
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_mcp_call_stage_protocol_failure_raises_typed_error(monkeypatch):
    from app.mcp import langchain_adapter
    from app.mcp.errors import MCPUnavailableError

    class CallFailureClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            raise RuntimeError("malformed MCP call response")

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: CallFailureClient())
    with pytest.raises(MCPUnavailableError) as caught:
        await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"})
    assert caught.value.category == "protocol"


@pytest.mark.asyncio
async def test_mcp_error_without_structured_content_remains_failure(monkeypatch):
    from app.mcp import langchain_adapter

    class ErrorTextClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            return {"content": [{"type": "text", "text": "protocol failure"}], "isError": True}

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: ErrorTextClient())
    payload = json.loads(await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"}))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "mcp_protocol_error"


@pytest.mark.asyncio
async def test_mcp_success_without_structured_content_raises_protocol_error(monkeypatch):
    from app.mcp import langchain_adapter
    from app.mcp.errors import MCPUnavailableError

    class TextOnlyClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            return {"content": [{"type": "text", "text": "looks successful"}], "isError": False}

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: TextOnlyClient())
    with pytest.raises(MCPUnavailableError) as caught:
        await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"})
    assert caught.value.category == "protocol"
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_malformed_structured_content_raises_protocol_error(monkeypatch):
    from app.mcp import langchain_adapter
    from app.mcp.errors import MCPUnavailableError

    class MalformedClient:
        async def request(self, method, params):
            if method == "tools/list":
                return {"tools": [{"name": "wikipedia", "description": "Wikipedia", "inputSchema": {"type": "object"}, "outputSchema": {"required": ["ok"]}, "_meta": {"com.askpdf/contract-id": "wikipedia_reference", "com.askpdf/contract-version": "1"}}]}
            return {"content": [{"type": "text", "text": "bad envelope"}], "structuredContent": {"ok": True, "content": "bad"}, "isError": False}

    monkeypatch.setattr(langchain_adapter, "get_mcp_client", lambda: MalformedClient())
    with pytest.raises(MCPUnavailableError) as caught:
        await langchain_adapter.call_mcp_tool("wikipedia", {"query": "Ada"})
    assert caught.value.category == "protocol"
    assert caught.value.retryable is False
import asyncio
import pytest

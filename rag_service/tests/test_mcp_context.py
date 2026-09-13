import pytest

from app.mcp.context_codec import RUNTIME_CONTEXT_KEY, decode_context, encode_context
from app.tools.context import ToolInvocationContext


def test_mcp_only_rejects_legacy_execution_settings(monkeypatch):
    from app.mcp.config import validate_mcp_configuration

    monkeypatch.setenv("MCP_TOOL_MODE", "legacy")
    try:
        validate_mcp_configuration()
    except RuntimeError as exc:
        assert "MCP-only" in str(exc)
    else:
        raise AssertionError("legacy MCP mode must be rejected")


def test_mcp_only_rejects_disabled_execution(monkeypatch):
    from app.mcp.config import validate_mcp_configuration

    monkeypatch.setenv("MCP_ENABLED", "false")
    try:
        validate_mcp_configuration()
    except RuntimeError as exc:
        assert "MCP-only" in str(exc)
    else:
        raise AssertionError("disabled MCP execution must be rejected")


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_mcp_request_timeout_fails_closed(monkeypatch, value):
    from app.mcp.config import validate_mcp_configuration

    monkeypatch.setenv("MCP_REQUEST_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError, match="MCP_REQUEST_TIMEOUT_SECONDS"):
        validate_mcp_configuration()


def test_context_round_trip_preserves_runtime_fields():
    original = ToolInvocationContext(
        thread_id="thread-1",
        run_id="run-1",
        tool_call_id="call-1",
        embedding_model="embed",
        context_window=32000,
        use_web_search=True,
        use_reranker=False,
        traceparent="00-trace",
    )

    encoded = encode_context(original)
    assert RUNTIME_CONTEXT_KEY in encoded
    assert decode_context(encoded) == original


def test_context_does_not_accept_malformed_scopes():
    context = ToolInvocationContext.from_mapping({"scopes": {"not": "a list"}})
    assert context.scopes == ()


def test_trace_context_injection_is_harmless_without_active_span():
    from app.mcp.telemetry import inject_trace_context

    carrier = {}
    assert inject_trace_context(carrier) is carrier


@pytest.mark.asyncio
async def test_active_parent_span_is_propagated_to_mcp_server_span(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from app.agent.tool_contract import ToolResult
    from app.mcp import server as server_module
    from app.mcp import tool_adapter
    from app.mcp.transport import InProcessMCPClient

    exporter = InMemorySpanExporter()
    monkeypatch.setenv("MCP_OTEL_ENABLED", "true")
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    tracer = provider.get_tracer("test")
    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]

    async def handler(request, context):
        return ToolResult(content="traceable")

    monkeypatch.setitem(
        server_module.MCP_TOOL_DEFINITIONS,
        "get_thread_shape",
        definition.__class__(definition.name, definition.request_model, handler, definition.registry_contract_id, definition.contract_version, definition.server_name),
    )
    monkeypatch.setattr(tool_adapter, "get_mcp_client", lambda: InProcessMCPClient())
    with tracer.start_as_current_span("parent") as parent:
        await tool_adapter.call_mcp_tool(
            "get_thread_shape",
            {},
            {"configurable": {"thread_id": "thread-1", "tool_call_id": "call-1"}},
        )

    spans = exporter.get_finished_spans()
    mcp_span = next(span for span in spans if span.name == "askpdf.mcp.tool")
    assert mcp_span.context.trace_id == parent.context.trace_id
    assert mcp_span.parent is not None
    assert mcp_span.parent.span_id == parent.context.span_id


@pytest.mark.asyncio
async def test_otel_disabled_does_not_create_mcp_spans(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from app.agent.tool_contract import ToolResult
    from app.mcp import server as server_module
    from app.mcp import tool_adapter
    from app.mcp.transport import InProcessMCPClient

    monkeypatch.setenv("MCP_OTEL_ENABLED", "false")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    definition = server_module.MCP_TOOL_DEFINITIONS["get_thread_shape"]

    async def handler(request, context):
        return ToolResult(content="trace-disabled")

    monkeypatch.setitem(
        server_module.MCP_TOOL_DEFINITIONS,
        "get_thread_shape",
        definition.__class__(definition.name, definition.request_model, handler, definition.registry_contract_id, definition.contract_version, definition.server_name),
    )
    monkeypatch.setattr(tool_adapter, "get_mcp_client", lambda: InProcessMCPClient())
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("parent"):
        await tool_adapter.call_mcp_tool(
            "get_thread_shape",
            {},
            {"configurable": {"thread_id": "thread-1", "tool_call_id": "call-1"}},
        )
    assert not [span for span in exporter.get_finished_spans() if span.name.startswith("askpdf.mcp")]


def test_mcp_result_decoder_extracts_nested_json_content():
    from app.mcp.result_decoder import decode_mcp_result

    decoded = decode_mcp_result({
        "ok": True,
        "content": '{"operations":[{"action":"create"}]}',
        "sources": [], "artifacts": {}, "metrics": {},
        "warnings": ["notice"],
        "trace": {"tool_call_id": "call-1"},
    })
    assert decoded.payload["operations"][0]["action"] == "create"
    assert decoded.warnings == ["notice"]
    assert decoded.envelope["trace"]["tool_call_id"] == "call-1"


def test_mcp_result_decoder_preserves_plain_and_text_only_results():
    from app.mcp.result_decoder import decode_mcp_result

    plain = decode_mcp_result("not JSON")
    assert plain.payload == {"content": "not JSON"}
    assert plain.ok is False
    assert plain.error["code"] == "mcp_protocol_error"
    text_only = decode_mcp_result('{"status":"approval_required"}')
    assert text_only.payload == {"status": "approval_required"}
    assert text_only.ok is False
    assert text_only.error["code"] == "mcp_protocol_error"


def test_mcp_result_decoder_preserves_failed_envelope():
    from app.mcp.result_decoder import decode_mcp_result

    decoded = decode_mcp_result({
        "ok": False,
        "content": '{"status":"disabled"}',
        "error": {"code": "web_search_disabled"},
        "warnings": ["WEB_SEARCH_DISABLED"],
    })
    assert decoded.ok is False
    assert decoded.payload["status"] == "disabled"
    assert decoded.error["code"] == "web_search_disabled"

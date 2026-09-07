import inspect

import httpx
import pytest


def _canonical_tool_result() -> dict:
    return {
        "ok": True,
        "content": "mcp-result",
        "sources": [],
        "artifacts": {},
        "warnings": [],
        "error": None,
        "metrics": {
            "elapsed_ms": 0.0,
            "result_chars": len("mcp-result"),
            "source_count": 0,
            "warning_count": 0,
        },
        "trace": {"tool_name": "search_documents"},
    }


@pytest.mark.asyncio
async def test_workflow_tool_invocation_dispatches_by_mcp_tool_name(monkeypatch):
    from langgraph_runtime.workflows import runtime_invocation

    calls = []

    class FakeExecutor:
        async def ainvoke(self, value, config=None):
            calls.append((value, config))
            return _canonical_tool_result()

    monkeypatch.setattr(
        runtime_invocation,
        "resolve_tool_executor",
        lambda tool_name, *, caller_node, config: (
            calls.append((tool_name, caller_node, config)) or FakeExecutor()
        ),
    )

    result = await runtime_invocation.invoke_tool_for_node(
        "search_documents",
        {"query": "question"},
        state={},
        config={},
        node="retrieval_worker",
        started=0.0,
    )

    assert result == _canonical_tool_result()
    assert calls[0][0] == "search_documents"
    assert calls[1][0] == {"query": "question"}
    assert "tool" not in inspect.signature(runtime_invocation.invoke_tool_for_node).parameters


def test_runtime_tool_fixture_is_not_accepted_without_the_canonical_envelope():
    from langgraph_runtime.agent.tool_contract import normalize_tool_result

    with pytest.raises(ValueError, match="canonical fields"):
        normalize_tool_result({"content": "mcp-result"}, tool_name="search_documents")


@pytest.mark.parametrize(
    "failure, expected",
    [
        (httpx.ConnectError("connection refused"), ("connection", True)),
        (httpx.ReadTimeout("read timed out"), ("timeout", True)),
        (ExceptionGroup("transport group", [ValueError("other"), httpx.ConnectError("down")]), ("connection", True)),
    ],
)
def test_mcp_transport_failures_are_retryable(failure, expected):
    from langgraph_runtime.mcp_client import classify_mcp_failure

    assert classify_mcp_failure(failure) == expected


def test_mcp_malformed_payloads_remain_non_retryable_protocol_failures():
    from langgraph_runtime.mcp_client import classify_mcp_failure

    assert classify_mcp_failure(ValueError("missing structuredContent")) == ("protocol", False)

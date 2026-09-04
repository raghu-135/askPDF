from types import SimpleNamespace

import httpx
import pytest


def test_graph_result_projection_does_not_expose_invocation_credentials():
    from langgraph_runtime.adapter import _result_from_graph

    token = "mcp-secret-test-token"
    result = _result_from_graph({
        "status": "completed",
        "agent_run_id": "run-1",
        "agent_workflow_id": "workflow-1",
        "answer": "safe",
        "structured_output": {"nested": {"mcp_execution_context_token": token}},
        "runtime_artifacts": [{"kind": "text", "content": "safe", "provenance": {"api_key": token}}],
        "usage": {"trace": {"authorization": token}},
    })

    serialized = str(result.to_dict())
    assert token not in serialized
    assert "agent_run_id" in result.runtime_metadata
    assert result.artifacts[0]["content"] == "safe"


def test_resume_and_continue_configs_require_and_install_fresh_mcp_grant():
    from langgraph_runtime.router_runtime import _runtime_config

    config = _runtime_config(
        app_thread_id="thread-1",
        checkpoint_thread_id="checkpoint-1",
        telemetry_sink={},
        deep_research_services_factory=lambda: None,
        mcp_execution_context_token="fresh-grant",
    )
    assert config["configurable"]["mcp_execution_context_token"] == "fresh-grant"


def test_missing_mcp_grant_fails_closed():
    from langgraph_runtime.adapter import LangGraphRuntimeAdapter

    request = SimpleNamespace(input={})
    with pytest.raises(Exception, match="fresh MCP execution grant"):
        LangGraphRuntimeAdapter()._mcp_token(request)


@pytest.mark.asyncio
async def test_provider_probe_uses_required_auth_header(monkeypatch):
    from langgraph_runtime.dependencies import probe_provider

    monkeypatch.setenv("LLM_AUTH_MODE", "required")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "model-1"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result["ok"] is True
    assert requests[0].headers["authorization"] == "Bearer provider-secret"


@pytest.mark.asyncio
async def test_provider_probe_omits_auth_for_keyless_mode(monkeypatch):
    from langgraph_runtime.dependencies import probe_provider

    monkeypatch.setenv("LLM_AUTH_MODE", "none")
    monkeypatch.setenv("LLM_KEYLESS_PROVIDER", "local")
    requests = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: (requests.append(request) or httpx.Response(200, json={"data": []}))
    ))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result["ok"] is True
    assert "authorization" not in requests[0].headers


def test_keyless_llm_client_uses_sdk_placeholder(monkeypatch):
    import langgraph_runtime.models.llm as llm_module

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)

    monkeypatch.setenv("LLM_API_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_AUTH_MODE", "none")
    monkeypatch.setenv("LLM_KEYLESS_PROVIDER", "local")
    llm_module.get_llm("local-model", own_async_transport=False)
    assert captured["api_key"] == "not-needed"


def test_outer_failure_result_contains_terminal_delta(monkeypatch):
    from langgraph_runtime.api import _terminal_result

    request = SimpleNamespace(agent_task_version=4, task_plan_revision=2)
    error = {"code": "runtime_execution_timeout", "retryable": True}
    result = _terminal_result(
        request,
        None,
        status="failed",
        error=error,
        operation_id="operation-1",
        attempt_id="run-1:attempt:1",
        boundary_event_id="run-1:terminal",
    )
    assert result.status == "failed"
    assert result.error == error
    assert result.orchestration_delta is not None
    assert result.orchestration_delta.result == {"status": "failed", "error": error}

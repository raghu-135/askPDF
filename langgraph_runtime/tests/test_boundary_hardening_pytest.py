import asyncio
from types import SimpleNamespace

import httpx
import pytest


def test_runtime_limits_are_configured_and_used_without_defaults(monkeypatch):
    from langgraph_runtime.models.llm import configure_runtime_limits, runtime_limits
    from langgraph_runtime.workflows.node_catalog import node_type_max_visits
    from langgraph_runtime.workflows.workflow_config_validation import collect_config_errors

    monkeypatch.setenv("DEFAULT_TOKEN_BUDGET", "16384")
    monkeypatch.setenv("REPLANS_LIMIT", "20")
    monkeypatch.setenv("MAX_CUSTOM_INSTRUCTIONS_CHARS", "4000")
    monkeypatch.setenv("MAX_SYSTEM_ROLE_CHARS", "1000")
    configure_runtime_limits()

    limits = runtime_limits()
    assert limits.default_token_budget == 16384
    assert limits.replans_limit == 20
    assert limits.max_custom_instructions_chars == 4000
    assert limits.max_system_role_chars == 1000
    assert node_type_max_visits("aggregator") == 21
    assert not any("replans" in error for error in collect_config_errors({"replans": 20}, "test"))


def test_task_policy_does_not_shadow_runtime_limits(monkeypatch):
    from langgraph_runtime.models.llm import configure_runtime_limits
    from langgraph_runtime.workflows.workflow_config_validation import collect_config_errors

    monkeypatch.setenv("DEFAULT_TOKEN_BUDGET", "16384")
    monkeypatch.setenv("REPLANS_LIMIT", "20")
    monkeypatch.setenv("MAX_CUSTOM_INSTRUCTIONS_CHARS", "4000")
    monkeypatch.setenv("MAX_SYSTEM_ROLE_CHARS", "1000")
    configure_runtime_limits()

    errors = collect_config_errors(
        {
            "replans": 20,
            "system_role": "",
            "custom_instructions": "",
            "task_policy": {
                "builtin_only": True,
                "profiles": ["document_researcher"],
                "limits": {"max_replans": 5},
            },
        },
        "deep_research_agent",
    )
    assert not any("attribute 'replans_limit'" in error for error in errors)
    assert not any("replans must be between" in error for error in errors)


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


def test_resumed_task_delta_uses_authoritative_plan_revision_after_planner_boundary():
    from langgraph_runtime.adapter import _result_from_graph

    result = _result_from_graph(
        {
            "status": "completed",
            "agent_task_id": "task-1",
            "agent_run_id": "run-1",
            "task_version": 2,
            # This value belongs to the original checkpoint and is stale after
            # the planner boundary was projected by the product service.
            "task_observed_plan_revision": 0,
            "task_plan_changes": [],
            "task_todos": [],
        },
        observed_plan_revision=1,
    )

    assert result.orchestration_delta is not None
    assert result.orchestration_delta.observed_plan_revision == 1


def test_ask_web_mode_blocks_context_prefetch_until_approval():
    from langgraph_runtime.graph import web_prefetch_allowed

    assert not web_prefetch_allowed({"use_web_search": True, "web_search_mode": "ask"})
    assert web_prefetch_allowed({"use_web_search": True, "web_search_mode": "on"})
    assert not web_prefetch_allowed({
        "use_web_search": True,
        "web_search_mode": "on",
        "hitl_policy": {
            "enabled": True,
            "tools": {"search_web": {"mode": "ask", "scope": "run"}},
        },
    })
    assert web_prefetch_allowed({"use_web_search": True, "web_search_mode": "on"})


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


@pytest.mark.asyncio
async def test_keyless_llm_client_uses_sdk_placeholder(monkeypatch):
    import langgraph_runtime.models.llm as llm_module

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ReasoningChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("LLM_API_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_AUTH_MODE", "none")
    monkeypatch.setenv("LLM_KEYLESS_PROVIDER", "local")
    client = httpx.AsyncClient()
    try:
        llm_module.get_llm("local-model", http_async_client=client)
        assert captured["api_key"] == "not-needed"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_required_auth_chat_client_sends_authorization_once(monkeypatch):
    from langchain_core.messages import HumanMessage

    import langgraph_runtime.models.llm as llm_module

    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_AUTH_MODE", "required")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        llm = llm_module.get_llm("deepseek/deepseek-chat", http_async_client=client)
        await llm.ainvoke([HumanMessage(content="hi there")])
    finally:
        await client.aclose()

    authorization = [value for key, value in requests[0].headers.raw if key.decode().lower() == "authorization"]
    assert authorization == [b"Bearer sk-or-test-key"]


def test_model_creation_requires_execution_client(monkeypatch):
    import langgraph_runtime.models.llm as llm_module

    monkeypatch.setenv("LLM_API_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_AUTH_MODE", "none")
    monkeypatch.setenv("LLM_KEYLESS_PROVIDER", "local")
    with pytest.raises(RuntimeError, match="outside an execution scope"):
        llm_module.get_llm("local-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_model_client_scope_closes_on_every_exit_path(monkeypatch, failure):
    import langgraph_runtime.models.llm as llm_module

    clients = []

    class FakeClient:
        async def aclose(self):
            self.closed = True

        def __init__(self):
            self.closed = False

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeClient)
    async def run():
        async with llm_module.model_client_scope() as client:
            clients.append(client)
            assert llm_module.execution_model_client({"configurable": {"model_client": client}}) is client
            if failure is not None:
                raise failure

    if failure is None:
        await run()
    else:
        with pytest.raises(type(failure)):
            await run()
    assert clients[-1].closed is True


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


def test_json_decision_parser_reads_reasoning_when_content_is_not_json():
    from types import SimpleNamespace

    from langgraph_runtime.agent.tool_contract import _safe_json_object
    from langgraph_runtime.workflows.decision_nodes import parse_json_decision

    response = SimpleNamespace(
        content="Hello there!\n",
        additional_kwargs={
            "reasoning_content": 'scratchpad\n{"route": "direct", "reason": "greeting", "tool_name": null, "query": null, "clarification_options": null}'
        },
        response_metadata={},
    )
    parsed = parse_json_decision(response, _safe_json_object)
    assert parsed["route"] == "direct"


def test_json_decision_parser_prefers_visible_json_content():
    from types import SimpleNamespace

    from langgraph_runtime.agent.tool_contract import _safe_json_object
    from langgraph_runtime.workflows.decision_nodes import parse_json_decision

    response = SimpleNamespace(
        content='{"route": "clarify"}',
        additional_kwargs={"reasoning_content": '{"route": "direct"}'},
        response_metadata={},
    )
    parsed = parse_json_decision(response, _safe_json_object)
    assert parsed["route"] == "clarify"

import uuid
import asyncio
from unittest.mock import AsyncMock

import pytest
from app.db.models_sqlmodel import AgentRun, AgentTask, AgentWorkflow
from app.services import tool_approval
from app.tools.context import ToolInvocationContext
from runtime_protocol.tool_approval import ApprovalMode, ApprovalScope, ToolApprovalPolicy, tool_approval_request
from runtime_protocol.tool_contract import ToolResult


def test_run_details_keep_identity_for_recovering_human_review():
    from app.api.agent_workflows import _run_payload

    run = AgentRun(id="paused-run", thread_id="thread", workflow_id="workflow", status="awaiting_human")
    assert _run_payload(run)["id"] == "paused-run"


@pytest.mark.asyncio
async def test_once_is_bound_to_run_invocation_tool_and_arguments(test_session_maker, sample_thread, monkeypatch):
    monkeypatch.setattr(tool_approval, "async_session_maker", test_session_maker)
    workflow_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    run = AgentRun(id=run_id, thread_id=sample_thread.id, workflow_id=workflow_id)
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(id=workflow_id, name=workflow_id, framework="langgraph", builder_id="langgraph_graph", spec_json={}))
            await session.flush()
            session.add(run)
    context = ToolInvocationContext(run_id=run_id, thread_id=sample_thread.id, extensions={
        "task_id": run_id,
        "tool_approval_policy": {"arbitrary_tool": {"mode": "ask", "scope": "run"}},
    })
    mode, request = await tool_approval.check_tool_approval("arbitrary_tool", {"value": 1}, context, invocation_id="call-1")
    assert mode is ApprovalMode.ASK
    request["interrupt_id"] = "interrupt-1"
    async with test_session_maker() as session:
        async with session.begin():
            await tool_approval.record_tool_decision(session, run, request, "approve")
    mode, _ = await tool_approval.check_tool_approval("arbitrary_tool", {"value": 1}, context, invocation_id="call-1")
    assert mode is ApprovalMode.ALLOW
    invoke = AsyncMock(return_value=ToolResult(content="executed"))
    for _ in range(2):
        result = await tool_approval.execute_tool_once("arbitrary_tool", {"value": 1}, context, invocation_id="call-1", invoke=invoke)
        assert result.content == "executed"
    invoke.assert_awaited_once()
    with pytest.raises(ValueError, match="different arguments"):
        await tool_approval.execute_tool_once("arbitrary_tool", {"value": 2}, context, invocation_id="call-1", invoke=invoke)
    for arguments, invocation in [({"value": 2}, "call-1"), ({"value": 1}, "call-2")]:
        mode, _ = await tool_approval.check_tool_approval("arbitrary_tool", arguments, context, invocation_id=invocation)
        assert mode is ApprovalMode.ASK
    cancelled = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await tool_approval.execute_tool_once("arbitrary_tool", {}, context, invocation_id="cancelled", invoke=cancelled)
    unknown = await tool_approval.execute_tool_once("arbitrary_tool", {}, context, invocation_id="cancelled", invoke=invoke)
    assert unknown.error.code == "tool_invocation_outcome_unknown"
    invoke.assert_awaited_once()
    started, finish = asyncio.Event(), asyncio.Event()

    async def slow_call():
        started.set()
        await finish.wait()
        return ToolResult(content="one side effect")

    running = asyncio.create_task(tool_approval.execute_tool_once("arbitrary_tool", {}, context, invocation_id="concurrent", invoke=slow_call))
    await started.wait()
    duplicate = await tool_approval.execute_tool_once("arbitrary_tool", {}, context, invocation_id="concurrent", invoke=invoke)
    assert duplicate.error.code == "tool_invocation_outcome_unknown"
    invoke.assert_awaited_once()
    finish.set()
    assert (await running).content == "one side effect"

    # A reopened approval can be resubmitted, and only the explicit scoped
    # action authorizes subsequent invocation identities.
    async with test_session_maker() as session:
        async with session.begin():
            await tool_approval.record_tool_decision(session, run, request, "approve_for_scope")
    mode, _ = await tool_approval.check_tool_approval("arbitrary_tool", {"value": 3}, context, invocation_id="call-3")
    assert mode is ApprovalMode.ALLOW


@pytest.mark.asyncio
async def test_denial_is_authoritative_for_future_tool_calls(test_session_maker, sample_thread, monkeypatch):
    monkeypatch.setattr(tool_approval, "async_session_maker", test_session_maker)
    workflow_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    run = AgentRun(id=run_id, thread_id=sample_thread.id, workflow_id=workflow_id)
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(id=workflow_id, name=workflow_id, framework="hermes", builder_id="hermes_agent", spec_json={}))
            await session.flush()
            session.add(run)
    request = tool_approval_request("search_web", {"query": "first"}, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="agent", response_operation="run.approval.respond")
    request["proposed_tool"]["invocation_id"] = "call-1"
    request["interrupt_id"] = "interrupt-1"
    async with test_session_maker() as session:
        async with session.begin():
            await tool_approval.record_tool_decision(session, run, request, "continue_without")
    context = ToolInvocationContext(run_id=run_id, thread_id=sample_thread.id, extensions={
        "tool_approval_policy": {"search_web": {"mode": "ask", "scope": "run"}},
    })
    mode, pending = await tool_approval.check_tool_approval("search_web", {"query": "different query"}, context, invocation_id="call-2")
    assert mode is ApprovalMode.DENY
    assert pending is None
    from app.mcp.transport import InProcessMCPClient
    from app.mcp import server
    definition = server.MCP_TOOL_DEFINITIONS["search_web"]
    handler = AsyncMock(return_value=ToolResult(content="must not run"))
    async def invoke(request, context):
        return await handler(request, context)
    monkeypatch.setitem(server.MCP_TOOL_DEFINITIONS, "search_web", definition.__class__(
        definition.name, definition.request_model, invoke, definition.registry_contract_id,
        definition.contract_version, definition.server_name,
    ))
    response = await InProcessMCPClient().request("tools/call", {
        "name": "search_web", "arguments": {"query": "different query", "_askpdf_invocation_id": "call-2"},
        "_meta": {"com.askpdf/runtime-context": context.as_dict()},
    })
    assert response["structuredContent"]["artifacts"]["approval_denied"]
    handler.assert_not_awaited()
    context.extensions["tool_approval_policy"]["search_documents"] = {"mode": "ask", "scope": "run"}
    response = await InProcessMCPClient().request("tools/call", {
        "name": "search_documents", "arguments": {"query": "requires approval", "_askpdf_invocation_id": "call-3"},
        "_meta": {"com.askpdf/runtime-context": context.as_dict()},
    })
    assert response["structuredContent"]["artifacts"]["approval_request"]["proposed_tool"]["name"] == "search_documents"


def test_disabled_web_setting_denies_all_external_tools():
    policies = tool_approval.invocation_policies(
        {"use_web_search": True}, permissions={"web_search_mode": "off"},
    )
    assert policies["search_web"]["mode"] == "deny"
    assert policies["arxiv"]["mode"] == "deny"


@pytest.mark.asyncio
async def test_explicit_task_grant_survives_new_run_without_leaking_to_other_tasks(test_session_maker, sample_thread):
    workflow_id, task_id, run_id = (str(uuid.uuid4()) for _ in range(3))
    run = AgentRun(id=run_id, thread_id=sample_thread.id, workflow_id=workflow_id, task_id=task_id)
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(id=workflow_id, name=workflow_id, framework="langgraph", builder_id="langgraph_graph", spec_json={}))
            await session.flush()
            session.add(AgentTask(id=task_id, thread_id=sample_thread.id, workflow_id=workflow_id, objective="test", objective_hash=task_id, create_idempotency_key=task_id))
            await session.flush()
            session.add(run)
    request = tool_approval_request("search_web", {"query": "first"}, policy=ToolApprovalPolicy(ApprovalMode.ASK, ApprovalScope.TASK), caller="agent", response_operation="run.resume")
    request["interrupt_id"] = "task-approval"
    request["proposed_tool"]["invocation_id"] = "first-call"
    async with test_session_maker() as session:
        async with session.begin():
            await tool_approval.record_tool_decision(session, run, request, "approve_for_scope")
    for scope, expected in [(task_id, ApprovalMode.ALLOW), ("different-task", ApprovalMode.ASK)]:
        context = ToolInvocationContext(run_id="next-run", thread_id=sample_thread.id, extensions={
            "task_id": scope, "tool_approval_policy": {"search_web": {"mode": "ask", "scope": "task"}},
        })
        mode, _ = await tool_approval.check_tool_approval("search_web", {"query": "second"}, context, invocation_id="second-call")
        assert mode is expected


def test_any_registered_tool_can_be_configured_for_human_approval():
    policies = tool_approval.invocation_policies({"hitl_policy": {"enabled": True, "tools": {
        "search_documents": {"mode": "ask", "scope": "run"},
    }}})
    assert policies["search_documents"] == {"mode": "ask", "scope": "run"}
    with pytest.raises(ValueError, match="Unknown tools"):
        tool_approval.invocation_policies({"hitl_policy": {"enabled": True, "tools": {
            "misspelled_tool": {"mode": "ask"},
        }}})


def test_approval_preserves_exact_arguments_and_rejects_oversized_requests():
    from app.product_orchestration.interrupts import normalize_pending_interrupt_payload

    arguments = {"code": "first\n  second", "nested": {"a": {"b": {"c": list(range(30))}}}}
    request = tool_approval_request("tool", arguments, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="agent", response_operation="run.resume")
    assert normalize_pending_interrupt_payload(request)["proposed_tool"]["arguments"] == arguments
    request["proposed_tool"]["arguments"]["code"] = "x" * 16000
    with pytest.raises(ValueError, match="too large"):
        normalize_pending_interrupt_payload(request)

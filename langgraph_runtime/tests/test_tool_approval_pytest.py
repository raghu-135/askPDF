import json
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, Send

from langgraph_runtime import mcp_client
from runtime_protocol.tool_approval import ApprovalMode, ToolApprovalPolicy, tool_approval_request


class State(TypedDict, total=False):
    result: str


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_mcp_tool_approval_resumes_exact_invocation(monkeypatch, approved):
    calls = []
    executed = []
    decisions = {}

    async def transport(name, arguments, config):
        invocation_id = arguments["_askpdf_invocation_id"]
        calls.append(invocation_id)
        if invocation_id not in decisions:
            request = tool_approval_request(name, {"query": arguments["query"]}, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="research", response_operation="run.resume")
            request["proposed_tool"]["invocation_id"] = invocation_id
            return json.dumps({"artifacts": {"approval_request": request}})
        if decisions[invocation_id]:
            executed.append(arguments["query"])
            return json.dumps({"content": "evidence", "artifacts": {}})
        return json.dumps({"content": "skipped", "artifacts": {"approval_denied": True}})

    monkeypatch.setattr(mcp_client, "_call_transport", transport)

    async def research(state, config):
        return {"result": await mcp_client._call("arbitrary_tool", {"query": "exact request"}, config)}

    graph = StateGraph(State)
    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-test", "agent_run_id": "run-1"}}
    paused = await app.ainvoke({}, config)
    assert paused["__interrupt__"]
    assert not executed
    pending = paused["__interrupt__"][0].value
    decisions[pending["proposed_tool"]["invocation_id"]] = approved
    completed = await app.ainvoke(Command(resume={"action": "approve" if approved else "continue_without"}), config)
    assert "__interrupt__" not in completed
    assert len(set(calls)) == 1
    assert executed == (["exact request"] if approved else [])


@pytest.mark.asyncio
async def test_resume_value_cannot_bypass_durable_permission(monkeypatch):
    async def transport(*args):
        return json.dumps({"artifacts": {"approval_request": {"proposed_tool": {}}}})
    monkeypatch.setattr(mcp_client, "_call_transport", transport)
    monkeypatch.setattr(mcp_client, "interrupt", lambda request: {"action": "approve"})
    with pytest.raises(mcp_client.MCPProtocolError):
        await mcp_client._call("tool", {}, {})


@pytest.mark.asyncio
async def test_deep_action_selection_is_checkpointed_without_double_config_injection(monkeypatch):
    from langgraph_runtime.workflows import deep_research_nodes
    calls = []

    async def model(state, config, node, messages, **kwargs):
        calls.append(node)
        assert config["configurable"]["thread_id"] == "model-selection"
        return "selected", {}

    monkeypatch.setattr(deep_research_nodes, "_call_model", model)

    async def research(state, config):
        result, _ = await deep_research_nodes._checkpointed_model_call(state, config, "research", [])
        mcp_client.interrupt({"type": "tool_approval"})
        return {"result": result}

    graph = StateGraph(State)
    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "model-selection"}}
    await app.ainvoke({}, config)
    completed = await app.ainvoke(Command(resume="approve"), config)
    assert completed["result"] == "selected"
    assert calls == ["research"]


@pytest.mark.asyncio
async def test_identical_sequential_calls_require_separate_approvals_and_do_not_replay(monkeypatch):
    decisions, executed = set(), []

    async def transport(name, arguments, config):
        identity = arguments["_askpdf_invocation_id"]
        if identity in decisions:
            executed.append(identity)
            return json.dumps({"content": "executed", "artifacts": {}})
        request = tool_approval_request(name, {}, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="research", response_operation="run.resume")
        request["proposed_tool"]["invocation_id"] = identity
        return json.dumps({"artifacts": {"approval_request": request}})

    monkeypatch.setattr(mcp_client, "_call_transport", transport)
    tool = mcp_client.create_mcp_langchain_tool("arbitrary_tool", checkpointed=True)

    async def research(state, config):
        from uuid import uuid4
        config = {**config, "configurable": {**config["configurable"], "subagent_id": str(uuid4())}}
        await tool.ainvoke({}, config)
        await tool.ainvoke({}, config)
        return {"result": "done"}

    graph = StateGraph(State)
    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "sequential", "agent_run_id": "run"}}
    result = await app.ainvoke({}, config)
    for expected in (1, 2):
        pending = result["__interrupt__"][0]
        identity = pending.value["proposed_tool"]["invocation_id"]
        assert identity not in decisions
        decisions.add(identity)
        result = await app.ainvoke(Command(resume={pending.id: {"action": "approve"}}), config)
        assert len(executed) == expected
    assert result["result"] == "done"


@pytest.mark.asyncio
async def test_parallel_interrupts_resume_only_the_approved_call(monkeypatch):
    decisions, executed = set(), []

    async def transport(name, arguments, config):
        identity = arguments["_askpdf_invocation_id"]
        if identity in decisions:
            executed.append(identity)
            return json.dumps({"content": "executed", "artifacts": {}})
        request = tool_approval_request(name, {}, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="research", response_operation="run.resume")
        request["proposed_tool"]["invocation_id"] = identity
        return json.dumps({"artifacts": {"approval_request": request}})

    monkeypatch.setattr(mcp_client, "_call_transport", transport)
    tool = mcp_client.create_mcp_langchain_tool("arbitrary_tool", checkpointed=True)

    async def research(state, config):
        await tool.ainvoke({}, config)
        return {}

    graph = StateGraph(State)
    graph.add_node("research", research)
    graph.add_conditional_edges(START, lambda state: [Send("research", {}), Send("research", {})])
    graph.add_edge("research", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "parallel", "agent_run_id": "run"}}
    result = await app.ainvoke({}, config)
    assert len(result["__interrupt__"]) == 2
    assert not executed
    for expected in (1, 2):
        pending = result["__interrupt__"][0]
        decisions.add(pending.value["proposed_tool"]["invocation_id"])
        result = await app.ainvoke(Command(resume={pending.id: {"action": "approve"}}), config)
        assert len(executed) == expected
    assert "__interrupt__" not in result

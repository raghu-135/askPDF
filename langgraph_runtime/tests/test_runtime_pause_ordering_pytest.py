from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from langgraph_runtime.graph import NodeRegistry
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


class _PauseState(TypedDict, total=False):
    answer: str


@pytest.mark.asyncio
async def test_hitl_gate_graph_interrupt_bypasses_node_failure_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = NodeRegistry().get_for_spec({
        "id": "web_approval_gate",
        "type": "hitl_gate",
    })
    interrupt = GraphInterrupt([Interrupt(value={"gate_id": "web_approval_gate"}, id="interrupt-1")])
    monkeypatch.setattr("langgraph_runtime.graph.interrupt", lambda _payload: (_ for _ in ()).throw(interrupt))

    with pytest.raises(GraphInterrupt) as raised:
        await gate({
        "hitl_policy": {
            "enabled": True,
            "gates": {
                "web_approval_gate": {
                    "mode": "approval",
                    "phase": "before",
                    "allowed_actions": ["approve", "approve_for_scope", "continue_without"],
                    "default_action": "continue_without",
                },
            },
        },
        "available_worker_nodes": [{"id": "web_worker", "type": "web_worker"}],
        "work_item_proposals": [{"worker_node_id": "web_worker", "worker_type": "web_worker"}],
        }, {"configurable": {"thread_id": "hitl-gate-bubble"}})

    assert raised.value is interrupt


@pytest.mark.asyncio
async def test_pause_resume_cannot_answer_following_hitl_interrupt() -> None:
    pause = {"token": "pause-1", "requested": True}
    calls: list[str] = []
    consumed: list[str] = []

    async def pause_checker() -> bool:
        return pause["requested"]

    async def pause_token_reader() -> str | None:
        return pause["token"] if pause["requested"] else None

    async def pause_consumer(token: str | None) -> bool:
        assert token == pause["token"]
        consumed.append(token)
        pause["requested"] = False
        return True

    registry = NodeRegistry()
    pause_gate = registry.get_for_spec({
        "id": "__task_pause_gate__business",
        "type": "task_pause_gate",
        "target_node_id": "business",
    })

    async def business(_state: _PauseState):
        calls.append("business")
        decision = interrupt({
            "type": "tool_approval",
            "allowed_actions": ["approve"],
        })
        return {"answer": str(decision["action"])}

    graph = StateGraph(_PauseState)
    graph.add_node("pause_gate", pause_gate)
    graph.add_node("business", business)
    graph.add_edge(START, "pause_gate")
    graph.add_edge("pause_gate", "business")
    graph.add_edge("business", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "pause-ordering",
            "pause_checker": pause_checker,
            "pause_token_reader": pause_token_reader,
            "pause_consumer": pause_consumer,
        }
    }

    first = await app.ainvoke({}, config=config)
    assert first["__interrupt__"][0].value["type"] == "task_pause"
    assert calls == []

    # This is the runtime resume lifecycle: claim the pending pause before
    # re-entering LangGraph.  The graph must not consume it after execution.
    await pause_consumer(await pause_token_reader())
    second = await app.ainvoke(Command(resume={"action": "approve"}), config=config)
    assert second["__interrupt__"][0].value["type"] == "tool_approval"
    assert calls == ["business"]

    third = await app.ainvoke(Command(resume={"action": "approve"}), config=config)
    assert third["answer"] == "approve"
    assert calls == ["business", "business"]
    assert consumed == ["pause-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reject", "unknown"])
async def test_task_pause_gate_rejects_invalid_decisions_without_consuming_pause(monkeypatch, action: str) -> None:
    pause = {"requested": True, "token": "pause-invalid"}

    async def pause_checker() -> bool:
        return pause["requested"]

    async def pause_token_reader() -> str:
        return pause["token"]

    gate = NodeRegistry().get_for_spec({
        "id": "__task_pause_gate__business",
        "type": "task_pause_gate",
        "target_node_id": "business",
    })
    monkeypatch.setattr(
        "langgraph_runtime.graph.interrupt",
        lambda _payload: {"action": action},
    )

    with pytest.raises(AgentRuntimeError, match="approve or resume"):
        await gate(
            {},
            {"configurable": {
                "pause_checker": pause_checker,
                "pause_token_reader": pause_token_reader,
            }},
        )
    assert pause["requested"] is True


def test_compiler_rejects_rejection_route_on_cooperative_pause_gate() -> None:
    from langgraph_runtime.compiler import WorkflowCompiler

    spec = {
        "config": {"graph": {
            "nodes": [{
                "id": "pause",
                "type": "task_pause_gate",
                "allowed_actions": ["approve", "reject"],
                "routes": {"reject": "END"},
            }],
            "edges": [],
        }},
    }
    with pytest.raises(ValueError, match="cannot advertise or route reject"):
        WorkflowCompiler()._with_pause_gates(spec["config"]["graph"])

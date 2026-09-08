from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_runtime.graph import NodeRegistry


class _PauseState(TypedDict, total=False):
    answer: str


@pytest.mark.asyncio
async def test_pause_resume_cannot_answer_following_hitl_interrupt() -> None:
    pause = {"token": "pause-1", "requested": True}
    calls: list[str] = []

    async def pause_checker() -> bool:
        return pause["requested"]

    async def pause_token_reader() -> str | None:
        return pause["token"] if pause["requested"] else None

    async def pause_consumer(token: str | None) -> bool:
        assert token == pause["token"]
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

    second = await app.ainvoke(Command(resume={"action": "approve"}), config=config)
    assert second["__interrupt__"][0].value["type"] == "tool_approval"
    assert calls == ["business"]
    await pause_consumer("pause-1")

    third = await app.ainvoke(Command(resume={"action": "approve"}), config=config)
    assert third["answer"] == "approve"
    assert calls == ["business"]

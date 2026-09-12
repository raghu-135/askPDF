from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_runtime.checkpointing import open_agent_checkpointer
from langgraph_runtime.adapter import _public_value
from langgraph_runtime.router_runtime import _require_resume_checkpoint


class _CheckpointState(TypedDict, total=False):
    question: str
    decision: str


def _approval_node(state: _CheckpointState) -> _CheckpointState:
    decision = interrupt({"kind": "approval", "question": state["question"]})
    return {"decision": str(decision)}


def _compile_approval_graph(checkpointer):
    graph = StateGraph(_CheckpointState)
    graph.add_node("approval", _approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=checkpointer)


def test_resume_checkpoint_guard_rejects_restart_without_pending_interrupt():
    snapshot = SimpleNamespace(next=("context_loader",), tasks=(SimpleNamespace(interrupts=()),))

    with pytest.raises(Exception) as caught:
        _require_resume_checkpoint(snapshot, run_id="run-1")

    assert caught.value.code == "runtime_resume_checkpoint_invalid"
    assert caught.value.retryable is False


def test_resume_checkpoint_guard_accepts_durable_loop_checkpoint_without_task_interrupt_metadata():
    _require_resume_checkpoint(
        SimpleNamespace(
            next=("web_approval_gate",),
            tasks=(SimpleNamespace(interrupts=()),),
            metadata={"source": "loop", "step": 10},
        ),
        run_id="run-1",
    )


def test_public_interrupt_keeps_resume_flag_but_hides_checkpoint_reference():
    projected = _public_value({
        "checkpoint_resume": True,
        "checkpoint_thread_id": "secret-thread",
        "prompt": "Approve",
    })

    assert projected == {"checkpoint_resume": True, "prompt": "Approve"}


@pytest.mark.asyncio
async def test_runtime_graph_resumes_after_postgres_checkpointer_reopen():
    thread_id = f"runtime-checkpoint-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    async with open_agent_checkpointer(setup=False) as first_checkpointer:
        first_graph = _compile_approval_graph(first_checkpointer)
        paused = await first_graph.ainvoke({"question": "Continue?"}, config=config)

    assert paused["__interrupt__"]
    assert paused["__interrupt__"][0].value == {"kind": "approval", "question": "Continue?"}

    async with open_agent_checkpointer(setup=False) as second_checkpointer:
        second_graph = _compile_approval_graph(second_checkpointer)
        resumed = await second_graph.ainvoke(Command(resume="approved"), config=config)
        snapshot = await second_graph.aget_state(config)
        await second_checkpointer.adelete_thread(thread_id)

    assert resumed["decision"] == "approved"
    assert snapshot.values["decision"] == "approved"
    assert not snapshot.next

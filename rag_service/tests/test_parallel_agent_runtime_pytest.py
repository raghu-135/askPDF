from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError

from app.agent_workflows.compiler import WorkflowCompiler
from app.agent_workflows.graph import NodeRegistry
from app.agent_workflows.parallel_runtime import (
    ParallelWorkerError,
    ParallelDispatchDeadlineExceeded,
    aggregate_parallel_results,
    cancelled_parallel_dispatch,
    dispatch_sends,
    normalize_work_items,
    normalized_parallel_policy,
    parallel_runtime_authorized,
)
from app.agent_workflows.state import merge_parallel_deltas
from app.agent_workflows.validator import WorkflowValidator


BUILTIN = Path(__file__).parents[1] / "app" / "agent_workflows" / "builtins" / "orchestrator_worker_rag_agent.json"


def _state():
    return {
        "agent_run_id": "run-1",
        "question": "Compare the uploaded paper with our saved preference",
        "use_web_search": False,
        "parallel_policy": normalized_parallel_policy({}),
        "available_worker_nodes": [
            {"id": "documents", "type": "retrieval_worker"},
            {"id": "memory", "type": "durable_memory_worker"},
            {"id": "web", "type": "web_worker"},
        ],
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "used_memory_ids": [],
        "evidence_packets": [],
        "node_events": [],
        "tool_events": [],
        "errors": [],
        "skipped_nodes": [],
        "node_visit_counts": {"planner": 1},
        "node_visit_sequence": [{"node": "planner", "node_type": "planner", "visit_index": 1}],
    }


def test_work_item_normalization_is_stable_bounded_and_read_only():
    state = _state()
    proposals = [
        {"worker_node_id": "documents", "query": "  paper   results "},
        {"worker_node_id": "documents", "query": "paper results"},
        {"worker_node_id": "memory", "query": "saved preference"},
        {"worker_node_id": "web", "query": "latest news"},
        {"worker_node_id": "finalizer", "query": "unsafe"},
    ]
    first = normalize_work_items(proposals, state=state, dispatch_node_id="dispatch", dispatch_visit=1)
    second = normalize_work_items(proposals, state=state, dispatch_node_id="dispatch", dispatch_visit=1)

    assert first == second
    assert [item["worker_node_id"] for item in first] == ["documents", "memory"]
    assert [item["ordinal"] for item in first] == [0, 1]
    assert len({item["work_id"] for item in first}) == 2
    assert first[0]["timeout_ms"] == 30_000


def test_dispatch_sends_only_incomplete_work_with_dynamic_timeout():
    state = _state()
    items = normalize_work_items(
        [{"worker_node_id": "documents", "query": "paper"}, {"worker_node_id": "memory", "query": "preference"}],
        state=state,
        dispatch_node_id="dispatch",
        dispatch_visit=1,
    )
    state.update({
        "work_items": items,
        "worker_result_packets": [{"work_id": items[0]["work_id"], "status": "completed"}],
        "parallel_aggregator_id": "aggregate",
    })
    sends = dispatch_sends(state)
    assert len(sends) == 1
    assert sends[0].node == "memory"
    assert sends[0].arg["work_item"]["work_id"] == items[1]["work_id"]
    assert sends[0].timeout.run_timeout == 30


def _packet(*, ordinal: int, worker: str, status: str = "completed"):
    return {
        "dispatch_id": "dispatch-1",
        "dispatch_visit": 1,
        "work_id": f"work-{ordinal}",
        "ordinal": ordinal,
        "worker_node_id": worker,
        "worker_type": "retrieval_worker",
        "attempt": 1,
        "status": status,
        "evidence_packets": [{
            "kind": "document",
            "content": f"evidence {ordinal}",
            "fingerprint": f"fingerprint-{ordinal}",
        }] if status == "completed" else [],
        "document_sources": [{"file_id": f"file-{ordinal}", "page": ordinal, "content": f"evidence {ordinal}"}] if status == "completed" else [],
        "web_sources": [],
        "chat_ids": [],
        "memory_refs": [],
        "node_events": [],
        "tool_events": [],
        "errors": [] if status == "completed" else [{"code": "worker_failed", "work_id": f"work-{ordinal}"}],
        "elapsed_ms": 10,
    }


def test_aggregation_is_deterministic_across_completion_order():
    packets = [_packet(ordinal=2, worker="c"), _packet(ordinal=0, worker="a"), _packet(ordinal=1, worker="b", status="failed")]
    snapshots = []
    for seed in range(8):
        shuffled = list(packets)
        random.Random(seed).shuffle(shuffled)
        state = _state()
        state.update({"dispatch_id": "dispatch-1", "work_items": [{}, {}, {}], "worker_result_packets": shuffled})
        snapshots.append(json.dumps(aggregate_parallel_results(state), sort_keys=True))

    assert len(set(snapshots)) == 1
    result = json.loads(snapshots[0])
    assert result["parallel_summary"]["completed"] == 2
    assert result["parallel_summary"]["failed"] == 1
    assert result["parallel_summary"]["partial_evidence"] is True
    assert [source["file_id"] for source in result["document_sources"]] == ["file-0", "file-2"]


def test_aggregation_rejects_all_failed_attempted_workers():
    state = _state()
    state.update({"dispatch_id": "dispatch-1", "work_items": [{}], "worker_result_packets": [_packet(ordinal=0, worker="a", status="failed")]})
    with pytest.raises(RuntimeError, match="parallel_dispatch_no_usable_results"):
        aggregate_parallel_results(state)


def test_orchestrator_builtin_validates_and_compiles_without_changing_other_builtins():
    payload = json.loads(BUILTIN.read_text(encoding="utf-8"))
    spec = payload["spec_json"]
    WorkflowValidator().validate(spec)
    graph = WorkflowCompiler().compile(spec).get_graph()

    assert spec["runtime"]["features"]["supports_parallel_dispatch"] is True
    assert set(graph.nodes) >= {"parallel_dispatch", "aggregator", "retrieval_worker"}


def test_parallel_validator_rejects_extra_worker_exit_and_hitl_inside_region():
    spec = json.loads(BUILTIN.read_text(encoding="utf-8"))["spec_json"]
    spec["config"]["graph"]["nodes"].append({"id": "inside_review", "type": "hitl_gate"})
    spec["config"]["graph"]["edges"].append({"from": "retrieval_worker", "to": "inside_review"})

    with pytest.raises(Exception, match="cannot have exits outside aggregator|parallel regions cannot contain HITL"):
        WorkflowValidator().validate(spec)


@pytest.mark.asyncio
async def test_compiled_orchestrator_fans_out_and_joins_once():
    spec = json.loads(BUILTIN.read_text(encoding="utf-8"))["spec_json"]
    registry = NodeRegistry()

    async def context_loader(state, _config):
        return {"pre_fetch_bundle": {}, "node_events": []}

    async def planner(state, _config):
        workers = [
            "retrieval_worker",
            "thread_conversation_history_worker",
            "durable_memory_worker",
            "thread_events_worker",
            "web_worker",
        ]
        return {
            "route": "execute",
            "route_reason": "parallel test",
            "execution_plan": workers,
            "work_item_proposals": [
                *({"worker_node_id": worker_id, "query": f"query-{index}"} for index, worker_id in enumerate(workers)),
                {"worker_node_id": "retrieval_worker", "query": "query-5"},
                {"worker_node_id": "retrieval_worker", "query": "query-6"},
                {"worker_node_id": "retrieval_worker", "query": "query-7"},
            ],
            "node_events": [],
        }

    active_workers = 0
    peak_workers = 0

    async def worker(state, _config):
        nonlocal active_workers, peak_workers
        active_workers += 1
        peak_workers = max(peak_workers, active_workers)
        try:
            await asyncio.sleep(0.02)
            item = state["work_item"]
            return {"worker_result_packets": [{
                **item,
                "status": "completed",
                "attempt": 1,
                "evidence_packets": [{"kind": item["evidence_kind"], "content": item["query"], "fingerprint": item["work_id"]}],
                "document_sources": [],
                "web_sources": [],
                "chat_ids": [],
                "memory_refs": [],
                "node_events": [],
                "tool_events": [],
                "errors": [],
                "elapsed_ms": 1,
            }]}
        finally:
            active_workers -= 1

    async def synthesizer(_state, _config):
        return {"final_answer": "joined", "node_events": []}

    async def finalizer(state, _config):
        return {"final_answer": state["final_answer"], "node_events": []}

    registry._nodes["context_loader"] = context_loader
    registry._nodes["planner"] = planner
    for node_type in (
        "retrieval_worker",
        "thread_conversation_history_worker",
        "durable_memory_worker",
        "thread_events_worker",
        "web_worker",
    ):
        registry._nodes[node_type] = worker
    registry._nodes["synthesizer"] = synthesizer
    registry._nodes["finalizer"] = finalizer

    initial = _state()
    initial.update({
        "agent_run_id": "run-graph",
        "thread_id": "thread-1",
        "llm_model": "unused",
        "embedding_model": "unused",
        "use_web_search": True,
        "parallel_enabled": True,
        "parallel_runtime_override": True,
        "parallel_aggregator_id": "aggregator",
        "worker_result_packets": [],
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "available_worker_nodes": [
            {"id": node_type, "type": node_type}
            for node_type in (
                "retrieval_worker",
                "thread_conversation_history_worker",
                "durable_memory_worker",
                "thread_events_worker",
                "web_worker",
            )
        ],
        "loop_policy": spec["config"]["loop_policy"],
    })
    result = await WorkflowCompiler(registry).compile(spec).ainvoke(
        initial,
        config={"configurable": {"thread_id": "checkpoint-1"}, "max_concurrency": 4},
    )

    assert result["final_answer"] == "joined"
    assert result["parallel_summary"]["planned"] == 8
    assert result["parallel_summary"]["completed"] == 8
    assert [packet["content"] for packet in result["evidence_packets"]] == [f"query-{index}" for index in range(8)]
    assert peak_workers == 4


@pytest.mark.asyncio
async def test_parallel_worker_returns_only_one_result_delta(monkeypatch):
    registry = NodeRegistry()
    item = {
        "dispatch_id": "dispatch-1",
        "dispatch_visit": 1,
        "work_id": "work-1",
        "ordinal": 0,
        "worker_node_id": "retrieval_worker",
        "worker_type": "retrieval_worker",
        "query": "paper",
        "evidence_kind": "document",
        "dedupe_key": "key",
        "attempt": 1,
        "timeout_ms": 30_000,
    }
    local_result = {
        "evidence_packets": [{"kind": "document", "content": "result", "fingerprint": "one"}],
        "document_sources": [{"file_id": "file-1", "page": 1}],
        "web_sources": [],
        "used_chat_ids": [],
        "used_memory_ids": [],
        "node_events": [{"status": "completed"}],
        "tool_events": [{"tool_name": "search_documents"}],
    }
    runner = AsyncMock(return_value=local_result)
    monkeypatch.setattr(registry, "_run_sequential_tool_worker", runner)
    state = {**_state(), "work_item": item, "dispatch_deadline_epoch_ms": 9_999_999_999_999}

    update = await registry._parallel_tool_worker("retrieval_worker", state, {})

    assert "worker_result_packets" in update
    assert "parallel_visit_records" in update
    assert "parallel_attempt_records" in update
    assert len(update["worker_result_packets"]) == 1
    assert update["worker_result_packets"][0]["work_id"] == "work-1"
    assert update["worker_result_packets"][0]["status"] == "completed"
    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_parallel_worker_does_not_retry_contract_errors(monkeypatch):
    registry = NodeRegistry()
    item = {
        "dispatch_id": "dispatch-1",
        "dispatch_visit": 1,
        "work_id": "work-1",
        "ordinal": 0,
        "worker_node_id": "retrieval_worker",
        "worker_type": "retrieval_worker",
        "query": "paper",
        "evidence_kind": "document",
        "dedupe_key": "key",
        "attempt": 1,
        "timeout_ms": 30_000,
    }
    runner = AsyncMock(side_effect=ValueError("tool permission denied"))
    monkeypatch.setattr(registry, "_run_sequential_tool_worker", runner)
    state = {**_state(), "work_item": item, "dispatch_deadline_epoch_ms": 9_999_999_999_999}

    with pytest.raises(ParallelWorkerError) as raised:
        await registry._parallel_tool_worker("retrieval_worker", state, {})

    assert raised.value.attempt == 1
    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_queued_worker_after_dispatch_deadline_is_non_retryable(monkeypatch):
    registry = NodeRegistry()
    item = {
        "dispatch_id": "dispatch-1",
        "dispatch_visit": 1,
        "work_id": "work-late",
        "ordinal": 0,
        "worker_node_id": "retrieval_worker",
        "worker_type": "retrieval_worker",
        "query": "paper",
        "evidence_kind": "document",
        "dedupe_key": "key",
        "attempt": 1,
        "timeout_ms": 30_000,
    }
    runner = AsyncMock()
    monkeypatch.setattr(registry, "_run_sequential_tool_worker", runner)
    state = {**_state(), "work_item": item, "dispatch_deadline_epoch_ms": 1}

    with pytest.raises(ParallelWorkerError) as raised:
        await registry._parallel_tool_worker("retrieval_worker", state, {})

    assert isinstance(raised.value.error, ParallelDispatchDeadlineExceeded)
    assert runner.await_count == 0


def test_parallel_reducer_is_identity_aware_across_checkpoint_replay():
    packet = {"work_id": "work-1", "attempt": 1, "status": "completed", "value": "first"}
    assert merge_parallel_deltas([packet], [dict(packet)]) == [packet]
    assert len(merge_parallel_deltas([packet], [{**packet, "attempt": 2}])) == 2


def test_parallel_rollout_gate_requires_reference_builtin_or_builder_override(monkeypatch):
    monkeypatch.delenv("ASKPDF_AGENT_WORKFLOW_PARALLEL_V1", raising=False)
    assert parallel_runtime_authorized({"workflow_id": "orchestrator_worker_rag_agent"}) is False
    assert parallel_runtime_authorized({"workflow_id": "custom", "parallel_runtime_override": True}) is True
    monkeypatch.setenv("ASKPDF_AGENT_WORKFLOW_PARALLEL_V1", "true")
    assert parallel_runtime_authorized({"workflow_id": "orchestrator_worker_rag_agent"}) is True
    assert parallel_runtime_authorized({"workflow_id": "custom"}) is False


def test_cancellation_materializes_queued_and_active_terminal_work():
    state = _state()
    items = normalize_work_items(
        [{"worker_node_id": "documents", "query": "paper"}, {"worker_node_id": "memory", "query": "preference"}],
        state=state,
        dispatch_node_id="dispatch",
        dispatch_visit=1,
    )
    events = [
        {"event": "worker.queued", "data": dict(items[0])},
        {"event": "worker.queued", "data": dict(items[1])},
        {"event": "worker.started", "data": {**items[0], "attempt": 1}},
    ]
    update = cancelled_parallel_dispatch({**state, "work_items": items}, events)
    packets = update["worker_result_packets"]
    assert [packet["status"] for packet in packets] == ["cancelled", "cancelled"]
    assert update["parallel_summary"]["cancelled"] == 2
    assert {event["reason"] for event in update["parallel_node_event_deltas"]} == {"active_cancelled", "queued_cancelled"}


@pytest.mark.asyncio
async def test_compiled_worker_retry_is_owned_by_langgraph(monkeypatch):
    spec = json.loads(BUILTIN.read_text(encoding="utf-8"))["spec_json"]
    registry = NodeRegistry()

    async def context_loader(_state, _config):
        return {"pre_fetch_bundle": {}, "node_events": []}

    async def planner(_state, _config):
        return {
            "route": "execute",
            "route_reason": "retry test",
            "execution_plan": ["retrieval_worker"],
            "work_item_proposals": [{"worker_node_id": "retrieval_worker", "query": "documents"}],
            "node_events": [],
        }

    async def synthesizer(_state, _config):
        return {"final_answer": "joined", "node_events": []}

    async def finalizer(state, _config):
        return {"final_answer": state["final_answer"], "node_events": []}

    registry._nodes["context_loader"] = context_loader
    registry._nodes["planner"] = planner
    registry._nodes["synthesizer"] = synthesizer
    registry._nodes["finalizer"] = finalizer
    worker_result = {
        "evidence_packets": [{"kind": "document", "content": "retried evidence"}],
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "used_memory_ids": [],
        "node_events": [{"status": "completed"}],
        "tool_events": [],
    }
    runner = AsyncMock(side_effect=[ConnectionError("temporary network failure"), worker_result])
    monkeypatch.setattr(registry, "_run_sequential_tool_worker", runner)
    initial = _state()
    initial.update({
        "agent_run_id": "run-retry",
        "workflow_id": "orchestrator_worker_rag_agent",
        "thread_id": "thread-1",
        "llm_model": "unused",
        "embedding_model": "unused",
        "parallel_enabled": True,
        "parallel_runtime_override": True,
        "parallel_aggregator_id": "aggregator",
        "worker_result_packets": [],
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "available_worker_nodes": [{"id": "retrieval_worker", "type": "retrieval_worker"}],
        "loop_policy": spec["config"]["loop_policy"],
    })

    result = await WorkflowCompiler(registry).compile(spec).ainvoke(
        initial,
        config={"configurable": {"thread_id": "retry-checkpoint"}, "max_concurrency": 4},
    )

    assert runner.await_count == 2
    assert result["worker_result_packets"][0]["attempt"] == 2
    assert [record["attempt"] for record in result["parallel_attempt_records"]] == [1, 2]


@pytest.mark.asyncio
async def test_checkpoint_resume_reuses_completed_parallel_pending_writes(monkeypatch):
    spec = json.loads(BUILTIN.read_text(encoding="utf-8"))["spec_json"]
    registry = NodeRegistry()

    async def context_loader(_state, _config):
        return {"pre_fetch_bundle": {}, "node_events": []}

    async def planner(_state, _config):
        return {
            "route": "execute",
            "execution_plan": ["retrieval_worker", "durable_memory_worker"],
            "work_item_proposals": [
                {"worker_node_id": "retrieval_worker", "query": "documents"},
                {"worker_node_id": "durable_memory_worker", "query": "memory"},
            ],
            "node_events": [],
        }

    async def passthrough_synthesizer(_state, _config):
        return {"final_answer": "joined", "node_events": []}

    async def passthrough_finalizer(state, _config):
        return {"final_answer": state["final_answer"], "node_events": []}

    registry._nodes.update({
        "context_loader": context_loader,
        "planner": planner,
        "synthesizer": passthrough_synthesizer,
        "finalizer": passthrough_finalizer,
    })
    calls = {"retrieval_worker": 0, "durable_memory_worker": 0}
    memory_can_finish = False

    async def worker_runner(node_name, _state, _config):
        nonlocal memory_can_finish
        calls[node_name] += 1
        if node_name == "durable_memory_worker" and not memory_can_finish:
            await asyncio.sleep(0.05)
            raise ValueError("forced first-superstep failure")
        return {
            "evidence_packets": [{"kind": "document", "content": node_name}],
            "document_sources": [],
            "web_sources": [],
            "used_chat_ids": [],
            "used_memory_ids": [],
            "node_events": [{"status": "completed"}],
            "tool_events": [],
        }

    monkeypatch.setattr(registry, "_run_sequential_tool_worker", worker_runner)
    normal_error_handler = registry.get_parallel_error_handler_for_spec

    def error_handler(node_spec):
        if node_spec.get("id") != "durable_memory_worker":
            return normal_error_handler(node_spec)

        async def fail_handler(_state, error: NodeError):
            del error
            raise RuntimeError("simulate process loss after pending writes")

        return fail_handler

    monkeypatch.setattr(registry, "get_parallel_error_handler_for_spec", error_handler)
    checkpointer = InMemorySaver()
    app = WorkflowCompiler(registry).compile(spec, checkpointer=checkpointer)
    initial = _state()
    initial.update({
        "agent_run_id": "run-resume",
        "workflow_id": "orchestrator_worker_rag_agent",
        "thread_id": "thread-1",
        "llm_model": "unused",
        "embedding_model": "unused",
        "parallel_enabled": True,
        "parallel_runtime_override": True,
        "parallel_aggregator_id": "aggregator",
        "worker_result_packets": [],
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "available_worker_nodes": [
            {"id": "retrieval_worker", "type": "retrieval_worker"},
            {"id": "durable_memory_worker", "type": "durable_memory_worker"},
        ],
        "loop_policy": spec["config"]["loop_policy"],
    })
    config = {"configurable": {"thread_id": "pending-write-resume"}, "max_concurrency": 4}

    with pytest.raises(RuntimeError, match="simulate process loss"):
        await app.ainvoke(initial, config=config)
    assert calls["retrieval_worker"] == 1

    memory_can_finish = True
    monkeypatch.setattr(registry, "get_parallel_error_handler_for_spec", normal_error_handler)
    resumed_app = WorkflowCompiler(registry).compile(spec, checkpointer=checkpointer)
    result = await resumed_app.ainvoke(None, config=config)

    assert result["final_answer"] == "joined"
    assert calls["retrieval_worker"] == 1
    assert calls["durable_memory_worker"] == 1
    assert result["parallel_summary"]["partial_evidence"] is True

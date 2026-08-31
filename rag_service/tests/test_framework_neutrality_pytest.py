from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent_workflows.execution_stream import AgentExecutionEventSink
from runtime_protocol.contracts import (
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeOperationId,
    native,
)
from langgraph_runtime.adapter import _event_from_graph, _result_from_graph
from app.runtime.operational_limits import (
    MAX_RUNTIME_JSON_COLLECTION_ITEMS,
    MAX_RUNTIME_JSON_DEPTH,
    validate_bounded_json,
)
import app.runtime.cleanup as cleanup


def test_langgraph_result_reports_confirmed_checkpoint_boundary() -> None:
    result = _result_from_graph({
        "status": "awaiting_human",
        "pending_interrupt": {"checkpoint_thread_id": "checkpoint-1"},
    })

    assert result.checkpoint_boundary_available is True
    assert result.continuation == ContinuationBinding(
        binding_type="langgraph.checkpoint",
        payload={"checkpoint_thread_id": "checkpoint-1"},
    )
    assert _result_from_graph({"status": "completed"}).checkpoint_boundary_available is None


def test_deep_agent_result_projects_observed_task_version_from_graph_state() -> None:
    result = _result_from_graph({
        "status": "completed",
        "agent_run_id": "run-1",
        "agent_task_id": "task-1",
        "task_version": 13,
        "task_observed_plan_revision": 2,
        "task_plan_revision": 2,
    })

    assert result.orchestration_delta is not None
    assert result.orchestration_delta.observed_task_version == 13
    assert result.orchestration_delta.observed_plan_revision == 2


def test_deep_agent_result_keeps_launch_revision_separate_from_runtime_plan() -> None:
    result = _result_from_graph({
        "status": "completed",
        "agent_run_id": "run-1",
        "agent_task_id": "task-1",
        "task_version": 4,
        "task_observed_plan_revision": 0,
        "task_plan_revision": 1,
        "task_plan": {"objective": "Research the document"},
    })

    assert result.orchestration_delta is not None
    assert result.orchestration_delta.observed_plan_revision == 0
    assert result.orchestration_delta.plan == {"objective": "Research the document"}


def test_langgraph_interrupt_event_supplies_source_and_checkpoint_fact() -> None:
    event = _event_from_graph(
        {
            "event": "interrupt.created",
            "data": {"checkpoint_thread_id": "checkpoint-1"},
        },
        run_id="run-1",
        sequence=1,
    )

    assert event.source_metadata["visualization_id"] == "langgraph.graph"
    assert event.checkpoint_boundary_available is True
    assert event.continuation is not None


@pytest.mark.asyncio
async def test_runtime_event_sink_persists_explicit_checkpoint_facts() -> None:
    binding_persister = AsyncMock()
    fact_persister = AsyncMock()
    event_persister = AsyncMock()
    sink = AgentExecutionEventSink()
    sink.detach_delivery()
    sink.bind_runtime_binding_persister(binding_persister)
    sink.bind_runtime_fact_persister(fact_persister)
    sink.bind_runtime_event_persister("run-1", event_persister)
    event = _event_from_graph(
        {"event": "interrupt.created", "data": {"checkpoint_thread_id": "checkpoint-1"}},
        run_id="run-1",
        sequence=1,
    )

    await sink.emit_runtime_event(event)
    await sink.finish_boundary()

    binding_persister.assert_awaited_once_with("run-1", event.continuation)
    fact_persister.assert_awaited_once_with(
        "run-1", {"checkpoint_boundary_available": True}
    )


@pytest.mark.asyncio
async def test_continuation_cleanup_does_not_treat_unavailable_as_cleaned(monkeypatch) -> None:
    run = SimpleNamespace(
        id="run-1",
        workflow_id="definition-1",
        framework="fake",
        builder_id="fake-builder",
        definition_category=None,
        resolved_spec_json={},
        runtime_binding_json={"binding_type": "fake.binding", "payload": {"id": "binding-1"}},
    )
    adapter = SimpleNamespace(framework="fake", builder_id="fake-builder")
    from app.runtime.registry import RuntimeRegistry

    monkeypatch.setattr(cleanup, "get_runtime_registry", lambda: RuntimeRegistry([adapter]))
    resolver = AsyncMock(return_value=SimpleNamespace(
        capabilities=RuntimeCapabilities(),
        error={"code": "runtime_unavailable"},
        runtime_available=False,
    ))
    monkeypatch.setattr(
        cleanup,
        "resolve_run_capability_resolution",
        resolver,
    )

    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.status == "unavailable"
    assert outcome.cleaned is False
    assert resolver.await_args.kwargs["adapter"] is adapter


@pytest.mark.asyncio
async def test_continuation_cleanup_accepts_opaque_binding_types(monkeypatch) -> None:
    deleted = AsyncMock(return_value={"status": "deleted"})
    adapter = SimpleNamespace(
        framework="fake",
        builder_id="fake-builder",
        delete_continuation=deleted,
    )
    capabilities = RuntimeCapabilities(
        operations={RuntimeOperationId.RUN_CONTINUATION_CLEANUP: native()}
    )
    run = SimpleNamespace(
        id="run-1",
        workflow_id="definition-1",
        framework="fake",
        builder_id="fake-builder",
        definition_category=None,
        resolved_spec_json={},
        runtime_binding_json={"binding_type": "vendor.opaque", "payload": {"token": "value"}},
    )
    from app.runtime.registry import RuntimeRegistry

    monkeypatch.setattr(cleanup, "get_runtime_registry", lambda: RuntimeRegistry([adapter]))
    resolver = AsyncMock(return_value=SimpleNamespace(
        capabilities=capabilities,
        error=None,
        runtime_available=True,
    ))
    monkeypatch.setattr(
        cleanup,
        "resolve_run_capability_resolution",
        resolver,
    )

    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.cleaned is True
    deleted.assert_awaited_once_with(
        ContinuationBinding(binding_type="vendor.opaque", payload={"token": "value"})
    )
    assert resolver.await_args.kwargs["adapter"] is adapter


def test_runtime_json_validation_rejects_coercion_depth_and_aggregate_size() -> None:
    with pytest.raises(ValueError, match="non-JSON"):
        validate_bounded_json({"value": object()}, field_name="input")

    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(MAX_RUNTIME_JSON_DEPTH + 1):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    with pytest.raises(ValueError, match="nesting depth"):
        validate_bounded_json(nested, field_name="update")

    values = [[index] for index in range(MAX_RUNTIME_JSON_COLLECTION_ITEMS // 2 + 1)]
    with pytest.raises(ValueError, match="collection items"):
        validate_bounded_json({"values": values}, field_name="input")


def test_shared_product_modules_do_not_branch_on_framework_names() -> None:
    app_root = Path(__file__).parents[1] / "app"
    shared_paths = (
        app_root / "services" / "agent_task_runtime.py",
        app_root / "services" / "agent_task_repository.py",
        app_root / "services" / "project_lifecycle_service.py",
        app_root / "agent_workflows" / "repository.py",
        app_root / "agent_workflows" / "run_store.py",
        app_root / "runtime" / "cleanup.py",
        app_root / "api" / "agent_tasks.py",
    )
    framework_names = {"langgraph", "hermes"}
    violations: list[str] = []
    for path in shared_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = (node.left, *node.comparators)
            if any(
                isinstance(operand, ast.Constant) and operand.value in framework_names
                for operand in operands
            ):
                violations.append(f"{path.name}:{node.lineno}: {ast.unparse(node)}")
    assert violations == []


def test_langgraph_runtime_has_no_product_persistence_execution_path() -> None:
    source = (
        Path(__file__).parents[1]
        / "app"
        / "runtime"
        / "langgraph"
        / "router_runtime.py"
    ).read_text()
    assert "persist_product_records" not in source
    assert "result_projector" not in source
    assert "create_chat_turn" not in source


def test_definition_authoring_has_no_default_runtime_fallback() -> None:
    source = (
        Path(__file__).parents[1]
        / "app"
        / "api"
        / "agent_workflows.py"
    ).read_text()
    assert "with_default_runtime" not in source
    assert "framework: str = Field(..., min_length=1)" in source
    assert "builder_id: str = Field(..., min_length=1)" in source


def test_run_creation_does_not_infer_checkpoint_or_synthesize_binding() -> None:
    source = (
        Path(__file__).parents[1]
        / "app"
        / "agent_workflows"
        / "run_store.py"
    ).read_text()
    assert 'setdefault("checkpoint_boundary_available"' not in source
    assert "langgraph_checkpoint" not in source
    assert 'f"{persisted_framework}' not in source
    assert 'runtime_binding_status="active" if runtime_binding_json else "unbound"' in source

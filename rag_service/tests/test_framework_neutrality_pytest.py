from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.runtime.operational_limits import (
    MAX_RUNTIME_JSON_COLLECTION_ITEMS,
    MAX_RUNTIME_JSON_DEPTH,
    validate_bounded_json,
)
import app.runtime.cleanup as cleanup


@pytest.mark.asyncio
async def test_continuation_cleanup_rejects_non_langgraph_frameworks(monkeypatch) -> None:
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
    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.status == "unsupported"
    assert outcome.cleaned is False


@pytest.mark.asyncio
async def test_continuation_cleanup_accepts_explicit_runtime_status(monkeypatch) -> None:
    adapter = SimpleNamespace(
        framework="langgraph",
        builder_id="langgraph_graph",
        cleanup_run=AsyncMock(return_value={"status": "cleaned"}),
    )
    run = SimpleNamespace(
        id="run-1",
        workflow_id="definition-1",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category=None,
        resolved_spec_json={"framework": "langgraph", "builder_id": "langgraph_graph"},
    )
    from app.runtime.registry import RuntimeRegistry

    monkeypatch.setattr(cleanup, "get_runtime_registry", lambda: RuntimeRegistry([adapter]))
    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.cleaned is True
    adapter.cleanup_run.assert_awaited_once_with("run-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, [], {}, {"status": "unknown"}, {"status": "deleted"}])
async def test_langgraph_cleanup_rejects_non_success_envelopes(monkeypatch, response) -> None:
    run = SimpleNamespace(
        id="run-cleanup",
        workflow_id="definition-1",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category=None,
        resolved_spec_json={"framework": "langgraph", "builder_id": "langgraph_graph"},
    )
    adapter = SimpleNamespace(
        framework="langgraph",
        builder_id="langgraph_graph",
        cleanup_run=AsyncMock(return_value=response),
    )
    registry = SimpleNamespace(get=lambda definition: adapter)
    monkeypatch.setattr(cleanup, "get_runtime_registry", lambda: registry)

    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.status == "failed"
    assert outcome.cleaned is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cleaned", "already_cleaned", "not_bound"])
async def test_langgraph_cleanup_accepts_only_explicit_success_statuses(monkeypatch, status) -> None:
    run = SimpleNamespace(
        id="run-cleanup",
        workflow_id="definition-1",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category=None,
        resolved_spec_json={"framework": "langgraph", "builder_id": "langgraph_graph"},
    )
    adapter = SimpleNamespace(
        framework="langgraph",
        builder_id="langgraph_graph",
        cleanup_run=AsyncMock(return_value={"status": status}),
    )
    registry = SimpleNamespace(get=lambda definition: adapter)
    monkeypatch.setattr(cleanup, "get_runtime_registry", lambda: registry)

    outcome = await cleanup.delete_run_continuation(run)

    assert outcome.status == status
    assert outcome.cleaned is True


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


def test_shared_product_modules_do_not_import_framework_implementations() -> None:
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
    violations: list[str] = []
    for path in shared_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [value.name for value in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "langgraph" or name.startswith("langgraph.") or name.startswith("langgraph_runtime"):
                    violations.append(f"{path.name}:{node.lineno}: {name}")
    assert violations == []


def test_langgraph_runtime_has_no_product_persistence_execution_path() -> None:
    packaged_root = Path(__file__).parents[1]
    runtime_root = packaged_root / "langgraph_runtime"
    if not runtime_root.exists():
        runtime_root = packaged_root.parent / "langgraph_runtime"
    if not (runtime_root / "router_runtime.py").exists():
        return
    source = (runtime_root / "router_runtime.py").read_text()
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

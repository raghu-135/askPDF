from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.agent_workflows.compiler import WorkflowCompiler
from app.agent_workflows import deep_research_nodes, router_runtime
from app.agent_workflows.debug_trace import AgentTraceRecorder
from app.agent_workflows.enums import WorkflowNodeType
from app.agent_workflows.graph import NodeRegistry
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.validator import WorkflowResolver, WorkflowValidator
from app.api import agent_tasks as agent_tasks_api
from app.api import agent_workflows as agent_workflows_api
from app.db.models_sqlmodel import AgentRun, AgentTaskTodo, AgentWorkflow
from app.models.deep_research import DeepResearchPlanProposal, DeepResearchSubagentResult
from app.services import agent_task_repository as repository
from app.services import agent_task_presentation
from app.services import agent_task_runtime
from app.services import agent_task_maintenance
from app.services.content_store import SharedVolumeContentStore, set_content_store
from app.services.task_artifact_service import artifact_ownership_key, persist_task_artifact
from app.time_utils import utc_now
from app.runtime.errors import RuntimeError as AgentRuntimeError
from app.runtime.events import create_runtime_event
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeCapabilityDisabledReason,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
    RuntimeCapabilities,
    native,
    unsupported,
)
from app.runtime.registry import RuntimeRegistry


def _spec() -> dict:
    return next(
        value["spec_json"]
        for value in load_builtin_workflows()
        if value["builtin_key"] == "deep_research_agent"
    )


def _valid_plan_text(profile: str = "document_researcher") -> str:
    return json.dumps({
        "objective": "Research Lamport clocks",
        "success_criteria": ["Evidence-backed report"],
        "todos": [{
            "id": "retrieve-evidence",
            "title": "Retrieve evidence",
            "description": "Search the available evidence",
            "completion_criteria": "Relevant sources are collected",
            "profile_id": profile,
        }],
    })


class TaskInvocationAdapter:
    framework = "hermes"
    builder_id = "hermes_agent"

    def __init__(self, *, resume_enabled: bool = True):
        self.resume_calls = 0
        self.start_calls = 0
        self.resume_enabled = resume_enabled

    async def capabilities(self, definition):
        resume = native() if self.resume_enabled else unsupported()
        return RuntimeCapabilities(operations={
            RuntimeOperationId.RUN_START: native(),
            RuntimeOperationId.RUN_RESUME: resume,
            RuntimeOperationId.RUN_APPROVAL_RESPOND: native(),
        })

    async def deployment_capabilities(self):
        return await self.capabilities(AgentDefinition("deployment", self.framework, self.builder_id))

    async def start(self, request, *, context, event_sink=None):
        self.start_calls += 1
        return AgentRuntimeResult(status="completed")

    async def resume(self, request, *, interrupt, context, event_sink=None):
        self.resume_calls += 1
        return AgentRuntimeResult(status="completed", output="resumed")

    async def continue_run(self, request, *, context, event_sink=None):
        return AgentRuntimeResult(status="completed", output="continued")

    async def respond_to_approval(self, request, response):
        return {"status": "accepted"}


@pytest.mark.asyncio
async def test_task_worker_start_gate_rejects_before_adapter_invocation(monkeypatch):
    rejection = AgentRuntimeError.capability_unavailable(
        operation_id="run.start",
        framework="hermes",
        builder_id="hermes_agent",
        support_level="conditional",
        disabled_reason=RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE,
    )
    require = AsyncMock(side_effect=rejection)
    monkeypatch.setattr(agent_task_runtime, "require_capability", require)
    adapter = SimpleNamespace(start=AsyncMock())
    repository = SimpleNamespace(mark_runtime_started=AsyncMock())
    run = SimpleNamespace(
        id="run-1",
        status="running",
        pending_interrupt_json={},
        runtime_binding_json={},
        runtime_binding_status="active",
        _fresh_runtime_run=True,
    )
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")
    request = AgentRuntimeRequest("run-1", "thread-1", definition.definition_id, definition.framework, definition.builder_id)

    with pytest.raises(AgentRuntimeError) as caught:
        await agent_task_runtime._invoke_task_runtime(
            adapter=adapter,
            definition=definition,
            run=run,
            runtime_request=request,
            runtime_context=RuntimeExecutionContext(),
        runtime_event_sink=None,
        repository=repository,
        registry=RuntimeRegistry(adapters=[]),
        )

    assert caught.value is rejection
    adapter.start.assert_not_awaited()
    repository.mark_runtime_started.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_worker_replays_persisted_terminal_result_without_hermes_continuation():
    adapter = SimpleNamespace(start=AsyncMock(), continue_run=AsyncMock(), resume=AsyncMock())
    repository = SimpleNamespace(mark_runtime_started=AsyncMock())
    run = SimpleNamespace(
        id="run-1",
        status="running",
        pending_interrupt_json={},
        run_metadata_json={
            "runtime_started": True,
            "projection": {
                "runtime_result": {
                    "status": "completed",
                    "answer": "durable answer",
                    "runtime_metadata": {"provider": "hermes"},
                },
            },
        },
        _fresh_runtime_run=False,
    )
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")

    result = await agent_task_runtime._invoke_task_runtime(
        adapter=adapter,
        definition=definition,
        run=run,
        runtime_request=AgentRuntimeRequest(
            "run-1", "thread-1", definition.definition_id, definition.framework, definition.builder_id,
        ),
        runtime_context=RuntimeExecutionContext(),
        runtime_event_sink=None,
        repository=repository,
        registry=RuntimeRegistry(adapters=[]),
    )

    assert result.status == "completed"
    assert result.output == "durable answer"
    adapter.start.assert_not_awaited()
    adapter.continue_run.assert_not_awaited()
    adapter.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_hermes_resolved_approval_continues_without_runtime_resume():
    result = AgentRuntimeResult(status="completed", output="approved")
    adapter = TaskInvocationAdapter()
    adapter.continue_run = AsyncMock(return_value=result)
    adapter.resume = AsyncMock()
    run = SimpleNamespace(
        id="run-1",
        status="running",
        runtime_binding_json={"binding_type": "hermes_session"},
        runtime_binding_status="active",
        pending_interrupt_json={
            "interrupt_id": "approval-1",
            "status": "resumed",
            "response_operation": "run.approval.respond",
            "decision": {"action": "approve"},
        },
        _fresh_runtime_run=False,
    )
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")
    request = AgentRuntimeRequest("run-1", "thread-1", definition.definition_id, definition.framework, definition.builder_id)
    registry = RuntimeRegistry(adapters=[adapter])

    actual = await agent_task_runtime._invoke_task_runtime(
        adapter=adapter,
        definition=definition,
        run=run,
        runtime_request=request,
        runtime_context=RuntimeExecutionContext(),
        runtime_event_sink=None,
        repository=SimpleNamespace(),
        registry=registry,
    )

    assert actual is result
    adapter.continue_run.assert_awaited_once()
    adapter.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_runtime_resume_is_rejected_by_real_registry_before_adapter_call():
    adapter = TaskInvocationAdapter(resume_enabled=False)
    registry = RuntimeRegistry(adapters=[adapter])
    run = SimpleNamespace(
        id="run-1",
        status="running",
        runtime_binding_json={"binding_type": "hermes_session"},
        runtime_binding_status="active",
        pending_interrupt_json={
            "status": "resumed",
            "response_operation": RuntimeOperationId.RUN_RESUME.value,
            "decision": {"action": "approve"},
        },
        _fresh_runtime_run=False,
    )
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")
    request = AgentRuntimeRequest("run-1", "thread-1", definition.definition_id, definition.framework, definition.builder_id)

    with pytest.raises(AgentRuntimeError) as caught:
        await agent_task_runtime._invoke_task_runtime(
            adapter=adapter,
            definition=definition,
            run=run,
            runtime_request=request,
            runtime_context=RuntimeExecutionContext(),
            runtime_event_sink=None,
            repository=SimpleNamespace(),
            registry=registry,
        )

    assert caught.value.code == "runtime_capability_unsupported"
    assert adapter.resume_calls == 0


@pytest.mark.asyncio
async def test_interrupted_hermes_prestart_gets_new_product_run(monkeypatch):
    task = SimpleNamespace(
        id="task-1", thread_id="thread-1", workflow_id="hermes_rag_agent",
        user_id="user-1", config_json={}, objective="research",
    )
    active = SimpleNamespace(
        id="old-run", thread_id="thread-1", framework="hermes", builder_id="hermes_agent",
        definition_id="hermes_rag_agent", status="running",
        run_metadata_json={"runtime_started": False},
        runtime_binding_json={
            "binding_type": "hermes_session",
            "payload": {"upstream_run_id": "upstream-1", "runtime_profile": "profile-1"},
        },
        runtime_binding_status="active",
    )
    replacement = SimpleNamespace(
        id="new-run", status="running", run_metadata_json={"runtime_started": False},
        pending_interrupt_json={}, runtime_binding_json={"binding_type": "hermes_session", "payload": {}},
        runtime_binding_status="active",
    )
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")
    adapter = SimpleNamespace(
        framework="hermes", builder_id="hermes_agent", cancel=AsyncMock(),
        start=AsyncMock(return_value=AgentRuntimeResult(status="completed")),
    )
    workflow = SimpleNamespace(
        id="hermes_rag_agent", spec_json={}, metadata_json={"version": 1},
        schema_version=1, category=None,
    )

    class FakeRepository:
        async def get_workflow(self, workflow_id, *, include_custom=False):
            return workflow

        async def create_run(self, **kwargs):
            return replacement

        async def complete_run(self, *args, **kwargs):
            return active

    class FakeBuilder:
        async def resolve(self, *args, **kwargs):
            return {"config": {}}

        async def normalize(self, definition_value, resolved):
            return resolved

    monkeypatch.setattr(agent_task_runtime.tasks, "get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=active))
    monkeypatch.setattr(agent_task_runtime.tasks, "attach_run", AsyncMock(return_value=replacement))
    monkeypatch.setattr(agent_task_runtime, "get_thread_settings", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_task_runtime, "definition_from_run", lambda run: definition)
    monkeypatch.setattr(agent_task_runtime, "definition_from_workflow", lambda value: definition)
    monkeypatch.setattr(agent_task_runtime, "adapter_for_definition", lambda value: adapter)
    monkeypatch.setattr(agent_task_runtime, "get_runtime_registry", lambda: RuntimeRegistry(adapters=[adapter]))
    monkeypatch.setattr(agent_task_runtime, "require_capability", AsyncMock())
    monkeypatch.setattr(agent_task_runtime, "builder_for_definition", lambda value: FakeBuilder())
    monkeypatch.setattr(agent_task_runtime, "AgentWorkflowRepository", FakeRepository)

    result = await agent_task_runtime.ensure_task_run(task.id)
    start_repository = SimpleNamespace(mark_runtime_started=AsyncMock())
    await agent_task_runtime._invoke_task_runtime(
        adapter=adapter,
        definition=definition,
        run=result,
        runtime_request=AgentRuntimeRequest(
            result.id, task.thread_id, definition.definition_id,
            definition.framework, definition.builder_id,
        ),
        runtime_context=RuntimeExecutionContext(),
        runtime_event_sink=None,
        repository=start_repository,
        registry=RuntimeRegistry(adapters=[adapter]),
    )

    assert result is replacement
    assert result._fresh_runtime_run is True
    adapter.cancel.assert_awaited_once()
    adapter.start.assert_awaited_once()
    start_repository.mark_runtime_started.assert_awaited_once_with(result.id)
    assert agent_task_runtime.require_capability.await_count == 2
    assert agent_task_runtime.require_capability.await_args_list[0].kwargs["run"] is active
    assert agent_task_runtime.require_capability.await_args_list[1].kwargs["run"] is replacement
    assert agent_task_runtime.tasks.attach_run.await_args.args[0] == task.id


@pytest.mark.asyncio
async def test_task_start_admission_rejection_does_not_create_command_or_run(monkeypatch):
    task = SimpleNamespace(id="task-1", thread_id="thread-1", workflow_id="hermes_rag_agent", version=3)
    monkeypatch.setattr(agent_tasks_api, "_owned_task", AsyncMock(return_value=task))
    monkeypatch.setattr(
        agent_tasks_api,
        "_require_task_capability",
        AsyncMock(side_effect=agent_tasks_api.HTTPException(
            status_code=409,
            detail={"code": "runtime_capability_unavailable"},
        )),
    )
    apply_command = AsyncMock()
    ensure_run = AsyncMock()
    monkeypatch.setattr(agent_tasks_api.repository, "apply_command", apply_command)
    monkeypatch.setattr(agent_tasks_api, "ensure_task_run", ensure_run)

    with pytest.raises(agent_tasks_api.HTTPException) as caught:
        await agent_tasks_api.command_agent_task(
            task.id,
            "start",
            agent_tasks_api.AgentTaskCommandRequest(expected_version=task.version),
            thread_id=task.thread_id,
            idempotency_key="start-1",
        )

    assert caught.value.detail["code"] == "runtime_capability_unavailable"
    apply_command.assert_not_awaited()
    ensure_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_deep_planner_repairs_invalid_output_once(monkeypatch):
    call_model = AsyncMock(side_effect=[
        ("{}", {"attempt": "initial"}),
        (_valid_plan_text(), {"attempt": "repair"}),
    ])
    monkeypatch.setattr(deep_research_nodes, "_call_model", call_model)
    sink = SimpleNamespace(emit=AsyncMock())
    state = {
        "runtime_execution_mode": True,
        "question": "Research Lamport clocks",
        "task_enabled_profiles": ["document_researcher", "evidence_critic"],
        "task_limits": {"max_todos": 5, "max_attempts": 2},
        "task_todos": [],
        "task_plan_revision": 0,
        "task_run_plan_count": 0,
        "llm_model": "small-test-model",
    }

    result = await deep_research_nodes.deep_task_planner(
        state, {"configurable": {"execution_event_sink": sink}},
    )

    assert result["task_plan"]["todos"][0]["profile_id"] == "document_researcher"
    assert call_model.await_count == 2
    assert [call.args[0] for call in sink.emit.await_args_list] == [
        "planner.validation_failed", "planner.repair_started",
    ]


@pytest.mark.asyncio
async def test_deep_planner_uses_bounded_fallback_after_invalid_repair(monkeypatch):
    secret_marker = "must-not-be-persisted"
    call_model = AsyncMock(side_effect=[
        (f"not json {secret_marker}", {}),
        ("{}", {}),
    ])
    monkeypatch.setattr(deep_research_nodes, "_call_model", call_model)
    sink = SimpleNamespace(emit=AsyncMock())
    state = {
        "runtime_execution_mode": True,
        "question": "Research Lamport clocks",
        "task_enabled_profiles": ["document_researcher"],
        "task_limits": {"max_todos": 5},
        "task_todos": [],
        "llm_model": "small-test-model",
    }

    result = await deep_research_nodes.deep_task_planner(
        state, {"configurable": {"execution_event_sink": sink}},
    )

    assert result["task_plan"]["todos"][0]["profile_id"] == "document_researcher"
    assert len(result["task_plan"]["todos"]) == 1
    assert call_model.await_count == 2
    assert [call.args[0] for call in sink.emit.await_args_list] == [
        "planner.validation_failed", "planner.repair_started", "planner.validation_failed",
        "planner.fallback_created",
    ]
    assert secret_marker not in json.dumps(sink.emit.await_args_list[-1].args[1])


async def _attach_test_run(test_session_maker, task, *, parent_run_id: str | None = None) -> AgentRun:
    run = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        resolved_spec_json=_spec(),
        checkpoint_thread_id=str(uuid.uuid4()),
        run_metadata_json={"run_kind": "agent_task"},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    await repository.attach_run(task.id, run, parent_run_id=parent_run_id)
    return run


@pytest.mark.asyncio
async def test_attach_run_converges_on_existing_active_run(test_session_maker, sample_thread):
    await _seed_deep_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="Research safely",
        idempotency_key=str(uuid.uuid4()),
        config={},
    )
    active = await _attach_test_run(test_session_maker, task)
    contender = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        resolved_spec_json=_spec(),
        checkpoint_thread_id=str(uuid.uuid4()),
        run_metadata_json={"run_kind": "agent_task"},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(contender)
    selected = await repository.attach_run(task.id, contender)

    assert selected.id == active.id
    async with test_session_maker() as session:
        discarded = await session.get(AgentRun, contender.id)
        refreshed_task = await session.get(type(task), task.id)
    assert discarded.status == "cancelled"
    assert discarded.task_id is None
    assert discarded.error_json["code"] == "concurrent_task_run_superseded"
    assert refreshed_task.active_run_id == active.id
    assert refreshed_task.latest_run_attempt == 1


@pytest.mark.asyncio
async def test_task_run_terminal_state_and_journals_commit_together(test_session_maker, sample_thread):
    await _seed_deep_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Atomic terminal commit",
        idempotency_key=str(uuid.uuid4()), config={},
    )
    task, _, _ = await repository.apply_command(
        task.id, action="start", idempotency_key="start-atomic", expected_version=task.version,
    )
    run = await _attach_test_run(test_session_maker, task)
    claimed = await repository.claim_next_task("atomic-worker", lease_seconds=60)
    assert claimed is not None and claimed.id == task.id
    terminal = create_runtime_event(
        event_id=f"askpdf-terminal:{run.id}:run.completed", run_id=run.id, sequence=1,
        kind="run.completed", payload={"status": "completed"},
    )
    await AgentWorkflowRepository().append_run_event_payload(
        run_id=run.id,
        event_id=f"{run.id}:1",
        sequence=1,
        attempt=1,
        kind="output.completed",
        payload_json={"answer": "done"},
    )

    finalized = await repository.finalize_task_run(
        task.id, run.id,
        run_status="completed", task_status="completed", metrics={"duration_ms": 1},
        error=None, debug_trace={"trace": {"run_id": run.id}}, terminal_reason="completed",
        terminal_event=terminal, final_artifact_id=None,
    )

    stored_run = await AgentWorkflowRepository().get_run(run.id)
    task_events = await repository.list_events(task.id, agent_run_id=run.id)
    run_events = await AgentWorkflowRepository().list_run_events(run.id)
    assert finalized.status == stored_run.status == "completed"
    assert finalized.lease_owner is None and finalized.lease_expires_at is None
    assert [event.event_type for event in task_events if event.terminal] == ["run.completed"]
    assert [(event.sequence, event.kind, event.terminal) for event in run_events] == [
        (1, "output.completed", False),
        (2, "run.completed", True),
    ]

async def _seed_deep_workflow(test_session_maker) -> None:
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent", name="Deep", description="test",
                visibility="builtin", is_builtin=True, schema_version=2,
                spec_json=_spec(), metadata_json={"version": 1},
            ))


def test_deep_research_builtin_is_valid_and_compilable():
    spec = _spec()
    assert WorkflowValidator().validate(spec) == {"valid": True, "errors": []}
    compiled = WorkflowCompiler().compile(spec)
    assert compiled is not None
    assert spec["config"]["task_policy"]["builtin_only"] is True
    assert spec["config"]["task_policy"]["limits"]["max_fanout"] == 2

    resolved = WorkflowResolver().resolve(spec, thread_settings={"replans": 10})
    assert resolved["config"]["replans"] == 5
    visit_limits = resolved["config"]["loop_policy"]["node_visit_limits"]
    assert visit_limits["deep_task_planner"] == 6
    assert visit_limits["deep_research_subagent"] == 100
    assert 1 <= visit_limits["deep_coordinator"] <= 60
    assert visit_limits["task_pause_gate"] == 16


@pytest.mark.asyncio
async def test_deep_research_web_capability_uses_profile_and_tool_grants_not_default(monkeypatch):
    spec = _spec()
    spec["config"]["use_web_search"] = True
    spec["config"]["allowed_tool_ids"] = [
        value for value in spec["config"]["allowed_tool_ids"] if value != "live_web_recon"
    ]
    workflow_repository = SimpleNamespace(
        get_workflow=AsyncMock(return_value=SimpleNamespace(spec_json=spec)),
        seed_builtin_workflows=AsyncMock(),
    )
    monkeypatch.setattr(agent_tasks_api, "AgentWorkflowRepository", lambda: workflow_repository)

    contract = await agent_tasks_api._deep_research_contract()

    assert contract["web_enabled"] is False


def test_deep_research_plan_rejects_cycles_and_unknown_dependencies():
    common = {
        "objective": "Research",
        "success_criteria": ["Grounded report"],
    }
    with pytest.raises(ValidationError, match="unknown dependencies"):
        DeepResearchPlanProposal.model_validate({
            **common,
            "todos": [{
                "id": "one", "title": "One", "description": "Research one",
                "completion_criteria": "Evidence", "dependency_ids": ["missing"],
                "profile_id": "document_researcher",
            }],
        })
    with pytest.raises(ValidationError, match="acyclic"):
        DeepResearchPlanProposal.model_validate({
            **common,
            "todos": [
                {"id": "one", "title": "One", "description": "One", "completion_criteria": "Done", "dependency_ids": ["two"], "profile_id": "document_researcher"},
                {"id": "two", "title": "Two", "description": "Two", "completion_criteria": "Done", "dependency_ids": ["one"], "profile_id": "memory_researcher"},
            ],
        })


def test_subagent_result_normalizes_common_model_schema_drift():
    result = DeepResearchSubagentResult.model_validate({
        "status": "completed",
        "summary": "Grounded result",
        "uncovered_gaps": "More clinical evidence is needed.",
        "usage": {
            "total_tokens": "120",
            "tool_calls": 2,
            "todo_id": "T1",
            "title": "Research task",
        },
    })

    assert result.uncovered_gaps == ["More clinical evidence is needed."]
    assert result.usage == {"total_tokens": 120, "tool_calls": 2}


def test_timeline_never_projects_active_subagents_as_failures():
    assert agent_tasks_api._subagent_timeline_type("queued") is None
    assert agent_tasks_api._subagent_timeline_type("running") is None
    assert agent_tasks_api._subagent_timeline_type("completed") == "todo_result"
    assert agent_tasks_api._subagent_timeline_type("failed") == "todo_failure"
    assert agent_tasks_api._subagent_timeline_type("timed_out") == "todo_failure"


def test_timeline_sources_flattens_deduplicates_and_bounds_artifact_provenance():
    artifact = SimpleNamespace(source_refs_json={
        "tools": [{
            "name": "search_web",
            "sources": [
                {"title": "Primary", "url": "https://example.test/a", "snippet": "Evidence"},
                {"title": "Primary", "url": "https://example.test/a", "snippet": "Evidence"},
                {"title": "Unsafe", "url": "javascript:alert(1)"},
                {"title": "Document evidence", "text": "Relevant passage", "file_hash": "abc123"},
            ],
        }],
    })

    sources = agent_task_presentation.timeline_sources([artifact], limit=3)
    assert [(source["kind"], source.get("title"), source.get("url"), source.get("file_hash")) for source in sources] == [
        ("web", "Primary", "https://example.test/a", None),
        ("document", "Document evidence", None, "abc123"),
    ]
    assert all(source["id"] for source in sources)


def test_timeline_sources_prefers_direct_evidence_and_preserves_all_origins():
    inherited = SimpleNamespace(
        id="artifact-1", agent_run_id="run-1", created_at="2026-01-01",
        provenance_json={"plan_revision": 1},
        source_refs_json={"sources": [{"title": "Older", "url": "https://example.test/source"}]},
    )
    direct = SimpleNamespace(
        id="artifact-2", agent_run_id="run-2", created_at="2026-01-02",
        provenance_json={"plan_revision": 3},
        source_refs_json={"sources": [{"title": "Current", "url": "https://example.test/source"}]},
    )

    sources = agent_task_presentation.timeline_sources(
        [inherited, direct], attempts_by_run={"run-1": 1, "run-2": 2}, selected_run_id="run-2",
    )

    assert len(sources) == 1
    assert sources[0]["title"] == "Current"
    assert sources[0]["artifact_id"] == "artifact-2"
    assert sources[0]["plan_revision"] == 3
    assert sources[0]["inherited"] is False
    assert [origin["attempt"] for origin in sources[0]["origins"]] == [2, 1]


def test_artifact_ownership_key_is_server_derived():
    assert artifact_ownership_key(todo_id=None, subagent_run_id=None) == "run"
    assert artifact_ownership_key(todo_id="todo-1", subagent_run_id=None) == "todo:todo-1"
    assert artifact_ownership_key(todo_id="todo-1", subagent_run_id="worker-1") == "subagent:worker-1"


@pytest.mark.asyncio
async def test_artifact_context_selects_only_completed_todo_evidence(monkeypatch):
    completed = SimpleNamespace(
        id="artifact-completed", validity="valid", kind="intermediate_report",
        sha256="hash-completed", byte_size=100, summary_json={}, todo_id="done",
        agent_run_id="run-1",
    )
    failed = SimpleNamespace(
        id="artifact-failed", validity="valid", kind="intermediate_report",
        sha256="hash-failed", byte_size=100, summary_json={}, todo_id="failed",
        agent_run_id="run-1",
    )
    monkeypatch.setattr(deep_research_nodes, "list_artifacts", AsyncMock(return_value=[completed, failed]))
    monkeypatch.setattr(
        deep_research_nodes,
        "list_task_runs",
        AsyncMock(return_value=[SimpleNamespace(id="run-1", task_attempt=1)]),
    )
    monkeypatch.setattr(deep_research_nodes, "invalidate_context_summaries", AsyncMock())

    result = await deep_research_nodes.assemble_artifact_context({
        "agent_task_id": "task-1",
        "agent_run_id": "run-1",
        "context_window": 8_192,
        "task_todos": [
            {"id": "done", "status": "completed", "artifact_ids": [completed.id]},
            {"id": "failed", "status": "failed", "artifact_ids": [failed.id]},
        ],
    }, {})

    assert [value["id"] for value in result["task_evidence_manifest"]] == [completed.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("validity", [None, "invalid", "deleted"])
async def test_artifact_context_reports_missing_and_invalid_completed_evidence(monkeypatch, validity):
    artifact = None if validity is None else SimpleNamespace(
        id="inherited-evidence", validity=validity, kind="intermediate_report",
        sha256="stored-hash", byte_size=100, summary_json={}, todo_id="done",
        agent_run_id="run-1",
    )
    monkeypatch.setattr(
        deep_research_nodes,
        "list_artifacts",
        AsyncMock(return_value=[] if artifact is None else [artifact]),
    )
    monkeypatch.setattr(
        deep_research_nodes,
        "list_task_runs",
        AsyncMock(return_value=[SimpleNamespace(id="run-1", task_attempt=1)]),
    )
    monkeypatch.setattr(deep_research_nodes, "invalidate_context_summaries", AsyncMock())

    result = await deep_research_nodes.assemble_artifact_context({
        "agent_task_id": "task-1",
        "agent_run_id": "run-2",
        "context_window": 8_192,
        "task_todos": [{
            "id": "done", "status": "completed", "artifact_ids": ["inherited-evidence"],
        }],
    }, {})

    expected_reason = "missing" if validity is None else validity
    assert result["task_evidence_manifest"] == []
    assert result["task_evidence_gaps"] == [f"inherited-evidence:{expected_reason}"]


@pytest.mark.asyncio
async def test_synthesizer_reports_hash_mismatched_inherited_evidence(monkeypatch, tmp_path):
    store = SharedVolumeContentStore(tmp_path / "content")
    set_content_store(store)
    await store.put("agent-tasks/task-1/run-1/evidence/1", b"original evidence")
    artifact = SimpleNamespace(
        id="inherited-evidence", validity="valid", kind="intermediate_report",
        sha256="not-the-object-hash", byte_size=17, summary_json={}, todo_id="done",
        agent_run_id="run-1", object_key="agent-tasks/task-1/run-1/evidence/1",
        provenance_json={}, source_refs_json={},
    )
    monkeypatch.setattr(deep_research_nodes, "list_artifacts", AsyncMock(return_value=[artifact]))
    monkeypatch.setattr(
        deep_research_nodes,
        "list_task_runs",
        AsyncMock(return_value=[
            SimpleNamespace(id="run-1", task_attempt=1),
            SimpleNamespace(id="run-2", task_attempt=2),
        ]),
    )
    monkeypatch.setattr(deep_research_nodes, "invalidate_context_summaries", AsyncMock())
    monkeypatch.setattr(deep_research_nodes, "_call_model", AsyncMock(return_value=("Incomplete report", {})))
    try:
        result = await deep_research_nodes.deep_task_synthesizer({
            "agent_task_id": "task-1", "agent_run_id": "run-2", "question": "Research",
            "context_window": 8_192, "task_memory_snapshot": {},
            "task_todos": [{
                "id": "done", "status": "completed", "required": True,
                "artifact_ids": [artifact.id],
            }],
        }, {})
    finally:
        set_content_store(None)

    assert "inherited-evidence:hash_mismatch" in result["task_incomplete_reasons"]


@pytest.mark.asyncio
async def test_child_retry_synthesizes_inherited_evidence_and_projects_source_lineage(
    monkeypatch,
    tmp_path,
    test_session_maker,
    sample_thread,
):
    store = SharedVolumeContentStore(tmp_path / "content")
    set_content_store(store)
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent", name="Deep", description="test",
                visibility="builtin", is_builtin=True, schema_version=2,
                spec_json=_spec(), metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Research inherited evidence",
        idempotency_key="inherited-evidence-task",
        config={"enabled_profiles": ["web_researcher"], "limits": {"max_attempts_per_todo": 2}},
    )
    run_one = await _attach_test_run(test_session_maker, task)
    proposal = DeepResearchPlanProposal.model_validate({
        "objective": task.objective, "success_criteria": ["Grounded report"],
        "todos": [
            {"id": "done", "title": "Completed evidence", "description": "Search", "completion_criteria": "Cited", "profile_id": "web_researcher"},
            {"id": "failed", "title": "Failed evidence", "description": "Search", "completion_criteria": "Cited", "profile_id": "web_researcher"},
        ],
    })
    await repository.persist_plan(task.id, proposal, agent_run_id=run_one.id, reason="initial", planner_visit=1)
    inherited = await persist_task_artifact(
        task_id=task.id,
        agent_run_id=run_one.id,
        todo_id="done",
        kind="intermediate_report",
        content="Evidence from the first attempt.",
        source_refs={"sources": [{
            "title": "Primary clinical source",
            "url": "https://example.test/clinical",
            "snippet": "Relevant evidence",
        }]},
    )
    async with test_session_maker() as session:
        async with session.begin():
            todos = {
                value.id: value
                for value in (await session.execute(
                    select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id)
                )).scalars().all()
            }
            todos["done"].status = "completed"
            todos["done"].progress = 100
            todos["done"].artifact_ids_json = [inherited.id]
            todos["failed"].status = "failed"
            todos["failed"].terminal_reason = "timeout"
    await AgentWorkflowRepository().complete_run(run_one.id, status="failed")
    failed_task = await repository.complete_task(task.id, status="failed", reason="required_work_failed")
    retried, _, _ = await repository.apply_command(
        task.id,
        action="retry",
        idempotency_key="inherited-evidence-retry",
        expected_version=failed_task.version,
    )
    run_two = await _attach_test_run(test_session_maker, retried, parent_run_id=run_one.id)
    todos = await repository.list_todos(task.id)
    state = {
        "agent_task_id": task.id, "agent_run_id": run_two.id, "question": task.objective,
        "context_window": 8_192, "task_memory_snapshot": {},
        "task_todos": [{
            "id": todo.id, "status": todo.status, "required": todo.required,
            "artifact_ids": list(todo.artifact_ids_json or []),
        } for todo in todos],
    }
    monkeypatch.setattr(
        deep_research_nodes,
        "_call_model",
        AsyncMock(return_value=("Final report grounded in attempt one.", {"model": "test"})),
    )
    try:
        synthesized = await deep_research_nodes.deep_task_synthesizer(state, {})
        assert synthesized["task_evidence_manifest"] == [{
            "id": inherited.id,
            "kind": "intermediate_report",
            "sha256": inherited.sha256,
            "byte_size": inherited.byte_size,
            "summary": {},
            "todo_id": "done",
            "plan_revision": 0,
            "origin_run_id": run_one.id,
            "origin_attempt": 1,
            "inherited": True,
            "validity": "valid",
        }]
        final_report = await persist_task_artifact(
            task_id=task.id,
            agent_run_id=run_two.id,
            kind="final_report",
            content=synthesized["final_answer"],
            provenance={
                "evidence_manifest": synthesized["task_evidence_manifest"],
                "evidence_gaps": synthesized["task_incomplete_reasons"],
                "quality_review": {},
            },
            source_refs={"artifact_ids": [inherited.id]},
        )
        timeline = await agent_tasks_api.get_agent_task_timeline(
            task.id, run_two.id, thread_id=sample_thread.id,
        )
    finally:
        set_content_store(None)

    final_item = next(item for item in timeline["items"] if item["id"] == f"final:{final_report.id}")
    assert final_item["primary_content"] == "Final report grounded in attempt one."
    assert final_item["evidence_manifest"][0]["origin_run_id"] == run_one.id
    assert final_item["sources"] == [{
        "id": final_item["sources"][0]["id"],
        "kind": "web",
        "title": "Primary clinical source",
        "url": "https://example.test/clinical",
        "snippet": "Relevant evidence",
        "artifact_id": inherited.id,
        "origin_run_id": run_one.id,
        "origin_attempt": 1,
        "plan_revision": 0,
        "inherited": True,
        "origins": [{
            "run_id": run_one.id,
            "attempt": 1,
            "artifact_id": inherited.id,
            "plan_revision": 0,
            "inherited": True,
        }],
    }]


def _ready_web_todo() -> SimpleNamespace:
    return SimpleNamespace(
        id="web-1", title="Current evidence", description="Search current evidence",
        completion_criteria="Cited current sources", status="running", priority=90,
        required=True, dependency_ids_json=[], profile_id="web_researcher", attempt=1,
        max_attempts=2, progress=0, result_summary=None, artifact_ids_json=[], version=1,
    )


@pytest.mark.asyncio
async def test_deep_scheduler_reuses_task_wide_web_approval(monkeypatch):
    todo = _ready_web_todo()
    monkeypatch.setattr(deep_research_nodes, "schedule_ready_todos", AsyncMock(return_value=[todo]))
    monkeypatch.setattr(deep_research_nodes, "list_todos", AsyncMock(return_value=[todo]))
    monkeypatch.setattr(deep_research_nodes, "interrupt", lambda *_args, **_kwargs: pytest.fail("approved task must not interrupt again"))

    result = await deep_research_nodes.deep_task_scheduler({
        "agent_task_id": "task-1", "agent_run_id": "run-1", "task_plan_revision": 2,
        "task_limits": {"max_concurrency": 2, "max_fanout": 2}, "web_search_mode": "ask",
        "task_web_access": "allowed_for_task",
    }, {})

    assert [item["todo"]["id"] for item in result["task_work_items"]] == [todo.id]


@pytest.mark.asyncio
async def test_deep_scheduler_ask_mode_offers_once_and_task_scope(monkeypatch):
    todo = _ready_web_todo()
    captured = {}
    monkeypatch.setattr(deep_research_nodes, "schedule_ready_todos", AsyncMock(return_value=[todo]))
    monkeypatch.setattr(deep_research_nodes, "list_todos", AsyncMock(return_value=[todo]))

    def approve_for_task(payload):
        captured.update(payload)
        return {"action": "approve_for_scope", "interrupt_id": "approval-1"}

    monkeypatch.setattr(deep_research_nodes, "interrupt", approve_for_task)
    result = await deep_research_nodes.deep_task_scheduler({
        "agent_task_id": "task-1", "agent_run_id": "run-1", "task_plan_revision": 1,
        "task_limits": {"max_concurrency": 2, "max_fanout": 2}, "web_search_mode": "ask",
        "task_web_access": "undecided",
    }, {})

    assert captured["allowed_actions"] == ["approve", "approve_for_scope", "continue_without"]
    assert captured["approval_scope_kind"] == "task"
    assert result["task_work_items"][0]["approval_ref"]["action"] == "approve_for_scope"
    assert result["task_web_access"] == "allowed_for_task"


@pytest.mark.asyncio
async def test_process_restart_continues_checkpoint_without_fresh_graph_input(monkeypatch):
    snapshot = SimpleNamespace(
        values={
            "question": "Research",
            "embedding_model": "embed",
            "context_window": 8192,
            "use_web_search": False,
            "use_reranker": True,
        },
        next=("deep_task_planner",),
    )
    app = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))
    invoke = AsyncMock(return_value={**snapshot.values, "final_answer": "Completed"})
    persist = AsyncMock(return_value={"status": "completed", "answer": "Completed"})
    monkeypatch.setattr(router_runtime, "WorkflowCompiler", lambda: SimpleNamespace(compile=lambda *_args, **_kwargs: app))
    monkeypatch.setattr(router_runtime, "_invoke_graph_with_partial_state", invoke)
    monkeypatch.setattr(router_runtime, "_persist_success_turn", persist)

    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="deep_research_agent",
        checkpoint_thread_id="run-1",
        resolved_spec_json=_spec(),
    )
    result = await router_runtime.continue_compiled_rag_chat(run, checkpointer=object())

    assert result == {"status": "completed", "answer": "Completed"}
    assert invoke.await_args.args[1] is None


@pytest.mark.asyncio
async def test_task_result_projector_never_persists_a_chat_turn(monkeypatch):
    snapshot = SimpleNamespace(
        values={"question": "Research", "embedding_model": "embed", "context_window": 8192},
        next=(),
    )
    app = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))
    persist = AsyncMock()
    monkeypatch.setattr(router_runtime, "WorkflowCompiler", lambda: SimpleNamespace(compile=lambda *_args, **_kwargs: app))
    monkeypatch.setattr(router_runtime, "_persist_success_turn", persist)
    run = SimpleNamespace(
        id="task-run", thread_id="thread", workflow_id="deep_research_agent",
        checkpoint_thread_id="task-run", resolved_spec_json=_spec(),
    )

    result = await router_runtime.continue_compiled_rag_chat(
        run,
        checkpointer=object(),
        result_projector=router_runtime.project_agent_task_result,
    )

    assert result["chat_turn_id"] is None
    assert result["status"] == "completed"
    persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_task_run_persists_debug_trace(monkeypatch):
    task = SimpleNamespace(
        id="task-trace",
        thread_id="thread-trace",
        status="running",
        objective="Research current evidence",
        version=2,
        config_json={
            "llm_model": "test-model",
            "context_window": 8192,
            "use_web_search": False,
            "web_search_mode": "off",
            "enabled_profiles": ["document_researcher"],
            "limits": {},
        },
        budgets_json={},
    )
    run = SimpleNamespace(
        id="run-trace",
        thread_id=task.thread_id,
        workflow_id="deep_research_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category="deep",
        workflow_version=1,
        checkpoint_thread_id="run-trace",
        resolved_spec_json=_spec(),
        pending_interrupt_json={},
        runtime_binding_json={},
        metrics_json={},
        debug_trace_json=None,
        status="running",
        started_at=utc_now(),
        completed_at=None,
    )
    completed_run = SimpleNamespace(**{**run.__dict__, "status": "completed", "completed_at": utc_now()})
    workflow_repository = SimpleNamespace(
        list_run_events=AsyncMock(return_value=[]),
        append_run_event=AsyncMock(return_value=True),
        update_runtime_binding=AsyncMock(),
    )

    monkeypatch.setattr(agent_task_runtime.tasks, "get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(agent_task_runtime, "ensure_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(agent_task_runtime, "get_thread", AsyncMock(return_value=SimpleNamespace(embedding_model="embed")))
    monkeypatch.setattr(agent_task_runtime.tasks, "list_todos", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_web_access", AsyncMock(return_value="undecided"))
    monkeypatch.setattr(agent_task_runtime.tasks, "list_artifacts", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_task_runtime.tasks, "complete_task", AsyncMock())
    finalize_task_run = AsyncMock(return_value=SimpleNamespace(**{**task.__dict__, "status": "completed"}))
    monkeypatch.setattr(agent_task_runtime.tasks, "finalize_task_run", finalize_task_run)
    monkeypatch.setattr(agent_task_runtime.tasks, "release_task_lease", AsyncMock())
    monkeypatch.setattr(agent_task_runtime, "record_terminal_result", AsyncMock())
    monkeypatch.setattr(agent_task_runtime, "AgentWorkflowRepository", lambda: workflow_repository)
    monkeypatch.setattr(
        "app.runtime.langgraph.router_runtime.continue_compiled_rag_chat",
        AsyncMock(return_value={"status": "completed", "answer": "Grounded report", "node_events": [], "tool_events": []}),
    )
    monkeypatch.setattr(
        agent_task_runtime,
        "persist_task_artifact",
        AsyncMock(return_value=SimpleNamespace(id="final-artifact")),
    )

    @asynccontextmanager
    async def checkpointer():
        yield object()

    monkeypatch.setattr("app.runtime.langgraph.checkpointing.open_agent_checkpointer", checkpointer)

    await agent_task_runtime.execute_claimed_task(task.id, "worker-trace")

    finalize_task_run.assert_awaited_once()
    assert not finalize_task_run.await_args.kwargs["error"]
    debug_payload = finalize_task_run.await_args.kwargs["debug_trace"]
    assert debug_payload["trace"]["run_id"] == run.id
    assert debug_payload["trace"]["status"] == "completed"
    assert debug_payload["summary"]["usedNodeCount"] == 0


@pytest.mark.asyncio
async def test_compiled_deep_graph_trace_covers_parallel_nodes_tools_and_hitl_branch():
    async def context_loader(_state, _config):
        return {"task_memory_snapshot": {}}

    async def planner(_state, _config):
        return {"task_plan_revision": 1}

    async def scheduler(_state, _config):
        todos = [
            {"id": "todo-a", "title": "A", "description": "A", "status": "running", "dependency_ids": []},
            {"id": "todo-b", "title": "B", "description": "B", "status": "running", "dependency_ids": []},
        ]
        return {
            "task_todos": todos,
            "task_work_items": [{
                "task_id": "task-trace", "agent_run_id": "run-trace-success",
                "todo": todo, "plan_revision": 1, "dispatch_id": "dispatch-deep",
                "ordinal": ordinal, "execution_key": f"execution-{ordinal}",
                "trace_visit_index": ordinal + 11,
            } for ordinal, todo in enumerate(todos)],
        }

    async def subagent(state, config):
        item = state["task_work_item"]
        recorder = config["configurable"]["trace_recorder"]
        recorder.record_tool_event({
            "tool_name": "search_documents",
            "caller_node": "deep_research_subagent",
            "caller_node_type": "deep_research_subagent",
            "caller_visit_index": item["trace_visit_index"],
            "dispatch_id": item["dispatch_id"],
            "work_id": item["execution_key"],
            "ordinal": item["ordinal"],
            "attempt": 1,
            "argument_hash": f"argument-{item['ordinal']}",
            "ok": True,
            "source_count": 1,
            "result_preview": {"todo_id": item["todo"]["id"]},
        })
        return {"task_result_packets": [{
            "dispatch_id": item["dispatch_id"],
            "work_id": item["execution_key"],
            "ordinal": item["ordinal"],
            "attempt": 1,
            "status": "completed",
            "todo_id": item["todo"]["id"],
        }]}

    async def coordinator(_state, _config):
        return {"task_controller_route": "synthesize", "task_controller_reason": "complete"}

    async def synthesizer(_state, _config):
        return {"final_answer": "Grounded report"}

    async def critic(_state, _config):
        return {"task_critic_report": {"pass": True}}

    async def finalizer(state, _config):
        return {"final_answer": state.get("final_answer") or "Stopped"}

    success_registry = NodeRegistry()
    success_registry._nodes.update({
        WorkflowNodeType.CONTEXT_LOADER.value: context_loader,
        WorkflowNodeType.DEEP_TASK_PLANNER.value: planner,
        WorkflowNodeType.DEEP_TASK_SCHEDULER.value: scheduler,
        WorkflowNodeType.DEEP_RESEARCH_SUBAGENT.value: subagent,
        WorkflowNodeType.DEEP_COORDINATOR.value: coordinator,
        WorkflowNodeType.DEEP_TASK_SYNTHESIZER.value: synthesizer,
        WorkflowNodeType.EVIDENCE_CRITIC.value: critic,
        WorkflowNodeType.FINALIZER.value: finalizer,
    })
    success_run = SimpleNamespace(
        id="run-trace-success", thread_id="thread-trace", user_id=None,
        workflow_id="deep_research_agent", workflow_version_id="deep_research_agent:v1",
        resolved_spec_json=_spec(), status="completed", started_at=utc_now(), completed_at=utc_now(),
    )
    success_recorder = AgentTraceRecorder(success_run)
    result = await WorkflowCompiler(success_registry).compile(_spec()).ainvoke(
        {
            "agent_run_id": success_run.id, "agent_task_id": "task-trace",
            "thread_id": success_run.thread_id, "question": "Research",
            "embedding_model": "test-embedding", "llm_model": "test-model",
            "context_window": 8_192, "node_events": [], "tool_events": [],
            "task_result_packets": [], "task_todos": [],
        },
        config={"configurable": {"trace_recorder": success_recorder}},
    )
    success_payload = success_recorder.finalize(
        run=success_run, chat_turn_id=None, metrics={}, result=result,
    )
    success_nodes = success_payload["summary"]["nodes"]
    node_types = [node["type"] for node in success_nodes]
    assert node_types.count("deep_research_subagent") == 2
    assert {
        "context_loader", "deep_task_planner", "deep_task_scheduler",
        "deep_research_subagent", "deep_coordinator", "deep_task_synthesizer",
        "evidence_critic", "finalizer",
    }.issubset(set(node_types))
    assert sorted(
        node["visitIndex"] for node in success_nodes if node["type"] == "deep_research_subagent"
    ) == [11, 12]
    tool_spans = [
        span for span in success_payload["trace"]["spans"]
        if str(span.get("span_id") or "").startswith("tool:")
    ]
    assert len(tool_spans) == 2
    assert {span["attributes"]["askpdf.parallel.work_id"] for span in tool_spans} == {
        "execution-0", "execution-1",
    }
    assert success_payload["summary"]["usedNodeCount"] == len(success_nodes)

    async def empty_scheduler(_state, _config):
        return {"task_work_items": [], "task_todos": []}

    async def pause_coordinator(_state, _config):
        return {"task_controller_route": "pause", "task_controller_reason": "requested"}

    pause_registry = NodeRegistry()
    pause_registry._nodes.update({
        WorkflowNodeType.CONTEXT_LOADER.value: context_loader,
        WorkflowNodeType.DEEP_TASK_PLANNER.value: planner,
        WorkflowNodeType.DEEP_TASK_SCHEDULER.value: empty_scheduler,
        WorkflowNodeType.DEEP_COORDINATOR.value: pause_coordinator,
        WorkflowNodeType.FINALIZER.value: finalizer,
    })

    async def hitl_gate(state, config, *, node_id):
        recorder = config["configurable"]["trace_recorder"]
        pending = {
            "interrupt_id": "pause-interrupt", "gate_id": node_id, "node_id": node_id,
            "type": "task_pause", "status": "pending", "title": "Deep research paused",
            "allowed_actions": ["approve", "reject"],
        }
        recorder.record_interrupted_snapshot(
            interrupt=pending,
            state=state,
        )
        recorder.record_interrupt_event(pending)
        return {
            "hitl_gate_route": "reject",
            "hitl_gate_routes": {node_id: "reject"},
        }

    pause_registry.hitl_gate = hitl_gate
    pause_run = SimpleNamespace(
        id="run-trace-pause", thread_id="thread-trace", user_id=None,
        workflow_id="deep_research_agent", workflow_version_id="deep_research_agent:v1",
        resolved_spec_json=_spec(), status="completed", started_at=utc_now(), completed_at=utc_now(),
    )
    pause_recorder = AgentTraceRecorder(pause_run)
    pause_result = await WorkflowCompiler(pause_registry).compile(_spec()).ainvoke(
        {
            "agent_run_id": pause_run.id, "agent_task_id": "task-trace",
            "thread_id": pause_run.thread_id, "question": "Research",
            "embedding_model": "test-embedding", "llm_model": "test-model",
            "context_window": 8_192, "node_events": [], "tool_events": [],
            "task_result_packets": [], "task_todos": [],
        },
        config={"configurable": {"trace_recorder": pause_recorder}},
    )
    pause_payload = pause_recorder.finalize(
        run=pause_run, chat_turn_id=None, metrics={}, result=pause_result,
    )
    assert any(node["type"] == "hitl_gate" for node in pause_payload["summary"]["nodes"])
    assert any(
        event.get("name") == "interrupt.requested"
        for span in pause_payload["trace"]["spans"]
        for event in span.get("events") or []
    )


@pytest.mark.asyncio
async def test_task_commands_are_idempotent_versioned_and_terminal_cancel(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent",
                name="Deep Research Agent",
                description="test",
                visibility="builtin",
                is_builtin=True,
                schema_version=2,
                spec_json=_spec(),
                metadata_json={"version": 1},
            ))

    task, duplicate = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="  Research   the evidence  ",
        idempotency_key="create-once",
        config={"enabled_profiles": ["document_researcher"], "limits": {}},
    )
    assert duplicate is False
    assert task.objective == "Research the evidence"
    same, duplicate = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="ignored duplicate body",
        idempotency_key="create-once",
        config={},
    )
    assert duplicate is True
    assert same.id == task.id

    started, command, duplicate = await repository.apply_command(
        task.id, action="start", idempotency_key="start-once", expected_version=task.version,
    )
    assert started.status == "queued"
    repeated, repeated_command, duplicate = await repository.apply_command(
        task.id, action="start", idempotency_key="start-once", expected_version=task.version,
    )
    assert duplicate is True
    assert repeated_command.id == command.id
    assert repeated.version == started.version
    with pytest.raises(repository.AgentTaskConflict, match="stale") as conflict:
        await repository.apply_command(
            task.id, action="pause", idempotency_key=str(uuid.uuid4()), expected_version=task.version,
        )
    assert conflict.value.code == "task_version_conflict"

    cancelled, _, _ = await repository.apply_command(
        task.id, action="cancel", idempotency_key="cancel-once", expected_version=started.version,
    )
    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None


@pytest.mark.asyncio
async def test_restarted_runner_terminalizes_reclaimed_cancellation(monkeypatch):
    task = SimpleNamespace(id="task-1", status="cancelling")
    active_run = SimpleNamespace(id="run-1", status="running")
    complete_run = AsyncMock()
    complete_task = AsyncMock()
    release_lease = AsyncMock()

    monkeypatch.setattr(agent_task_runtime.tasks, "get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=active_run))
    monkeypatch.setattr(agent_task_runtime.tasks, "complete_task", complete_task)
    monkeypatch.setattr(agent_task_runtime.tasks, "release_task_lease", release_lease)
    monkeypatch.setattr(
        agent_task_runtime,
        "AgentWorkflowRepository",
        lambda: SimpleNamespace(complete_run=complete_run),
    )

    await agent_task_runtime.execute_claimed_task(task.id, "replacement-worker")

    complete_run.assert_awaited_once_with(
        active_run.id,
        status="cancelled",
        error_json={"code": "agent_task_cancelled", "retryable": False},
    )
    complete_task.assert_awaited_once_with(
        task.id,
        status="cancelled",
        reason="cancelled_by_user",
    )
    release_lease.assert_awaited_once_with(task.id, "replacement-worker", lease_seconds=60)


@pytest.mark.asyncio
async def test_cancelling_task_is_reclaimable_without_returning_to_running(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent",
                name="Deep Research Agent",
                description="test",
                visibility="builtin",
                is_builtin=True,
                schema_version=2,
                spec_json=_spec(),
                metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="Cancel after restart",
        idempotency_key="cancel-reclaim",
        config={},
    )
    queued, _, _ = await repository.apply_command(
        task.id,
        action="start",
        idempotency_key="cancel-reclaim-start",
        expected_version=task.version,
    )
    claimed = await repository.claim_next_task("original-worker")
    assert claimed is not None and claimed.id == task.id
    cancelling, _, _ = await repository.apply_command(
        task.id,
        action="cancel",
        idempotency_key="cancel-reclaim-command",
        expected_version=claimed.version,
    )
    assert cancelling.status == "cancelling"

    await repository.release_task_lease(task.id, "original-worker")
    reclaimed = await repository.claim_next_task("replacement-worker")

    assert reclaimed is not None and reclaimed.id == task.id
    assert reclaimed.status == "cancelling"
    assert reclaimed.current_phase == "cancelling"


@pytest.mark.asyncio
async def test_scheduler_respects_dependencies_and_replays_unstarted_claim(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent", name="Deep", description="test",
                visibility="builtin", is_builtin=True, schema_version=2,
                spec_json=_spec(), metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Research", idempotency_key="schedule",
        config={"enabled_profiles": ["document_researcher"], "limits": {"max_attempts_per_todo": 2}},
    )
    proposal = DeepResearchPlanProposal.model_validate({
        "objective": "Research", "success_criteria": ["Done"],
        "todos": [
            {"id": "first", "title": "First", "description": "First", "completion_criteria": "Done", "profile_id": "document_researcher", "priority": 90},
            {"id": "second", "title": "Second", "description": "Second", "completion_criteria": "Done", "dependency_ids": ["first"], "profile_id": "document_researcher"},
        ],
    })
    run = await _attach_test_run(test_session_maker, task)
    assert await repository.get_task_web_access(task.id) == "undecided"
    await repository.set_task_web_access(
        task.id,
        repository.WEB_ACCESS_ALLOWED,
        agent_run_id=run.id,
        interrupt_id="web-approval-1",
    )
    await repository.set_task_web_access(
        task.id,
        repository.WEB_ACCESS_ALLOWED,
        agent_run_id=run.id,
        interrupt_id="web-approval-1",
    )
    assert await repository.get_task_web_access(task.id) == "allowed_for_task"
    access_events = [
        event for event in await repository.list_events(task.id)
        if event.event_type == "approval.responded"
        and event.payload_json.get("status") == "allowed_for_task"
    ]
    assert len(access_events) == 1
    await repository.persist_plan(task.id, proposal, agent_run_id=run.id, reason="initial", planner_visit=1)
    await repository.persist_plan(task.id, proposal, agent_run_id=run.id, reason="bounded_replan", planner_visit=2)
    persisted = {todo.id: todo for todo in await repository.list_todos(task.id)}
    assert persisted["second"].dependency_ids_json == ["first"]
    assert all(
        isinstance(event.payload_json, dict)
        for event in await repository.list_events(task.id)
    )
    claimed = await repository.schedule_ready_todos(task.id, limit=4)
    assert [todo.id for todo in claimed] == ["first"]
    assert claimed[0].attempt == 1
    replayed = await repository.schedule_ready_todos(task.id, limit=4)
    assert [todo.id for todo in replayed] == ["first"]
    assert replayed[0].attempt == 1


@pytest.mark.asyncio
async def test_terminal_retry_preserves_completed_todos_and_resets_failed_projection(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent", name="Deep", description="test",
                visibility="builtin", is_builtin=True, schema_version=2,
                spec_json=_spec(), metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Research", idempotency_key="retry-projection",
        config={"enabled_profiles": ["document_researcher"], "limits": {"max_attempts_per_todo": 2}},
    )
    run = await _attach_test_run(test_session_maker, task)
    proposal = DeepResearchPlanProposal.model_validate({
        "objective": "Research", "success_criteria": ["Done"],
        "todos": [
            {"id": "done", "title": "Done", "description": "Done", "completion_criteria": "Done", "profile_id": "document_researcher"},
            {"id": "failed", "title": "Failed", "description": "Failed", "completion_criteria": "Done", "profile_id": "document_researcher"},
        ],
    })
    await repository.persist_plan(task.id, proposal, agent_run_id=run.id, reason="initial", planner_visit=1)
    async with test_session_maker() as session:
        async with session.begin():
            rows = {
                value.id: value
                for value in (await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id))).scalars().all()
            }
            rows["done"].status = "completed"
            rows["done"].progress = 100
            rows["done"].artifact_ids_json = ["artifact-done"]
            rows["failed"].status = "failed"
            rows["failed"].attempt = 2
            rows["failed"].progress = 80
            rows["failed"].result_summary = "Unusable partial result"
            rows["failed"].terminal_reason = "timeout"
            rows["failed"].artifact_ids_json = ["artifact-failed"]
            rows["failed"].evidence_ids_json = ["evidence-failed"]
    failed_task = await repository.complete_task(task.id, status="failed", reason="required_work_failed")
    retried, _, _ = await repository.apply_command(
        task.id,
        action="retry",
        idempotency_key="retry-projection-command",
        expected_version=failed_task.version,
    )
    todos = {value.id: value for value in await repository.list_todos(task.id)}

    assert retried.status == "queued"
    assert retried.progress == 50
    assert todos["done"].status == "completed"
    assert todos["done"].artifact_ids_json == ["artifact-done"]
    assert todos["failed"].status == "pending"
    assert todos["failed"].attempt == 0
    assert todos["failed"].progress == 0
    assert todos["failed"].artifact_ids_json == []
    assert todos["failed"].evidence_ids_json == []


@pytest.mark.asyncio
async def test_todo_identity_is_task_scoped_and_budget_terminal_guards_are_atomic(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent", name="Deep", description="test",
                visibility="builtin", is_builtin=True, schema_version=2,
                spec_json=_spec(), metadata_json={"version": 1},
            ))

    tasks = []
    for suffix in ("one", "two"):
        task, _ = await repository.create_task(
            thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
            workflow_id="deep_research_agent", objective=f"Research {suffix}",
            idempotency_key=f"task-{suffix}",
            config={
                "enabled_profiles": ["document_researcher"],
                "limits": {"max_model_tokens": 10, "max_tool_calls": 2},
            },
        )
        proposal = DeepResearchPlanProposal.model_validate({
            "objective": task.objective,
            "success_criteria": ["Done"],
            "todos": [{
                "id": "shared-logical-id", "title": "Research", "description": "Research",
                "completion_criteria": "Done", "profile_id": "document_researcher",
            }],
        })
        run = await _attach_test_run(test_session_maker, task)
        await repository.persist_plan(task.id, proposal, agent_run_id=run.id, reason="initial", planner_visit=1)
        assert [todo.id for todo in await repository.list_todos(task.id)] == ["shared-logical-id"]
        tasks.append(task)

    await repository.consume_budget(tasks[0].id, model_tokens=10, tool_calls=2)
    with pytest.raises(repository.AgentTaskConflict, match="token budget"):
        await repository.consume_budget(tasks[0].id, model_tokens=1)
    with pytest.raises(repository.AgentTaskConflict, match="tool-call budget"):
        await repository.consume_budget(tasks[0].id, tool_calls=1)

    cancelled = await repository.complete_task(tasks[0].id, status="cancelled", reason="user")
    unchanged = await repository.complete_task(tasks[0].id, status="failed", reason="late_worker")
    assert unchanged.status == "cancelled"
    assert unchanged.version == cancelled.version


def test_task_api_enforces_idempotency_ownership_and_builtin_contract(api_client, sample_thread):
    capabilities = api_client.get("/api/deep-research/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["enabled"] is True
    assert capabilities.json()["web_enabled"] is True
    builtin_limits = _spec()["config"]["task_policy"]["limits"]
    assert capabilities.json()["limits"] == builtin_limits

    payload = {
        "objective": "Research the uploaded evidence",
        "llm_model": "test-model",
        "context_window": 8192,
        "web_search_mode": "off",
    }
    headers = {"Idempotency-Key": "api-create-once"}
    created = api_client.post(
        f"/api/threads/{sample_thread.id}/agent-tasks",
        json=payload,
        headers=headers,
    )
    assert created.status_code == 201
    task = created.json()["task"]
    assert task["workflow_id"] == "deep_research_agent"
    assert task["configuration"]["engine"] == "langgraph"
    assert task["configuration"]["limits"]["max_concurrency"] == builtin_limits["max_concurrency"]

    web_created = api_client.post(
        f"/api/threads/{sample_thread.id}/agent-tasks",
        json={**payload, "objective": "Research current web evidence", "web_search_mode": "on"},
        headers={"Idempotency-Key": "api-create-web"},
    )
    assert web_created.status_code == 201, web_created.text
    assert web_created.json()["task"]["configuration"]["web_search_mode"] == "on"
    assert web_created.json()["task"]["configuration"]["use_web_search"] is True
    assert "web_researcher" in web_created.json()["task"]["configuration"]["enabled_profiles"]
    duplicate = api_client.post(
        f"/api/threads/{sample_thread.id}/agent-tasks",
        json={**payload, "objective": "A changed duplicate body"},
        headers=headers,
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["task"]["id"] == task["id"]

    missing = api_client.get(
        f"/api/agent-tasks/{task['id']}",
        params={"thread_id": str(uuid.uuid4())},
    )
    assert missing.status_code == 404
    started = api_client.post(
        f"/api/agent-tasks/{task['id']}/start",
        params={"thread_id": sample_thread.id},
        json={"expected_version": task["version"]},
        headers={"Idempotency-Key": "builtin-start"},
    )
    assert started.status_code == 200, started.text
    assert started.json()["task"]["status"] == "queued"
    assert started.json()["task"]["active_run_id"]


def test_task_api_explicitly_selects_hermes_and_uses_deployment_context(api_client, sample_thread, monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    monkeypatch.setenv("HERMES_MCP_CONTEXT_SECRET", "x" * 32)
    payload = {
        "objective": "Research with Hermes",
        "llm_model": "test-model",
        "context_window": 8192,
        "web_search_mode": "off",
        "engine": "hermes",
    }
    created = api_client.post(
        f"/api/threads/{sample_thread.id}/agent-tasks",
        json=payload,
        headers={"Idempotency-Key": "api-create-hermes"},
    )
    assert created.status_code == 201, created.text
    task = created.json()["task"]
    assert task["workflow_id"] == "hermes_rag_agent"
    assert task["configuration"]["engine"] == "hermes"
    assert task["configuration"]["context_window"] == 8192

    conflicting = api_client.post(
        f"/api/threads/{sample_thread.id}/agent-tasks",
        json={**payload, "context_window": 4096},
        headers={"Idempotency-Key": "api-create-hermes-normalized"},
    )
    assert conflicting.status_code == 409, conflicting.text
    assert conflicting.json()["detail"]["code"] == "hermes_context_length_conflict"
    assert conflicting.json()["detail"]["configured_context_length"] == 8192


@pytest.mark.asyncio
async def test_selected_run_event_stream_advances_cursor_across_filtered_events(monkeypatch):
    task = SimpleNamespace(id="task-1", active_run_id="run-selected")
    monkeypatch.setattr(agent_tasks_api, "_owned_task", AsyncMock(return_value=task))
    monkeypatch.setattr(
        repository,
        "list_task_runs",
        AsyncMock(return_value=[
            SimpleNamespace(id="run-selected"),
            SimpleNamespace(id="run-other"),
        ]),
    )
    calls: list[int] = []

    async def list_events(_task_id, *, after_sequence=0, **_kwargs):
        calls.append(after_sequence)
        if after_sequence == 0:
            return [
                SimpleNamespace(
                    id="event-1", sequence=1, event_type="todo.completed", task_id="task-1",
                    agent_run_id="run-other", todo_id="todo-other", subagent_run_id=None,
                    artifact_id=None, payload_json={"ignored": True}, created_at=utc_now(),
                ),
                SimpleNamespace(
                    id="event-2", sequence=2, event_type="task.updated", task_id="task-1",
                    agent_run_id=None, todo_id=None, subagent_run_id=None,
                    artifact_id=None, payload_json={"ignored": True}, created_at=utc_now(),
                ),
            ]
        assert after_sequence == 2
        return [SimpleNamespace(
            id="event-3", sequence=3, event_type="artifact.created", task_id="task-1",
            agent_run_id="run-selected", todo_id="todo-selected", subagent_run_id=None,
            artifact_id="artifact-selected", payload_json={"kind": "intermediate_report"},
            created_at=utc_now(),
        )]

    monkeypatch.setattr(repository, "list_events", list_events)
    monkeypatch.setattr(agent_tasks_api.asyncio, "sleep", AsyncMock())
    response = await agent_tasks_api.stream_agent_task_events(
        "task-1",
        thread_id="thread-1",
        after_sequence=0,
        run_id="run-selected",
        scope="run",
    )
    iterator = response.body_iterator
    try:
        chunk = await iterator.__anext__()
    finally:
        await iterator.aclose()

    assert calls == [0, 2]
    assert "id: 3" in chunk
    assert '"run_id":"run-selected"' in chunk
    assert "event-1" not in chunk and "event-2" not in chunk


@pytest.mark.asyncio
async def test_product_run_event_stream_reads_persisted_run_journal(monkeypatch):
    run = SimpleNamespace(id="run-selected")
    row = SimpleNamespace(
        id="journal-1",
        event_id="event-1",
        sequence=3,
        attempt=1,
        kind="run.completed",
        payload_json={"answer": "done"},
        occurred_at=utc_now(),
        created_at=utc_now(),
    )
    calls: list[str] = []

    class Repository:
        async def list_run_events(self, run_id):
            calls.append(run_id)
            return [row]

    monkeypatch.setattr(agent_workflows_api, "_owned_run_for_operation", AsyncMock(return_value=run))
    monkeypatch.setattr(agent_workflows_api, "AgentWorkflowRepository", Repository)
    monkeypatch.setattr(agent_workflows_api.asyncio, "sleep", AsyncMock())
    response = await agent_workflows_api.stream_agent_run_events(
        run.id,
        thread_id="thread-1",
        after_sequence=0,
    )
    iterator = response.body_iterator
    try:
        chunk = await iterator.__anext__()
    finally:
        await iterator.aclose()

    assert calls == [run.id]
    assert "event: run_event" in chunk
    assert '"event_id":"event-1"' in chunk
    assert '"terminal":true' in chunk


@pytest.mark.asyncio
async def test_identical_artifact_content_deduplicates_only_within_one_owner(
    tmp_path, test_session_maker, sample_thread,
):
    await _seed_deep_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Artifact ownership",
        idempotency_key="artifact-owner-scope",
        config={"limits": {"max_artifacts": 10, "max_artifact_bytes": 100_000, "max_single_artifact_bytes": 10_000}},
    )
    run = await _attach_test_run(test_session_maker, task)
    set_content_store(SharedVolumeContentStore(tmp_path / "content"))
    try:
        first = await persist_task_artifact(
            task_id=task.id, agent_run_id=run.id, todo_id="todo-a",
            kind="intermediate_report", content="same report",
        )
        second = await persist_task_artifact(
            task_id=task.id, agent_run_id=run.id, todo_id="todo-b",
            kind="intermediate_report", content="same report",
        )
        replay = await persist_task_artifact(
            task_id=task.id, agent_run_id=run.id, todo_id="todo-a",
            kind="intermediate_report", content="same report",
        )
    finally:
        set_content_store(None)

    assert first.id != second.id
    assert first.ownership_key == "todo:todo-a"
    assert second.ownership_key == "todo:todo-b"
    assert replay.id == first.id


@pytest.mark.asyncio
async def test_active_runtime_is_accrued_and_exhaustion_cancels_dispatch(
    test_session_maker, sample_thread,
):
    await _seed_deep_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id, project_id=sample_thread.project_id, user_id=None,
        workflow_id="deep_research_agent", objective="Runtime budget",
        idempotency_key="runtime-budget",
        config={"limits": {"max_active_runtime_ms": 5_000}},
    )
    task, _, _ = await repository.apply_command(
        task.id, action="start", idempotency_key="start-runtime", expected_version=task.version,
    )
    claimed = await repository.claim_next_task("worker-1", lease_seconds=60)
    assert claimed is not None
    async with test_session_maker() as session:
        async with session.begin():
            stored = await session.get(type(task), task.id)
            stored.heartbeat_at = utc_now() - timedelta(seconds=10)

    assert await repository.heartbeat_task(task.id, "worker-1", lease_seconds=60) is False
    exhausted = await repository.get_task(task.id)
    assert exhausted.status == "cancelling"
    assert exhausted.terminal_reason == "active_runtime_budget_exhausted"
    assert exhausted.budgets_json["elapsed_active_ms"] >= 5_000


@pytest.mark.asyncio
async def test_task_maintenance_runs_all_bounded_cleanup_classes(monkeypatch):
    artifact = SimpleNamespace(id="artifact", task_id="task", object_key="agent-tasks/task/run/artifact/1")
    store = SimpleNamespace(
        delete=AsyncMock(), exists=AsyncMock(return_value=True),
        list_keys=AsyncMock(return_value=[artifact.object_key, "agent-tasks/orphan/run/artifact/1"]),
    )
    monkeypatch.setattr(agent_task_maintenance, "get_content_store", lambda: store)
    monkeypatch.setattr(agent_task_maintenance.tasks, "expire_stale_tasks", AsyncMock(return_value=2))
    monkeypatch.setattr(agent_task_maintenance.tasks, "release_stale_task_leases", AsyncMock(return_value=1))
    monkeypatch.setattr(agent_task_maintenance.tasks, "list_pending_task_deletions", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_task_maintenance.tasks, "list_expired_artifacts", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_task_maintenance.tasks, "list_live_artifacts", AsyncMock(return_value=[artifact]))
    monkeypatch.setattr(agent_task_maintenance, "run_runtime_reconciliation", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_task_maintenance.tasks, "list_terminal_task_checkpoint_ids_before", AsyncMock(return_value=["checkpoint-1"]))
    monkeypatch.setattr(agent_task_maintenance.tasks, "clear_task_checkpoint_ids", AsyncMock(return_value=1))
    monkeypatch.setattr(
        "app.runtime.langgraph.checkpointing.delete_agent_checkpoints",
        AsyncMock(return_value=["checkpoint-1"]),
    )

    result = await agent_task_maintenance.run_task_maintenance(batch_size=10)

    assert result["expired_tasks"] == 2
    assert result["recovered_leases"] == 1
    assert result["orphaned_content"] == 1
    assert result["deleted_checkpoints"] == 1
    store.delete.assert_awaited_once_with("agent-tasks/orphan/run/artifact/1")


@pytest.mark.asyncio
async def test_task_worker_runs_maintenance_before_processing_a_busy_queue(monkeypatch):
    maintenance = AsyncMock(return_value={})
    claimed = SimpleNamespace(id="task-1", active_run_id="run-1")
    claim = AsyncMock(side_effect=[claimed, None])
    run = SimpleNamespace(id="run-1", task_id="task-1", framework="langgraph", builder_id="langgraph_graph")
    execute = AsyncMock()
    monkeypatch.setattr(agent_task_runtime, "run_task_maintenance", maintenance)
    monkeypatch.setattr(agent_task_runtime.tasks, "claim_next_task", claim)
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(agent_task_runtime, "execute_claimed_task", execute)

    await agent_task_runtime.run_task_worker(once=True, poll_seconds=0.01)

    maintenance.assert_awaited_once()
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_worker_honors_pre_signalled_shutdown(monkeypatch):
    maintenance = AsyncMock(return_value={})
    claim = AsyncMock()
    stop_event = asyncio.Event()
    stop_event.set()
    monkeypatch.setattr(agent_task_runtime, "run_task_maintenance", maintenance)
    monkeypatch.setattr(agent_task_runtime.tasks, "claim_next_task", claim)

    await agent_task_runtime.run_task_worker(stop_event=stop_event)

    maintenance.assert_awaited_once()
    claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_worker_stops_after_active_claim_without_claiming_more(monkeypatch):
    maintenance = AsyncMock(return_value={})
    claim = AsyncMock(return_value=SimpleNamespace(id="task-1", active_run_id="run-1"))
    run = SimpleNamespace(id="run-1", task_id="task-1", framework="langgraph", builder_id="langgraph_graph")
    stop_event = asyncio.Event()

    async def execute(_task_id, _worker_id):
        stop_event.set()

    monkeypatch.setattr(agent_task_runtime, "run_task_maintenance", maintenance)
    monkeypatch.setattr(agent_task_runtime.tasks, "claim_next_task", claim)
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(agent_task_runtime, "execute_claimed_task", execute)

    await agent_task_runtime.run_task_worker(stop_event=stop_event, poll_seconds=60)

    claim.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_worker_uses_persisted_framework_for_custom_hermes_definition(monkeypatch):
    task = SimpleNamespace(id="task-1", workflow_id="custom-hermes", active_run_id="run-1")
    run = SimpleNamespace(id="run-1", task_id="task-1", framework="hermes", builder_id="hermes_agent")
    selected_frameworks = []
    monkeypatch.setattr(agent_task_runtime, "run_task_maintenance", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_task_runtime.tasks, "claim_next_task", AsyncMock(side_effect=[task, None]))
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(agent_task_runtime, "deep_agent_budgets", lambda framework: selected_frameworks.append(framework) or {"wake_limit_seconds": 30})
    execute = AsyncMock()
    monkeypatch.setattr(agent_task_runtime, "execute_claimed_task", execute)

    await agent_task_runtime.run_task_worker(once=True)

    assert selected_frameworks == ["hermes"]
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_worker_fails_claim_without_persisted_runtime_identity(monkeypatch):
    task = SimpleNamespace(id="task-1", active_run_id="run-1")
    complete = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(agent_task_runtime, "run_task_maintenance", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_task_runtime.tasks, "claim_next_task", AsyncMock(side_effect=[task, None]))
    monkeypatch.setattr(agent_task_runtime.tasks, "get_task_run", AsyncMock(return_value=None))
    monkeypatch.setattr(agent_task_runtime.tasks, "complete_task", complete)
    monkeypatch.setattr(agent_task_runtime.tasks, "release_task_lease", release)
    execute = AsyncMock()
    monkeypatch.setattr(agent_task_runtime, "execute_claimed_task", execute)

    await agent_task_runtime.run_task_worker(once=True)

    complete.assert_awaited_once_with(task.id, status="failed", reason="task_runtime_identity_invalid")
    release.assert_awaited_once()
    execute.assert_not_awaited()

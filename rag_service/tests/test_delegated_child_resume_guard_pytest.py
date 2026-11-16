from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime_protocol.contracts import AgentRuntimeResult

from app.product_orchestration.interrupts import run_interrupt_resume_guard
from app.product_orchestration.service import AgentRunService


@pytest.mark.asyncio
async def test_delegate_sync_stops_sink_before_stamping_resume_guard(monkeypatch):
  service = AgentRunService(repository=MagicMock())
  call_order: list[str] = []

  child_run = SimpleNamespace(
      id="child-run",
      thread_id="thread-1",
      workflow_id="external_research",
      parent_run_id="parent-run",
      status="awaiting_human",
      runtime_binding_json={"binding_type": "hermes.session", "payload": {"session_id": "s1"}},
      resolved_spec_json={"schema_version": 1},
      workflow_version_id="wv-1",
      workflow_version=1,
      pending_interrupt_json={
          "interrupt_id": "int-1",
          "status": "pending",
          "response_operation": "run.approval.respond",
          "checkpoint_resume": True,
      },
  )
  parent_run = SimpleNamespace(id="parent-run", thread_id="thread-1", user_id="user-1")

  service.repository.get_run = AsyncMock(return_value=parent_run)
  service.repository.seed_builtin_workflows = AsyncMock()
  workflow = SimpleNamespace(
      id="external_research",
      spec_json={"runtime": {"chat_eligible": True}},
      framework="hermes",
      builder_id="hermes_agent",
      category="chat",
  )
  service.repository.get_workflow = AsyncMock(return_value=workflow)
  service.repository.create_run = AsyncMock(return_value=child_run)
  service.repository.update_runtime_binding = AsyncMock()
  service.repository.update_run_metadata_fields = AsyncMock()

  async def close_sink(*args, **kwargs):
      call_order.append("close_sink")

  async def mark_awaiting(*args, **kwargs):
      call_order.append("mark_awaiting")
      return child_run

  async def stamp_guard(run_id):
      call_order.append("stamp_guard")
      guard = run_interrupt_resume_guard(child_run)
      child_run.pending_interrupt_json = {
          **(child_run.pending_interrupt_json or {}),
          "resume_guard": guard,
      }
      return child_run

  service.repository.mark_run_awaiting_human = mark_awaiting
  service.repository.stamp_pending_interrupt_resume_guard = stamp_guard

  monkeypatch.setattr("app.product_orchestration.service.close_agent_execution_sink", close_sink)
  monkeypatch.setattr("app.product_orchestration.service.get_thread_settings", AsyncMock(return_value={"browser_access_mode": "ask"}))
  monkeypatch.setattr("app.product_orchestration.service.workflow_is_chat_eligible", lambda spec: True)
  monkeypatch.setattr(
      "app.product_orchestration.service.definition_from_workflow",
      lambda w: SimpleNamespace(definition_id="external_research", framework="hermes", builder_id="hermes_agent"),
  )
  monkeypatch.setattr(
      "app.product_orchestration.service.builder_for_definition",
      lambda d: MagicMock(resolve=AsyncMock(return_value={}), normalize=AsyncMock(return_value={})),
  )

  adapter = MagicMock()
  adapter.prepare_request = AsyncMock(side_effect=lambda req, **kwargs: req)
  adapter.start = AsyncMock(
      return_value=AgentRuntimeResult(
          status="awaiting_human",
          continuation=MagicMock(),
          output={
              "status": "awaiting_human",
              "pending_interrupt": {
                  "interrupt_id": "int-1",
                  "status": "pending",
                  "response_operation": "run.approval.respond",
                  "checkpoint_resume": True,
              },
          },
      ),
  )
  monkeypatch.setattr("app.product_orchestration.service.adapter_for_definition", lambda d: adapter)
  monkeypatch.setattr(
      "app.product_orchestration.service.result_to_product_payload",
      lambda runtime_result: {
          "status": "awaiting_human",
          "pending_interrupt": {
              "interrupt_id": "int-1",
              "status": "pending",
              "response_operation": "run.approval.respond",
              "checkpoint_resume": True,
          },
      },
  )
  monkeypatch.setattr("app.product_orchestration.service.build_run_metrics", lambda result, **kwargs: {})
  monkeypatch.setattr(
      "app.product_orchestration.service.AgentTraceRecorder",
      lambda run: MagicMock(record_interrupted_snapshot=MagicMock(), finalize=MagicMock(return_value={})),
  )

  response = await service.run_delegate_sync(
      parent_run_id="parent-run",
      thread_id="thread-1",
      child_workflow_id="external_research",
      question="Browse scholar",
      pre_fetch_bundle={},
      embedding_model="embed-model",
  )

  assert response["status"] == "awaiting_human"
  assert call_order.index("close_sink") < call_order.index("mark_awaiting")
  assert call_order.index("mark_awaiting") < call_order.index("stamp_guard")
  guard = (response["pending_interrupt"] or {}).get("resume_guard") or {}
  expected = run_interrupt_resume_guard(child_run)
  assert guard.get("runtime_binding_hash") == expected.get("runtime_binding_hash")


@pytest.mark.asyncio
async def test_read_terminal_chat_answer_prefers_parent_turn_over_placeholder(monkeypatch):
    from app.services.agent_runtime_projection import read_terminal_chat_answer_for_run

    child = SimpleNamespace(id="child-run", parent_run_id="parent-run")
    parent_turn = SimpleNamespace(
        agent_run_turn_kind="assistant_final",
        agent_run_sequence=0,
        payload={
            "question": "similar papers",
            "answer": "Found https://example.com/paper-one and https://example.com/paper-two",
        },
    )

    class Repository:
        async def list_chat_turns_for_run(self, run_id):
            if run_id == "parent-run":
                return [parent_turn]
            return []

    monkeypatch.setattr("app.product_orchestration.repository.AgentWorkflowRepository", Repository)
    answer = await read_terminal_chat_answer_for_run(child)
    assert "https://example.com/paper-one" in answer


@pytest.mark.asyncio
async def test_delegated_child_resume_json_surfaces_parent_final_answer(monkeypatch):
    from app.api import agent_workflows as agent_workflows_api

    child_run = SimpleNamespace(
        id="child-run",
        parent_run_id="parent-run",
        thread_id="thread-1",
        workflow_id="external_research",
        status="completed",
        pending_interrupt_json=None,
    )

    async def ready_thread(thread_id):
        return SimpleNamespace(id=thread_id)

    class CompletingService:
        async def resume_agent_run(self, *args, **kwargs):
            return SimpleNamespace(
                run=child_run,
                interrupt=None,
                outcome="completed",
                duplicate=False,
            )

    async def parent_answer(run):
        assert run.parent_run_id == "parent-run"
        return "Similar work: https://arxiv.org/abs/1706.03762"

    monkeypatch.setattr(agent_workflows_api, "get_thread", ready_thread)
    monkeypatch.setattr(agent_workflows_api, "AgentRunService", CompletingService)
    monkeypatch.setattr(
        agent_workflows_api,
        "AgentWorkflowRepository",
        lambda: SimpleNamespace(list_chat_turns_for_run=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        "app.services.agent_runtime_projection.read_terminal_chat_answer_for_run",
        parent_answer,
    )
    monkeypatch.setattr(
        agent_workflows_api,
        "_run_payload_for_read",
        AsyncMock(return_value={"id": "child-run", "status": "completed"}),
    )

    request = agent_workflows_api.AgentRunResumeRequest(
        thread_id="thread-1",
        interrupt_id="interrupt-1",
        action="approve",
    )
    payload = await agent_workflows_api.resume_agent_run("child-run", request)
    assert "arxiv.org" in payload["answer"]
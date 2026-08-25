"""Opt-in black-box checks against the deployed LangGraph runtime service."""

from __future__ import annotations

import asyncio
from collections import deque
import os
import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from app.runtime.adapter import RuntimeExecutionContext


_phase5_enabled = os.getenv("PHASE5_EXTERNAL_SMOKE", "").lower() in {"1", "true", "yes", "on"}
if _phase5_enabled and not os.getenv("PHASE5_EXTERNAL_LLM_MODEL"):
    raise RuntimeError("PHASE5_EXTERNAL_SMOKE=true requires PHASE5_EXTERNAL_LLM_MODEL")
pytestmark = pytest.mark.skipif(not _phase5_enabled, reason="requires PHASE5_EXTERNAL_SMOKE=true")


def _workflow(workflow_id: str) -> dict:
    return next(item["spec_json"] for item in load_builtin_workflows() if item["builtin_key"] == workflow_id)


class _RecentEvents:
    def __init__(self, limit: int = 12) -> None:
        self.events: deque[dict[str, Any]] = deque(maxlen=limit)

    async def emit_runtime_event(self, event: Any) -> None:
        self.events.append(event.to_dict())


async def _timeout_diagnostics(adapter: HttpLangGraphRuntimeAdapter, request: AgentRuntimeRequest) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        async with asyncio.timeout(5):
            diagnostics["inspection"] = dict(await adapter.inspect_state(request))
    except Exception as exc:
        diagnostics["inspection_error"] = type(exc).__name__
    try:
        async with asyncio.timeout(5):
            diagnostics["cancellation"] = dict(await adapter.cancel(request))
    except Exception as exc:
        diagnostics["cancellation_error"] = type(exc).__name__
    return diagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_id", ["router_rag_agent", "evaluator_replanner_rag_agent"])
async def test_external_runtime_executes_builtin_workflows(workflow_id):
    run_id = f"phase5-smoke-{workflow_id}-{uuid.uuid4().hex}"
    spec = _workflow(workflow_id)
    definition = AgentDefinition(
        definition_id=workflow_id,
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    request = AgentRuntimeRequest(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        definition_id=workflow_id,
        framework="langgraph",
        builder_id="langgraph_graph",
        input={"question": "Summarize the available evidence."},
        options={
            "embedding_model": os.getenv("PHASE5_EXTERNAL_EMBEDDING_MODEL", "phase5-deterministic-embedding"),
            "llm_model": os.environ["PHASE5_EXTERNAL_LLM_MODEL"],
            "use_web_search": False,
            "use_reranker": False,
        },
    )
    context = RuntimeExecutionContext(
        request=SimpleNamespace(
            question="Summarize the available evidence.",
            llm_model=os.environ["PHASE5_EXTERNAL_LLM_MODEL"],
            use_web_search=False,
            use_reranker=False,
            context_window=8192,
            runtime_execution_mode=True,
        ),
        embedding_model=request.options["embedding_model"],
        resolved_spec=spec,
        agent_run_context={"agent_run_id": run_id, "agent_workflow_id": workflow_id},
    )
    adapter = HttpLangGraphRuntimeAdapter()
    recent_events = _RecentEvents()
    smoke_timeout = float(os.getenv("PHASE5_SMOKE_TIMEOUT_SECONDS", "120"))
    try:
        capabilities = await adapter.capabilities(definition)
        assert capabilities.operations["run.events"].enabled is True
        validation = await adapter.validate(definition, spec)
        assert validation.valid is True
        try:
            async with asyncio.timeout(smoke_timeout):
                result = await adapter.start(request, context=context, event_sink=recent_events)
        except TimeoutError:
            diagnostics = await _timeout_diagnostics(adapter, request)
            pytest.fail(
                "Phase 5 external workflow timed out "
                f"after {smoke_timeout:.0f}s; run_id={run_id}; workflow_id={workflow_id}; "
                f"recent_events={list(recent_events.events)!r}; diagnostics={diagnostics!r}"
            )
        assert result.status in {"completed", "clarification", "awaiting_human"}
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_production_control_plane_executes_external_runtime_via_product_api():
    base_url = os.getenv("PHASE5_CONTROL_PLANE_URL", "http://rag-service:8000")
    unique = uuid.uuid4().hex
    timeout = httpx.Timeout(float(os.getenv("PHASE5_SMOKE_TIMEOUT_SECONDS", "120")))
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        project_response = await client.post(
            "/api/projects",
            json={
                "name": f"Phase 5 production smoke {unique}",
                "embedding_model": os.getenv(
                    "PHASE5_EXTERNAL_EMBEDDING_MODEL",
                    "phase5-deterministic-embedding",
                ),
            },
        )
        project_response.raise_for_status()
        project_id = project_response.json()["id"]

        thread_response = await client.post(
            f"/api/projects/{project_id}/threads",
            json={"name": "Production artifact external-runtime smoke"},
        )
        thread_response.raise_for_status()
        thread_id = thread_response.json()["id"]

        chat_response = await client.post(
            f"/api/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": "Summarize the available evidence.",
                "llm_model": os.environ["PHASE5_EXTERNAL_LLM_MODEL"],
                "bypass_clarification": True,
            },
        )
        chat_response.raise_for_status()
        chat_result = chat_response.json()
        assert chat_result["status"] in {"completed", "clarification", "awaiting_human"}
        assert chat_result["agent_run_id"]

        runs_response = await client.get(f"/api/threads/{thread_id}/agent-runs")
        runs_response.raise_for_status()
        runs = runs_response.json()["agent_runs"]
        persisted = next(run for run in runs if run["id"] == chat_result["agent_run_id"])
        assert persisted["status"] in {"completed", "clarification", "awaiting_human"}

        health_response = await client.get("/health")
        health_response.raise_for_status()
        assert health_response.json()["status"] == "ok"

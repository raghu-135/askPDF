"""Opt-in black-box checks against the deployed LangGraph runtime service."""

from __future__ import annotations

import asyncio
from collections import deque
import os
import uuid
from types import SimpleNamespace
from typing import Any

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

    async def emit(self, event: Any, payload: Any = None) -> None:
        if hasattr(event, "to_dict"):
            value = event.to_dict()
        elif isinstance(event, str):
            value = {"kind": event, "payload": dict(payload or {})}
        else:
            value = {"value": repr(event)}
        self.events.append(value)


async def _timeout_diagnostics(adapter: HttpLangGraphRuntimeAdapter, request: AgentRuntimeRequest) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        async with asyncio.timeout(5):
            diagnostics["inspection"] = dict(await adapter.inspect(request))
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
        assert capabilities.streaming is True
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

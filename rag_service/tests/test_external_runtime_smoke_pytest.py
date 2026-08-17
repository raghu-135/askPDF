"""Opt-in black-box checks against the deployed LangGraph runtime service."""

from __future__ import annotations

import os
from types import SimpleNamespace

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


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_id", ["router_rag_agent", "evaluator_replanner_rag_agent"])
async def test_external_runtime_executes_builtin_workflows(workflow_id):
    run_id = f"phase5-smoke-{workflow_id}"
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
            "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
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
    try:
        capabilities = await adapter.capabilities(definition)
        assert capabilities.streaming is True
        validation = await adapter.validate(definition, spec)
        assert validation.valid is True
        result = await adapter.start(request, context=context)
        assert result.status in {"completed", "clarification", "awaiting_human"}
    finally:
        await adapter.aclose()

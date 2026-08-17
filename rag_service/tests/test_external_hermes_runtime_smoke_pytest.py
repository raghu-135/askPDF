"""Opt-in black-box checks against the deployed Hermes runtime gateway."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.hermes_adapter import HermesRuntimeAdapter


_enabled = os.getenv("PHASE7_HERMES_SMOKE", "").lower() in {"1", "true", "yes", "on"}
_required = ("HERMES_RUNTIME_URL", "HERMES_MODEL", "ASKPDF_MCP_URL")
if _enabled:
    missing = [name for name in _required if not os.getenv(name)]
    if os.getenv("HERMES_RUNTIME_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        missing.append("HERMES_RUNTIME_ENABLED=true")
    if missing:
        raise RuntimeError("PHASE7_HERMES_SMOKE=true requires: " + ", ".join(missing))

pytestmark = pytest.mark.skipif(not _enabled, reason="requires PHASE7_HERMES_SMOKE=true")


def _definition_and_spec() -> tuple[AgentDefinition, dict]:
    value = next(item for item in load_builtin_workflows() if item["builtin_key"] == "hermes_rag_agent")
    return AgentDefinition(
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        category="router",
    ), dict(value["spec_json"])


@pytest.mark.asyncio
async def test_external_hermes_runtime_contract_and_execution():
    definition, spec = _definition_and_spec()
    request = AgentRuntimeRequest(
        run_id="phase7-smoke-hermes",
        thread_id="phase7-smoke-thread",
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input={"question": "Use the approved document evidence tool and summarize the available evidence."},
        options={"llm_model": os.environ["HERMES_MODEL"]},
    )
    adapter = HermesRuntimeAdapter(base_url=os.environ["HERMES_RUNTIME_URL"])
    try:
        capabilities = await adapter.capabilities(definition)
        assert capabilities.streaming is True
        assert capabilities.cancellation is True
        validation = await adapter.validate(definition, spec)
        assert validation.valid is True
        result = await adapter.start(
            request,
            context=RuntimeExecutionContext(
                request=SimpleNamespace(question=request.input["question"], runtime_execution_mode=True),
                resolved_spec=spec,
                agent_run_context={"agent_run_id": request.run_id, "agent_workflow_id": definition.definition_id},
            ),
        )
        assert result.status in {"completed", "failed", "cancelled"}
        assert result.runtime_metadata.get("upstream_run_id") or result.continuation is not None
    finally:
        await adapter.aclose()

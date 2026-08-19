"""Opt-in black-box checks against the deployed Hermes runtime gateway."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
import uuid

import asyncpg
import httpx
import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.hermes_adapter import HermesRuntimeAdapter


_enabled = os.getenv("PHASE7_HERMES_SMOKE", "").lower() in {"1", "true", "yes", "on"}
_required = (
    "HERMES_RUNTIME_URL",
    "PHASE7_PRODUCT_DATABASE_URL",
)
_TEST_MODEL = "phase5-deterministic"
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
    unique = uuid.uuid4().hex
    run_id = f"phase7-smoke-hermes-{unique}"
    thread_id = f"phase7-smoke-thread-{unique}"
    request = AgentRuntimeRequest(
        run_id=run_id,
        thread_id=thread_id,
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input={"question": "Use the approved document evidence tool and summarize the available evidence."},
        options={"llm_model": _TEST_MODEL},
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
        assert result.status == "completed", result
        assert result.output
        assert result.continuation is not None
        assert result.continuation.binding_type == "hermes_session"
        assert result.continuation.payload["session_id"]
        assert result.continuation.payload["upstream_run_id"]
        assert result.runtime_metadata["upstream_run_id"] == result.continuation.payload["upstream_run_id"]
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_product_api_executes_and_persists_hermes_run():
    unique = uuid.uuid4().hex
    base_url = os.getenv("PHASE7_CONTROL_PLANE_URL", "http://rag-service:8000")
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        project = await client.post(
            "/api/projects",
            json={"name": f"Phase 7 Hermes smoke {unique}", "embedding_model": "phase5-deterministic-embedding"},
        )
        project.raise_for_status()
        thread = await client.post(
            f"/api/projects/{project.json()['id']}/threads",
            json={"name": "Hermes product API smoke"},
        )
        thread.raise_for_status()
        thread_id = thread.json()["id"]
        settings = await client.put(
            f"/api/threads/{thread_id}/settings",
            json={"agent_workflow": {"workflow_id": "hermes_rag_agent"}},
        )
        settings.raise_for_status()
        chat = await client.post(
            f"/api/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": "Use document evidence and provide the deterministic answer.",
                "llm_model": _TEST_MODEL,
                "bypass_clarification": True,
            },
        )
        chat.raise_for_status()
        result = chat.json()
        assert result["status"] == "completed", result
        assert result["answer"]
        run_id = result["agent_run_id"]
        run_response = await client.get(f"/api/agent-runs/{run_id}", params={"thread_id": thread_id})
        run_response.raise_for_status()
        persisted = run_response.json()["agent_run"]
        assert persisted["framework"] == "hermes"
        assert persisted["builder_id"] == "hermes_agent"
        assert persisted["status"] == "completed"
        assert persisted["final_output"]["answer"]

    connection = await asyncpg.connect(os.environ["PHASE7_PRODUCT_DATABASE_URL"])
    try:
        binding = await connection.fetchval("SELECT runtime_binding_json FROM agent_runs WHERE id = $1", run_id)
    finally:
        await connection.close()
    if isinstance(binding, str):
        binding = json.loads(binding)
    assert binding["binding_type"] == "hermes_session"
    assert binding["payload"]["upstream_run_id"]

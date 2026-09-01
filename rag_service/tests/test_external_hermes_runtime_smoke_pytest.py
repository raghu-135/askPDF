"""Opt-in black-box checks against the deployed Hermes runtime gateway."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
import uuid

import asyncpg
import httpx
import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.adapter import RuntimeInvocationContext
from runtime_protocol.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.runtime.hermes_builder import HermesBuilderProvider
from app.mcp.execution_context_token import issue_execution_context_token
from app.tools.context import ToolInvocationContext


_enabled = os.getenv("HERMES_RUNTIME_SMOKE", "").lower() in {"1", "true", "yes", "on"}
_required = (
    "HERMES_RUNTIME_URL",
    "HERMES_RUNTIME_PRODUCT_DATABASE_URL",
)
_TEST_MODEL = "hermes-runtime-deterministic"
if _enabled:
    missing = [name for name in _required if not os.getenv(name)]
    if "hermes" not in {value.strip().lower() for value in os.getenv("COMPOSE_PROFILES", "").split(",")}:
        missing.append("COMPOSE_PROFILES=hermes")
    if missing:
        raise RuntimeError("HERMES_RUNTIME_SMOKE=true requires: " + ", ".join(missing))

pytestmark = pytest.mark.skipif(not _enabled, reason="requires HERMES_RUNTIME_SMOKE=true")


def _definition_and_spec() -> tuple[AgentDefinition, dict]:
    value = next(item for item in load_builtin_workflows() if item["builtin_key"] == "hermes_rag_agent")
    return AgentDefinition(
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        category="deep",
    ), dict(value["spec_json"])


@pytest.mark.asyncio
async def test_external_hermes_runtime_contract_and_execution():
    definition, spec = _definition_and_spec()
    unique = uuid.uuid4().hex
    run_id = f"hermes-runtime-smoke-hermes-{unique}"
    thread_id = f"hermes-runtime-smoke-thread-{unique}"
    configured_context = int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"])
    token = issue_execution_context_token(
        ToolInvocationContext(
            thread_id=thread_id,
            run_id=run_id,
            embedding_model="hermes-runtime-deterministic-embedding",
            context_window=configured_context,
            extensions={"task_id": run_id, "llm_model": _TEST_MODEL},
        ),
        task_id=run_id,
        allowed_tools=list(spec["config"]["allowed_tool_ids"]),
    )
    request = AgentRuntimeRequest(
        run_id=run_id,
        thread_id=thread_id,
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input={
            "question": "Use the approved document evidence tool and summarize the available evidence.",
            "mcp_execution_context_token": token,
        },
        options={"llm_model": _TEST_MODEL, "context_window": configured_context},
    )
    adapter = HermesRuntimeAdapter(base_url=os.environ["HERMES_RUNTIME_URL"])
    try:
        capabilities = await adapter.capabilities(definition)
        assert capabilities.operations["run.events"].enabled is True
        assert capabilities.operations["run.cancel"].enabled is True
        assert capabilities.operations["run.steer_live"].enabled is False
        for operation in ("run.send_followup", "run.interrupt_with_input"):
            assert capabilities.operations[operation].enabled is False
        validation = await adapter.validate(definition, spec)
        assert validation.valid is True
        resolved_spec = await HermesBuilderProvider().resolve(
            definition,
            spec,
            request_overrides={"llm_model": _TEST_MODEL, "context_window": configured_context},
        )
        result = await adapter.start(
            request,
            context=RuntimeInvocationContext(
                request_payload={"question": request.input["question"], "runtime_execution_mode": True},
                resolved_spec=resolved_spec,
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
async def test_product_api_executes_and_persists_hermes_deep_research_task():
    unique = uuid.uuid4().hex
    base_url = os.getenv("HERMES_RUNTIME_CONTROL_PLANE_URL", "http://rag-service:8000")
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        project = await client.post(
            "/api/projects",
            json={"name": f"Hermes runtime smoke {unique}", "embedding_model": "hermes-runtime-deterministic-embedding"},
        )
        project.raise_for_status()
        thread = await client.post(
            f"/api/projects/{project.json()['id']}/threads",
            json={"name": "Hermes product API smoke"},
        )
        thread.raise_for_status()
        thread_id = thread.json()["id"]
        created = await client.post(
            f"/api/threads/{thread_id}/agent-tasks",
            headers={"Idempotency-Key": f"hermes-runtime-hermes-{unique}"},
            json={
                "objective": "Use document evidence and provide the deterministic answer.",
                "llm_model": _TEST_MODEL,
                "context_window": int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"]),
                "web_search_mode": "off",
                "engine": "hermes",
            },
        )
        created.raise_for_status()
        task = created.json()["task"]
        started = await client.post(
            f"/api/agent-tasks/{task['id']}/start",
            params={"thread_id": thread_id},
            headers={"Idempotency-Key": f"hermes-runtime-hermes-start-{unique}"},
            json={"expected_version": task["version"]},
        )
        started.raise_for_status()
        task = started.json()["task"]
        for _ in range(120):
            current = await client.get(f"/api/agent-tasks/{task['id']}", params={"thread_id": thread_id})
            current.raise_for_status()
            task = current.json()["task"]
            if task["status"] in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.5)
        assert task["status"] == "completed", json.dumps(task, indent=2, default=str)
        run_id = task["active_run_id"]
        run_response = await client.get(f"/api/agent-runs/{run_id}", params={"thread_id": thread_id})
        run_response.raise_for_status()
        persisted = run_response.json()["agent_run"]
        assert persisted["framework"] == "hermes"
        assert persisted["builder_id"] == "hermes_agent"
        assert persisted["status"] == "completed"
        assert persisted["workflow_id"] == "hermes_rag_agent"

    connection = await asyncpg.connect(os.environ["HERMES_RUNTIME_PRODUCT_DATABASE_URL"])
    try:
        binding = await connection.fetchval("SELECT runtime_binding_json FROM agent_runs WHERE id = $1", run_id)
    finally:
        await connection.close()
    if isinstance(binding, str):
        binding = json.loads(binding)
    assert binding["binding_type"] == "hermes_session"
    assert binding["payload"]["upstream_run_id"]

from __future__ import annotations

import os
import uuid
import json

import httpx
import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.mcp.execution_context_token import issue_execution_context_token
from app.runtime.hermes_builder import HermesBuilderProvider
from app.tools.context import ToolInvocationContext
from runtime_protocol.contracts import AgentDefinition, AgentRuntimeRequest
from hermes_test_helpers import RUNTIME_URL, read_sse


pytestmark = pytest.mark.skipif(
    os.getenv("HERMES_RUNTIME_INTEGRATION", "").lower() not in {"1", "true", "yes", "on"},
    reason="requires the Hermes runtime Hermes integration Compose profile",
)


RECOVERY_RUN_ID = os.getenv("HERMES_RUNTIME_RECOVERY_RUN_ID") or f"hermes-runtime-recovery-{uuid.uuid4().hex}"
TEST_MODEL = os.getenv("HERMES_MODEL", "hermes-runtime-deterministic-hermes")


async def _recovery_payload() -> dict:
    thread_id = f"thread-{RECOVERY_RUN_ID}"
    definition = AgentDefinition(
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        category="deep",
    )
    builtin = next(
        item for item in load_builtin_workflows()
        if item["builtin_key"] == "hermes_rag_agent"
    )
    spec = dict(builtin["spec_json"])
    context_window = int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"])
    token = issue_execution_context_token(
        ToolInvocationContext(
            thread_id=thread_id,
            run_id=RECOVERY_RUN_ID,
            embedding_model="hermes-runtime-deterministic-embedding",
            context_window=context_window,
            extensions={"task_id": RECOVERY_RUN_ID, "llm_model": TEST_MODEL},
        ),
        task_id=RECOVERY_RUN_ID,
        allowed_tools=list(spec["config"]["allowed_tool_ids"]),
    )
    request = AgentRuntimeRequest(
        run_id=RECOVERY_RUN_ID,
        thread_id=thread_id,
        task_id=RECOVERY_RUN_ID,
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input={
            "question": "Work slowly and continue until stopped.",
            "mcp_execution_context_token": token,
        },
        options={"llm_model": TEST_MODEL, "context_window": context_window},
    )
    resolved = await HermesBuilderProvider().resolve(
        definition,
        spec,
        request_overrides={"llm_model": TEST_MODEL, "context_window": context_window},
    )
    return {"request": request.to_dict(), "context": {"resolved_spec": resolved}}


@pytest.mark.asyncio
async def test_seed_restart_recovery_record() -> None:
    payload = await _recovery_payload()
    started = False
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        async with client.stream("POST", "/v1/runs/start", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip()).get("event") or {}
                if event.get("kind") == "run.started":
                    started = True
                    break
    assert started


@pytest.mark.asyncio
async def test_recovered_run_reconnects_without_another_upstream_start() -> None:
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await read_sse(client, "GET", f"/v1/runs/{RECOVERY_RUN_ID}/events")
    assert any(item["event"].get("terminal") for item in events)

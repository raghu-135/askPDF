from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import Response

pytest.importorskip("opentelemetry.sdk.trace")

import app.api.agent_workflows as agent_workflows_api
from app.runtime.capability_resolver import CapabilityResolution
from app.runtime.contracts import AgentDefinition, RuntimeCapabilities, RuntimeOperationId, native
from app.runtime.registry import RuntimeRegistry


class _DiscoveryAdapter:
    def __init__(self, framework: str, delay: float = 0.0) -> None:
        self.framework = framework
        self.builder_id = f"{framework}_builder"
        self.implemented_operations = frozenset({RuntimeOperationId.RUN_START})
        self.delay = delay

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        if self.delay:
            await asyncio.sleep(self.delay)
        return RuntimeCapabilities(
            operations={RuntimeOperationId.RUN_START: native()},
            deployment={"runtime_available": True},
        )


@pytest.mark.asyncio
async def test_runtime_catalog_resolves_deployments_concurrently_and_preserves_order(monkeypatch) -> None:
    slow = _DiscoveryAdapter("slow", delay=0.08)
    fast = _DiscoveryAdapter("fast")
    monkeypatch.setattr(
        agent_workflows_api,
        "get_runtime_registry",
        lambda: RuntimeRegistry(adapters=[slow, fast]),
    )

    started = time.monotonic()
    payload = await agent_workflows_api.list_agent_runtimes(Response())
    elapsed = time.monotonic() - started

    assert elapsed < 0.14
    assert [entry["runtime_id"] for entry in payload["agent_runtimes"]] == [
        "fast:fast_builder",
        "slow:slow_builder",
    ]
    assert all(entry["resource"] == "deployment" for entry in payload["agent_runtimes"])
    assert all(entry["runtime_available"] is True for entry in payload["agent_runtimes"])
    assert all(entry["capabilities"] is not None for entry in payload["agent_runtimes"])


@pytest.mark.asyncio
async def test_runtime_catalog_isolates_discovery_failure_in_deployment_envelope(monkeypatch) -> None:
    healthy = _DiscoveryAdapter("healthy")
    failed = _DiscoveryAdapter("failed")
    original_resolve = agent_workflows_api.resolve_deployment_capability_resolution

    async def resolve(adapter):
        if adapter.framework == "failed":
            raise RuntimeError("unexpected discovery failure")
        return await original_resolve(adapter)

    monkeypatch.setattr(
        agent_workflows_api,
        "get_runtime_registry",
        lambda: RuntimeRegistry(adapters=[healthy, failed]),
    )
    monkeypatch.setattr(agent_workflows_api, "resolve_deployment_capability_resolution", resolve)

    payload = await agent_workflows_api.list_agent_runtimes(Response())
    failed_entry = next(item for item in payload["agent_runtimes"] if item["framework"] == "failed")

    assert failed_entry["runtime_available"] is False
    assert failed_entry["capabilities"] is None
    assert failed_entry["error"]["code"] == "runtime_capability_discovery_failed"


@pytest.mark.asyncio
async def test_definition_and_run_routes_return_their_authoritative_envelopes(monkeypatch) -> None:
    adapter = _DiscoveryAdapter("fake")
    definition = AgentDefinition("definition-1", "fake", "fake_builder")
    capabilities = RuntimeCapabilities(
        operations={RuntimeOperationId.RUN_START: native()},
        deployment={"runtime_available": True},
    )
    resolution = CapabilityResolution(capabilities)
    monkeypatch.setattr(agent_workflows_api, "get_runtime_registry", lambda: RuntimeRegistry(adapters=[adapter]))
    async def resolve_definition(*args, **kwargs):
        return resolution

    async def resolve_run(*args, **kwargs):
        return resolution

    monkeypatch.setattr(agent_workflows_api, "resolve_definition_capability_resolution", resolve_definition)
    monkeypatch.setattr(agent_workflows_api, "resolve_run_capability_resolution", resolve_run)
    monkeypatch.setattr(agent_workflows_api, "definition_from_workflow", lambda workflow: definition)
    monkeypatch.setattr(agent_workflows_api, "definition_from_run", lambda run: definition)
    monkeypatch.setattr(agent_workflows_api, "workflow_is_chat_eligible", lambda spec: True)
    monkeypatch.setattr(agent_workflows_api, "_is_valid_workflow_for_service", lambda workflow: True)
    monkeypatch.setattr(agent_workflows_api, "builtin_workflow_keys", lambda: set())

    class FakeRepository:
        async def seed_builtin_workflows(self):
            return None

        async def get_workflow(self, workflow_id, include_custom):
            return SimpleNamespace(spec_json={})

        async def get_run(self, run_id):
            return SimpleNamespace(id=run_id, thread_id="thread-1", task_id=None, status="running")

    monkeypatch.setattr(agent_workflows_api, "AgentWorkflowRepository", FakeRepository)
    monkeypatch.setattr(agent_workflows_api, "get_thread", lambda thread_id: asyncio.sleep(0, result=True))

    definition_payload = await agent_workflows_api.get_agent_workflow_capabilities("definition-1", Response())
    run_payload = await agent_workflows_api.get_agent_run_capabilities("run-1", Response(), "thread-1")

    assert definition_payload["resource"] == "definition"
    assert definition_payload["definition_id"] == "definition-1"
    assert run_payload["resource"] == "run"
    assert run_payload["run_id"] == "run-1"
    assert run_payload["runtime_available"] is True


@pytest.mark.asyncio
async def test_run_state_route_is_owned_and_capability_gated(monkeypatch) -> None:
    run = SimpleNamespace(id="run-1", thread_id="thread-1")

    class FakeRepository:
        async def get_run(self, run_id):
            return run if run_id == run.id else None

    class FakeService:
        async def inspect_agent_run(self, value):
            assert value is run
            return {"checkpoint": "state-1"}

    monkeypatch.setattr(agent_workflows_api, "AgentWorkflowRepository", FakeRepository)
    monkeypatch.setattr(agent_workflows_api, "AgentRunService", FakeService)
    monkeypatch.setattr(agent_workflows_api, "get_thread", lambda thread_id: asyncio.sleep(0, result=True))

    payload = await agent_workflows_api.get_agent_run_state("run-1", "thread-1")

    assert payload == {"run_id": "run-1", "state": {"checkpoint": "state-1"}}

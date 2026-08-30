"""Executable Phase 0 freeze for runtime protocol v1.

Changing this fixture is a protocol change. Additive implementation work must
continue to produce this shape until a separately versioned contract exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.runtime.contracts import (
    CONTRACT_VERSION,
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
)
from app.runtime.transport import WIRE_VERSION, sse_encode
from hermes_runtime.api import create_app as create_hermes_app
from runtime_service.api import create_app as create_langgraph_app


FIXTURE = Path(__file__).parent / "fixtures" / "runtime_contract_v1.json"


def _frozen_values():
    continuation = ContinuationBinding(
        binding_type="langgraph_checkpoint",
        payload={"checkpoint_thread_id": "checkpoint-1"},
        runtime_version="fixture-runtime-1",
    )
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
        display_name="Router RAG Agent",
        capabilities={"supports_hitl": True},
        definition_version="1",
    )
    request = AgentRuntimeRequest(
        run_id="run-1",
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        input={"question": "What is in the document?"},
        options={"llm_model": "fixture-model"},
        continuation=continuation,
        trace_id="trace-1",
        permissions={"tool_ids": ["document_evidence"]},
    )
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="run.interrupted",
        payload={"interrupt_id": "interrupt-1"},
        occurred_at="2026-01-01T00:00:00Z",
        trace_id="trace-1",
        runtime_version="fixture-runtime-1",
        continuation=continuation,
    )
    result = AgentRuntimeResult(
        status="interrupted",
        interruption={"interrupt_id": "interrupt-1", "kind": "approval"},
        runtime_metadata={"framework": "langgraph"},
        continuation=continuation,
    )
    capabilities = RuntimeCapabilities(
        streaming=True,
        resume=True,
        cancellation=True,
        inspection=True,
        continuation_cleanup=True,
        task_execution=True,
        native_checkpoints=True,
        runtime_version="fixture-runtime-1",
    )
    return definition, request, event, result, capabilities


def _http_surface(app) -> list[str]:
    surface = []
    for route in app.routes:
        if not route.path.startswith(("/healthz", "/startupz", "/readyz", "/v1/")):
            continue
        for method in sorted(route.methods or ()):
            if method not in {"HEAD", "OPTIONS"}:
                surface.append(f"{method} {route.path}")
    return sorted(surface)


def test_runtime_contract_v1_matches_checked_in_wire_fixture():
    frozen = json.loads(FIXTURE.read_text())
    definition, request, event, result, capabilities = _frozen_values()

    assert CONTRACT_VERSION == WIRE_VERSION == frozen["contract_version"] == 1
    assert definition.to_dict() == frozen["definition"]
    assert request.to_dict() == frozen["request"]
    assert event.to_dict() == frozen["event"]
    assert result.to_dict() == frozen["result"]
    assert capabilities.to_dict() == frozen["capabilities"]
    assert sse_encode(event, result=result) == frozen["sse"]


def test_runtime_http_v1_surface_matches_checked_in_fixture(monkeypatch, tmp_path):
    frozen = json.loads(FIXTURE.read_text())["http_surface"]
    monkeypatch.setenv("HERMES_API_URL", "http://hermes-fixture.invalid")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes-runtime.json"))

    assert _http_surface(create_langgraph_app()) == frozen["langgraph"]
    assert _http_surface(create_hermes_app()) == frozen["hermes"]

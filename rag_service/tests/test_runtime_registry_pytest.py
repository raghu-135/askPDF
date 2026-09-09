from types import SimpleNamespace

import pytest

from runtime_protocol.contracts import AgentDefinition
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError


class FakeAdapter:
    framework = "fake"
    builder_id = "fake_builder"


def test_default_registry_uses_external_adapter_without_importing_in_process(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_RUNTIME_URL", "http://langgraph-runtime.test")
    for name, value in {
        "AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS": "30",
        "AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS": "300",
        "AGENT_RUNTIME_READ_TIMEOUT_SECONDS": "600",
        "AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS": "3",
        "AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS": "1",
        "AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS": "30",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS": "0.5",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_BYTES": "8192",
    }.items():
        monkeypatch.setenv(name, value)
    for suffix in (
        "MAX_MODEL_CALLS", "MAX_MODEL_TOKENS", "MAX_TOOL_CALLS", "MAX_ACTIVE_RUNTIME_MS",
        "MAX_DURATION_MS", "MAX_OUTPUT_CHARS", "MAX_EVENT_COUNT",
        "SUBAGENT_TIMEOUT_MS", "DISPATCH_TIMEOUT_MS", "WORKER_TIMEOUT_MS", "WEB_WORKER_TIMEOUT_MS",
    ):
        monkeypatch.setenv(f"DEEP_AGENT_{suffix}", "7200000" if suffix == "MAX_DURATION_MS" else "100")
    registry = RuntimeRegistry()
    registry.initialize()
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    adapter = registry.get(definition)
    assert adapter.__class__.__name__ == "HttpLangGraphRuntimeAdapter"
    assert adapter.framework == "langgraph"


def test_registry_requires_concrete_framework_and_builder_identity():
    registry = RuntimeRegistry(adapters=[FakeAdapter()])

    assert registry.get(
        AgentDefinition(
            definition_id="definition-1",
            framework="fake",
            builder_id="fake_builder",
        )
    ).builder_id == "fake_builder"

    with pytest.raises(RuntimeSelectionError):
        registry.get(
            AgentDefinition(
                definition_id="definition-1",
                framework="fake",
                builder_id="unknown",
            )
        )


def test_registry_exposes_deterministic_deployment_identity():
    first = FakeAdapter()
    second = SimpleNamespace(framework="langgraph", builder_id="graph")
    registry = RuntimeRegistry(adapters=[first, second])

    assert [registry.deployment_id(adapter) for adapter in registry.adapters()] == [
        "fake:fake_builder",
        "langgraph:graph",
    ]
    assert registry.get_deployment("fake:fake_builder") is first
    assert registry.get_deployment("https://runtime.example") is None


def test_neutral_runtime_modules_have_no_framework_imports():
    from pathlib import Path

    root = Path(__file__).parents[1] / "app" / "runtime"
    forbidden = ("langgraph", "langchain", "RunnableConfig", "StateGraph")
    paths = [root / "catalog.py", root / "adapter.py", Path(__file__).parents[1] / "runtime_protocol/contracts.py", Path(__file__).parents[1] / "runtime_protocol/errors.py"]
    for path in paths:
        source = path.read_text()
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any(token in line for line in import_lines for token in forbidden), path.name
@pytest.mark.asyncio
async def test_projection_is_idempotent_for_existing_chat_turn(monkeypatch):
    from app.services.agent_runtime_projection import AgentRuntimeProjection

    class Turn:
        id = "turn-1"
        agent_run_turn_kind = "assistant_final"
        agent_run_sequence = 0
        completed_at = None
        created_at = None

    class Repository:
        async def get_run(self, _run_id):
            return None

        async def list_chat_turns_for_run(self, _run_id):
            return [Turn()]

    monkeypatch.setattr("app.agent_workflows.repository.AgentWorkflowRepository", Repository)
    result = await AgentRuntimeProjection().project_chat_result(
        thread_id="thread-1",
        question="hello",
        result={"status": "completed", "answer": "ok"},
        run_context={"agent_run_id": "run-1"},
        duration_ms=1.0,
    )

    assert result["chat_turn_id"] == "turn-1"

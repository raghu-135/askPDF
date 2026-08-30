from types import SimpleNamespace

import pytest

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError
from app.runtime.mode import AgentRuntimeMode, agent_runtime_mode


class FakeAdapter:
    framework = "fake"
    builder_id = "fake_builder"


def test_runtime_mode_requires_explicit_configuration(monkeypatch):
    monkeypatch.delenv("AGENT_RUNTIME_MODE", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_RUNTIME_MODE"):
        agent_runtime_mode()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("external", AgentRuntimeMode.EXTERNAL), ("in_process", AgentRuntimeMode.IN_PROCESS)],
)
def test_runtime_mode_accepts_explicit_values(monkeypatch, value, expected):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", value)
    assert agent_runtime_mode() is expected


def test_runtime_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "automatic")
    with pytest.raises(RuntimeError, match="external.*in_process"):
        agent_runtime_mode()


def test_default_registry_uses_external_adapter_without_importing_in_process(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "external")
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
        "MAX_DURATION_MS", "MAX_OUTPUT_CHARS", "MAX_EVENT_COUNT", "WAKE_LIMIT_SECONDS",
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


def test_explicit_in_process_initialization_imports_adapter_immediately(monkeypatch):
    import app.runtime.registry as module

    monkeypatch.setenv("AGENT_RUNTIME_MODE", "in_process")

    def missing_adapter():
        raise ModuleNotFoundError("No module named 'langgraph'")

    monkeypatch.setattr(module, "_default_langgraph_adapter", missing_adapter)
    with pytest.raises(RuntimeError, match="AGENT_RUNTIME_MODE=in_process.*external"):
        RuntimeRegistry().initialize()


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
    for name in ("contracts.py", "errors.py", "catalog.py", "adapter.py"):
        source = (root / name).read_text()
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any(token in line for line in import_lines for token in forbidden), name
@pytest.mark.asyncio
async def test_langgraph_adapter_start_projects_typed_result(monkeypatch):
    import app.runtime.langgraph_adapter as module
    import app.runtime.langgraph.router_runtime as router_module

    class Checkpointer:
        pass

    class CheckpointerContext:
        async def __aenter__(self):
            return Checkpointer()

        async def __aexit__(self, *_args):
            return False

    captured = {}

    monkeypatch.setattr(module.checkpointing, "open_agent_checkpointer", lambda: CheckpointerContext())

    async def fake_execute(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "status": "completed",
            "answer": "adapter result",
            "agent_run_id": "run-1",
        }

    monkeypatch.setattr(router_module, "execute_compiled_rag_chat", fake_execute)

    adapter = LangGraphRuntimeAdapter()
    request = AgentRuntimeRequest(
        run_id="run-1",
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    result = await adapter.start(
        request,
        context=RuntimeExecutionContext(
            request=SimpleNamespace(question="hello"),
            embedding_model="embed-model",
            resolved_spec={"workflow_id": "router_rag_agent"},
            agent_run_context={"checkpoint_thread_id": "checkpoint-1"},
        ),
    )

    assert result.status == "completed"
    assert result.output == "adapter result"
    assert captured["args"][0] == "thread-1"
    assert captured["args"][2] == "embed-model"
    assert captured["kwargs"]["checkpointer"].__class__ is Checkpointer
    assert captured["kwargs"]["agent_run_context"]["agent_run_id"] == "run-1"
    assert captured["kwargs"]["persist_product_records"] is False


@pytest.mark.asyncio
async def test_langgraph_adapter_validates_and_projects_neutral_events():
    adapter = LangGraphRuntimeAdapter()
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    validation = await adapter.validate(definition, {"schema_version": 1})
    events = await adapter.project_trace(
        [{"event": "run.completed", "data": {"answer": "ok"}}],
        run_id="run-1",
    )

    assert validation.valid is False
    assert validation.issues
    assert events[0].run_id == "run-1"
    assert events[0].terminal is True


@pytest.mark.asyncio
async def test_projection_is_idempotent_for_existing_chat_turn(monkeypatch):
    from app.services.agent_runtime_projection import AgentRuntimeProjection

    monkeypatch.setenv("AGENT_RUNTIME_MODE", "in_process")

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

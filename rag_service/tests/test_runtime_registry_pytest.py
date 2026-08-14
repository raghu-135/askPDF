from types import SimpleNamespace

import pytest

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError


class FakeAdapter:
    framework = "fake"
    builder_id = "fake_builder"


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


def test_neutral_runtime_modules_have_no_framework_imports():
    from pathlib import Path

    root = Path(__file__).parents[1] / "app" / "runtime"
    forbidden = ("langgraph", "langchain", "RunnableConfig", "StateGraph")
    for name in ("contracts.py", "errors.py", "catalog.py", "adapter.py"):
        source = (root / name).read_text()
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any(token in line for line in import_lines for token in forbidden), name
@pytest.mark.asyncio
async def test_langgraph_adapter_start_projects_legacy_result(monkeypatch):
    import app.runtime.langgraph_adapter as module

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

    monkeypatch.setattr(module.router_runtime, "execute_compiled_rag_chat", fake_execute)

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

    class Turn:
        id = "turn-1"
        agent_run_turn_kind = "assistant_final"
        agent_run_sequence = 0
        completed_at = None
        created_at = None

    class Repository:
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

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

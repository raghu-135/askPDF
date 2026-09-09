from types import SimpleNamespace

import pytest

from app.runtime.builder import BuilderCapabilities, BuilderCatalog
from app.runtime.builder_registry import BuilderRegistry, BuilderSelectionError
from runtime_protocol.contracts import AgentDefinition, RuntimeCapabilities, RuntimeValidationIssue, RuntimeValidationResult
from app.runtime.langgraph_builder import LangGraphBuilderProvider


class FakeBuilder:
    framework = "fake"
    builder_id = "fake_builder"


class FakeLangGraphRuntime:
    def __init__(self):
        self.resolve_calls = []

    async def capabilities(self, definition):
        return RuntimeCapabilities()

    async def validate(self, definition, spec, *, options=None):
        return RuntimeValidationResult(
            valid=False,
            issues=(RuntimeValidationIssue("invalid", "invalid schema", "schema_version"),),
        )

    async def resolve_definition(
        self,
        definition,
        spec,
        *,
        thread_settings,
        request_overrides,
        options=None,
    ):
        self.resolve_calls.append({
            "thread_settings": dict(thread_settings),
            "request_overrides": dict(request_overrides),
        })
        return {**dict(spec), "config": {**dict(spec.get("config") or {}), **dict(thread_settings), **dict(request_overrides)}}


def test_builder_registry_is_keyed_by_framework_and_builder_not_category():
    registry = BuilderRegistry(providers=[FakeBuilder()])
    definition = AgentDefinition(
        definition_id="definition-1",
        framework="fake",
        builder_id="fake_builder",
        category="deep",
    )

    assert registry.get(definition).builder_id == "fake_builder"

    with pytest.raises(BuilderSelectionError):
        registry.get(
            AgentDefinition(
                definition_id="definition-1",
                framework="fake",
                builder_id="unknown",
                category="deep",
            )
        )


def test_builder_contract_types_are_json_compatible():
    capabilities = BuilderCapabilities(
        framework="langgraph",
        builder_id="langgraph_graph",
        runtime_capabilities={"supports_replans": True},
    )
    catalog = BuilderCatalog(
        framework="langgraph",
        builder_id="langgraph_graph",
        capabilities=capabilities,
        payload={"node_catalog": {}},
    )

    assert catalog.to_dict()["capabilities"]["runtime_capabilities"]["supports_replans"] is True
    assert catalog.to_dict()["payload"]["node_catalog"] == {}


def test_neutral_builder_modules_have_no_framework_imports():
    from pathlib import Path

    root = Path(__file__).parents[1] / "app" / "runtime"
    forbidden = ("langgraph", "langchain", "RunnableConfig", "StateGraph")
    for name in ("builder.py", "builder_registry.py"):
        source = (root / name).read_text()
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any(token in line for line in import_lines for token in forbidden), name


def test_langgraph_provider_owns_task_web_tool_mapping():
    provider = LangGraphBuilderProvider(adapter=FakeLangGraphRuntime())
    with_web = AgentDefinition(
        definition_id="deep_research_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_metadata={"allowed_tool_ids": ["live_web_recon"]},
    )
    without_web = AgentDefinition(
        definition_id="deep_research_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_metadata={"allowed_tool_ids": ["document_evidence"]},
    )

    assert provider.supports_task_web_search(with_web) is True
    assert provider.supports_task_web_search(without_web) is False


@pytest.mark.asyncio
async def test_langgraph_provider_preserves_concrete_identity():
    provider = LangGraphBuilderProvider(adapter=FakeLangGraphRuntime())
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
    )

    capabilities = await provider.capabilities(definition)

    assert capabilities.framework == "langgraph"
    assert capabilities.builder_id == "langgraph_graph"
    assert capabilities.authoring is True
    assert capabilities.transient_tests is True


@pytest.mark.asyncio
async def test_langgraph_provider_does_not_send_product_thread_settings_to_runtime():
    runtime = FakeLangGraphRuntime()
    provider = LangGraphBuilderProvider(adapter=runtime)
    definition = AgentDefinition("deep_research_agent", "langgraph", "langgraph_graph")

    await provider.resolve(
        definition,
        {"schema_version": 1, "config": {}},
        thread_settings={
            "agent_workflow": {"workflow_id": "deep_research_agent"},
            "memory": {"memory_enabled": True},
            "replans_limit": 20,
            "replans": 3,
            "system_role": "product role",
        },
        request_overrides={"context_window": 8192},
    )

    assert runtime.resolve_calls == [{
        "thread_settings": {"replans": 3, "system_role": "product role"},
        "request_overrides": {"context_window": 8192},
    }]


@pytest.mark.asyncio
async def test_langgraph_provider_rejects_invalid_spec_without_compiling():
    provider = LangGraphBuilderProvider(adapter=FakeLangGraphRuntime())
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )

    result = await provider.validate(definition, {"schema_version": 1})

    assert result.valid is False
    assert result.issues

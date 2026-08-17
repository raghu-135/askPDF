from types import SimpleNamespace

import pytest

from app.runtime.builder import BuilderCapabilities, BuilderCatalog
from app.runtime.builder_registry import BuilderRegistry, BuilderSelectionError
from app.runtime.contracts import AgentDefinition
from app.runtime.langgraph_builder import LangGraphBuilderProvider


class FakeBuilder:
    framework = "fake"
    builder_id = "fake_builder"


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


@pytest.mark.asyncio
async def test_langgraph_provider_preserves_concrete_identity():
    provider = LangGraphBuilderProvider()
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
async def test_langgraph_provider_rejects_invalid_spec_without_compiling():
    provider = LangGraphBuilderProvider()
    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )

    result = await provider.validate(definition, {"schema_version": 1})

    assert result.valid is False
    assert result.issues
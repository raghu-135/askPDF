import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.builder_registry import get_builder_registry
from app.runtime.contracts import AgentDefinition
from app.runtime.hermes_builder import HermesBuilderProvider


def _definition() -> AgentDefinition:
    return AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent", category="router")


def test_hermes_builtin_is_concrete_and_not_a_graph():
    workflow = next(item for item in load_builtin_workflows() if item["builtin_key"] == "hermes_rag_agent")
    assert workflow["framework"] == "hermes"
    assert workflow["builder_id"] == "hermes_agent"
    assert "graph" not in workflow["spec_json"]["config"]


def test_hermes_provider_is_registered_without_changing_langgraph_provider():
    assert get_builder_registry().get(_definition()).framework == "hermes"
    assert get_builder_registry().get(AgentDefinition("router_rag_agent", "langgraph", "langgraph_graph")).framework == "langgraph"


@pytest.mark.asyncio
async def test_hermes_provider_rejects_graph_fields():
    provider = HermesBuilderProvider()
    spec = {
        "schema_version": 2,
        "runtime": {"kind": "hermes_agent"},
        "config": {"system_prompt": "x", "mcp_server": "askpdf", "allowed_tool_ids": ["x"], "graph": {}},
    }
    result = await provider.validate(_definition(), spec)
    assert result.valid is False
    assert any(issue.code == "graph_fields_not_supported" for issue in result.issues)


@pytest.mark.asyncio
async def test_hermes_provider_catalog_is_framework_specific():
    catalog = await HermesBuilderProvider().catalog(_definition())
    assert catalog.framework == "hermes"
    assert catalog.builder_id == "hermes_agent"
    assert catalog.payload["definition_ids"] == ["hermes_rag_agent"]

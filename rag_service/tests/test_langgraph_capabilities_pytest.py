from runtime_protocol.contracts import AgentDefinition

from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from app.runtime.registry import RuntimeRegistry


def _definition() -> AgentDefinition:
    return AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )


def test_control_plane_langgraph_registry_is_http_only():
    registry = RuntimeRegistry()
    registry.initialize()
    adapter = registry.get(_definition())
    assert isinstance(adapter, HttpLangGraphRuntimeAdapter)

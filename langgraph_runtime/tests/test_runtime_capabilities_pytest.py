from fastapi.testclient import TestClient

import pytest

from runtime_protocol.contracts import AgentDefinition, RuntimeOperationId, RuntimeSupportLevel
from langgraph_runtime.api import create_app
from langgraph_runtime.capabilities import LangGraphDeploymentProfile, langgraph_capabilities


def _definition(**kwargs):
    return AgentDefinition(
        definition_id=kwargs.pop("definition_id", "router_rag_agent"),
        framework="langgraph",
        builder_id="langgraph_graph",
        category=kwargs.pop("category", None),
        capabilities=kwargs.pop("capabilities", {}),
        definition_metadata=kwargs.pop("definition_metadata", {}),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("backend", "url", "saver", "checkpoint_available", "durable"),
    [("memory", "", False, False, False), ("postgres", "postgresql://db/runtime", True, True, True), ("postgres", "", True, False, False), ("postgres", "postgresql://db/runtime", False, False, False)],
)
def test_deployment_profile_fails_closed_without_checkpoint_support(monkeypatch, backend, url, saver, checkpoint_available, durable):
    monkeypatch.setattr("langgraph_runtime.capabilities._module_available", lambda name: saver if name == "langgraph.checkpoint.postgres.aio" else True)
    values = {"ASKPDF_AGENT_CHECKPOINTER": backend}
    if url:
        values["AGENT_CHECKPOINT_DATABASE_URL"] = url
    profile = LangGraphDeploymentProfile.from_environment(values)
    assert profile.checkpoint_available is checkpoint_available
    assert profile.durable_persistence is durable
    assert profile.deployment_metadata()["checkpointer_backend"] == backend
    if backend == "postgres" and not checkpoint_available:
        assert profile.runtime_available is False
        assert profile.configuration_error


def test_deep_research_capabilities_keep_runtime_subagent_control_explicitly_unsupported():
    definition = _definition(
        definition_id="deep_research_agent",
        capabilities={"supports_replans": True, "supports_parallel_dispatch": True, "supports_long_running_tasks": True, "supports_artifacts": True},
        definition_metadata={"graph_node_types": ["deep_task_planner", "deep_research_subagent"], "allowed_tool_ids": ["durable_memory", "document_evidence"]},
    )
    capabilities = langgraph_capabilities(definition, profile=LangGraphDeploymentProfile("in_process", "memory", True, False, True))
    assert {"planning", "parallel_dispatch", "artifacts", "subagent_orchestration", "memory", "tools"} <= set(capabilities.features)
    for operation in (RuntimeOperationId.SUBAGENT_LIST, RuntimeOperationId.SUBAGENT_SEND, RuntimeOperationId.SUBAGENT_CANCEL):
        assert capabilities.operations[operation.value].support is RuntimeSupportLevel.UNSUPPORTED
        assert capabilities.operations[operation.value].enabled is False


def test_runtime_capabilities_endpoint_returns_neutral_capabilities(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "loopback_http")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "http://127.0.0.1:8000/internal/mcp/")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.post("/v1/capabilities", json={"definition": _definition().to_dict()})
    assert response.status_code == 200
    assert "operations" in response.json()["result"]["capabilities"]

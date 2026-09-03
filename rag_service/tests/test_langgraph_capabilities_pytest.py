import pytest
from fastapi.testclient import TestClient

from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    ContinuationBinding,
    RuntimeOperationId,
    RuntimeFeatureId,
    RuntimeSupportLevel,
)
from langgraph_runtime.capabilities import (
    LangGraphDeploymentProfile,
    langgraph_capabilities,
)
from langgraph_runtime.adapter import LangGraphRuntimeAdapter
from app.runtime.capability_resolver import capabilities_for_definition, resolve_capabilities
from app.runtime.registry import RuntimeRegistry
from langgraph_runtime.api import create_app


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
    [
        ("memory", "", False, True, False),
        ("postgres", "postgresql://db/runtime", True, True, True),
        ("postgres", "", True, False, False),
        ("postgres", "postgresql://db/runtime", False, False, False),
    ],
)
def test_deployment_profile_derives_checkpoint_support_without_fallback(
    monkeypatch,
    backend,
    url,
    saver,
    checkpoint_available,
    durable,
):
    monkeypatch.setattr(
        "langgraph_runtime.capabilities._module_available",
        lambda name: saver if name == "langgraph.checkpoint.postgres.aio" else True,
    )
    values = {
        "ASKPDF_AGENT_CHECKPOINTER": backend,
    }
    if url:
        values["AGENT_CHECKPOINT_DATABASE_URL"] = url
    profile = LangGraphDeploymentProfile.from_environment(values)

    assert profile.checkpoint_available is checkpoint_available
    assert profile.durable_persistence is durable
    assert profile.deployment_metadata()["checkpointer_backend"] == backend
    if backend == "postgres" and not checkpoint_available:
        assert profile.runtime_available is False
        assert profile.configuration_error


@pytest.mark.asyncio
async def test_langgraph_adapter_uses_profile_for_memory_and_postgres(monkeypatch):
    definition = _definition()
    adapter = LangGraphRuntimeAdapter()

    memory = LangGraphDeploymentProfile("in_process", "memory", True, False, True)
    postgres = LangGraphDeploymentProfile("external", "postgres", True, True, True)
    unavailable = LangGraphDeploymentProfile("in_process", "postgres", False, False, False, "missing")

    monkeypatch.setattr(
        "langgraph_runtime.capabilities.LangGraphDeploymentProfile.from_environment",
        classmethod(lambda cls: memory),
    )
    memory_caps = await adapter.capabilities(definition)
    assert memory_caps.deployment["durable_persistence"] is False
    assert memory_caps.operations[RuntimeOperationId.RUN_RESUME.value].enabled is True

    monkeypatch.setattr(
        "langgraph_runtime.capabilities.LangGraphDeploymentProfile.from_environment",
        classmethod(lambda cls: postgres),
    )
    postgres_caps = await adapter.capabilities(definition)
    assert postgres_caps.deployment["durable_persistence"] is True

    monkeypatch.setattr(
        "langgraph_runtime.capabilities.LangGraphDeploymentProfile.from_environment",
        classmethod(lambda cls: unavailable),
    )
    unavailable_caps = await adapter.capabilities(definition)
    assert unavailable_caps.operations[RuntimeOperationId.RUN_START.value].enabled is False
    assert unavailable_caps.operations[RuntimeOperationId.RUN_RESUME.value].enabled is False
    assert unavailable_caps.operations[RuntimeOperationId.RUN_RESUME.value].disabled_reason == "checkpoint_store_unavailable"


def test_deep_research_profile_derives_features_but_keeps_runtime_subagent_control_unsupported():
    definition = _definition(
        definition_id="deep_research_agent",
        category="deep",
        capabilities={
            "supports_replans": True,
            "supports_parallel_dispatch": True,
            "supports_long_running_tasks": True,
            "supports_artifacts": True,
        },
        definition_metadata={
            "graph_node_types": ["deep_task_planner", "deep_research_subagent"],
            "allowed_tool_ids": ["durable_memory", "document_evidence"],
            "task_profiles": ["document_researcher"],
        },
    )
    profile = LangGraphDeploymentProfile("in_process", "memory", True, False, True)
    capabilities = langgraph_capabilities(definition, profile=profile)

    assert {"planning", "parallel_dispatch", "artifacts", "subagent_orchestration", "memory", "tools"} <= set(capabilities.features)
    assert all(capabilities.features[key].enabled for key in capabilities.features)
    for operation in (RuntimeOperationId.SUBAGENT_LIST, RuntimeOperationId.SUBAGENT_SEND, RuntimeOperationId.SUBAGENT_CANCEL):
        descriptor = capabilities.operations[operation.value]
        assert descriptor.support is RuntimeSupportLevel.UNSUPPORTED
        assert descriptor.enabled is False


@pytest.mark.asyncio
async def test_definition_resolver_uses_deep_definition_features_and_task_operations(monkeypatch):
    definition = _definition(
        definition_id="deep_research_agent",
        category="deep",
        capabilities={
            "supports_replans": True,
            "supports_parallel_dispatch": True,
            "supports_long_running_tasks": True,
            "supports_artifacts": True,
        },
        definition_metadata={
            "graph_node_types": ["deep_task_planner", "deep_research_subagent"],
            "allowed_tool_ids": ["durable_memory", "document_evidence"],
            "task_profiles": ["document_researcher"],
        },
    )
    monkeypatch.setattr(
        "langgraph_runtime.capabilities.LangGraphDeploymentProfile.from_environment",
        classmethod(lambda cls: LangGraphDeploymentProfile("in_process", "memory", True, False, True)),
    )

    capabilities = await capabilities_for_definition(
        definition,
        registry=RuntimeRegistry(adapters=[LangGraphRuntimeAdapter()]),
    )

    assert capabilities.features[RuntimeFeatureId.PLANNING].enabled is True
    assert capabilities.features[RuntimeFeatureId.SUBAGENT_ORCHESTRATION].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].owner.value == "product"
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].owner.value == "product"
    assert capabilities.operations[RuntimeOperationId.RUN_START.value].owner.value == "runtime"

    deployment = langgraph_capabilities(definition)
    assert RuntimeOperationId.TASK_PAUSE.value not in deployment.operations
    assert RuntimeOperationId.TASK_RETRY.value not in deployment.operations


def test_external_runtime_capabilities_use_the_same_profile(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER_SETUP", "false")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.post("/v1/capabilities", json={"definition": _definition().to_dict()})

    assert response.status_code == 200
    payload = response.json()["result"]["capabilities"]
    expected = langgraph_capabilities(
        _definition(),
        profile=LangGraphDeploymentProfile.from_environment(),
    ).to_dict()
    assert payload == expected


@pytest.mark.asyncio
async def test_state_update_is_absent_at_run_level():
    definition = _definition()
    registry = RuntimeRegistry(adapters=[LangGraphRuntimeAdapter()])

    unbound = await resolve_capabilities(
        definition,
        registry=registry,
        run=type("Run", (), {
            "status": "running",
            "pending_interrupt_json": None,
            "runtime_binding_json": {},
            "runtime_binding_status": "active",
        })(),
    )

    bound = await resolve_capabilities(
        definition,
        registry=registry,
        run=type("Run", (), {
            "status": "running",
            "pending_interrupt_json": None,
            "runtime_binding_json": {"binding_type": "langgraph_checkpoint"},
            "runtime_binding_status": "active",
            "run_metadata_json": {"runtime_started": True, "checkpoint_boundary_available": True},
        })(),
    )
    assert "run.update_state" not in unbound.operations
    assert "run.update_state" not in bound.operations


@pytest.mark.asyncio
async def test_langgraph_inspect_state_reads_checkpoint_snapshot(monkeypatch):
    import langgraph_runtime.adapter as module

    class Snapshot:
        values = {"messages": ["hello"]}
        next = ("answer",)
        metadata = {"source": "loop"}

    class Graph:
        async def aget_state(self, config):
            assert config == {"configurable": {"thread_id": "checkpoint-1"}}
            return Snapshot()

    class CheckpointerContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    class Compiler:
        def compile(self, spec, *, checkpointer):
            assert spec == {"workflow_id": "router_rag_agent"}
            assert checkpointer is not None
            return Graph()

    monkeypatch.setattr(module, "checkpointing", type("Checkpointing", (), {
        "open_agent_checkpointer": lambda: CheckpointerContext(),
    }))
    monkeypatch.setattr("langgraph_runtime.compiler.WorkflowCompiler", Compiler)

    request = AgentRuntimeRequest(
        run_id="run-1", thread_id="thread-1", definition_id="router_rag_agent",
        framework="langgraph", builder_id="langgraph_graph",
        options={"resolved_spec": {"workflow_id": "router_rag_agent"}},
        continuation=ContinuationBinding("langgraph.checkpoint", {
            "binding_id": __import__("langgraph_runtime.bindings", fromlist=["issue_binding"]).issue_binding(
                checkpoint_thread_id="checkpoint-1", run_id="run-1"
            )
        }),
    )
    state = await LangGraphRuntimeAdapter().inspect_state(request)

    assert state["state"] == {"messages": ["hello"]}
    assert state["next"] == ["answer"]
    assert state["metadata"] == {"source": "loop"}

import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.runtime.builder_registry import get_builder_registry
from app.runtime.builder import UnsupportedRequestOverrideError
from app.runtime.contracts import AgentDefinition
from app.runtime.hermes_builder import HermesBuilderProvider
from app.runtime.langgraph_builder import LangGraphBuilderProvider


def _definition() -> AgentDefinition:
    return AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent", category="deep")


def _spec() -> dict:
    workflow = next(
        item for item in load_builtin_workflows()
        if item["builtin_key"] == "hermes_rag_agent"
    )
    return dict(workflow["spec_json"])


def test_hermes_builtin_is_concrete_and_not_a_graph():
    workflow = next(item for item in load_builtin_workflows() if item["builtin_key"] == "hermes_rag_agent")
    assert workflow["framework"] == "hermes"
    assert workflow["builder_id"] == "hermes_agent"
    assert "graph" not in workflow["spec_json"]["config"]


def test_hermes_prompt_uses_pinned_progressive_tool_disclosure_protocol():
    prompt = _spec()["config"]["system_prompt"]
    assert "tool_search" in prompt
    assert "tool_describe" in prompt
    assert "tool_call" in prompt
    assert "exact namespaced name" in prompt
    assert "do not invent a namespaced name" in prompt
    assert "call search_document_by_id" not in prompt


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


def test_langgraph_provider_retains_owned_request_overrides():
    definition = AgentDefinition("router_rag_agent", "langgraph", "langgraph_graph")
    filtered = LangGraphBuilderProvider().filter_request_overrides(
        definition,
        {"use_web_search": True, "replans": 2, "unknown": "drop", "use_reranker": None},
        reject_unsupported=False,
    )
    assert filtered == {"use_web_search": True, "replans": 2}


@pytest.mark.asyncio
async def test_hermes_resolution_drops_langgraph_request_overrides():
    provider = HermesBuilderProvider()
    resolved = await provider.resolve(
        _definition(),
        _spec(),
        request_overrides={
            "use_web_search": True,
            "replans": 3,
            "system_role": "LangGraph-only role",
            "arbitrary": "must-not-persist",
        },
    )
    assert set(resolved["config"]) <= provider._allowed_config_keys
    assert resolved["config"]["use_web_search"] is True
    assert "replans" not in resolved["config"]
    assert "system_role" not in resolved["config"]
    assert "arbitrary" not in resolved["config"]


@pytest.mark.asyncio
async def test_hermes_resolution_inherits_thread_model_through_deployment_provider(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    resolved = await HermesBuilderProvider().resolve(
        _definition(),
        _spec(),
        thread_settings={"llm_model": "askpdf-selected-model"},
    )

    assert resolved["config"]["model"] == "askpdf-selected-model"
    assert resolved["config"]["provider"] == "lmstudio"
    assert resolved["managed_profile"]["model_policy"] == {
        "model": "askpdf-selected-model",
        "provider": "lmstudio",
    }
    assert "# askPDF Deep Research Policy (v1)" in resolved["config"]["system_prompt"]
    assert "Hermes MCP execution protocol" in resolved["config"]["system_prompt"]
    assert resolved["config"]["research_policy_id"] == "deep_research_v1"


@pytest.mark.asyncio
async def test_hermes_resolution_prefers_request_selected_model(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    resolved = await HermesBuilderProvider().resolve(
        _definition(),
        _spec(),
        thread_settings={"llm_model": "older-thread-model"},
        request_overrides={"llm_model": "askpdf-request-model"},
    )

    assert resolved["config"]["model"] == "askpdf-request-model"
    assert resolved["config"]["provider"] == "lmstudio"


def test_hermes_explicit_unsupported_override_is_rejected():
    provider = HermesBuilderProvider()
    with pytest.raises(UnsupportedRequestOverrideError) as exc_info:
        provider.filter_request_overrides(
            _definition(),
            {"use_web_search": True, "malicious": {"graph": "payload"}},
            reject_unsupported=True,
        )
    assert exc_info.value.keys == ("malicious",)


@pytest.mark.asyncio
async def test_hermes_supported_override_is_merged_and_final_spec_revalidated():
    provider = HermesBuilderProvider()
    provider._supported_request_override_keys = frozenset({"model"})
    resolved = await provider.resolve(
        _definition(),
        _spec(),
        request_overrides={"model": "hermes-proof-model"},
    )
    assert resolved["config"]["model"] == "hermes-proof-model"

    provider._supported_request_override_keys = frozenset({"graph"})
    with pytest.raises(ValueError, match="does not support"):
        await provider.resolve(
            _definition(),
            _spec(),
            request_overrides={"graph": {}},
        )

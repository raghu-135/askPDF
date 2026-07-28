from app.agent.tool_registry import (
    TOOL_CONTRACT_METADATA,
    TOOL_FRIENDLY_CONFIG,
    get_tool_contract_metadata,
    known_tool_contract_ids,
    list_tool_contract_metadata,
    validate_tool_call_allowed,
)
from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.agent_workflows.validator import WorkflowValidator


def _builtin_spec(builtin_key: str):
    for workflow in load_builtin_workflows():
        if workflow.get("builtin_key") == builtin_key:
            return workflow["spec_json"]
    raise AssertionError(f"Missing builtin workflow fixture: {builtin_key}")


def test_tool_contract_metadata_covers_user_facing_tool_ids():
    friendly_ids = {
        config["id"]
        for config in TOOL_FRIENDLY_CONFIG.values()
        if isinstance(config, dict) and config.get("id")
    }

    assert friendly_ids <= known_tool_contract_ids()


def test_router_rag_allowed_tool_ids_are_contract_ids():
    allowed_tool_ids = set(_builtin_spec("router_rag_agent")["config"]["allowed_tool_ids"])

    assert allowed_tool_ids <= known_tool_contract_ids()


def test_tool_contract_metadata_exposes_graph_integration_fields():
    document_contract = get_tool_contract_metadata("search_documents")
    memory_contract = get_tool_contract_metadata("search_long_term_memory")
    web_contract = get_tool_contract_metadata("search_web")
    records = list_tool_contract_metadata()

    assert document_contract["id"] == "document_evidence"
    assert document_contract["category"] == "retrieval"
    assert document_contract["allowed_caller_nodes"] == ["retrieval_worker"]
    assert document_contract["artifact_keys"] == ["document_sources", "web_sources"]
    assert "missing_thread_context" in document_contract["warning_codes"]

    assert memory_contract["id"] == "memory_recall"
    assert memory_contract["allowed_caller_nodes"] == ["long_term_memory_worker"]
    assert memory_contract["artifact_keys"] == [
        "memory_refs",
        "memory_scopes",
        "memory_scope_policy",
    ]
    assert "no_relevant_memory" in memory_contract["warning_codes"]

    assert web_contract["id"] == "live_web_recon"
    assert web_contract["allowed_caller_nodes"] == ["web_worker"]
    assert "web_search_disabled" in web_contract["warning_codes"]

    assert any(record["tool_name"] == "search_documents" and record["display_name"] == "Document Evidence" for record in records)
    assert records == sorted(records, key=lambda record: record["tool_name"])


def test_workflow_validator_accepts_external_contract_ids():
    spec = _builtin_spec("router_rag_agent")
    spec["config"]["allowed_tool_ids"] = [
        *spec["config"]["allowed_tool_ids"],
        "wikipedia_reference",
        "semantic_scholar_research",
    ]

    assert WorkflowValidator().validate(spec) == {"valid": True, "errors": []}


def test_tool_contract_records_are_schema_like():
    for tool_name, contract in TOOL_CONTRACT_METADATA.items():
        assert contract["id"]
        assert contract["category"]
        assert isinstance(contract["allowed_caller_nodes"], list)
        assert isinstance(contract["artifact_keys"], list)
        assert isinstance(contract["warning_codes"], list)
        metadata = get_tool_contract_metadata(tool_name)
        assert metadata["tool_name"] == tool_name
        assert metadata["id"] == contract["id"]
        assert metadata["category"] == contract["category"]
        assert metadata["allowed_caller_nodes"] == contract["allowed_caller_nodes"]
        assert metadata["artifact_keys"] == contract["artifact_keys"]
        assert metadata["warning_codes"] == contract["warning_codes"]


def test_tool_call_validation_enforces_allowed_caller_nodes():
    validate_tool_call_allowed("search_documents", "retrieval_worker")
    validate_tool_call_allowed("search_conversation_history", "memory_worker")
    validate_tool_call_allowed("search_long_term_memory", "long_term_memory_worker")
    validate_tool_call_allowed("search_thread_timeline", "timeline_worker")
    validate_tool_call_allowed("search_web", "web_worker")

    try:
        validate_tool_call_allowed("search_documents", "memory_worker")
    except ValueError as exc:
        assert "search_documents is not allowed from caller node memory_worker" in str(exc)
        assert "retrieval_worker" in str(exc)
    else:
        raise AssertionError("Expected disallowed caller node to raise")

    try:
        validate_tool_call_allowed("unknown_tool", "retrieval_worker")
    except ValueError as exc:
        assert "Unknown tool contract: unknown_tool" in str(exc)
    else:
        raise AssertionError("Expected unknown tool contract to raise")


def test_tool_contracts_endpoint(api_client):
    response = api_client.get("/api/tools/contracts")

    assert response.status_code == 200
    tools = response.json()["tools"]
    by_name = {tool["tool_name"]: tool for tool in tools}

    assert by_name["search_documents"]["id"] == "document_evidence"
    assert by_name["search_documents"]["display_name"] == "Document Evidence"
    assert by_name["search_documents"]["allowed_caller_nodes"] == ["retrieval_worker"]
    assert by_name["search_documents"]["artifact_keys"] == ["document_sources", "web_sources"]
    assert "missing_thread_context" in by_name["search_documents"]["warning_codes"]
    assert by_name["search_long_term_memory"]["id"] == "memory_recall"
    assert by_name["search_long_term_memory"]["allowed_caller_nodes"] == ["long_term_memory_worker"]
    assert by_name["search_web"]["category"] == "web"
    assert "web_search_disabled" in by_name["search_web"]["warning_codes"]

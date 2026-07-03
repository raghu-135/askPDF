from app.agent.tool_registry import (
    TOOL_CONTRACT_METADATA,
    TOOL_FRIENDLY_CONFIG,
    get_tool_contract_metadata,
    known_tool_contract_ids,
    list_tool_contract_metadata,
)
from app.agent_patterns.templates import builtin_router_rag_spec
from app.agent_patterns.validator import TemplateValidator


def test_tool_contract_metadata_covers_user_facing_tool_ids():
    friendly_ids = {
        config["id"]
        for config in TOOL_FRIENDLY_CONFIG.values()
        if isinstance(config, dict) and config.get("id")
    }

    assert friendly_ids <= known_tool_contract_ids()


def test_router_rag_allowed_tool_ids_are_contract_ids():
    allowed_tool_ids = set(builtin_router_rag_spec()["config"]["allowed_tool_ids"])

    assert allowed_tool_ids <= known_tool_contract_ids()


def test_tool_contract_metadata_exposes_graph_integration_fields():
    document_contract = get_tool_contract_metadata("search_documents")
    web_contract = get_tool_contract_metadata("search_web")
    records = list_tool_contract_metadata()

    assert document_contract["id"] == "document_evidence"
    assert document_contract["category"] == "retrieval"
    assert document_contract["allowed_caller_nodes"] == ["retrieval_worker"]
    assert document_contract["artifact_keys"] == ["document_sources", "web_sources"]
    assert "missing_thread_context" in document_contract["warning_codes"]

    assert web_contract["id"] == "live_web_recon"
    assert web_contract["allowed_caller_nodes"] == ["web_worker"]
    assert "web_search_disabled" in web_contract["warning_codes"]

    assert any(record["tool_name"] == "search_documents" and record["display_name"] == "Document Evidence" for record in records)
    assert records == sorted(records, key=lambda record: record["tool_name"])


def test_template_validator_accepts_external_contract_ids():
    spec = builtin_router_rag_spec()
    spec["config"]["allowed_tool_ids"] = [
        "document_evidence",
        "wikipedia_reference",
        "semantic_scholar_research",
    ]

    assert TemplateValidator().validate(spec) == {"valid": True, "errors": []}


def test_tool_contract_records_are_schema_like():
    for tool_name, contract in TOOL_CONTRACT_METADATA.items():
        assert contract["id"]
        assert contract["category"]
        assert isinstance(contract["allowed_caller_nodes"], list)
        assert isinstance(contract["artifact_keys"], list)
        assert isinstance(contract["warning_codes"], list)
        assert get_tool_contract_metadata(tool_name) == contract

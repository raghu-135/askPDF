from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.registry import MCP_TOOL_DEFINITIONS, descriptor, validate_registry
from app.services.document_pipeline import stable_source_id
from app.tools.contracts import ReadContextRequest


def test_mcp_registry_matches_authoritative_tool_registry():
    validate_registry()
    for name, (model, _) in MCP_TOOL_DEFINITIONS.items():
        item = descriptor(name, model)
        assert item["name"] == TOOL_FRIENDLY_CONFIG[name]["mcp_tool"]
        assert item["_meta"]["com.askpdf/contract-id"] == TOOL_FRIENDLY_CONFIG[name]["id"]
        assert "inputSchema" in item
        assert item["outputSchema"]["required"] == ["ok", "content", "sources", "artifacts", "warnings", "error", "metrics", "trace"]


def test_document_source_ids_round_trip_through_context_contract():
    source_id = stable_source_id("file-hash", "generation", "chunk")
    request = ReadContextRequest(source_id=source_id, token_budget=2000)
    assert request.source_id == source_id
    assert request.token_budget == 2000

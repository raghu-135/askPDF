from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.registry import MCP_TOOL_DEFINITIONS, descriptor, validate_registry


def test_mcp_registry_matches_authoritative_tool_registry():
    validate_registry()
    for name, (model, _) in MCP_TOOL_DEFINITIONS.items():
        item = descriptor(name, model)
        assert item["name"] == TOOL_FRIENDLY_CONFIG[name]["mcp_tool"]
        assert item["_meta"]["com.askpdf/contract-id"] == TOOL_FRIENDLY_CONFIG[name]["id"]
        assert "inputSchema" in item
        assert item["outputSchema"]["required"] == ["ok", "content", "sources", "artifacts", "warnings", "metrics", "trace"]

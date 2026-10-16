import copy

import pytest
from jsonschema import Draft202012Validator

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.registry import MCP_TOOL_DEFINITIONS, descriptor, validate_registry
from app.services.document_pipeline import stable_source_id
from app.tools.contracts import ReadContextRequest
from tests.test_canvas_spec_pytest import valid_spec


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


def test_publish_canvas_input_schema_accepts_string_table_cells():
    validate_registry()
    schema = descriptor("publish_canvas", MCP_TOOL_DEFINITIONS["publish_canvas"])["inputSchema"]
    validator = Draft202012Validator(schema)
    payload = {"spec": valid_spec()}
    validator.validate(payload)

    invalid = copy.deepcopy(payload)
    invalid["spec"]["sections"][0]["blocks"][1]["rows"] = [["Latency bound", 42, "Partial"]]
    with pytest.raises(Exception):
        validator.validate(invalid)

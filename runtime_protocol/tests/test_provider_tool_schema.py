import json

import pytest

from runtime_protocol.provider_tool_schema import (
    provider_compatible_function_tools,
    provider_compatible_json_schema,
)


def test_inline_defs_and_remove_ref_keywords():
    schema = {
        "$defs": {
            "Claim": {
                "type": "object",
                "properties": {"text": {"type": "string", "default": "x"}},
                "required": ["text"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {"$ref": "#/$defs/Claim"},
            }
        },
        "title": "Draft",
    }

    normalized = provider_compatible_json_schema(schema)
    encoded = json.dumps(normalized)

    assert "$defs" not in encoded
    assert "$ref" not in encoded
    assert "default" not in encoded
    assert normalized["properties"]["claims"]["items"]["required"] == ["text"]


def test_provider_compatible_function_tools_sanitizes_parameters():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "publish_canvas",
                "parameters": {
                    "$defs": {"Node": {"type": "object"}},
                    "properties": {"node": {"$ref": "#/$defs/Node"}},
                },
            },
        }
    ]

    sanitized = provider_compatible_function_tools(tools)
    payload = json.dumps(sanitized)

    assert "$defs" not in payload
    assert "$ref" not in payload


def test_unresolvable_ref_falls_back_to_object():
    normalized = provider_compatible_json_schema({"properties": {"x": {"$ref": "#/$defs/Missing"}}})
    assert normalized["properties"]["x"] == {"type": "object"}


def test_cycle_breaks_to_object():
    schema = {
        "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
        "type": "object",
        "properties": {"loop": {"$ref": "#/$defs/Loop"}},
    }
    normalized = provider_compatible_json_schema(schema)
    assert normalized["properties"]["loop"] == {"type": "object"}

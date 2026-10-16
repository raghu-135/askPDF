from runtime_protocol.tool_contract import ToolResult, normalize_tool_result
from datetime import datetime, timezone


def test_tool_result_uses_canonical_source_metadata_keys():
    result = ToolResult(
        content="evidence",
        artifacts={
            "document_sources": [{"file_hash": "file-1"}],
            "web_sources": [{"url": "https://example.com"}],
            "used_chat_ids": ["chat-1"],
        },
    )
    payload = result.to_json()
    assert '"document_sources"' in payload
    assert '"web_sources"' in payload
    assert '"used_chat_ids"' in payload
    assert "_" * 2 + "document_sources" not in payload
    assert "_" * 2 + "web_sources" not in payload
    assert "_" * 2 + "used_chat_ids" not in payload


def test_tool_result_preserves_error_and_structured_artifacts():
    result = ToolResult(
        content="No evidence",
        ok=False,
        error={"code": "lookup_failed", "message": "lookup failed"},
        artifacts={"thread_shape": {"documents": []}},
    )
    payload = normalize_tool_result(result.to_payload())
    assert payload["ok"] is False
    assert payload["error"]["code"] == "lookup_failed"
    assert payload["artifacts"]["thread_shape"] == {"documents": []}


def test_tool_result_normalizes_datetime_artifacts_for_mcp_json():
    result = ToolResult(
        content="shape",
        artifacts={"thread_shape": {"last_qa_at": datetime(2026, 8, 11, tzinfo=timezone.utc)}},
    )
    payload = normalize_tool_result(result.to_payload())
    assert payload["artifacts"]["thread_shape"]["last_qa_at"] == "2026-08-11T00:00:00Z"

from app.tools.results import ToolResult
from datetime import datetime, timezone


def test_tool_result_preserves_existing_source_metadata_keys():
    result = ToolResult(
        text="evidence",
        structured_content={
            "document_sources": [{"file_hash": "file-1"}],
            "web_sources": [{"url": "https://example.com"}],
            "used_chat_ids": ["chat-1"],
        },
    )
    payload = result.legacy_payload(contract_id="document_evidence")
    assert '"__document_sources__"' in payload
    assert '"__web_sources__"' in payload
    assert '"__used_chat_ids__"' in payload


def test_tool_result_preserves_error_and_structured_artifacts():
    result = ToolResult(
        text="No evidence",
        ok=False,
        error={"code": "lookup_failed"},
        artifacts={"thread_shape": {"documents": []}},
    )
    payload = result.structured(contract_id="thread_shape")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "lookup_failed"
    assert payload["artifacts"]["thread_shape"] == {"documents": []}


def test_tool_result_normalizes_datetime_artifacts_for_mcp_json():
    result = ToolResult(
        text="shape",
        artifacts={"thread_shape": {"last_qa_at": datetime(2026, 8, 11, tzinfo=timezone.utc)}},
    )
    payload = result.structured(contract_id="thread_shape")
    assert payload["artifacts"]["thread_shape"]["last_qa_at"] == "2026-08-11T00:00:00+00:00"

import pytest

from app.agent.tool_contract import (
    compact_tool_event,
    make_tool_error_result,
    make_tool_result,
    normalize_tool_result,
    tool_started,
)


class TestAskPdfToolContract:
    def test_make_tool_result_records_trace_metrics_and_legacy_fields(self):
        config = {
            "configurable": {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "caller_node": "retrieval_worker",
                "route": "document",
            }
        }
        started = tool_started()

        raw = make_tool_result(
            tool_name="search_documents",
            content="Evidence",
            config=config,
            started=started,
            sources=[{"file_hash": "file-1"}],
            artifacts={"document_sources": [{"file_hash": "file-1"}]},
        ).to_json(legacy_fields={"__document_sources__": [{"file_hash": "file-1"}]})
        payload = normalize_tool_result(raw, tool_name="search_documents", config=config)

        assert payload["ok"] is True
        assert payload["content"] == "Evidence"
        assert payload["__document_sources__"] == [{"file_hash": "file-1"}]
        assert payload["trace"]["agent_run_id"] == "run-1"
        assert payload["trace"]["caller_node"] == "retrieval_worker"
        assert payload["metrics"]["result_chars"] == len("Evidence")

    def test_normalize_tool_result_accepts_legacy_json_and_plain_strings(self):
        legacy = normalize_tool_result(
            '{"content":"Memory","__used_chat_ids__":["turn-1"]}',
            tool_name="search_conversation_history",
        )
        plain = normalize_tool_result("No thread context found.", tool_name="get_thread_shape")

        assert legacy["content"] == "Memory"
        assert legacy["__used_chat_ids__"] == ["turn-1"]
        assert plain["content"] == "No thread context found."
        assert plain["ok"] is True

    def test_normalize_tool_result_warns_for_missing_content(self):
        payload = normalize_tool_result({"ok": True}, tool_name="bad_tool")

        assert payload["content"] == ""
        assert "tool_output_missing_content" in payload["warnings"]

    def test_make_tool_error_result_is_recoverable_and_compact(self):
        config = {"configurable": {"agent_run_id": "run-1", "caller_node": "web_worker"}}

        result = make_tool_error_result(
            tool_name="search_web",
            error=RuntimeError("network unavailable"),
            config=config,
            started=tool_started(),
            user_message="Web search failed: network unavailable",
        )
        payload = normalize_tool_result(result)
        event = compact_tool_event(payload)

        assert payload["ok"] is False
        assert payload["content"] == "Web search failed: network unavailable"
        assert payload["error"]["type"] == "RuntimeError"
        assert event["tool_name"] == "search_web"
        assert event["caller_node"] == "web_worker"
        assert event["ok"] is False

    @pytest.mark.asyncio
    async def test_contract_protocol_matches_langchain_style_ainvoke(self):
        class FakeTool:
            name = "fake_tool"

            async def ainvoke(self, input, config=None):
                return make_tool_result(
                    tool_name=self.name,
                    content=input["query"],
                    config=config,
                ).to_json()

        raw = await FakeTool().ainvoke(
            {"query": "hello"},
            config={"configurable": {"caller_node": "test_node"}},
        )
        payload = normalize_tool_result(raw, tool_name="fake_tool")

        assert payload["content"] == "hello"
        assert payload["trace"]["caller_node"] == "test_node"

import pytest
from unittest.mock import AsyncMock, patch

from app.agent.tool_contract import (
    ToolErrorCode,
    ToolWarningCode,
    compact_tool_event,
    make_tool_error_result,
    make_tool_result,
    normalize_tool_result,
    tool_started,
)
from app.rag.agent_tools import search_durable_memory


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
            tool_name="search_thread_conversation_history",
        )
        plain = normalize_tool_result("No thread context found.", tool_name="get_thread_shape")

        assert legacy["content"] == "Memory"
        assert legacy["__used_chat_ids__"] == ["turn-1"]
        assert plain["content"] == "No thread context found."
        assert plain["ok"] is True

    def test_normalize_tool_result_warns_for_missing_content(self):
        payload = normalize_tool_result({"ok": True}, tool_name="bad_tool")

        assert payload["content"] == ""
        assert ToolWarningCode.TOOL_OUTPUT_MISSING_CONTENT in payload["warnings"]

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
        assert payload["error"]["code"] == ToolErrorCode.failed("search_web")
        assert payload["error"]["type"] == "RuntimeError"
        assert event["tool_name"] == "search_web"
        assert event["caller_node"] == "web_worker"
        assert event["ok"] is False

    def test_compact_tool_event_keeps_input_preview_and_artifact_refs(self):
        config = {"configurable": {"caller_node": "retrieval_worker"}}
        raw = make_tool_result(
            tool_name="search_documents",
            content="Document evidence body",
            config=config,
            sources=[{"file_hash": "file-1"}],
            artifacts={
                "document_sources": [
                    {
                        "file_hash": "file-1",
                        "file_name": "paper.pdf",
                        "chunk_id": 7,
                        "pages": "2-3",
                        "text": "Relevant passage",
                    }
                ]
            },
        )
        payload = normalize_tool_result(raw)

        event = compact_tool_event(payload, tool_input={"query": "paper", "max_results": 10})

        assert event["tool_input"] == {"query": "paper", "max_results": 10}
        assert event["result_preview"] == "Document evidence body"
        assert event["artifact_summary"] == {"document_sources": 1}
        assert event["artifact_refs"]["document_matches"][0]["file_hash"] == "file-1"
        assert event["artifact_refs"]["document_matches"][0]["chunk_id"] == 7

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

    @pytest.mark.asyncio
    async def test_long_term_memory_tool_preserves_flat_memory_evidence_and_debug_ids(self):
        search_result = {
            "memories": [
                {
                    "id": "memory-1",
                    "scope_type": "project",
                    "scope_id": "project-1",
                    "content": "The launch codename is Atlas.",
                    "score": 0.91,
                    "score_type": "similarity",
                    "raw_score": 0.83,
                    "embedding_model": "BAAI/bge-m3",
                }
            ],
            "scopes": [{"scope_type": "project", "scope_id": "project-1"}],
            "scope_policy": {
                "requested_scopes": ["thread", "project", "user"],
                "searched_scopes": [{"scope_type": "project", "scope_id": "project-1"}],
                "skipped_scopes": [{"scope_type": "user", "reason": "thread_opt_out"}],
            },
        }
        with patch(
            "app.services.memory_tool_service.build_memory_tool_context",
            new_callable=AsyncMock,
            return_value=(object(), None, None),
        ), patch(
            "app.services.memory_tool_service.search_memory_tool",
            new_callable=AsyncMock,
            return_value=search_result,
        ):
            raw = await search_durable_memory.ainvoke(
                {"query": "launch codename"},
                config={"configurable": {"app_thread_id": "thread-1"}},
            )

        payload = normalize_tool_result(raw, tool_name="search_durable_memory")
        assert "The launch codename is Atlas." in payload["content"]
        assert payload["artifacts"]["memory_refs"] == [
            {
                "memory_id": "memory-1",
                "scope_type": "project",
                "scope_id": "project-1",
                "score": 0.91,
                "score_type": "similarity",
                "raw_score": 0.83,
                "embedding_model": "BAAI/bge-m3",
                    "scope_rank": None,
                    "attributes": {},
            }
        ]
        assert payload["artifacts"]["memory_scope_policy"]["skipped_scopes"] == [
            {"scope_type": "user", "reason": "thread_opt_out"}
        ]

    @pytest.mark.asyncio
    async def test_long_term_memory_tool_reuses_sufficient_prefetch_without_second_search(self):
        prefetched = [
            {
                "id": f"memory-{index}",
                "scope_type": "thread",
                "scope_id": "thread-1",
                "content": f"Preference {index}",
                "excerpt": f"Preference {index}",
                "score": 0.9 - (index * 0.01),
                "attributes": {"kind": "preference"},
            }
            for index in range(3)
        ]
        expanded_search = AsyncMock()
        with patch(
            "app.services.memory_tool_service.build_memory_tool_context",
            new_callable=AsyncMock,
            return_value=(object(), None, None),
        ), patch(
            "app.services.memory_tool_service.search_memory_tool",
            new=expanded_search,
        ):
            raw = await search_durable_memory.ainvoke(
                {"query": "preferences"},
                config={"configurable": {
                    "app_thread_id": "thread-1",
                    "prefetched_durable_memories": prefetched,
                    "prefetched_durable_memory_scopes": [{"scope_type": "thread", "scope_id": "thread-1"}],
                    "prefetched_durable_memory_scope_policy": {
                        "searched_scopes": [{"scope_type": "thread", "scope_id": "thread-1"}],
                    },
                    "prefetched_durable_memory_debug": {"rejection_reasons": {}},
                }},
            )

        payload = normalize_tool_result(raw, tool_name="search_durable_memory")
        expanded_search.assert_not_awaited()
        assert payload["artifacts"]["memory_retrieval_debug"]["reused_prefetch"] is True
        assert payload["artifacts"]["memory_retrieval_debug"]["expanded_search"] is False

    @pytest.mark.asyncio
    async def test_long_term_memory_tool_reports_policy_when_no_memory_matches(self):
        search_result = {
            "memories": [],
            "scopes": [{"scope_type": "thread", "scope_id": "thread-1"}],
            "scope_policy": {
                "requested_scopes": ["thread", "project", "user"],
                "searched_scopes": [{"scope_type": "thread", "scope_id": "thread-1"}],
                "skipped_scopes": [
                    {"scope_type": "project", "reason": "thread_opt_out"},
                    {"scope_type": "user", "reason": "project_opt_out"},
                ],
            },
        }
        with patch(
            "app.services.memory_tool_service.build_memory_tool_context",
            new_callable=AsyncMock,
            return_value=(object(), None, None),
        ), patch(
            "app.services.memory_tool_service.search_memory_tool",
            new_callable=AsyncMock,
            return_value=search_result,
        ):
            raw = await search_durable_memory.ainvoke(
                {"query": "preference"},
                config={"configurable": {"app_thread_id": "thread-1"}},
            )

        payload = normalize_tool_result(raw, tool_name="search_durable_memory")
        assert payload["artifacts"]["memory_refs"] == []
        assert payload["artifacts"]["memory_scopes"] == search_result["scopes"]
        assert payload["artifacts"]["memory_scope_policy"] == search_result["scope_policy"]
        event = compact_tool_event(payload)
        assert event["artifacts"] == {
            "memory_scopes": search_result["scopes"],
            "memory_scope_policy": search_result["scope_policy"],
        }
        assert "memory_refs" not in event["artifacts"]

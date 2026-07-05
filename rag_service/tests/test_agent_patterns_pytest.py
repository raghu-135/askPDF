import os
import uuid
import logging
import asyncio
import json
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent_patterns.router_runtime import handle_router_rag_chat
from app.agent_patterns.graph import NodeRegistry, TemplateCompiler, _llm_result_metadata
from app.agent_patterns.graph import build_planner_prompt, infer_required_plan_steps, normalize_execution_plan
from app.agent_patterns.debug_trace import build_debug_trace
from app.agent_patterns.metrics import build_run_metrics
from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.service import AgentRunService
from app.agent_patterns.templates import (
    PLAN_EXECUTE_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_AGENT_VERSION,
    ROUTER_RAG_AGENT_ID,
    ROUTER_RAG_AGENT_VERSION,
    builtin_plan_execute_rag_spec,
    builtin_router_rag_spec,
)
from app.agent_patterns.validator import TemplateResolver, TemplateValidationError, TemplateValidator
from app.db.models_sqlmodel import AgentRun, ChatTurn
from app.models.retry import invoke_with_retry
from app.time_utils import utc_now


SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
TRACE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "agent_debug_trace_v1.schema.json"


class TestModelRetry:
    @pytest.mark.asyncio
    async def test_invoke_with_retry_reports_retry_attempts_without_changing_behavior(self, monkeypatch):
        calls = 0
        observed = []
        delays = []

        async def fake_sleep(delay):
            delays.append(delay)

        async def flaky_call():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("status_code=429 temporary overload")
            return "ok"

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await invoke_with_retry(flaky_call, retry_observer=lambda event: observed.append(event))

        assert result == "ok"
        assert calls == 3
        assert delays == [2, 4]
        assert [event["attempt"] for event in observed] == [1, 2]
        assert observed[0]["delay_ms"] == 2000
        assert observed[0]["http_status_code"] == 429
        assert observed[0]["exception_type"] == "RuntimeError"
        assert observed[0]["reason"] == "Retryable OpenAI-compatible API error (429)"


class TestAgentRunMetrics:
    def test_build_run_metrics_rolls_up_node_tool_and_error_counts(self):
        metrics = build_run_metrics(
            {
                "route": "document",
                "node_events": [
                    {"node": "router", "elapsed_ms": 2.25},
                    {"node": "retrieval_worker", "elapsed_ms": 7.75},
                    {"node": "retrieval_worker", "elapsed_ms": 1.0},
                ],
                "tool_events": [
                    {"tool_name": "search_documents", "ok": True, "elapsed_ms": 11.5, "warnings": ["low_sources"]},
                    {"tool_name": "search_web", "ok": False, "elapsed_ms": 3.0, "warnings": []},
                ],
                "errors": [{"code": "worker_error"}],
                "document_sources": [{"id": "doc-1"}],
                "web_sources": [],
                "used_chat_ids": ["chat-1"],
            },
            duration_ms=25.123,
        )

        assert metrics["duration_ms"] == 25.12
        assert metrics["route"] == "document"
        assert metrics["node_event_count"] == 3
        assert metrics["node_elapsed_ms"] == {"router": 2.25, "retrieval_worker": 8.75}
        assert metrics["node_total_elapsed_ms"] == 11.0
        assert metrics["tool_event_count"] == 2
        assert metrics["tool_warning_count"] == 1
        assert metrics["tool_error_count"] == 1
        assert metrics["tool_elapsed_ms"] == 14.5
        assert metrics["error_count"] == 1
        assert metrics["document_source_count"] == 1
        assert metrics["used_chat_id_count"] == 1

    def test_build_debug_trace_represents_skipped_nodes_without_warnings(self):
        run = SimpleNamespace(
            id="run-1",
            thread_id="thread-1",
            user_id="user-1",
            template_id=PLAN_EXECUTE_RAG_AGENT_ID,
            template_version_id=f"{PLAN_EXECUTE_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_plan_execute_rag_spec(),
            status="completed",
            started_at=utc_now(),
            completed_at=utc_now(),
        )

        trace = build_debug_trace(
            run=run,
            chat_turn=SimpleNamespace(id="turn-1"),
            node_events=[
                {
                    "node": "memory_worker",
                    "status": "skipped",
                    "skipped": True,
                    "skip_reason": "not_selected_by_plan",
                    "elapsed_ms": 0.5,
                    "start_time": "2026-07-04T02:30:08.000Z",
                    "end_time": "2026-07-04T02:30:08.001Z",
                },
                {
                    "node": "router",
                    "status": "completed",
                    "route": "direct",
                    "elapsed_ms": 4.0,
                    "llm_result_summary": {
                        "llm": {
                            "model_name": "test-llm",
                            "response_chars": 42,
                            "token_counts": {
                                "prompt": 11,
                                "completion": 7,
                                "total": 18,
                                "reasoning": 3,
                                "cached": 2,
                            },
                            "reasoning_available": True,
                            "reasoning_format": "structured",
                            "reasoning_chars": 12,
                            "reasoning_preview": "Short reasoning",
                            "retry_count": 1,
                            "retry_attempts": [
                                {
                                    "attempt": 1,
                                    "delay_ms": 2000,
                                    "reason": "Retryable OpenAI-compatible API error (429)",
                                    "http_status_code": 429,
                                    "exception_type": "RuntimeError",
                                    "exception_message": "status_code=429 temporary overload",
                                }
                            ],
                        }
                    },
                }
            ],
            tool_events=[
                {
                    "tool_name": "search_documents",
                    "caller_node": "retrieval_worker",
                    "ok": True,
                    "elapsed_ms": 2.25,
                    "start_time": "2026-07-04T02:30:09.000Z",
                    "end_time": "2026-07-04T02:30:09.002Z",
                }
            ],
            metrics={"duration_ms": 3.0, "route": "execute", "tool_warning_count": 0, "error_count": 0},
            route="execute",
            route_reason="Document evidence is enough.",
        )

        skipped_span = next(span for span in trace["spans"] if span["span_id"] == "node:memory_worker:0")
        assert skipped_span["status"] == "skipped"
        assert skipped_span["start_time"] == "2026-07-04T02:30:08Z"
        assert skipped_span["end_time"] == "2026-07-04T02:30:08.001000Z"
        assert skipped_span["attributes"]["askpdf.skip_reason"] == "not_selected_by_plan"
        assert any(event["name"] == "skipped" for event in skipped_span["events"])
        assert not any(event["name"] == "warning" for event in skipped_span["events"])
        router_span = next(span for span in trace["spans"] if span["span_id"] == "node:router:1")
        assert router_span["attributes"]["llm.model_name"] == "test-llm"
        assert router_span["attributes"]["llm.token_count.total"] == 18
        assert router_span["attributes"]["llm.retry_count"] == 1
        assert trace["metrics"]["llm_span_count"] == 1
        assert trace["metrics"]["llm_token_count_prompt"] == 11
        assert trace["metrics"]["llm_token_count_completion"] == 7
        assert trace["metrics"]["llm_token_count_total"] == 18
        assert trace["metrics"]["llm_token_count_reasoning"] == 3
        assert trace["metrics"]["llm_token_count_cached"] == 2
        assert trace["metrics"]["llm_retry_count"] == 1
        retry_event = next(event for event in router_span["events"] if event["name"] == "llm.retry")
        assert retry_event["attributes"]["llm.retry.attempt"] == 1
        assert retry_event["attributes"]["llm.retry.delay_ms"] == 2000
        assert retry_event["attributes"]["http.status_code"] == 429
        llm_event = next(event for event in router_span["events"] if event["name"] == "llm.completed")
        assert llm_event["attributes"]["llm.response_chars"] == 42
        assert llm_event["attributes"]["llm.token_count.reasoning"] == 3
        assert llm_event["attributes"]["llm.reasoning_format"] == "structured"
        assert llm_event["output"]["reasoning_preview"] == "Short reasoning"
        tool_span = next(span for span in trace["spans"] if span["span_id"] == "tool:search_documents:0")
        assert tool_span["start_time"] == "2026-07-04T02:30:09Z"
        assert tool_span["end_time"] == "2026-07-04T02:30:09.002000Z"

    def test_llm_result_metadata_includes_bounded_reasoning_preview(self):
        reasoning = "Reasoning detail. " * 200
        summary = _llm_result_metadata(
            SimpleNamespace(content="answer", usage_metadata={}, response_metadata={}),
            normalized_response={
                "reasoning": reasoning,
                "reasoning_available": True,
                "reasoning_format": "structured",
            },
        )

        assert summary["reasoning_available"] is True
        assert summary["reasoning_format"] == "structured"
        assert summary["reasoning_chars"] == len(reasoning)
        assert summary["reasoning_preview"].startswith("Reasoning detail.")
        assert len(summary["reasoning_preview"]) <= 1803

    def test_build_debug_trace_bounds_preview_values_but_preserves_raw_events(self):
        long_text = "A" * 5000
        run = SimpleNamespace(
            id="run-size",
            thread_id="thread-1",
            user_id="user-1",
            template_id=ROUTER_RAG_AGENT_ID,
            template_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_router_rag_spec(),
            status="completed",
            started_at=utc_now(),
            completed_at=utc_now(),
        )

        trace = build_debug_trace(
            run=run,
            chat_turn=SimpleNamespace(id="turn-1"),
            node_events=[
                {
                    "node": "context_loader",
                    "status": "completed",
                    "input_preview": {"question": long_text},
                    "output_preview": {"document_evidence": long_text},
                    "output_refs": {
                        "document_matches": [
                            {
                                "file_hash": "file-1",
                                "chunk_id": "chunk-1",
                                "preview": long_text,
                                "text": long_text,
                            }
                        ]
                    },
                }
            ],
            tool_events=[],
            metrics={"duration_ms": 1.0, "route": "direct", "tool_warning_count": 0, "error_count": 0},
            route="direct",
        )

        node_span = next(span for span in trace["spans"] if span["span_id"] == "node:context_loader:0")
        assert len(node_span["input"]["value"]["question"]) <= 903
        assert len(node_span["output"]["value"]["document_evidence"]) <= 903
        ref = node_span["output"]["refs"]["document_matches"][0]
        assert len(ref["preview"]) <= 903
        assert len(ref["text"]) <= 903
        assert node_span["raw"]["output_refs"]["document_matches"][0]["text"] == long_text

    def test_build_debug_trace_v1_shape_for_direct_tool_plan_and_failed_runs(self):
        run = SimpleNamespace(
            id="run-shape",
            thread_id="thread-1",
            user_id="user-1",
            template_id=PLAN_EXECUTE_RAG_AGENT_ID,
            template_version_id=f"{PLAN_EXECUTE_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_plan_execute_rag_spec(),
            status="failed",
            started_at=utc_now(),
            completed_at=utc_now(),
        )

        trace = build_debug_trace(
            run=run,
            chat_turn=SimpleNamespace(id="turn-1"),
            node_events=[
                {"node": "planner", "status": "completed", "route": "execute", "execution_plan": ["retrieval_worker"]},
                {"node": "retrieval_worker", "status": "failed", "error": {"code": "worker_failed", "message": "boom", "retryable": True}},
                {"node": "memory_worker", "status": "skipped", "skipped": True, "skip_reason": "not_selected_by_plan"},
            ],
            tool_events=[
                {
                    "tool_name": "search_documents",
                    "tool_id": "document_evidence",
                    "tool_category": "document",
                    "caller_node": "retrieval_worker",
                    "ok": False,
                    "error": {"code": "search_documents_failed", "message": "tool boom", "retryable": True},
                }
            ],
            metrics={"duration_ms": 4.0, "route": "execute", "tool_warning_count": 0, "error_count": 1},
            route="execute",
            error={"code": "agent_run_failed", "raw_message": "run boom", "retryable": True},
        )

        assert trace["schema_version"] == 1
        span_ids = {span["span_id"]: span for span in trace["spans"]}
        assert span_ids["node:planner:0"]["raw"]["node"] == "planner"
        assert span_ids["run:run-shape"]["events"][0]["name"] == "exception"
        assert span_ids["run:run-shape"]["events"][0]["attributes"]["askpdf.error.retryable"] is True
        assert span_ids["node:planner:0"]["events"][0]["name"] == "decision.made"
        assert span_ids["node:retrieval_worker:1"]["status"] == "error"
        assert any(event["name"] == "exception" for event in span_ids["node:retrieval_worker:1"]["events"])
        assert span_ids["node:memory_worker:2"]["status"] == "skipped"
        assert any(event["name"] == "skipped" for event in span_ids["node:memory_worker:2"]["events"])
        assert span_ids["tool:search_documents:0"]["status"] == "error"
        assert span_ids["tool:search_documents:0"]["attributes"]["tool.id"] == "document_evidence"

    def test_debug_trace_schema_contract_matches_generated_shape(self):
        schema = json.loads(TRACE_SCHEMA_PATH.read_text())
        span_schema = schema["$defs"]["span"]
        run = SimpleNamespace(
            id="run-contract",
            thread_id="thread-1",
            user_id="user-1",
            template_id=ROUTER_RAG_AGENT_ID,
            template_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_router_rag_spec(),
            status="completed",
            started_at=utc_now(),
            completed_at=utc_now(),
        )

        trace = build_debug_trace(
            run=run,
            chat_turn=SimpleNamespace(id="turn-1"),
            node_events=[{"node": "router", "status": "completed", "route": "direct"}],
            tool_events=[],
            metrics={"duration_ms": 1.0, "route": "direct", "tool_warning_count": 0, "error_count": 0},
            route="direct",
        )

        assert schema["properties"]["schema_version"]["const"] == 1
        for field in schema["required"]:
            assert field in trace
        assert trace["schema_version"] == 1
        assert "raw" not in trace

        for span in trace["spans"]:
            for field in span_schema["required"]:
                assert field in span
            assert span["kind"] in span_schema["properties"]["kind"]["enum"]
            for event in span["events"]:
                assert "name" in event


class TestRouterRagTemplateValidator:
    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda spec: spec.update({"pattern_type": "simple_rag_agent"}), "pattern_type must be one of:"),
            (lambda spec: spec["config"].update({"surprise": True}), "unknown config keys: surprise"),
            (lambda spec: spec["config"].update({"allowed_tool_ids": ["not_a_tool"]}), "unknown allowed_tool_ids: not_a_tool"),
            (lambda spec: spec["config"].update({"max_iterations": 999}), "max_iterations must be between"),
        ],
    )
    def test_rejects_invalid_router_rag_specs(self, mutate, expected):
        spec = builtin_router_rag_spec()
        mutate(spec)

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert expected in str(exc.value)

    def test_resolver_freezes_thread_and_request_overrides(self):
        resolved = TemplateResolver().resolve(
            builtin_router_rag_spec(),
            thread_settings={"max_iterations": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )

        assert resolved["config"]["max_iterations"] == 3
        assert resolved["config"]["use_reranker"] is False
        assert resolved["config"]["use_web_search"] is True

    def test_accepts_builtin_router_rag_spec(self):
        result = TemplateValidator().validate(builtin_router_rag_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_plan_execute_rag_spec(self):
        result = TemplateValidator().validate(builtin_plan_execute_rag_spec())

        assert result == {"valid": True, "errors": []}

    def test_rejects_router_rag_graph_topology_changes(self):
        spec = builtin_router_rag_spec()
        spec["config"]["graph"]["nodes"].append({"id": "surprise", "type": "retrieval_worker"})

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "graph nodes must match" in str(exc.value)

    def test_rejects_router_rag_specs_missing_required_tools(self):
        spec = builtin_router_rag_spec()
        spec["config"]["allowed_tool_ids"].remove("document_evidence")

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "missing required allowed_tool_ids: document_evidence" in str(exc.value)

    def test_compiles_builtin_router_rag_spec(self):
        graph = TemplateCompiler().compile(builtin_router_rag_spec())

        assert graph is not None

    def test_compiles_builtin_plan_execute_rag_spec(self):
        graph = TemplateCompiler().compile(builtin_plan_execute_rag_spec())

        assert graph is not None

    def test_rejects_plan_execute_graph_topology_changes(self):
        spec = builtin_plan_execute_rag_spec()
        spec["config"]["graph"]["edges"].append({"from": "planner", "to": "synthesizer"})

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "plan_execute_rag_agent graph edges must match" in str(exc.value)

    def test_normalize_execution_plan_clamps_invalid_plans_to_document_execution(self):
        normalized = normalize_execution_plan(
            {"route": "execute", "execution_plan": ["unknown_worker", "web_worker"]},
            use_web_search=False,
        )

        assert normalized["route"] == "execute"
        assert normalized["execution_plan"] == ["retrieval_worker"]

    @pytest.mark.parametrize(
        "question, expected_steps",
        [
            ("What is the latest document about?", ["timeline_worker", "retrieval_worker"]),
            ("What did we discuss previously about reranking?", ["memory_worker"]),
            ("What does the uploaded PDF say about risks?", ["retrieval_worker"]),
            ("What changed since the first upload?", ["timeline_worker", "retrieval_worker"]),
        ],
    )
    def test_infers_required_plan_steps_from_question_intent(self, question, expected_steps):
        assert infer_required_plan_steps(question) == expected_steps

    def test_normalize_execution_plan_adds_timeline_for_temporal_questions(self):
        normalized = normalize_execution_plan(
            {"route": "execute", "execution_plan": ["retrieval_worker"], "reason": "needs content"},
            use_web_search=False,
            question="What changed since the first upload?",
        )

        assert normalized["route"] == "execute"
        assert normalized["execution_plan"] == ["retrieval_worker", "timeline_worker"]

    def test_normalize_execution_plan_uses_memory_for_non_temporal_conversation_recall(self):
        normalized = normalize_execution_plan(
            {"route": "execute", "execution_plan": [], "reason": "needs prior chat"},
            use_web_search=False,
            question="What did we discuss previously about embeddings?",
        )

        assert normalized["execution_plan"] == ["memory_worker"]

    @pytest.mark.parametrize(
        "question, expected_steps",
        [
            ("What is the latest document about?", ["retrieval_worker", "timeline_worker"]),
            ("What changed since the first upload?", ["retrieval_worker", "timeline_worker"]),
            ("What did we discuss previously about reranking?", ["memory_worker"]),
        ],
    )
    def test_normalize_execution_plan_clamps_direct_for_obvious_retrieval_intent(self, question, expected_steps):
        normalized = normalize_execution_plan(
            {"route": "direct", "execution_plan": [], "reason": "prefetch is enough"},
            use_web_search=False,
            question=question,
        )

        assert normalized["route"] == "execute"
        assert normalized["execution_plan"] == expected_steps

    def test_normalize_execution_plan_keeps_direct_for_generic_question_without_intent_cues(self):
        normalized = normalize_execution_plan(
            {"route": "direct", "execution_plan": [], "reason": "prefetch is enough"},
            use_web_search=False,
            question="Can you answer this briefly?",
        )

        assert normalized["route"] == "direct"
        assert normalized["execution_plan"] == []

    def test_normalize_execution_plan_does_not_clamp_clarify_for_temporal_questions(self):
        normalized = normalize_execution_plan(
            {"route": "clarify", "execution_plan": [], "reason": "ambiguous"},
            use_web_search=False,
            question="Which latest document do you mean?",
        )

        assert normalized["route"] == "clarify"
        assert normalized["execution_plan"] == []
        assert normalized["clarification_options"] == [
            "Do I want an answer based on the uploaded document evidence?",
            "Do I want an answer based on what we discussed earlier in this thread?",
            "Do I want an answer based on the timeline or order of events in this thread?",
        ]
        assert all(option.startswith("Do I want") for option in normalized["clarification_options"])

    def test_build_planner_prompt_contains_temporal_memory_document_rules(self):
        prompt = build_planner_prompt(
            {
                "question": "What is the latest document about?",
                "use_web_search": False,
                "pre_fetch_bundle": {"documents": [{"file_name": "paper.pdf", "file_hash": "file-1"}]},
            }
        )

        assert "latest, first, earliest, oldest, since, before, after" in prompt
        assert "include `timeline_worker`" in prompt
        assert "prior conversation recall without time/order wording" in prompt
        assert "include `memory_worker` rather than `timeline_worker`" in prompt
        assert "uploaded document/PDF/page/quote/citation/content" in prompt
        assert "Choose `direct` only when pre-fetched context directly answers the question" in prompt
        assert "Do not choose `direct` for latest, first, since, before, after, or current questions" in prompt
        assert "`timeline_worker` queries should preserve temporal anchor words" in prompt


class TestRouterRagGraphToolConsumers:
    def test_tool_config_enforces_registry_contracts(self):
        from app.agent_patterns.graph import _tool_config

        state = {
            "agent_run_id": "run-1",
            "route": "document",
            "allowed_tool_ids": builtin_router_rag_spec()["config"]["allowed_tool_ids"],
        }
        config = {"configurable": {"thread_id": "thread-1"}}

        allowed = _tool_config(
            state,
            config,
            caller_node="retrieval_worker",
            tool_name="search_documents",
        )
        assert allowed["configurable"]["caller_node"] == "retrieval_worker"
        assert allowed["configurable"]["tool_name"] == "search_documents"

        with pytest.raises(ValueError, match="search_documents is not allowed from caller node memory_worker"):
            _tool_config(
                state,
                config,
                caller_node="memory_worker",
                tool_name="search_documents",
            )

        with pytest.raises(ValueError, match="is not enabled for this agent run"):
            _tool_config(
                dict(state, allowed_tool_ids=["deep_memory"]),
                config,
                caller_node="retrieval_worker",
                tool_name="search_documents",
            )

    @pytest.mark.asyncio
    async def test_workers_consume_tool_artifacts_without_legacy_fields(self, monkeypatch):
        class FakeTool:
            def __init__(self, payload):
                self.payload = payload

            async def ainvoke(self, _args, config=None):
                assert "__document_sources__" not in self.payload
                assert "__web_sources__" not in self.payload
                assert "__used_chat_ids__" not in self.payload
                assert "__timeline_events__" not in self.payload
                return self.payload

        registry = NodeRegistry()
        base_state = {
            "agent_run_id": "run-1",
            "thread_id": "thread-1",
            "question": "What does the document say?",
            "route": "document",
            "document_sources": [],
            "web_sources": [],
            "used_chat_ids": [],
            "node_events": [],
            "tool_events": [],
            "allowed_tool_ids": builtin_router_rag_spec()["config"]["allowed_tool_ids"],
        }
        config = {"configurable": {"thread_id": "thread-1"}}

        monkeypatch.setattr(
            "app.agent_patterns.graph.search_documents",
            FakeTool(
                {
                    "content": "Document evidence.",
                    "artifacts": {
                        "document_sources": [{"file_hash": "file-1", "file_name": "paper.pdf"}],
                        "web_sources": [{"url": "https://cached.example", "title": "Cached"}],
                    },
                    "sources": [{"file_hash": "file-1", "file_name": "paper.pdf"}],
                }
            ),
        )
        document_update = await registry.retrieval_worker(dict(base_state), config)
        assert document_update["document_sources"] == [{"file_hash": "file-1", "file_name": "paper.pdf"}]
        assert document_update["web_sources"] == [{"url": "https://cached.example", "title": "Cached"}]
        assert document_update["tool_events"][0]["tool_name"] == "search_documents"

        monkeypatch.setattr(
            "app.agent_patterns.graph.search_conversation_history",
            FakeTool(
                {
                    "content": "Memory evidence.",
                    "artifacts": {"used_chat_ids": ["turn-1:assistant"]},
                }
            ),
        )
        memory_update = await registry.memory_worker(dict(base_state, route="memory"), config)
        assert memory_update["used_chat_ids"] == ["turn-1:assistant"]
        assert memory_update["tool_events"][0]["tool_name"] == "search_conversation_history"

        monkeypatch.setattr(
            "app.agent_patterns.graph.search_thread_timeline",
            FakeTool(
                {
                    "content": "Timeline evidence.",
                    "artifacts": {
                        "timeline_events": [
                            {
                                "timeline_event_type": "message_created",
                                "timeline_event_at": "2026-07-01T00:00:00Z",
                            }
                        ]
                    },
                }
            ),
        )
        timeline_update = await registry.timeline_worker(dict(base_state, route="timeline"), config)
        assert timeline_update["node_events"][-1]["timeline_event_count"] == 1
        assert timeline_update["tool_events"][0]["tool_name"] == "search_thread_timeline"

        monkeypatch.setattr(
            "app.agent_patterns.graph.search_web",
            FakeTool(
                {
                    "content": "Web evidence.",
                    "artifacts": {"web_sources": [{"url": "https://example.com", "title": "Example"}]},
                    "sources": [{"url": "https://example.com", "title": "Example"}],
                }
            ),
        )
        web_update = await registry.web_worker(dict(base_state, route="web"), config)
        assert web_update["web_sources"] == [{"url": "https://example.com", "title": "Example"}]
        assert web_update["tool_events"][0]["tool_name"] == "search_web"

    @pytest.mark.asyncio
    async def test_unselected_plan_workers_emit_skipped_event_without_calling_tool(self, monkeypatch):
        class ExplodingTool:
            async def ainvoke(self, _args, config=None):
                raise AssertionError("tool should not be called for unselected plan worker")

        monkeypatch.setattr("app.agent_patterns.graph.search_conversation_history", ExplodingTool())

        registry = NodeRegistry()
        update = await registry.memory_worker(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What is this about?",
                "route": "execute",
                "execution_plan": ["retrieval_worker"],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "allowed_tool_ids": builtin_plan_execute_rag_spec()["config"]["allowed_tool_ids"],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert update["node_events"][-1]["node"] == "memory_worker"
        assert update["node_events"][-1]["skipped"] is True
        assert update["node_events"][-1]["skip_reason"] == "not_selected_by_plan"
        assert "warnings" not in update["node_events"][-1]
        assert "tool_events" not in update


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentPatternRepository:
    @pytest_asyncio.fixture
    async def repo(self, engine):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            yield AgentPatternRepository(repo_session)

    @pytest.mark.asyncio
    async def test_seed_builtin_router_rag_template_is_idempotent(self, repo):
        await repo.seed_builtin_templates()
        await repo.seed_builtin_templates()

        templates = await repo.list_templates()
        router_template, router_version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        plan_template, plan_version = await repo.get_template_with_current_version(PLAN_EXECUTE_RAG_AGENT_ID)

        assert {template.id for template in templates} == {ROUTER_RAG_AGENT_ID, PLAN_EXECUTE_RAG_AGENT_ID}
        assert router_template.current_version_id == router_version.id
        assert router_version.version == ROUTER_RAG_AGENT_VERSION
        assert router_version.validation_result_json == {"valid": True, "errors": []}
        assert plan_template.current_version_id == plan_version.id
        assert plan_version.version == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert plan_version.validation_result_json == {"valid": True, "errors": []}

    @pytest.mark.asyncio
    async def test_run_lifecycle_persists_resolved_spec(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)

        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )
        completed = await repo.complete_run(
            run.id,
            status="completed",
            metrics_json={"duration_ms": 12.5},
        )

        assert completed.status == "completed"
        assert completed.metrics_json == {"duration_ms": 12.5}
        assert completed.resolved_spec_json == {"pattern_type": ROUTER_RAG_AGENT_ID}

    @pytest.mark.asyncio
    async def test_list_runs_for_thread_orders_recent_first_and_limits(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        first = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "n": 1},
        )
        second = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "n": 2},
        )

        runs = await repo.list_runs_for_thread(sample_thread.id, limit=1)

        assert [run.id for run in runs] == [second.id]
        assert first.id != second.id

    @pytest.mark.asyncio
    async def test_prune_runs_before_deletes_only_matching_old_statuses(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        old_completed = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "old_completed"},
        )
        old_running = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "old_running"},
        )
        recent_completed = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "recent_completed"},
        )
        old_at = utc_now() - timedelta(days=45)
        recent_at = utc_now() - timedelta(days=1)

        session = await repo._get_session()
        async with session.begin():
            old_completed_row = await session.get(AgentRun, old_completed.id)
            old_running_row = await session.get(AgentRun, old_running.id)
            recent_completed_row = await session.get(AgentRun, recent_completed.id)
            old_completed_row.started_at = old_at
            old_completed_row.status = "completed"
            old_running_row.started_at = old_at
            old_running_row.status = "running"
            recent_completed_row.started_at = recent_at
            recent_completed_row.status = "completed"

        deleted_ids = await repo.prune_runs_before(
            utc_now() - timedelta(days=30),
            statuses=["completed", "failed"],
            thread_id=sample_thread.id,
        )

        assert deleted_ids == [old_completed.id]
        assert await repo.get_run(old_completed.id) is None
        assert await repo.get_run(old_running.id) is not None
        assert await repo.get_run(recent_completed.id) is not None

    @pytest.mark.asyncio
    async def test_prune_runs_before_requires_explicit_statuses(self, repo):
        with pytest.raises(ValueError, match="statuses must contain at least one status"):
            await repo.prune_runs_before(utc_now(), statuses=[])

    @pytest.mark.asyncio
    async def test_fail_stale_running_runs_marks_only_old_running_rows_failed(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        stale_running = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "stale_running"},
        )
        recent_running = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "recent_running"},
        )
        stale_completed = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "stale_completed"},
        )
        old_at = utc_now() - timedelta(hours=6)
        recent_at = utc_now() - timedelta(minutes=5)

        session = await repo._get_session()
        async with session.begin():
            stale_running_row = await session.get(AgentRun, stale_running.id)
            recent_running_row = await session.get(AgentRun, recent_running.id)
            stale_completed_row = await session.get(AgentRun, stale_completed.id)
            stale_running_row.started_at = old_at
            recent_running_row.started_at = recent_at
            stale_completed_row.started_at = old_at
            stale_completed_row.status = "completed"

        failed_ids = await repo.fail_stale_running_runs(utc_now() - timedelta(hours=1))

        assert failed_ids == [stale_running.id]
        stale_running_after = await repo.get_run(stale_running.id)
        recent_running_after = await repo.get_run(recent_running.id)
        stale_completed_after = await repo.get_run(stale_completed.id)
        assert stale_running_after.status == "failed"
        assert stale_running_after.completed_at is not None
        assert stale_running_after.error_json["code"] == "agent_run_stale"
        assert stale_running_after.metrics_json["error_count"] == 1
        assert recent_running_after.status == "running"
        assert stale_completed_after.status == "completed"

    @pytest.mark.asyncio
    async def test_unsupported_simple_rag_template_is_not_exposed(self, repo):
        await repo.seed_builtin_templates()

        template, version = await repo.get_template_with_current_version("simple_rag_agent")

        assert template is None
        assert version is None


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentRunService:
    @pytest.mark.asyncio
    async def test_run_thread_chat_falls_back_to_router_for_unsupported_simple_setting(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_context = {}

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": "simple_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                captured_context.update(agent_run_context or {})
                return {
                    "answer": "router fallback",
                    "document_sources": [{"id": "doc"}],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID
        assert result["agent_pattern_version"] == ROUTER_RAG_AGENT_VERSION
        assert captured_context["agent_run_id"] == result["agent_run_id"]
        assert run.status == "completed"
        assert run.metrics_json["document_source_count"] == 1
        assert run.resolved_spec_json["pattern_type"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_defaults_to_router_rag(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                return {
                    "answer": "router default",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_router_rag_when_selected(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": ROUTER_RAG_AGENT_ID}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "router ok",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [{"node": "router", "elapsed_ms": 3.5}],
                    "tool_events": [
                        {
                            "tool_name": "search_documents",
                            "caller_node": "retrieval_worker",
                            "ok": True,
                            "elapsed_ms": 9.25,
                            "warnings": [],
                        }
                    ],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID
        assert captured_spec["pattern_type"] == ROUTER_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.metrics_json["route"] == "direct"
        assert run.metrics_json["node_event_count"] == 1
        assert run.metrics_json["node_elapsed_ms"] == {"router": 3.5}
        assert run.metrics_json["tool_event_count"] == 1
        assert run.metrics_json["tool_elapsed_ms"] == 9.25

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_plan_execute_rag_when_selected(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": PLAN_EXECUTE_RAG_AGENT_ID}}

            async def fake_handle_plan_execute_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "plan execute ok",
                    "document_sources": [{"id": "doc"}],
                    "web_sources": [],
                    "used_chat_ids": ["turn-1:assistant"],
                    "clarification_options": None,
                    "route": "execute",
                    "node_events": [{"node": "planner", "elapsed_ms": 2.0, "execution_plan": ["retrieval_worker"]}],
                    "tool_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr(
                "app.agent_patterns.router_runtime.handle_plan_execute_rag_chat",
                fake_handle_plan_execute_rag_chat,
            )

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert result["agent_pattern_version"] == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert captured_spec["pattern_type"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.resolved_spec_json["pattern_type"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert run.metrics_json["route"] == "execute"
        assert run.metrics_json["node_elapsed_ms"] == {"planner": 2.0}
        assert run.metrics_json["document_source_count"] == 1
        assert run.metrics_json["used_chat_id_count"] == 1

    @pytest.mark.asyncio
    async def test_run_thread_chat_persists_failed_run_metrics(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": ROUTER_RAG_AGENT_ID}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                return {
                    "answer": "fallback",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "document",
                    "node_events": [{"node": "router", "elapsed_ms": 4.0}],
                    "tool_events": [],
                    "errors": [{"code": "router_rag_execution_failed"}],
                    "agent_error": {"code": "router_rag_execution_failed", "raw_message": "boom", "retryable": True},
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert run.status == "failed"
        assert run.error_json["code"] == "router_rag_execution_failed"
        assert run.metrics_json["route"] == "document"
        assert run.metrics_json["node_event_count"] == 1
        assert run.metrics_json["error_count"] == 1


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestRouterRagRuntime:
    @pytest.mark.asyncio
    async def test_handle_router_rag_chat_runs_compiled_direct_route_and_persists_turn(
        self,
        engine,
        sample_thread,
        monkeypatch,
        caplog,
    ):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        class FakeEmbeddingModel:
            async def aembed_query(self, query):
                return [0.1, 0.2, 0.3]

        class FakeVectorDb:
            async def search_knowledge_sources(self, **kwargs):
                return [
                    {
                        "text": "DiffusionBlocks is about modular diffusion model research.",
                        "file_hash": "file-1",
                        "chunk_id": 1,
                        "score": 0.9,
                    }
                ]

        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        content='{"route":"direct","reason":"prefetched context is sufficient","clarification_options":null}',
                        usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
                        response_metadata={"model_name": "test-llm"},
                    )
                return SimpleNamespace(
                    content="DiffusionBlocks focuses on modular diffusion model research.",
                    usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    response_metadata={
                        "model_name": "test-llm",
                        "token_usage": {
                            "completion_tokens_details": {"reasoning_tokens": 2},
                            "prompt_tokens_details": {"cached_tokens": 4},
                        },
                    },
                )

        fake_llm = FakeLlm()

        async def fake_get_recent_messages(_thread_id, limit):
            return []

        async def fake_get_thread_shape(_thread_id):
            return {
                "total_qa_pairs": 0,
                "total_qa_chars": 0,
                "documents": {
                    "file-1": {
                        "file_name": "diffusionblocks.pdf",
                        "source_type": "pdf",
                        "document_available_in_thread_at": "2026-07-02T00:00:00Z",
                        "chunk_count": 1,
                        "total_chars": 128,
                        "word_count": 18,
                        "page_count": 1,
                        "sentence_count": 1,
                    }
                },
            }

        async def fake_fetch_semantic_history(**kwargs):
            return "", []

        async def fake_get_document_metadata_lookup(_thread_id):
            return {
                "file-1": {
                    "file_name": "diffusionblocks.pdf",
                    "source_type": "pdf",
                    "document_available_in_thread_at": "2026-07-02T00:00:00Z",
                }
            }

        def fake_group_document_chunks(chunks, lookup, char_budget=None):
            return (
                "[Source: PDF: diffusionblocks.pdf]\nDiffusionBlocks is about modular diffusion model research.",
                [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}],
            )

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {"memory_compact_text": "Q/A compact"}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        async def fake_create_chat_turn(
            *,
            thread_id,
            question,
            answer,
            rewritten_question=None,
            status="completed",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=None,
            document_sources=None,
            used_chat_ids=None,
            clarification_options=None,
            error=None,
            metadata=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                status=status,
                payload={
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "reasoning_available": reasoning_available,
                    "reasoning_format": reasoning_format,
                    "web_sources": web_sources or [],
                    "document_sources": document_sources or [],
                    "used_chat_ids": used_chat_ids or [],
                    "clarification_options": clarification_options,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
            async with session_factory() as write_session:
                write_session.add(turn)
                await write_session.commit()
                await write_session.refresh(turn)
            return turn

        monkeypatch.setattr("app.rag.chat_service.get_embedding_model", lambda _name: FakeEmbeddingModel())
        monkeypatch.setattr("app.rag.chat_service.get_recent_messages", fake_get_recent_messages)
        monkeypatch.setattr("app.rag.chat_service.get_thread_shape", fake_get_thread_shape)
        monkeypatch.setattr("app.rag.chat_service.fetch_semantic_history", fake_fetch_semantic_history)
        monkeypatch.setattr("app.rag.chat_service.get_document_metadata_lookup", fake_get_document_metadata_lookup)
        monkeypatch.setattr("app.rag.chat_service.group_document_chunks", fake_group_document_chunks)
        monkeypatch.setattr("app.db.vector.get_vector_db", lambda: FakeVectorDb())
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        req = SimpleNamespace(
            question="What is this document about?",
            llm_model="test-llm",
            use_web_search=False,
            use_reranker=False,
            context_window=8192,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone="America/Chicago",
            client_locale="en-US",
            client_now_iso="2026-07-02T12:00:00.000Z",
        )

        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=builtin_router_rag_spec(),
            agent_run_context={
                "agent_run_id": "run-1",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["answer"] == "DiffusionBlocks focuses on modular diffusion model research."
        assert result["chat_turn_id"] == turn.id
        assert result["route"] == "direct"
        assert [event["node"] for event in result["node_events"]] == [
            "context_loader",
            "router",
            "direct_answer",
            "finalizer",
        ]
        context_event = result["node_events"][0]
        router_event = result["node_events"][1]
        answer_event = result["node_events"][2]
        assert context_event["status"] == "completed"
        assert context_event["output_refs"]["document_matches"][0]["file_hash"] == "file-1"
        assert context_event["output_refs"]["available_documents"][0]["file_hash"] == "file-1"
        assert router_event["prompt_summary"]["section"] == "Router Node Prompt"
        assert router_event["llm_result_summary"]["route"] == "direct"
        assert router_event["llm_result_summary"]["llm"]["model_name"] == "test-llm"
        assert router_event["llm_result_summary"]["llm"]["token_counts"] == {
            "prompt": 12,
            "completion": 5,
            "total": 17,
        }
        assert answer_event["prompt_summary"]["section"] == "Final Answer Prompt"
        assert answer_event["llm_result_summary"]["llm"]["token_counts"] == {
            "prompt": 20,
            "completion": 8,
            "total": 28,
            "reasoning": 2,
            "cached": 4,
        }
        assert answer_event["output_preview"]["answer"] == result["answer"]
        assert all(isinstance(event.get("elapsed_ms"), (int, float)) for event in result["node_events"])
        assert result["agent_run_id"] == "run-1"
        assert turn is not None
        assert turn.status == "completed"
        assert turn.payload["metadata"]["agent_run_id"] == "run-1"
        assert turn.payload["metadata"]["agent_route"] == "direct"
        assert turn.payload["metadata"]["agent_debug_trace"]["schema_version"] == 1
        assert "agent_node_events" not in turn.payload["metadata"]
        assert "agent_tool_events" not in turn.payload["metadata"]
        assert result["tool_events"] == []

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Router RAG run started | run_id=run-1" in log_text
        assert "Router RAG run completed | run_id=run-1" in log_text
        for node in ("context_loader", "router", "direct_answer", "finalizer"):
            assert f"Router RAG node completed | run_id=run-1" in log_text
            assert f"node={node}" in log_text
        assert "route=direct" in log_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "route, expected_nodes, expected_status",
        [
            ("document", ["context_loader", "router", "retrieval_worker", "synthesizer", "finalizer"], "completed"),
            ("memory", ["context_loader", "router", "memory_worker", "synthesizer", "finalizer"], "completed"),
            ("timeline", ["context_loader", "router", "timeline_worker", "synthesizer", "finalizer"], "completed"),
            ("web", ["context_loader", "router", "web_worker", "synthesizer", "finalizer"], "completed"),
            ("clarify", ["context_loader", "router", "finalizer"], "clarification"),
        ],
    )
    async def test_handle_router_rag_chat_covers_compiled_routes(
        self,
        engine,
        sample_thread,
        monkeypatch,
        caplog,
        route,
        expected_nodes,
        expected_status,
    ):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    options = '["Which uploaded document?","Which previous answer?"]' if route == "clarify" else "null"
                    return SimpleNamespace(
                        content=f'{{"route":"{route}","reason":"test route","clarification_options":{options}}}'
                    )
                return SimpleNamespace(content=f"Final answer from {route} route.")

        class FakeTool:
            def __init__(self, payload):
                self.payload = payload

            async def ainvoke(self, _args, config=None):
                return self.payload

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "Prior answer about DiffusionBlocks.",
                "document_evidence_text": "Document evidence about DiffusionBlocks.",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [{"file_name": "diffusionblocks.pdf", "file_hash": "file-1"}],
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
            }

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        async def fake_create_chat_turn(
            *,
            thread_id,
            question,
            answer,
            rewritten_question=None,
            status="completed",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=None,
            document_sources=None,
            used_chat_ids=None,
            clarification_options=None,
            error=None,
            metadata=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                status=status,
                payload={
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "reasoning_available": reasoning_available,
                    "reasoning_format": reasoning_format,
                    "web_sources": web_sources or [],
                    "document_sources": document_sources or [],
                    "used_chat_ids": used_chat_ids or [],
                    "clarification_options": clarification_options,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
            async with session_factory() as write_session:
                write_session.add(turn)
                await write_session.commit()
                await write_session.refresh(turn)
            return turn

        document_payload = {
            "content": "Document worker evidence.",
            "__document_sources__": [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}],
        }
        memory_payload = {
            "content": "Memory worker evidence.",
            "__used_chat_ids__": ["turn-1"],
        }
        timeline_payload = {
            "content": "Timeline worker evidence.",
            "__timeline_events__": [{"timeline_event_type": "document_added", "timeline_event_at": "2026-07-01T00:00:00Z"}],
        }
        web_payload = {
            "content": "Web worker evidence.",
            "__web_sources__": [{"url": "https://example.com", "title": "Example"}],
        }
        fake_llm = FakeLlm()

        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.graph.search_documents", FakeTool(document_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_conversation_history", FakeTool(memory_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_thread_timeline", FakeTool(timeline_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_web", FakeTool(web_payload))
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        req = SimpleNamespace(
            question="Route coverage?",
            llm_model="test-llm",
            use_web_search=route == "web",
            use_reranker=False,
            context_window=8192,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone=None,
            client_locale=None,
            client_now_iso=None,
        )

        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=builtin_router_rag_spec(),
            agent_run_context={
                "agent_run_id": f"run-{route}",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["route"] == route
        assert [event["node"] for event in result["node_events"]] == expected_nodes
        assert all(isinstance(event.get("elapsed_ms"), (int, float)) for event in result["node_events"])
        assert turn is not None
        assert turn.status == expected_status
        assert turn.payload["metadata"]["agent_route"] == route
        assert turn.payload["metadata"]["agent_debug_trace"]["schema_version"] == 1
        assert "agent_node_events" not in turn.payload["metadata"]
        assert "agent_tool_events" not in turn.payload["metadata"]
        if route == "clarify":
            assert result["tool_events"] == []
        else:
            assert len(result["tool_events"]) == 1
            assert result["tool_events"][0]["caller_node"] == expected_nodes[2]
            assert result["tool_events"][0]["ok"] is True
            assert result["tool_events"][0]["result_preview"].endswith("worker evidence.")
        if route == "document":
            assert result["tool_events"][0]["tool_name"] == "search_documents"
            assert result["tool_events"][0]["tool_input"]["query"] == "Route coverage?"
            assert result["document_sources"] == [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}]
            assert result["answer"] == "Final answer from document route."
        elif route == "memory":
            assert result["tool_events"][0]["tool_name"] == "search_conversation_history"
            assert result["tool_events"][0]["tool_input"]["query"] == "Route coverage?"
            assert result["used_chat_ids"] == ["turn-1"]
            assert result["answer"] == "Final answer from memory route."
        elif route == "timeline":
            assert result["tool_events"][0]["tool_name"] == "search_thread_timeline"
            assert result["tool_events"][0]["tool_input"]["query"] == "Route coverage?"
            assert result["answer"] == "Final answer from timeline route."
        elif route == "web":
            assert result["tool_events"][0]["tool_name"] == "search_web"
            assert result["tool_events"][0]["tool_input"] == "Route coverage?"
            assert result["web_sources"] == [{"url": "https://example.com", "title": "Example"}]
            assert result["answer"] == "Final answer from web route."
        else:
            assert result["clarification_options"] == ["Which uploaded document?", "Which previous answer?"]
            assert result["answer"].startswith("I need a bit more clarification.")

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert f"Router RAG run completed | run_id=run-{route}" in log_text
        assert f"route={route}" in log_text
        for node in expected_nodes:
            assert f"node={node}" in log_text

    @pytest.mark.asyncio
    async def test_handle_router_rag_chat_failed_run_keeps_partial_telemetry(
        self,
        engine,
        sample_thread,
        monkeypatch,
    ):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        class FakeLlm:
            async def ainvoke(self, messages):
                return SimpleNamespace(content='{"route":"document","reason":"needs docs","clarification_options":null}')

        class FailingTool:
            async def ainvoke(self, _args, config=None):
                raise RuntimeError("document tool exploded")

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "",
                "document_evidence_text": "",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [{"file_name": "paper.pdf", "file_hash": "file-1"}],
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
            }

        async def fake_create_chat_turn(
            *,
            thread_id,
            question,
            answer,
            rewritten_question=None,
            status="completed",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=None,
            document_sources=None,
            used_chat_ids=None,
            clarification_options=None,
            error=None,
            metadata=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                status=status,
                payload={
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "reasoning_available": reasoning_available,
                    "reasoning_format": reasoning_format,
                    "web_sources": web_sources or [],
                    "document_sources": document_sources or [],
                    "used_chat_ids": used_chat_ids or [],
                    "clarification_options": clarification_options,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
            async with session_factory() as write_session:
                write_session.add(turn)
                await write_session.commit()
                await write_session.refresh(turn)
            return turn

        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: FakeLlm())
        monkeypatch.setattr("app.agent_patterns.graph.search_documents", FailingTool())
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)

        req = SimpleNamespace(
            question="What is in the document?",
            llm_model="test-llm",
            use_web_search=False,
            use_reranker=False,
            context_window=8192,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone=None,
            client_locale=None,
            client_now_iso=None,
        )

        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=builtin_router_rag_spec(),
            agent_run_context={
                "agent_run_id": "run-failed",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["agent_error"]["code"] == "router_rag_execution_failed"
        assert result["chat_turn_id"] == turn.id
        assert result["route"] == "document"
        assert [event["node"] for event in result["node_events"]] == ["context_loader", "router"]
        assert result["tool_events"] == []
        assert result["errors"][0]["raw_message"] == "document tool exploded"
        assert turn.status == "failed"
        assert turn.payload["metadata"]["agent_run_id"] == "run-failed"
        assert turn.payload["metadata"]["agent_route"] == "document"
        assert turn.payload["metadata"]["agent_debug_trace"]["schema_version"] == 1
        assert "agent_node_events" not in turn.payload["metadata"]
        assert "agent_tool_events" not in turn.payload["metadata"]
        assert turn.payload["metadata"]["agent_error"]["raw_message"] == "document tool exploded"


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentPatternApi:
    def test_list_and_get_builtin_agent_pattern(self, api_client):
        listed = api_client.get("/api/agent-patterns")
        assert listed.status_code == 200
        assert {item["id"] for item in listed.json()["agent_patterns"]} == {
            ROUTER_RAG_AGENT_ID,
            PLAN_EXECUTE_RAG_AGENT_ID,
        }

        detail = api_client.get(f"/api/agent-patterns/{ROUTER_RAG_AGENT_ID}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["agent_pattern"]["id"] == ROUTER_RAG_AGENT_ID
        assert payload["current_version"]["version"] == ROUTER_RAG_AGENT_VERSION
        assert payload["current_version"]["validation"]["valid"] is True
        assert "document_evidence" in payload["capabilities"]["required_tool_ids"]
        assert payload["capabilities"]["node_tool_requirements"]["retrieval_worker"] == "document_evidence"

        plan_detail = api_client.get(f"/api/agent-patterns/{PLAN_EXECUTE_RAG_AGENT_ID}")
        assert plan_detail.status_code == 200
        plan_payload = plan_detail.json()
        assert plan_payload["agent_pattern"]["id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert plan_payload["current_version"]["version"] == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert plan_payload["current_version"]["validation"]["valid"] is True
        assert plan_payload["capabilities"]["node_tool_requirements"]["planner"] == "clarify_intent"

        stale_detail = api_client.get("/api/agent-patterns/simple_rag_agent")
        assert stale_detail.status_code == 404

    def test_validate_agent_pattern_endpoint(self, api_client):
        valid = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": builtin_router_rag_spec()},
        )
        invalid_spec = builtin_router_rag_spec()
        invalid_spec["config"]["allowed_tool_ids"] = ["mystery_tool"]
        stale_spec = builtin_router_rag_spec()
        stale_spec["pattern_type"] = "simple_rag_agent"
        invalid = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": invalid_spec},
        )
        stale = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": stale_spec},
        )

        assert valid.status_code == 200
        valid_payload = valid.json()
        assert valid_payload["valid"] is True
        assert valid_payload["errors"] == []
        assert valid_payload["pattern_type"] == ROUTER_RAG_AGENT_ID
        assert "document_evidence" in valid_payload["required_tool_ids"]
        assert invalid.status_code == 200
        invalid_payload = invalid.json()
        assert invalid_payload["valid"] is False
        assert invalid_payload["unknown_allowed_tool_ids"] == ["mystery_tool"]
        assert "document_evidence" in invalid_payload["missing_required_tool_ids"]
        assert stale.status_code == 200
        assert stale.json()["valid"] is False

    def test_validate_thread_agent_config_endpoint_resolves_without_running_chat(self, api_client, sample_thread):
        response = api_client.post(
            f"/api/threads/{sample_thread.id}/agent-config/validate",
            json={"overrides": {"use_web_search": True, "max_iterations": 2}},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is True
        assert payload["template_id"] == ROUTER_RAG_AGENT_ID
        assert payload["template_version"] == ROUTER_RAG_AGENT_VERSION
        assert payload["validation"]["valid"] is True
        assert payload["resolved_spec_json"]["config"]["use_web_search"] is True
        assert payload["resolved_spec_json"]["config"]["max_iterations"] == 2

    def test_validate_thread_agent_config_endpoint_reports_invalid_overrides(self, api_client, sample_thread):
        response = api_client.post(
            f"/api/threads/{sample_thread.id}/agent-config/validate",
            json={"overrides": {"allowed_tool_ids": ["mystery_tool"]}},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is False
        assert payload["template_id"] == ROUTER_RAG_AGENT_ID
        assert payload["validation"]["valid"] is False
        assert payload["validation"]["unknown_allowed_tool_ids"] == ["mystery_tool"]

    @pytest.mark.asyncio
    async def test_list_thread_agent_runs_returns_recent_compact_summaries(self, api_client, engine, sample_thread):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            first = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            await repo.complete_run(
                first.id,
                status="completed",
                metrics_json={"duration_ms": 10.0, "route": "direct", "node_event_count": 2},
            )
            second = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            await repo.complete_run(
                second.id,
                status="failed",
                metrics_json={"duration_ms": 5.0, "route": "document", "error_count": 1},
                error_json={"code": "router_rag_execution_failed", "raw_message": "boom", "retryable": True},
            )

        response = api_client.get(f"/api/threads/{sample_thread.id}/agent-runs?limit=1")

        assert response.status_code == 200
        payload = response.json()
        assert payload["thread_id"] == sample_thread.id
        assert payload["limit"] == 1
        assert len(payload["agent_runs"]) == 1
        summary = payload["agent_runs"][0]
        assert summary["id"] == second.id
        assert summary["status"] == "failed"
        assert summary["metrics"]["route"] == "document"
        assert summary["metrics"]["error_count"] == 1
        assert summary["error"]["code"] == "router_rag_execution_failed"
        assert "resolved_spec_json" not in summary

    def test_list_thread_agent_runs_returns_404_for_missing_thread(self, api_client):
        response = api_client.get(f"/api/threads/{uuid.uuid4()}/agent-runs")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_agent_run_includes_debug_telemetry(self, api_client, engine, sample_thread):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )

        turn_id = str(uuid.uuid4())
        node_telemetry = [{"node": "router", "route": "web", "elapsed_ms": 4.5}]
        tool_telemetry = [
            {
                "tool_name": "search_web",
                "caller_node": "web_worker",
                "ok": True,
                "elapsed_ms": 12.3,
                "warnings": [],
            }
        ]
        debug_trace = build_debug_trace(
            run=run,
            chat_turn=SimpleNamespace(id=turn_id),
            node_events=node_telemetry,
            tool_events=tool_telemetry,
            metrics={
                "duration_ms": 42.0,
                "route": "web",
                "node_event_count": 1,
                "node_elapsed_ms": {"router": 4.5},
                "node_total_elapsed_ms": 4.5,
                "tool_event_count": 1,
                "tool_warning_count": 0,
                "tool_error_count": 0,
                "tool_elapsed_ms": 12.3,
            },
            route="web",
            route_reason="Needs live evidence.",
        )

        turn = ChatTurn(
            id=turn_id,
            thread_id=sample_thread.id,
            status="completed",
            payload={
                "question": "What happened?",
                "answer": "Answer",
                "metadata": {
                    "agent_run_id": run.id,
                    "agent_route": "web",
                    "agent_route_reason": "Needs live evidence.",
                    "agent_debug_trace": debug_trace,
                },
            },
        )
        async with session_factory() as write_session:
            write_session.add(turn)
            await write_session.commit()
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.complete_run(
                run.id,
                status="completed",
                metrics_json={
                    "duration_ms": 42.0,
                    "route": "web",
                    "node_event_count": 1,
                    "node_elapsed_ms": {"router": 4.5},
                    "node_total_elapsed_ms": 4.5,
                    "tool_event_count": 1,
                    "tool_warning_count": 0,
                    "tool_error_count": 0,
                    "tool_elapsed_ms": 12.3,
                },
                chat_turn_id=turn.id,
            )

        response = api_client.get(f"/api/agent-runs/{run.id}")

        assert response.status_code == 200
        payload = response.json()["agent_run"]
        assert payload["id"] == run.id
        assert payload["chat_turn_id"] == turn.id
        assert payload["metrics_json"]["tool_event_count"] == 1
        assert set(payload["debug"]) == {"trace"}
        assert "node_events" not in payload["debug"]
        assert "tool_events" not in payload["debug"]
        trace = payload["debug"]["trace"]
        assert trace["metrics"]["duration_ms"] == 42.0
        assert trace["metrics"]["route"] == "web"
        assert trace["metrics"]["node_event_count"] == 1
        assert trace["metrics"]["node_elapsed_ms"] == {"router": 4.5}
        assert trace["metrics"]["tool_elapsed_ms"] == 12.3
        assert trace["schema_version"] == 1
        assert trace["trace_id"] == run.id
        assert trace["run_id"] == run.id
        assert trace["thread_id"] == sample_thread.id
        assert trace["chat_turn_id"] == turn.id
        assert trace["template_id"] == ROUTER_RAG_AGENT_ID
        assert trace["pattern_type"] == ROUTER_RAG_AGENT_ID
        assert trace["attributes"]["session.id"] == sample_thread.id
        assert trace["attributes"]["askpdf.route"] == "web"
        assert trace["attributes"]["askpdf.route_reason"] == "Needs live evidence."
        span_ids = {span["span_id"]: span for span in trace["spans"]}
        assert span_ids[f"run:{run.id}"]["kind"] == "AGENT"
        assert span_ids["node:router:0"]["kind"] == "AGENT"
        assert span_ids["node:router:0"]["raw"] == {"node": "router", "route": "web", "elapsed_ms": 4.5}
        assert span_ids["node:router:0"]["parent_span_id"] == f"run:{run.id}"
        assert span_ids["node:router:0"]["attributes"]["askpdf.route"] == "web"
        assert any(event["name"] == "decision.made" for event in span_ids["node:router:0"]["events"])
        assert span_ids["tool:search_web:0"]["parent_span_id"] == f"run:{run.id}"
        assert span_ids["tool:search_web:0"]["attributes"]["tool.id"] == "live_web_recon"
        assert span_ids["tool:search_web:0"]["raw"]["tool_name"] == "search_web"
        assert span_ids["tool:search_web:0"]["raw"]["tool_id"] == "live_web_recon"
        assert span_ids["tool:search_web:0"]["raw"]["tool_category"] == "web"
        assert span_ids["tool:search_web:0"]["raw"]["tool_display_name"] == "Internet Search"
        assert span_ids["tool:search_web:0"]["raw"]["artifact_keys"] == ["web_sources"]
        assert "web_search_disabled" in span_ids["tool:search_web:0"]["raw"]["known_warning_codes"]
        assert any(event["name"] == "tool.completed" for event in span_ids["tool:search_web:0"]["events"])

    @pytest.mark.asyncio
    async def test_get_failed_agent_run_without_turn_includes_error_debug(self, api_client, engine, sample_thread):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            await repo.complete_run(
                run.id,
                status="failed",
                metrics_json={
                    "duration_ms": 3.0,
                    "route": None,
                    "node_event_count": 0,
                    "tool_event_count": 0,
                    "tool_warning_count": 0,
                    "tool_error_count": 0,
                    "error_count": 1,
                },
                error_json={"code": "agent_run_failed", "raw_message": "compile failed", "retryable": True},
            )

        response = api_client.get(f"/api/agent-runs/{run.id}")

        assert response.status_code == 200
        payload = response.json()["agent_run"]
        assert payload["status"] == "failed"
        assert set(payload["debug"]) == {"trace"}
        trace = payload["debug"]["trace"]
        assert trace["schema_version"] == 1
        assert trace["status"] == "failed"
        assert trace["chat_turn_id"] is None
        assert trace["metrics"]["error_count"] == 1
        assert "raw" not in trace
        root_span = trace["spans"][0]
        assert root_span["span_id"] == f"run:{run.id}"
        assert root_span["status"] == "failed"
        assert root_span["events"][0]["name"] == "exception"
        assert root_span["events"][0]["attributes"]["askpdf.error.code"] == "agent_run_failed"

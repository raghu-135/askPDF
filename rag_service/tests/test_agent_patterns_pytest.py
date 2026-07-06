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

from app.agent_patterns.checkpointing import open_agent_checkpointer
from app.agent_patterns.router_runtime import handle_router_rag_chat
from app.agent_patterns.graph import NodeRegistry, TemplateCompiler, _llm_result_metadata
from app.agent_patterns.graph import (
    build_planner_prompt,
    infer_required_plan_steps,
    normalize_execution_plan,
    normalize_evaluator_report,
)
from app.agent_patterns.debug_trace import AgentTraceRecorder, build_debug_payload, build_debug_trace, build_runtime_trace_event
from app.agent_patterns.metrics import build_run_metrics
from app.agent_patterns.repository import AgentPatternRepository, AgentRunInterruptError
from app.agent_patterns.service import AgentRunService
from app.agent_patterns.templates import (
    EVALUATOR_REPLANNER_RAG_AGENT_ID,
    EVALUATOR_REPLANNER_RAG_AGENT_VERSION,
    EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION,
    PLAN_EXECUTE_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_AGENT_VERSION,
    PLAN_EXECUTE_RAG_AGENT_V2_VERSION,
    ROUTER_RAG_AGENT_ID,
    ROUTER_RAG_AGENT_VERSION,
    ROUTER_RAG_AGENT_V2_VERSION,
    builtin_evaluator_replanner_rag_spec,
    builtin_evaluator_replanner_rag_v2_spec,
    builtin_plan_execute_rag_spec,
    builtin_plan_execute_rag_v2_spec,
    builtin_router_rag_hitl_web_spec,
    builtin_router_rag_spec,
    builtin_router_rag_v2_spec,
)
from app.agent_patterns.validator import TemplateResolver, TemplateValidationError, TemplateValidator
from app.db.models_sqlmodel import AgentPatternTemplate, AgentPatternTemplateVersion, AgentRun, ChatTurn, Thread
from app.models.llm_server_client import REPLANS_LIMIT
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z, utc_now


SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
TRACE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "agent_debug_trace_v1.schema.json"


def make_trace_recorder(run_id: str, thread_id: str, spec: dict, template_id: str = ROUTER_RAG_AGENT_ID) -> AgentTraceRecorder:
    return AgentTraceRecorder(
        SimpleNamespace(
            id=run_id,
            thread_id=thread_id,
            user_id=None,
            template_id=template_id,
            template_version_id=f"{template_id}:v1",
            resolved_spec_json=spec,
            status="running",
            started_at=utc_now(),
            completed_at=None,
        )
    )


async def create_agent_run_record(
    session_factory,
    *,
    run_id: str,
    thread_id: str,
    spec: dict,
    template_id: str = ROUTER_RAG_AGENT_ID,
) -> AgentRun:
    async with session_factory() as repo_session:
        repo = AgentPatternRepository(repo_session)
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(template_id)
        run = AgentRun(
            id=run_id,
            thread_id=thread_id,
            template_id=template.id,
            run_metadata_json={"template_version_id": version.id, "template_version": version.version},
            resolved_spec_json=spec,
            status="running",
            started_at=utc_now(),
        )
        repo_session.add(run)
        await repo_session.commit()
        return run


class TestAgentCheckpointing:
    @pytest.mark.asyncio
    async def test_explicit_postgres_checkpointer_fails_without_database_url(self, monkeypatch):
        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "postgres")
        monkeypatch.delenv("AGENT_CHECKPOINT_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("ASKPDF_AGENT_CHECKPOINTER_ALLOW_MEMORY_FALLBACK", raising=False)

        with pytest.raises(RuntimeError, match="requires AGENT_CHECKPOINT_DATABASE_URL or DATABASE_URL"):
            async with open_agent_checkpointer():
                pass


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

    def test_build_debug_trace_bounds_preview_values_and_raw_events(self):
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
        assert len(node_span["raw"]["output_refs"]["document_matches"][0]["text"]) <= 903

    def test_debug_payload_redacts_sensitive_values_and_bounds_large_values(self):
        long_text = "secret-adjacent context " * 1000
        run = SimpleNamespace(
            id="run-redact",
            thread_id="thread-1",
            user_id="user-1",
            template_id=ROUTER_RAG_AGENT_ID,
            template_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_router_rag_spec(),
            status="completed",
            started_at=utc_now(),
            completed_at=utc_now(),
        )

        payload = build_debug_payload(
            run=run,
            chat_turn_id="turn-1",
            node_events=[
                {
                    "node": "router",
                    "status": "completed",
                    "input_preview": {
                        "question": long_text,
                        "authorization": "Bearer should-not-survive",
                    },
                    "output_preview": {"answer": long_text, "api_key": "sk-should-not-survive"},
                    "prompt_summary": {
                        "section": "Router",
                        "system_message": long_text,
                        "preview": long_text,
                    },
                }
            ],
            tool_events=[
                {
                    "tool_name": "search_web",
                    "caller_node": "router",
                    "ok": True,
                    "tool_input": {"query": "x", "access_token": "token-should-not-survive"},
                    "result_preview": long_text,
                    "artifact_summary": {"cookie": "cookie-should-not-survive", "text": long_text},
                }
            ],
            metrics={"duration_ms": 1.0, "route": "web", "tool_warning_count": 0, "error_count": 0},
            route="web",
        )

        encoded = json.dumps(payload, ensure_ascii=True, default=str)
        assert "should-not-survive" not in encoded
        router_span = next(span for span in payload["trace"]["spans"] if span["span_id"] == "node:router:0")
        assert len(router_span["input"]["value"]["question"]) <= 903
        assert len(router_span["output"]["value"]["answer"]) <= 903

    def test_runtime_trace_redaction_preserves_token_usage_counters(self):
        event = build_runtime_trace_event(
            "llm.completed",
            attributes={
                "access_token": "should-not-survive",
                "token_usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            },
        )

        encoded = json.dumps(event, ensure_ascii=True, default=str)
        assert "should-not-survive" not in encoded
        assert event["attributes"]["access_token"] == "[redacted]"
        assert event["attributes"]["token_usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

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

    def test_interrupt_trace_events_live_on_root_span_and_are_summarized(self):
        run = SimpleNamespace(
            id="run-hitl",
            thread_id="thread-1",
            user_id="user-1",
            template_id=ROUTER_RAG_AGENT_ID,
            template_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
            resolved_spec_json=builtin_router_rag_spec(),
            status="awaiting_human",
            started_at=utc_now(),
            completed_at=None,
        )
        recorder = AgentTraceRecorder(run)
        recorder.record_interrupt_event(
            {
                "interrupt_id": "interrupt-trace-1",
                "gate_id": "before_web",
                "node_id": "web_worker",
                "type": "tool_approval",
                "status": "pending",
                "requested_at": "2026-07-05T12:00:00Z",
                "expires_at": "2026-07-05T12:05:00Z",
                "resume_token": "not-for-trace",
                "resume_version": 1,
                "allowed_actions": ["approve", "reject"],
                "title": "Approve web search?",
                "input_summary": {"query": "latest evidence"},
                "proposed_tool": {"name": "search_web"},
            }
        )

        payload = recorder.finalize(
            run=run,
            chat_turn_id=None,
            metrics={"duration_ms": 1.0, "tool_warning_count": 0, "error_count": 0},
        )
        trace = payload["trace"]
        root_span = next(span for span in trace["spans"] if span["span_id"] == "run:run-hitl")
        event = next(item for item in root_span["events"] if item["name"] == "interrupt.requested")

        assert event["attributes"]["askpdf.interrupt.id"] == "interrupt-trace-1"
        assert event["attributes"]["askpdf.interrupt.gate_id"] == "before_web"
        assert event["attributes"]["askpdf.node.id"] == "web_worker"
        assert event["input"]["title"] == "Approve web search?"
        assert event["output"]["proposed_tool"] == {"name": "search_web"}
        assert "resume_token" not in json.dumps(event)
        assert payload["summary"]["interruptCount"] == 1
        assert payload["summary"]["lastInterruptStatus"] == "pending"


class TestRouterRagTemplateValidator:
    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda spec: spec.update({"pattern_type": "simple_rag_agent"}), "pattern_type must be one of:"),
            (lambda spec: spec["config"].update({"surprise": True}), "unknown config keys: surprise"),
            (lambda spec: spec["config"].update({"allowed_tool_ids": ["not_a_tool"]}), "unknown allowed_tool_ids: not_a_tool"),
            (lambda spec: spec["config"].update({"replans": 999}), "replans is only supported"),
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
            thread_settings={"replans": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )

        assert "replans" not in resolved["config"]
        assert resolved["config"]["use_reranker"] is False
        assert resolved["config"]["use_web_search"] is True

        evaluator_resolved = TemplateResolver().resolve(
            builtin_evaluator_replanner_rag_spec(),
            thread_settings={"replans": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )
        assert evaluator_resolved["config"]["replans"] == 3

    def test_rejects_zero_replan_budget(self):
        spec = builtin_evaluator_replanner_rag_spec()
        spec["config"]["replans"] = 0

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "replans must be between" in str(exc.value)

    def test_accepts_builtin_router_rag_spec(self):
        result = TemplateValidator().validate(builtin_router_rag_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_router_rag_hitl_web_spec(self):
        result = TemplateValidator().validate(builtin_router_rag_hitl_web_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_plan_execute_rag_spec(self):
        result = TemplateValidator().validate(builtin_plan_execute_rag_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_evaluator_replanner_rag_spec(self):
        result = TemplateValidator().validate(builtin_evaluator_replanner_rag_spec())

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

    def test_compiles_builtin_router_rag_hitl_web_spec(self):
        graph = TemplateCompiler().compile(builtin_router_rag_hitl_web_spec())

        assert graph is not None

    def test_materializes_generic_hitl_gate_overlay_for_action_node(self):
        spec = builtin_router_rag_spec()
        spec["config"]["hitl_policy"] = {
            "enabled": True,
            "gates": {
                "review_before_documents": {
                    "enabled": True,
                    "mode": "approval",
                    "phase": "before",
                    "target": {"node_id": "retrieval_worker"},
                    "allowed_actions": ["approve", "continue_without"],
                    "default_action": "continue_without",
                    "routes": {"approve": "retrieval_worker", "continue_without": "synthesizer"},
                }
            },
        }

        TemplateValidator().validate(spec)
        materialized = TemplateCompiler().materialize_spec(spec)
        TemplateValidator().validate(materialized)
        graph_spec = materialized["config"]["graph"]

        assert {"id": "review_before_documents", "type": "hitl_gate"} in graph_spec["nodes"]
        router_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "router")
        assert router_edge["routes"]["document"] == "review_before_documents"
        gate_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "review_before_documents")
        assert gate_edge["routes"] == {"approve": "retrieval_worker", "continue_without": "synthesizer"}

    def test_materializes_multi_select_choice_gate_overlay(self):
        spec = builtin_plan_execute_rag_spec()
        spec["config"]["hitl_policy"] = {
            "enabled": True,
            "gates": {
                "research_source_choice": {
                    "enabled": True,
                    "mode": "choice",
                    "phase": "before",
                    "target": {"node_id": "retrieval_worker"},
                    "selection_mode": "multi",
                    "allowed_actions": ["approve_selected", "continue_without", "reject"],
                    "default_action": "continue_without",
                    "options": [
                        {"id": "document_search", "label": "Document search", "target_node_id": "retrieval_worker"},
                        {"id": "web_search", "label": "Web search", "target_node_id": "web_worker"},
                    ],
                    "routes": {
                        "continue_without": "synthesizer",
                        "reject": "synthesizer",
                    },
                }
            },
        }

        TemplateValidator().validate(spec)
        materialized = TemplateCompiler().materialize_spec(spec)
        TemplateValidator().validate(materialized)
        graph_spec = materialized["config"]["graph"]

        assert {"id": "research_source_choice", "type": "hitl_gate"} in graph_spec["nodes"]
        planner_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "planner")
        assert planner_edge["routes"]["execute"] == "research_source_choice"
        gate_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "research_source_choice")
        assert gate_edge["routes"] == {
            "document_search": "retrieval_worker",
            "web_search": "web_worker",
            "continue_without": "synthesizer",
            "reject": "synthesizer",
        }

    def test_materializes_final_review_as_hitl_policy_overlay(self):
        spec = builtin_router_rag_spec()
        spec["config"]["hitl_policy"] = {
            "enabled": True,
            "gates": {
                "human_review_gate": {
                    "enabled": True,
                    "mode": "review",
                    "phase": "after",
                    "target": {"node_id": "finalizer", "node_type": "finalizer"},
                    "allowed_actions": ["approve", "edit", "continue_without"],
                    "default_action": "approve",
                    "routes": {"approve": "END", "edit": "END", "continue_without": "END"},
                    "editable_fields": ["final_answer"],
                },
            },
        }
        materialized = TemplateCompiler().materialize_spec(spec)
        TemplateValidator().validate(materialized)
        config = materialized["config"]
        graph_spec = config["graph"]

        assert config["hitl_policy"]["enabled"] is True
        assert config["hitl_policy"]["gates"]["human_review_gate"]["mode"] == "review"
        assert {"id": "human_review_gate", "type": "hitl_gate"} in graph_spec["nodes"]
        assert {"from": "finalizer", "to": "human_review_gate"} in graph_spec["edges"]
        gate_edge = next(
            edge
            for edge in graph_spec["edges"]
            if edge.get("from") == "human_review_gate" and edge.get("conditional") is True
        )
        assert gate_edge["routes"] == {
            "approve": "END",
            "edit": "END",
            "continue_without": "END",
        }

    def test_compiles_builtin_plan_execute_rag_spec(self):
        graph = TemplateCompiler().compile(builtin_plan_execute_rag_spec())

        assert graph is not None

    def test_compiles_builtin_evaluator_replanner_rag_spec(self):
        graph = TemplateCompiler().compile(builtin_evaluator_replanner_rag_spec())

        assert graph is not None

    def test_rejects_plan_execute_graph_topology_changes(self):
        spec = builtin_plan_execute_rag_spec()
        spec["config"]["graph"]["edges"].append({"from": "planner", "to": "synthesizer"})

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "plan_execute_rag_agent graph edges must match" in str(exc.value)

    def test_rejects_evaluator_replanner_graph_topology_changes(self):
        spec = builtin_evaluator_replanner_rag_spec()
        spec["config"]["graph"]["edges"].append({"from": "evidence_evaluator", "to": "finalizer"})

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "evaluator_replanner_rag_agent graph edges must match" in str(exc.value)

    def test_rejects_evaluator_replanner_unbounded_replans(self):
        spec = builtin_evaluator_replanner_rag_spec()
        spec["config"]["replans"] = REPLANS_LIMIT + 1

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "replans must be between" in str(exc.value)

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

    def test_normalize_evaluator_report_bounds_payload(self):
        report = normalize_evaluator_report(
            {
                "sufficient": False,
                "confidence": 2,
                "missing_evidence": ["missing citations"] * 10,
                "citation_risk": "severe",
                "contradiction_risk": "high",
                "recommended_next_steps": ["search documents"] * 10,
                "reason": "x" * 1000,
            },
            {"evidence": ""},
        )

        assert report["sufficient"] is False
        assert report["confidence"] == 1.0
        assert len(report["missing_evidence"]) == 5
        assert report["citation_risk"] == "medium"
        assert report["contradiction_risk"] == "high"
        assert len(report["recommended_next_steps"]) == 5
        assert len(report["reason"]) <= 503

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

    def test_v2_custom_graph_validates_and_compiles_with_instance_ids(self):
        spec = {
            "schema_version": 2,
            "pattern_type": "custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "router_1", "type": "router"},
                        {"id": "retrieval_1", "type": "retrieval_worker"},
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "router_1"},
                        {
                            "from": "router_1",
                            "conditional": True,
                            "route_fn": "router_route",
                            "routes": {
                                "document": "retrieval_1",
                                "clarify": "final_1",
                            },
                        },
                        {"from": "retrieval_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        assert TemplateValidator().validate(spec)["valid"] is True
        assert TemplateCompiler().compile(spec) is not None

    @pytest.mark.parametrize(
        "edge_update,match",
        [
            ({"route_fn": None}, "must declare route_fn"),
            ({"route_fn": "evaluator_route"}, "route_fn evaluator_route is not allowed"),
            ({"routes": {"document": "missing_node"}}, "target is unknown: missing_node"),
        ],
    )
    def test_v2_custom_graph_rejects_unsafe_conditional_edges(self, edge_update, match):
        edge = {
            "from": "router_1",
            "conditional": True,
            "route_fn": "router_route",
            "routes": {"document": "retrieval_1"},
        }
        if edge_update.get("route_fn") is None:
            edge.pop("route_fn")
        else:
            edge.update(edge_update)
        if "routes" in edge_update:
            edge["routes"] = edge_update["routes"]
        spec = {
            "schema_version": 2,
            "pattern_type": "custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "router_1", "type": "router"},
                        {"id": "retrieval_1", "type": "retrieval_worker"},
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "router_1"},
                        edge,
                        {"from": "retrieval_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        with pytest.raises(TemplateValidationError, match=match):
            TemplateValidator().validate(spec)

    def test_v2_custom_graph_rejects_tool_ids_not_supported_by_graph_nodes(self):
        spec = {
            "schema_version": 2,
            "pattern_type": "custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "router_1", "type": "router"},
                        {"id": "memory_1", "type": "memory_worker"},
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "router_1"},
                        {
                            "from": "router_1",
                            "conditional": True,
                            "route_fn": "router_route",
                            "routes": {"memory": "memory_1"},
                        },
                        {"from": "memory_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        with pytest.raises(TemplateValidationError, match="not supported by any node"):
            TemplateValidator().validate(spec)

    def test_v2_custom_graph_rejects_unbounded_cycles(self):
        spec = {
            "schema_version": 2,
            "pattern_type": "custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "planner_1", "type": "planner"},
                        {"id": "retrieval_1", "type": "retrieval_worker"},
                        {"id": "evaluator_1", "type": "evidence_evaluator"},
                        {"id": "replanner_1", "type": "replanner"},
                        {"id": "synth_1", "type": "synthesizer"},
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "planner_1"},
                        {
                            "from": "planner_1",
                            "conditional": True,
                            "route_fn": "planner_route",
                            "routes": {"execute": "retrieval_1", "direct": "final_1", "clarify": "final_1"},
                        },
                        {"from": "retrieval_1", "to": "evaluator_1"},
                        {
                            "from": "evaluator_1",
                            "conditional": True,
                            "route_fn": "evaluator_route",
                            "routes": {"answer": "synth_1", "replan": "replanner_1", "answer_budget_exhausted": "synth_1"},
                        },
                        {"from": "replanner_1", "to": "retrieval_1"},
                        {"from": "synth_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        with pytest.raises(TemplateValidationError, match="requires loop_policy"):
            TemplateValidator().validate(spec)

    def test_v2_custom_graph_accepts_bounded_cycles(self):
        spec = {
            "schema_version": 2,
            "pattern_type": "custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                "loop_policy": {
                    "max_total_visits": 9,
                    "default_max_node_visits": 1,
                    "node_visit_limits": {
                        "retrieval_1": 2,
                        "evaluator_1": 2,
                    },
                },
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "planner_1", "type": "planner"},
                        {"id": "retrieval_1", "type": "retrieval_worker"},
                        {"id": "evaluator_1", "type": "evidence_evaluator"},
                        {"id": "replanner_1", "type": "replanner"},
                        {"id": "synth_1", "type": "synthesizer"},
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "planner_1"},
                        {
                            "from": "planner_1",
                            "conditional": True,
                            "route_fn": "planner_route",
                            "routes": {"execute": "retrieval_1", "direct": "final_1", "clarify": "final_1"},
                        },
                        {"from": "retrieval_1", "to": "evaluator_1"},
                        {
                            "from": "evaluator_1",
                            "conditional": True,
                            "route_fn": "evaluator_route",
                            "routes": {"answer": "synth_1", "replan": "replanner_1", "answer_budget_exhausted": "synth_1"},
                        },
                        {"from": "replanner_1", "to": "retrieval_1"},
                        {"from": "synth_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        assert TemplateValidator().validate(spec)["valid"] is True
        assert TemplateCompiler().compile(spec) is not None

    @pytest.mark.asyncio
    async def test_bound_node_spec_enforces_visit_limits(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {"content": "Document evidence."}

        monkeypatch.setattr("app.agent_patterns.graph.search_documents", FakeTool())
        bound = NodeRegistry().get_for_spec({"id": "retrieval_1", "type": "retrieval_worker"})
        with pytest.raises(ValueError, match="exceeded visit limit 1"):
            await bound(
                {
                    "agent_run_id": "run-1",
                    "thread_id": "thread-1",
                    "question": "What does the document say?",
                    "route": "document",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "node_events": [],
                    "tool_events": [],
                    "allowed_tool_ids": ["document_evidence"],
                    "loop_policy": {
                        "max_total_visits": 4,
                        "default_max_node_visits": 1,
                        "node_visit_limits": {"retrieval_1": 1},
                    },
                    "node_visit_counts": {"retrieval_1": 1},
                    "node_visit_sequence": [{"node": "retrieval_1", "node_type": "retrieval_worker", "visit_index": 1}],
                },
                {"configurable": {"thread_id": "thread-1"}},
            )

    @pytest.mark.asyncio
    async def test_hitl_gate_respects_interrupt_limit_without_interrupting(self):
        bound = NodeRegistry().get_for_spec({"id": "approval_1", "type": "hitl_gate"})
        update = await bound(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "Should web run?",
                "route": "web",
                "node_events": [],
                "tool_events": [],
                "hitl_interrupt_counts": {"approval_1": 1},
                "hitl_policy": {
                    "enabled": True,
                    "gates": {
                        "approval_1": {
                            "enabled": True,
                            "mode": "approval",
                            "target": {"node_id": "web_worker", "node_type": "web_worker"},
                            "allowed_actions": ["approve", "continue_without"],
                            "default_action": "continue_without",
                            "max_interrupts_per_run": 1,
                            "routes": {"approve": "web_worker", "continue_without": "synthesizer"},
                        }
                    },
                },
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert update["hitl_gate_route"] == "continue_without"
        assert update["hitl_gate_routes"]["approval_1"] == "continue_without"
        assert update["hitl_interrupt_counts"]["approval_1"] == 1
        assert update["node_events"][-1]["skip_reason"] == "hitl_interrupt_limit_exhausted"

    @pytest.mark.asyncio
    async def test_bound_node_spec_emits_instance_id_and_node_type(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {
                    "content": "Document evidence.",
                    "artifacts": {"document_sources": [{"file_hash": "file-1"}]},
                }

        monkeypatch.setattr("app.agent_patterns.graph.search_documents", FakeTool())
        bound = NodeRegistry().get_for_spec({"id": "retrieval_1", "type": "retrieval_worker"})
        update = await bound(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What does the document say?",
                "route": "document",
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "allowed_tool_ids": ["document_evidence"],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert update["node_events"][-1]["node"] == "retrieval_1"
        assert update["node_events"][-1]["node_type"] == "retrieval_worker"
        assert update["node_events"][-1]["visit_index"] == 1
        assert update["node_visit_counts"]["retrieval_1"] == 1
        assert update["tool_events"][0]["caller_node"] == "retrieval_1"
        assert update["tool_events"][0]["caller_node_type"] == "retrieval_worker"
        assert update["tool_events"][0]["caller_visit_index"] == 1
        assert update["evidence_packets"][0]["producer_node_id"] == "retrieval_1"
        assert update["evidence_packets"][0]["producer_node_type"] == "retrieval_worker"

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

    @pytest.mark.asyncio
    async def test_evidence_evaluator_routes_to_answer_when_sufficient(self, monkeypatch):
        class FakeLlm:
            async def ainvoke(self, _messages):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "sufficient": True,
                            "confidence": 0.9,
                            "missing_evidence": [],
                            "citation_risk": "low",
                            "contradiction_risk": "low",
                            "recommended_next_steps": [],
                            "reason": "enough evidence",
                        }
                    )
                )

        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: FakeLlm())

        update = await NodeRegistry().evidence_evaluator(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What does the PDF say?",
                "llm_model": "test-llm",
                "use_web_search": False,
                "context_window": 8192,
                "execution_plan": ["retrieval_worker"],
                "evidence": "Document evidence.",
                "document_sources": [{"file_hash": "file-1"}],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "replan_count": 0,
                "replans": 1,
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert update["evaluator_route"] == "answer"
        assert update["evaluation_confidence"] == 0.9
        assert update["node_events"][-1]["event_name"] == "evaluation.completed"

    @pytest.mark.asyncio
    async def test_evidence_evaluator_routes_to_replan_or_budget_exhausted(self, monkeypatch):
        class FakeLlm:
            async def ainvoke(self, _messages):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "sufficient": False,
                            "confidence": 0.3,
                            "missing_evidence": ["Need timeline evidence."],
                            "citation_risk": "medium",
                            "contradiction_risk": "low",
                            "recommended_next_steps": ["Run timeline_worker."],
                            "reason": "missing chronology",
                        }
                    )
                )

        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: FakeLlm())
        base_state = {
            "agent_run_id": "run-1",
            "thread_id": "thread-1",
            "question": "What changed since the first upload?",
            "llm_model": "test-llm",
            "use_web_search": False,
            "context_window": 8192,
            "execution_plan": ["retrieval_worker"],
            "evidence": "Document evidence.",
            "document_sources": [{"file_hash": "file-1"}],
            "web_sources": [],
            "used_chat_ids": [],
            "node_events": [],
            "tool_events": [],
            "replans": 1,
        }

        replan_update = await NodeRegistry().evidence_evaluator(
            dict(base_state, replan_count=0),
            {"configurable": {"thread_id": "thread-1"}},
        )
        exhausted_update = await NodeRegistry().evidence_evaluator(
            dict(base_state, replan_count=1),
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert replan_update["evaluator_route"] == "replan"
        assert replan_update["node_events"][-1]["event_name"] == "replan.requested"
        assert exhausted_update["evaluator_route"] == "answer_budget_exhausted"
        assert exhausted_update["node_events"][-1]["event_name"] == "replan.budget_exhausted"
        assert "replan budget is exhausted" in exhausted_update["evidence"]

    @pytest.mark.asyncio
    async def test_replanner_clamps_web_when_disabled_or_disallowed(self, monkeypatch):
        class FakeLlm:
            async def ainvoke(self, _messages):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "reason": "Need broader evidence.",
                            "execution_plan": ["web_worker", "retrieval_worker"],
                        }
                    )
                )

        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: FakeLlm())

        update = await NodeRegistry().replanner(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What is current?",
                "llm_model": "test-llm",
                "use_web_search": False,
                "context_window": 8192,
                "execution_plan": ["retrieval_worker"],
                "evaluator_report": {"sufficient": False, "missing_evidence": ["current web evidence"]},
                "evidence": "Document evidence.",
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "replan_count": 0,
                "replans": 1,
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert update["execution_plan"] == ["retrieval_worker"]
        assert update["replan_count"] == 1
        assert "web_worker_removed_when_web_search_disabled" in update["node_events"][-1]["llm_result_summary"]["normalization_notes"]


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
        evaluator_template, evaluator_version = await repo.get_template_with_current_version(EVALUATOR_REPLANNER_RAG_AGENT_ID)

        assert {template.id for template in templates} == {
            ROUTER_RAG_AGENT_ID,
            PLAN_EXECUTE_RAG_AGENT_ID,
            EVALUATOR_REPLANNER_RAG_AGENT_ID,
        }
        assert router_template.current_version_id == router_version.id
        assert router_version.version == ROUTER_RAG_AGENT_VERSION
        assert router_version.validation_result_json == {"valid": True, "errors": []}
        assert plan_template.current_version_id == plan_version.id
        assert plan_version.version == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert plan_version.validation_result_json == {"valid": True, "errors": []}
        assert evaluator_template.current_version_id == evaluator_version.id
        assert evaluator_version.version == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert evaluator_version.validation_result_json == {"valid": True, "errors": []}

    @pytest.mark.asyncio
    async def test_seed_builtin_v2_preview_versions_validate_and_compile(self, repo):
        await repo.seed_builtin_templates()

        preview_specs = [
            (ROUTER_RAG_AGENT_ID, ROUTER_RAG_AGENT_V2_VERSION, builtin_router_rag_v2_spec),
            (PLAN_EXECUTE_RAG_AGENT_ID, PLAN_EXECUTE_RAG_AGENT_V2_VERSION, builtin_plan_execute_rag_v2_spec),
            (
                EVALUATOR_REPLANNER_RAG_AGENT_ID,
                EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION,
                builtin_evaluator_replanner_rag_v2_spec,
            ),
        ]
        for template_id, version_number, spec_factory in preview_specs:
            template, preview_version = await repo.get_template_version(
                template_id,
                version_number,
                include_preview=True,
            )

            assert template.id == template_id
            assert preview_version.version == version_number
            assert preview_version.schema_version == 2
            assert preview_version.spec_json == spec_factory()
            assert preview_version.validation_result_json == {"valid": True, "errors": []}
            TemplateCompiler().compile(preview_version.spec_json)

        _, hidden_preview = await repo.get_template_version(ROUTER_RAG_AGENT_ID, ROUTER_RAG_AGENT_V2_VERSION)
        assert hidden_preview is None

    @pytest.mark.asyncio
    async def test_db_loaded_invalid_v2_spec_fails_validation(self, repo):
        bad_spec = builtin_router_rag_v2_spec()
        bad_spec["config"]["graph"]["nodes"].append({"id": "unsafe_1", "type": "unsafe_type"})
        bad_spec["config"]["graph"]["edges"].append({"from": "router", "to": "unsafe_1"})

        async with repo._session.begin():
            repo._session.add(
                AgentPatternTemplate(
                    id="internal_bad_agent",
                    name="Internal Bad Agent",
                    description="Invalid internal test agent.",
                    visibility="internal",
                    is_builtin=False,
                    current_version_id="internal_bad_agent:v1",
                )
            )
            repo._session.add(
                AgentPatternTemplateVersion(
                    id="internal_bad_agent:v1",
                    template_id="internal_bad_agent",
                    version=1,
                    schema_version=2,
                    spec_json=bad_spec,
                    validation_result_json={},
                    changelog="Invalid test spec.",
                )
            )

        template, version = await repo.get_template_with_current_version("internal_bad_agent", include_custom=True)

        assert template.id == "internal_bad_agent"
        with pytest.raises(TemplateValidationError, match="unknown type"):
            TemplateValidator().validate(version.spec_json)

    @pytest.mark.asyncio
    async def test_create_internal_custom_v2_template_version_validates_and_stores_current_version(self, repo):
        spec = builtin_router_rag_v2_spec()
        spec["pattern_type"] = "internal_custom_rag_agent"

        template, version = await repo.create_internal_template_version(
            template_id="internal_custom_rag_agent",
            name="Internal Custom RAG Agent",
            description="Internal JSON-authored custom pattern.",
            spec_json=spec,
            changelog="Initial internal custom pattern.",
        )
        public_template = await repo.get_template("internal_custom_rag_agent")
        loaded_template, loaded_version = await repo.get_template_with_current_version(
            "internal_custom_rag_agent",
            include_custom=True,
        )

        assert template.id == "internal_custom_rag_agent"
        assert template.visibility == "internal"
        assert template.is_builtin is False
        assert template.current_version_id == "internal_custom_rag_agent:v1"
        assert version.schema_version == 2
        assert version.validation_result_json == {"valid": True, "errors": []}
        assert public_template is None
        assert loaded_template.id == template.id
        assert loaded_version.id == version.id
        assert TemplateCompiler().compile(loaded_version.spec_json) is not None

    @pytest.mark.asyncio
    async def test_create_internal_custom_template_rejects_invalid_or_non_v2_specs(self, repo):
        invalid_spec = builtin_router_rag_v2_spec()
        invalid_spec["config"]["graph"]["edges"][2].pop("route_fn")
        with pytest.raises(TemplateValidationError, match="must declare route_fn"):
            await repo.create_internal_template_version(
                template_id="internal_invalid_agent",
                name="Internal Invalid Agent",
                spec_json=invalid_spec,
            )

        with pytest.raises(TemplateValidationError, match="schema_version 2"):
            await repo.create_internal_template_version(
                template_id="internal_v1_agent",
                name="Internal v1 Agent",
                spec_json=builtin_router_rag_spec(),
            )

        missing_template, missing_version = await repo.get_template_with_current_version(
            "internal_invalid_agent",
            include_custom=True,
        )
        assert missing_template is None
        assert missing_version is None

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
        assert completed.run_metadata_json == {"template_version_id": version.id}
        assert completed.template_version_id == version.id

    @pytest.mark.asyncio
    async def test_mark_run_awaiting_human_persists_bounded_pending_interrupt(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )

        paused = await repo.mark_run_awaiting_human(
            run.id,
            {
                "interrupt_id": "interrupt-1",
                "gate_id": "before_web",
                "node_id": "web_worker",
                "type": "tool_approval",
                "allowed_actions": ["approve", "reject", "continue_without"],
                "prompt": "x" * 3000,
                "input_summary": {"source_text": "body " * 1200},
            },
        )

        assert paused.status == "awaiting_human"
        assert paused.completed_at is None
        assert paused.pending_interrupt_json["interrupt_id"] == "interrupt-1"
        assert paused.pending_interrupt_json["status"] == "pending"
        assert len(paused.pending_interrupt_json["prompt"]) <= 2003
        assert len(paused.pending_interrupt_json["input_summary"]["source_text"]) <= 2003

        awaiting_runs = await repo.list_runs_for_thread(sample_thread.id, status="awaiting_human")
        assert [item.id for item in awaiting_runs] == [run.id]

    @pytest.mark.asyncio
    async def test_resolve_pending_interrupt_resumes_atomically_and_idempotently(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )
        debug_payload = build_debug_payload(
            run=run,
            node_events=[],
            tool_events=[],
            metrics={"duration_ms": 1.0, "tool_warning_count": 0, "error_count": 0},
        )
        await repo.mark_run_awaiting_human(
            run.id,
            {
                "interrupt_id": "interrupt-approve",
                "allowed_actions": ["approve", "reject"],
                "resume_token": "resume-token",
                "resume_version": 3,
            },
            debug_trace_json=debug_payload,
        )

        first = await repo.resolve_pending_interrupt(
            run.id,
            interrupt_id="interrupt-approve",
            action="approve",
            resume_token="resume-token",
            resume_version=3,
            client_metadata={"button": "approve"},
        )
        second = await repo.resolve_pending_interrupt(
            run.id,
            interrupt_id="interrupt-approve",
            action="approve",
            resume_token="resume-token",
            resume_version=3,
        )

        assert first.outcome == "resumed"
        assert first.duplicate is False
        assert first.run.status == "running"
        assert first.run.completed_at is None
        assert first.interrupt["status"] == "resumed"
        assert first.interrupt["decision"]["action"] == "approve"
        assert first.run.metrics_json["interrupt_resolution_count"] == 1
        assert second.outcome == "resumed"
        assert second.duplicate is True
        assert second.run.metrics_json["interrupt_resolution_count"] == 1
        root_events = [
            event
            for span in second.run.debug_trace_json["trace"]["spans"]
            if span["span_id"] == f"run:{run.id}"
            for event in span["events"]
        ]
        interrupt_events = [event for event in root_events if event["name"].startswith("interrupt.")]
        assert [event["name"] for event in interrupt_events] == ["interrupt.requested", "interrupt.resumed"]
        assert second.run.debug_trace_json["summary"]["interruptCount"] == 2
        assert second.run.debug_trace_json["summary"]["lastInterruptStatus"] == "resumed"

        with pytest.raises(AgentRunInterruptError, match="already been resolved"):
            await repo.resolve_pending_interrupt(
                run.id,
                interrupt_id="interrupt-approve",
                action="reject",
            )

    @pytest.mark.asyncio
    async def test_resolve_pending_interrupt_validates_selected_options(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )
        run_id = run.id
        await repo.mark_run_awaiting_human(
            run_id,
            {
                "interrupt_id": "interrupt-choice",
                "type": "option_review",
                "mode": "choice",
                "allowed_actions": ["approve_selected", "continue_without"],
                "selection_mode": "single_or_multi",
                "options": [
                    {"id": "document_search", "label": "Document search", "target_node_id": "retrieval_worker"},
                    {"id": "web_search", "label": "Web search", "target_node_id": "web_worker"},
                ],
                "resume_version": 1,
            },
        )

        with pytest.raises(AgentRunInterruptError) as exc:
            await repo.resolve_pending_interrupt(
                run_id,
                interrupt_id="interrupt-choice",
                action="approve_selected",
                selected_option_ids=["unknown"],
                resume_version=1,
            )
        assert exc.value.code == "interrupt_selection_invalid"

        result = await repo.resolve_pending_interrupt(
            run_id,
            interrupt_id="interrupt-choice",
            action="approve_selected",
            selected_option_ids=["document_search", "web_search"],
            resume_version=1,
        )

        assert result.outcome == "resumed"
        assert result.interrupt["decision"]["selected_option_ids"] == ["document_search", "web_search"]

    @pytest.mark.asyncio
    async def test_reject_pending_interrupt_marks_run_terminal(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )
        await repo.mark_run_awaiting_human(
            run.id,
            {"interrupt_id": "interrupt-reject", "allowed_actions": ["approve", "reject"]},
        )

        result = await repo.resolve_pending_interrupt(
            run.id,
            interrupt_id="interrupt-reject",
            action="reject",
        )

        assert result.outcome == "rejected"
        assert result.run.status == "rejected"
        assert result.run.completed_at is not None
        assert result.run.error_json["code"] == "agent_run_rejected_by_human"
        assert result.interrupt["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_chat_turns_can_share_one_agent_run_and_null_on_delete(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )

        session = await repo._get_session()
        async with session.begin():
            session.add(
                ChatTurn(
                    id="agent-run-turn-1",
                    thread_id=sample_thread.id,
                    agent_run_id=run.id,
                    agent_run_turn_kind="assistant_progress",
                    agent_run_sequence=0,
                    agent_trace_refs_json={"node_ids": ["planner"]},
                    status="completed",
                    payload={"question": "Q1", "answer": "A1", "metadata": {}},
                )
            )
            session.add(
                ChatTurn(
                    id="agent-run-turn-2",
                    thread_id=sample_thread.id,
                    agent_run_id=run.id,
                    agent_run_turn_kind="assistant_final",
                    agent_run_sequence=1,
                    agent_trace_refs_json={"span_ids": ["node:finalizer:0"]},
                    status="completed",
                    payload={"question": "Q2", "answer": "A2", "metadata": {}},
                )
            )

        turns = await repo.list_chat_turns_for_run(run.id)
        assert [turn.id for turn in turns] == ["agent-run-turn-1", "agent-run-turn-2"]
        assert [turn.agent_run_turn_kind for turn in turns] == ["assistant_progress", "assistant_final"]

        async with session.begin():
            persisted_run = await session.get(AgentRun, run.id)
            await session.delete(persisted_run)

        session.expire_all()
        async with session.begin():
            first = await session.get(ChatTurn, "agent-run-turn-1")
            second = await session.get(ChatTurn, "agent-run-turn-2")

        assert first.agent_run_id is None
        assert second.agent_run_id is None

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
    async def test_prune_checkpoints_for_terminal_runs_only(self, repo, sample_thread):
        class FakeCheckpointer:
            def __init__(self):
                self.deleted_thread_ids = []

            async def adelete_thread(self, thread_id):
                self.deleted_thread_ids.append(thread_id)

        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        old_completed = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "old_completed"},
        )
        old_failed = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "old_failed"},
        )
        old_awaiting = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "old_awaiting"},
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
            old_failed_row = await session.get(AgentRun, old_failed.id)
            old_awaiting_row = await session.get(AgentRun, old_awaiting.id)
            recent_completed_row = await session.get(AgentRun, recent_completed.id)
            old_completed_row.started_at = old_at
            old_completed_row.status = "completed"
            old_failed_row.started_at = old_at
            old_failed_row.status = "failed"
            old_awaiting_row.started_at = old_at
            old_awaiting_row.status = "awaiting_human"
            recent_completed_row.started_at = recent_at
            recent_completed_row.status = "completed"

        fake_checkpointer = FakeCheckpointer()
        deleted_checkpoint_thread_ids = await repo.prune_checkpoints_for_runs_before(
            utc_now() - timedelta(days=30),
            statuses=["completed", "failed"],
            thread_id=sample_thread.id,
            checkpointer=fake_checkpointer,
        )

        assert set(deleted_checkpoint_thread_ids) == {old_completed.checkpoint_thread_id, old_failed.checkpoint_thread_id}
        assert set(fake_checkpointer.deleted_thread_ids) == set(deleted_checkpoint_thread_ids)

        with pytest.raises(ValueError, match="terminal run statuses"):
            await repo.prune_checkpoints_for_runs_before(
                utc_now(),
                statuses=["completed", "awaiting_human"],
                checkpointer=fake_checkpointer,
            )

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
    async def test_expire_pending_interrupts_is_separate_from_stale_running_cleanup(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
        awaiting = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "awaiting"},
        )
        running = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID, "case": "running"},
        )
        now = utc_now()
        await repo.mark_run_awaiting_human(
            awaiting.id,
            {
                "interrupt_id": "interrupt-expired",
                "allowed_actions": ["approve", "reject"],
                "expires_at": iso_utc_z(now - timedelta(minutes=1)),
            },
        )

        expired_ids = await repo.expire_pending_interrupts(now=now, thread_id=sample_thread.id)
        expired_run = await repo.get_run(awaiting.id)
        running_after_expire = await repo.get_run(running.id)

        assert expired_ids == [awaiting.id]
        assert expired_run.status == "expired"
        assert expired_run.completed_at is not None
        assert expired_run.pending_interrupt_json["status"] == "expired"
        assert running_after_expire.status == "running"

        failed_ids = await repo.fail_stale_running_runs(now + timedelta(seconds=1), thread_id=sample_thread.id)
        assert failed_ids == [running.id]
        assert (await repo.get_run(awaiting.id)).status == "expired"

    @pytest.mark.asyncio
    async def test_unsupported_simple_rag_template_is_not_exposed(self, repo):
        await repo.seed_builtin_templates()

        template, version = await repo.get_template_with_current_version("simple_rag_agent")

        assert template is None
        assert version is None


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentRunService:
    def _agent_req(self, question: str = "Needs current research?") -> SimpleNamespace:
        return SimpleNamespace(
            question=question,
            llm_model="test-llm",
            use_web_search=True,
            use_reranker=False,
            context_window=8192,
            replans=1,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone="America/Chicago",
            client_locale="en-US",
            client_now_iso="2026-07-05T12:00:00.000Z",
        )

    async def _run_hitl_web_gate_flow(
        self,
        session_factory,
        sample_thread,
        monkeypatch,
        *,
        action: str,
        enable_web_approval: bool = True,
        duplicate_after_resume: bool = True,
    ):
        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content='{"route":"web","reason":"Needs live evidence."}')
                if action == "approve":
                    return SimpleNamespace(content="Answer with approved web evidence.")
                return SimpleNamespace(content="Answer without live web evidence.")

        class FakeWebTool:
            name = "search_web"

            def __init__(self):
                self.calls = 0
                self.inputs = []

            async def ainvoke(self, tool_input, config=None):
                self.calls += 1
                self.inputs.append(tool_input)
                return {
                    "content": "Live web evidence.",
                    "sources": [{"url": "https://example.test/result", "title": "Example"}],
                    "artifacts": {
                        "web_sources": [
                            {
                                "url": "https://example.test/result",
                                "title": "Example",
                                "preview": "Live web evidence.",
                            }
                        ]
                    },
                    "trace": {"tool_name": "search_web", "caller_node": "web_worker"},
                    "metrics": {"result_chars": 18, "source_count": 1, "warning_count": 0},
                }

        fake_llm = FakeLlm()
        fake_web = FakeWebTool()
        created_turn_ids = []
        stats_calls = []
        index_calls = []

        async def fake_get_thread_settings(_thread_id):
            return {
                "agent_pattern": {"template_id": ROUTER_RAG_AGENT_ID},
                "hitl_web_approval": enable_web_approval,
            }

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "",
                "document_evidence_text": "",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [],
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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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
            created_turn_ids.append(turn.id)
            return turn

        async def fake_index_chat_memory_for_thread(**kwargs):
            index_calls.append(kwargs)
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(thread_id, qa_chars):
            stats_calls.append((thread_id, qa_chars))

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
        monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            template_ref = SimpleNamespace(id=template.id)
            version_ref = SimpleNamespace(
                id=version.id,
                version=version.version,
                spec_json=builtin_router_rag_hitl_web_spec(),
            )

            async def fake_get_template_with_current_version(_template_id):
                return template_ref, version_ref

            repo.get_template_with_current_version = fake_get_template_with_current_version
            service = AgentRunService(repository=repo)
            paused = await service.run_thread_chat(sample_thread.id, self._agent_req(), sample_thread.embed_model)
            paused_run = await repo.get_run(paused["agent_run_id"])
            paused_debug = paused_run.debug_trace_json
            paused_turns = await repo.list_chat_turns_for_run(paused_run.id)
            pending = dict(paused_run.pending_interrupt_json or {})
            resumed = await service.resume_agent_run(
                paused_run.id,
                interrupt_id=pending["interrupt_id"],
                action=action,
                resume_version=pending["resume_version"],
                expected_thread_id=sample_thread.id,
            )
            duplicate = None
            if duplicate_after_resume:
                duplicate = await service.resume_agent_run(
                    paused_run.id,
                    interrupt_id=pending["interrupt_id"],
                    action=action,
                    resume_version=pending["resume_version"],
                    expected_thread_id=sample_thread.id,
                )
            turns = await repo.list_chat_turns_for_run(paused_run.id)

        return {
            "paused": paused,
            "paused_run": paused_run,
            "paused_debug": paused_debug,
            "paused_turns": paused_turns,
            "pending": pending,
            "resumed": resumed,
            "duplicate": duplicate,
            "turns": turns,
            "created_turn_ids": created_turn_ids,
            "stats_calls": stats_calls,
            "index_calls": index_calls,
            "fake_llm": fake_llm,
            "fake_web": fake_web,
        }

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

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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
                replans=1,
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

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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
                replans=1,
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

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                node_event = {"node": "router", "elapsed_ms": 3.5, "route": "direct"}
                tool_event = {
                    "tool_name": "search_documents",
                    "caller_node": "retrieval_worker",
                    "ok": True,
                    "elapsed_ms": 9.25,
                    "warnings": [],
                }
                trace_recorder.record_node_event(node_event)
                trace_recorder.record_tool_event(tool_event)
                return {
                    "answer": "router ok",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [node_event],
                    "tool_events": [tool_event],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                replans=1,
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
        assert run.debug_trace_json["version"] == 1
        assert run.debug_trace_json["trace"]["run_id"] == run.id
        assert "graph" not in run.debug_trace_json

    @pytest.mark.asyncio
    async def test_run_thread_chat_can_load_v2_preview_version_when_opted_in(self, engine, sample_thread, monkeypatch):
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
                return {
                    "agent_pattern": {
                        "template_id": ROUTER_RAG_AGENT_ID,
                        "template_version": ROUTER_RAG_AGENT_V2_VERSION,
                    }
                }

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "router v2 ok",
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
                replans=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(
                repository=repo,
                allow_preview_agent_patterns=True,
            ).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        router_edge = next(
            edge
            for edge in captured_spec["config"]["graph"]["edges"]
            if edge.get("from") == "router" and edge.get("conditional")
        )
        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID
        assert result["agent_pattern_version"] == ROUTER_RAG_AGENT_V2_VERSION
        assert run.template_version_id == f"{ROUTER_RAG_AGENT_ID}:v{ROUTER_RAG_AGENT_V2_VERSION}"
        assert run.resolved_spec_json["schema_version"] == 2
        assert router_edge["route_fn"] == "router_route"
        assert captured_spec["config"]["loop_policy"]["max_total_visits"] == 9

    @pytest.mark.asyncio
    async def test_run_thread_chat_falls_back_for_custom_db_pattern_without_opt_in(self, engine, sample_thread, monkeypatch):
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
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["pattern_type"] = "internal_custom_rag_agent"
            await repo.create_internal_template_version(
                template_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "router fallback",
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
                replans=1,
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
        assert run.template_id == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_can_load_custom_db_pattern_when_opted_in(self, engine, sample_thread, monkeypatch):
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
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["pattern_type"] = "internal_custom_rag_agent"
            await repo.create_internal_template_version(
                template_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "custom ok",
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
                replans=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(
                repository=repo,
                allow_custom_agent_patterns=True,
            ).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == "internal_custom_rag_agent"
        assert result["agent_pattern_version"] == 1
        assert run.template_version_id == "internal_custom_rag_agent:v1"
        assert run.resolved_spec_json["schema_version"] == 2
        assert captured_spec["pattern_type"] == "internal_custom_rag_agent"

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

            async def fake_handle_plan_execute_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                node_event = {"node": "planner", "elapsed_ms": 2.0, "route": "execute", "execution_plan": ["retrieval_worker"]}
                trace_recorder.record_node_event(node_event)
                return {
                    "answer": "plan execute ok",
                    "document_sources": [{"id": "doc"}],
                    "web_sources": [],
                    "used_chat_ids": ["turn-1:assistant"],
                    "clarification_options": None,
                    "route": "execute",
                    "node_events": [node_event],
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
                replans=1,
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
        assert run.debug_trace_json["summary"]["route"] == "execute"
        assert "graph" not in run.debug_trace_json

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_evaluator_replanner_rag_when_selected(self, engine, sample_thread, monkeypatch):
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
                return {"agent_pattern": {"template_id": EVALUATOR_REPLANNER_RAG_AGENT_ID}}

            async def fake_handle_evaluator_replanner_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                node_event = {
                    "node": "evidence_evaluator",
                    "elapsed_ms": 2.0,
                    "route": "execute",
                    "evaluator_route": "answer",
                    "evaluation_confidence": 0.8,
                    "replan_count": 0,
                    "evaluator_report": {"sufficient": True, "confidence": 0.8},
                }
                trace_recorder.record_node_event(node_event)
                return {
                    "answer": "evaluator replanner ok",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "execute",
                    "node_events": [node_event],
                    "tool_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr(
                "app.agent_patterns.router_runtime.handle_evaluator_replanner_rag_chat",
                fake_handle_evaluator_replanner_rag_chat,
            )

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_web_search=False,
                use_reranker=True,
                replans=1,
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

        assert result["agent_pattern_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert result["agent_pattern_version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert captured_spec["pattern_type"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.resolved_spec_json["pattern_type"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert run.metrics_json["route"] == "execute"
        assert run.metrics_json["replan_count"] == 0
        assert run.metrics_json["evaluation_confidence"] == 0.8
        assert run.debug_trace_json["summary"]["evaluatorRoute"] == "answer"
        assert "graph" not in run.debug_trace_json

    @pytest.mark.asyncio
    async def test_run_thread_chat_pauses_before_web_and_resumes_from_checkpoint_once(self, engine, sample_thread, monkeypatch):
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
                    return SimpleNamespace(content='{"route":"web","reason":"Needs live evidence.","clarification_options":null}')
                return SimpleNamespace(content="Checkpointed final answer.")

        class FakeWebTool:
            name = "search_web"

            def __init__(self):
                self.calls = 0

            async def ainvoke(self, tool_input, config=None):
                self.calls += 1
                return {
                    "content": "Checkpointed web evidence.",
                    "sources": [{"url": "https://example.test/checkpoint", "title": "Checkpoint"}],
                    "artifacts": {
                        "web_sources": [
                            {
                                "url": "https://example.test/checkpoint",
                                "title": "Checkpoint",
                                "preview": "Checkpointed web evidence.",
                            }
                        ]
                    },
                    "trace": {"tool_name": "search_web", "caller_node": "web_worker"},
                    "metrics": {"result_chars": 27, "source_count": 1, "warning_count": 0},
                }

        fake_llm = FakeLlm()
        fake_web = FakeWebTool()
        created_turn_ids = []

        async def fake_get_thread_settings(_thread_id):
            return {
                "agent_pattern": {"template_id": ROUTER_RAG_AGENT_ID},
                "hitl_web_approval": True,
            }

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "",
                "document_evidence_text": "",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [],
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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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
            created_turn_ids.append(turn.id)
            return turn

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
        monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            service = AgentRunService(repository=repo)
            req = SimpleNamespace(
                question="Pause before web?",
                llm_model="test-llm",
                use_web_search=True,
                use_reranker=False,
                context_window=8192,
                replans=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
                client_timezone="America/Chicago",
                client_locale="en-US",
                client_now_iso="2026-07-05T12:00:00.000Z",
            )

            paused = await service.run_thread_chat(sample_thread.id, req, sample_thread.embed_model)
            run = await repo.get_run(paused["agent_run_id"])
            pending = run.pending_interrupt_json
            paused_run_status = run.status
            paused_completed_at = run.completed_at
            paused_checkpoint_thread_id = run.checkpoint_thread_id
            resumed = await service.resume_agent_run(
                run.id,
                interrupt_id=pending["interrupt_id"],
                action="approve",
                resume_version=pending["resume_version"],
                expected_thread_id=sample_thread.id,
            )
            duplicate = await service.resume_agent_run(
                run.id,
                interrupt_id=pending["interrupt_id"],
                action="approve",
                resume_version=pending["resume_version"],
                expected_thread_id=sample_thread.id,
            )
            turns = await repo.list_chat_turns_for_run(run.id)

        assert paused["status"] == "awaiting_human"
        assert "chat_turn_id" not in paused
        assert paused_run_status == "awaiting_human"
        assert paused_completed_at is None
        assert paused_checkpoint_thread_id == run.id
        assert pending["checkpoint_resume"] is True
        assert pending["checkpoint_thread_id"] == paused_checkpoint_thread_id
        assert pending["type"] == "tool_approval"
        assert pending["gate_id"] == "web_approval_gate"
        assert pending["proposed_tool"]["name"] == "search_web"
        gates = run.resolved_spec_json["config"]["hitl_policy"]["gates"]
        assert gates["web_approval_gate"]["target"] == {"node_id": "web_worker", "node_type": "web_worker"}
        assert fake_llm.calls == 2

        assert resumed is not None
        assert resumed.duplicate is False
        assert resumed.run.status == "completed"
        assert resumed.run.pending_interrupt_json["status"] == "resumed"
        assert resumed.run.metrics_json["interrupt_resolution_count"] == 1
        assert duplicate is not None
        assert duplicate.duplicate is True
        assert duplicate.run.status == "completed"
        assert len(created_turn_ids) == 1
        assert len(turns) == 1
        assert turns[0].payload["answer"] == "Checkpointed final answer."
        assert turns[0].payload["web_sources"] == [
            {
                "url": "https://example.test/checkpoint",
                "title": "Checkpoint",
                "preview": "Checkpointed web evidence.",
            }
        ]
        assert turns[0].agent_run_id == run.id
        assert fake_web.calls == 1

        root_events = [
            event
            for span in resumed.run.debug_trace_json["trace"]["spans"]
            if span["span_id"] == f"run:{run.id}"
            for event in span["events"]
        ]
        root_event_names = [event["name"] for event in root_events]
        interrupt_events = [event for event in root_events if event["name"].startswith("interrupt.")]
        assert [event["name"] for event in interrupt_events] == ["interrupt.requested", "interrupt.resumed"]
        for event_name in ["checkpoint.created", "resume.requested", "resume.applied", "graph.resumed"]:
            assert event_name in root_event_names
        node_span_ids = {
            span["attributes"]["askpdf.node.id"]
            for span in resumed.run.debug_trace_json["trace"]["spans"]
            if span.get("attributes", {}).get("askpdf.node.id")
        }
        assert "web_approval_gate" in node_span_ids
        assert "web_worker" in node_span_ids
        assert resumed.run.debug_trace_json["summary"]["usedNodeCount"] == 6
        assert any(node["id"] == "web_approval_gate" for node in resumed.run.debug_trace_json["summary"]["nodes"])
        assert resumed.run.debug_trace_json["trace"]["status"] == "completed"
        assert resumed.run.debug_trace_json["trace"]["chat_turn_id"] == turns[0].id
        assert resumed.run.debug_trace_json["summary"]["lastInterruptStatus"] == "resumed"

    @pytest.mark.asyncio
    async def test_hitl_web_gate_approve_resumes_and_executes_web_once(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        result = await self._run_hitl_web_gate_flow(session_factory, sample_thread, monkeypatch, action="approve")

        assert result["paused"]["status"] == "awaiting_human"
        assert "chat_turn_id" not in result["paused"]
        assert result["paused_turns"] == []
        assert result["pending"]["type"] == "tool_approval"
        assert result["pending"]["gate_id"] == "web_approval_gate"
        assert result["pending"]["allowed_actions"] == ["approve", "continue_without"]
        assert result["pending"]["proposed_tool"]["name"] == "search_web"
        assert any(
            event.get("node") == "web_approval_gate" and event.get("status") == "interrupted"
            for event in result["paused"]["node_events"]
        )
        paused_node_ids = {
            span["attributes"]["askpdf.node.id"]
            for span in result["paused_debug"]["trace"]["spans"]
            if span.get("attributes", {}).get("askpdf.node.id")
        }
        assert "web_approval_gate" in paused_node_ids
        assert result["created_turn_ids"] == [result["turns"][0].id]
        assert len(result["stats_calls"]) == 1
        assert len(result["index_calls"]) == 1
        assert result["fake_web"].calls == 1
        assert result["resumed"].run.status == "completed"
        assert result["resumed"].run.pending_interrupt_json["status"] == "resumed"
        assert result["resumed"].run.pending_interrupt_json["decision"]["action"] == "approve"
        assert result["duplicate"].duplicate is True
        assert result["fake_web"].calls == 1
        assert len(result["turns"]) == 1
        assert result["turns"][0].payload["answer"] == "Answer with approved web evidence."
        assert result["turns"][0].payload["web_sources"] == [
            {
                "url": "https://example.test/result",
                "title": "Example",
                "preview": "Live web evidence.",
            }
        ]
        node_ids = {
            span["attributes"]["askpdf.node.id"]
            for span in result["resumed"].run.debug_trace_json["trace"]["spans"]
            if span.get("attributes", {}).get("askpdf.node.id")
        }
        assert {"web_approval_gate", "web_worker"} <= node_ids
        root_events = [
            event
            for span in result["resumed"].run.debug_trace_json["trace"]["spans"]
            if span["span_id"] == f"run:{result['resumed'].run.id}"
            for event in span["events"]
        ]
        assert [event["name"] for event in root_events if event["name"].startswith("interrupt.")] == [
            "interrupt.requested",
            "interrupt.resumed",
        ]
        assert "graph.resumed" in [event["name"] for event in root_events]

    @pytest.mark.asyncio
    async def test_hitl_web_gate_continue_without_skips_web_and_completes(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        result = await self._run_hitl_web_gate_flow(session_factory, sample_thread, monkeypatch, action="continue_without")

        assert result["paused"]["status"] == "awaiting_human"
        assert result["paused_turns"] == []
        assert any(
            event.get("node") == "web_approval_gate" and event.get("status") == "interrupted"
            for event in result["paused"]["node_events"]
        )
        assert result["fake_web"].calls == 0
        assert result["resumed"].run.status == "completed"
        assert result["resumed"].run.pending_interrupt_json["decision"]["action"] == "continue_without"
        assert result["duplicate"].duplicate is True
        assert len(result["turns"]) == 1
        assert result["turns"][0].payload["answer"] == "Answer without live web evidence."
        assert result["turns"][0].payload["web_sources"] == []
        node_ids = {
            span["attributes"]["askpdf.node.id"]
            for span in result["resumed"].run.debug_trace_json["trace"]["spans"]
            if span.get("attributes", {}).get("askpdf.node.id")
        }
        assert "web_approval_gate" in node_ids
        assert "web_worker" not in node_ids
        gate_spans = [
            span
            for span in result["resumed"].run.debug_trace_json["trace"]["spans"]
            if span.get("attributes", {}).get("askpdf.node.id") == "web_approval_gate"
        ]
        assert gate_spans[-1]["output"]["value"]["next"] == "synthesizer"

    @pytest.mark.asyncio
    async def test_hitl_web_gate_thread_setting_does_not_add_final_review_on_resume(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        result = await self._run_hitl_web_gate_flow(
            session_factory,
            sample_thread,
            monkeypatch,
            action="approve",
            enable_web_approval=True,
            duplicate_after_resume=False,
        )

        assert result["pending"]["gate_id"] == "web_approval_gate"
        assert "web_approval_gate" in result["paused_run"].resolved_spec_json["config"]["hitl_policy"]["gates"]
        assert result["fake_web"].calls == 1
        assert result["resumed"].run.status == "completed"
        assert result["turns"][0].payload["answer"] == "Answer with approved web evidence."

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST") != "1",
        reason="set ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST=1 to run the Postgres checkpoint persistence test",
    )
    async def test_run_thread_chat_resumes_after_postgres_checkpointer_reopen(
        self,
        engine,
        test_database_url,
        monkeypatch,
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
                    return SimpleNamespace(content='{"route":"web","reason":"Needs live evidence.","clarification_options":null}')
                return SimpleNamespace(content="Postgres checkpoint answer.")

        class FakeWebTool:
            name = "search_web"

            def __init__(self):
                self.calls = 0

            async def ainvoke(self, tool_input, config=None):
                self.calls += 1
                return {
                    "content": "Postgres web evidence.",
                    "sources": [{"url": "https://example.test/postgres", "title": "Postgres"}],
                    "artifacts": {
                        "web_sources": [
                            {
                                "url": "https://example.test/postgres",
                                "title": "Postgres",
                                "preview": "Postgres web evidence.",
                            }
                        ]
                    },
                    "trace": {"tool_name": "search_web", "caller_node": "web_worker"},
                    "metrics": {"result_chars": 22, "source_count": 1, "warning_count": 0},
                }

        fake_llm = FakeLlm()
        fake_web = FakeWebTool()
        created_turn_ids = []

        async def fake_get_thread_settings(_thread_id):
            return {
                "agent_pattern": {"template_id": EVALUATOR_REPLANNER_RAG_AGENT_ID},
                "hitl_web_approval": True,
            }

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "",
                "document_evidence_text": "",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [],
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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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
            created_turn_ids.append(turn.id)
            return turn

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        async with session_factory() as setup_session:
            thread = Thread(
                id=str(uuid.uuid4()),
                name="Postgres checkpoint test",
                embed_model="BAAI/bge-m3",
                settings={},
            )
            setup_session.add(thread)
            await setup_session.commit()
            await setup_session.refresh(thread)
            thread_id = thread.id
            embed_model = thread.embed_model

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "postgres")
        monkeypatch.setenv("AGENT_CHECKPOINT_DATABASE_URL", test_database_url)
        monkeypatch.delenv("ASKPDF_AGENT_CHECKPOINTER_ALLOW_MEMORY_FALLBACK", raising=False)
        monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as first_session:
            first_repo = AgentPatternRepository(first_session)
            await first_repo.seed_builtin_templates()
            req = SimpleNamespace(
                question="Pause and survive restart?",
                llm_model="test-llm",
                use_web_search=True,
                use_reranker=False,
                context_window=8192,
                replans=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
                client_timezone="America/Chicago",
                client_locale="en-US",
                client_now_iso="2026-07-05T12:00:00.000Z",
            )
            paused = await AgentRunService(repository=first_repo).run_thread_chat(
                thread_id,
                req,
                embed_model,
            )
            paused_run = await first_repo.get_run(paused["agent_run_id"])
            pending = paused_run.pending_interrupt_json
            checkpoint_thread_id = paused_run.checkpoint_thread_id

        async with session_factory() as second_session:
            second_repo = AgentPatternRepository(second_session)
            resumed = await AgentRunService(repository=second_repo).resume_agent_run(
                paused_run.id,
                interrupt_id=pending["interrupt_id"],
                action="approve",
                resume_version=pending["resume_version"],
                expected_thread_id=thread_id,
            )
            duplicate = await AgentRunService(repository=second_repo).resume_agent_run(
                paused_run.id,
                interrupt_id=pending["interrupt_id"],
                action="approve",
                resume_version=pending["resume_version"],
                expected_thread_id=thread_id,
            )
            turns = await second_repo.list_chat_turns_for_run(paused_run.id)
            deleted_checkpoint_thread_ids = await second_repo.prune_checkpoints_for_runs_before(
                utc_now() + timedelta(seconds=1),
                statuses=["completed"],
                thread_id=thread_id,
            )

        assert paused["status"] == "awaiting_human"
        assert pending["checkpoint_resume"] is True
        assert pending["checkpoint_thread_id"] == checkpoint_thread_id
        assert checkpoint_thread_id == paused_run.id
        assert resumed is not None
        assert resumed.duplicate is False
        assert resumed.run.status == "completed"
        assert duplicate is not None
        assert duplicate.duplicate is True
        assert len(created_turn_ids) == 1
        assert len(turns) == 1
        assert turns[0].payload["answer"] == "Postgres checkpoint answer."
        assert fake_web.calls == 1
        assert fake_llm.calls == 2
        assert deleted_checkpoint_thread_ids == [checkpoint_thread_id]

    @pytest.mark.asyncio
    async def test_resume_that_pauses_again_preserves_trace_continuity(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                template_version=version.version,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            debug_payload = build_debug_payload(
                run=run,
                node_events=[],
                tool_events=[],
                metrics={"duration_ms": 1.0, "tool_warning_count": 0, "error_count": 0},
            )
            await repo.mark_run_awaiting_human(
                run.id,
                {
                    "interrupt_id": "interrupt-1",
                    "allowed_actions": ["approve", "reject"],
                    "checkpoint_resume": True,
                    "checkpoint_thread_id": run.checkpoint_thread_id,
                    "resume_version": 1,
                },
                debug_trace_json=debug_payload,
            )

            async def fake_resume_compiled_rag_chat(run, *, interrupt, checkpointer, trace_recorder):
                trace_recorder.record_runtime_event(
                    "graph.resumed",
                    attributes={
                        "askpdf.run.id": run.id,
                        "askpdf.thread.id": run.thread_id,
                        "askpdf.interrupt.id": interrupt.get("interrupt_id"),
                        "askpdf.resume.action": "approve",
                        "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                    },
                )
                return {
                    "status": "awaiting_human",
                    "pending_interrupt": {
                        "interrupt_id": "interrupt-2",
                        "allowed_actions": ["approve", "reject"],
                        "checkpoint_resume": True,
                        "checkpoint_thread_id": run.checkpoint_thread_id,
                        "resume_version": 2,
                    },
                    "duration_ms": 2.0,
                    "route": "direct",
                    "route_reason": "second review gate",
                    "node_events": [],
                    "tool_events": [],
                }

            monkeypatch.setattr("app.agent_patterns.router_runtime.resume_compiled_rag_chat", fake_resume_compiled_rag_chat)

            result = await AgentRunService(repository=repo).resume_agent_run(
                run.id,
                interrupt_id="interrupt-1",
                action="approve",
                resume_version=1,
                expected_thread_id=sample_thread.id,
            )
            updated_run = await repo.get_run(run.id)

        assert result is not None
        assert result.duplicate is False
        assert updated_run.status == "awaiting_human"
        assert updated_run.pending_interrupt_json["interrupt_id"] == "interrupt-2"
        assert updated_run.pending_interrupt_json["status"] == "pending"
        root_events = [
            event
            for span in updated_run.debug_trace_json["trace"]["spans"]
            if span["span_id"] == f"run:{run.id}"
            for event in span["events"]
        ]
        root_event_names = [event["name"] for event in root_events]
        requested_interrupt_ids = [
            event["attributes"]["askpdf.interrupt.id"]
            for event in root_events
            if event["name"] == "interrupt.requested"
        ]
        assert requested_interrupt_ids == ["interrupt-1", "interrupt-2"]
        assert "interrupt.resumed" in root_event_names
        assert "resume.requested" in root_event_names
        assert "resume.applied" in root_event_names
        assert "graph.resumed" in root_event_names
        assert "checkpoint.created" in root_event_names
        assert updated_run.debug_trace_json["trace"]["status"] == "awaiting_human"
        assert updated_run.debug_trace_json["summary"]["lastInterruptStatus"] == "pending"

    @pytest.mark.asyncio
    async def test_duplicate_resume_submission_does_not_invoke_graph_twice(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        calls = []
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()
            template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                template_version=version.version,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            debug_payload = build_debug_payload(
                run=run,
                node_events=[],
                tool_events=[],
                metrics={"duration_ms": 1.0, "tool_warning_count": 0, "error_count": 0},
            )
            await repo.mark_run_awaiting_human(
                run.id,
                {
                    "interrupt_id": "duplicate-interrupt",
                    "allowed_actions": ["approve", "reject"],
                    "checkpoint_resume": True,
                    "checkpoint_thread_id": run.checkpoint_thread_id,
                    "resume_version": 1,
                },
                debug_trace_json=debug_payload,
            )

            async def fake_resume_compiled_rag_chat(run, *, interrupt, checkpointer, trace_recorder):
                calls.append(interrupt["interrupt_id"])
                trace_recorder.record_runtime_event(
                    "graph.resumed",
                    attributes={
                        "askpdf.run.id": run.id,
                        "askpdf.thread.id": run.thread_id,
                        "askpdf.interrupt.id": interrupt.get("interrupt_id"),
                        "askpdf.resume.action": "approve",
                        "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                    },
                )
                return {
                    "status": "completed",
                    "duration_ms": 2.0,
                    "route": "direct",
                    "route_reason": "approved once",
                    "node_events": [],
                    "tool_events": [],
                    "chat_turn_id": "turn-1",
                }

            monkeypatch.setattr("app.agent_patterns.router_runtime.resume_compiled_rag_chat", fake_resume_compiled_rag_chat)

            service = AgentRunService(repository=repo)
            first = await service.resume_agent_run(
                run.id,
                interrupt_id="duplicate-interrupt",
                action="approve",
                resume_version=1,
                expected_thread_id=sample_thread.id,
            )
            second = await service.resume_agent_run(
                run.id,
                interrupt_id="duplicate-interrupt",
                action="approve",
                resume_version=1,
                expected_thread_id=sample_thread.id,
            )
            updated_run = await repo.get_run(run.id)

        assert first is not None
        assert first.duplicate is False
        assert second is not None
        assert second.duplicate is True
        assert calls == ["duplicate-interrupt"]
        assert updated_run.status == "completed"
        assert updated_run.metrics_json["interrupt_resolution_count"] == 1

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

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                node_event = {"node": "router", "elapsed_ms": 4.0, "route": "document"}
                trace_recorder.record_node_event(node_event)
                return {
                    "answer": "fallback",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "document",
                    "node_events": [node_event],
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
                replans=1,
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
        assert run.debug_trace_json["trace"]["status"] == "failed"
        assert "graph" not in run.debug_trace_json


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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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

        spec = builtin_router_rag_spec()
        await create_agent_run_record(
            session_factory,
            run_id="run-1",
            thread_id=sample_thread.id,
            spec=spec,
        )
        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": "run-1",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
            trace_recorder=make_trace_recorder("run-1", sample_thread.id, spec),
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
        assert turn.agent_run_id == "run-1"
        assert turn.agent_run_turn_kind == "assistant_final"
        assert turn.agent_run_sequence == 0
        assert turn.agent_trace_refs_json is None
        assert "agent_run_id" not in turn.payload["metadata"]
        assert turn.payload["metadata"]["agent_route"] == "direct"
        assert "agent_debug_trace" not in turn.payload["metadata"]
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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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

        run_id = f"run-{route}"
        spec = builtin_router_rag_spec()
        await create_agent_run_record(
            session_factory,
            run_id=run_id,
            thread_id=sample_thread.id,
            spec=spec,
        )
        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": run_id,
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
            trace_recorder=make_trace_recorder(run_id, sample_thread.id, spec),
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["route"] == route
        assert [event["node"] for event in result["node_events"]] == expected_nodes
        assert all(isinstance(event.get("elapsed_ms"), (int, float)) for event in result["node_events"])
        assert turn is not None
        assert turn.status == expected_status
        assert turn.agent_run_id == run_id
        assert turn.agent_run_turn_kind == "assistant_final"
        assert turn.agent_run_sequence == 0
        assert turn.agent_trace_refs_json is None
        assert turn.payload["metadata"]["agent_route"] == route
        assert "agent_debug_trace" not in turn.payload["metadata"]
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
            agent_run_id=None,
            agent_run_turn_kind=None,
            agent_run_sequence=None,
            agent_trace_refs_json=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                agent_run_id=agent_run_id,
                agent_run_turn_kind=agent_run_turn_kind,
                agent_run_sequence=agent_run_sequence,
                agent_trace_refs_json=agent_trace_refs_json,
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

        spec = builtin_router_rag_spec()
        await create_agent_run_record(
            session_factory,
            run_id="run-failed",
            thread_id=sample_thread.id,
            spec=spec,
        )
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": "run-failed",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
            trace_recorder=make_trace_recorder("run-failed", sample_thread.id, spec),
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["agent_error"]["code"] == "router_rag_execution_failed"
        assert result["chat_turn_id"] == turn.id
        assert result["route"] == "document"
        assert [event["node"] for event in result["node_events"]] == ["context_loader", "router", "retrieval_worker"]
        assert result["node_events"][-1]["status"] == "failed"
        assert result["node_events"][-1]["error"]["raw_message"] == "document tool exploded"
        assert result["tool_events"] == []
        assert result["errors"][0]["raw_message"] == "document tool exploded"
        assert turn.status == "failed"
        assert turn.agent_run_id == "run-failed"
        assert turn.agent_run_turn_kind == "assistant_final"
        assert turn.agent_run_sequence == 0
        assert turn.agent_trace_refs_json is None
        assert "agent_run_id" not in turn.payload["metadata"]
        assert turn.payload["metadata"]["agent_route"] == "document"
        assert "agent_debug_trace" not in turn.payload["metadata"]
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
            EVALUATOR_REPLANNER_RAG_AGENT_ID,
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

        evaluator_detail = api_client.get(f"/api/agent-patterns/{EVALUATOR_REPLANNER_RAG_AGENT_ID}")
        assert evaluator_detail.status_code == 200
        evaluator_payload = evaluator_detail.json()
        assert evaluator_payload["agent_pattern"]["id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert evaluator_payload["current_version"]["version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert evaluator_payload["current_version"]["validation"]["valid"] is True
        assert evaluator_payload["capabilities"]["node_tool_requirements"]["evidence_evaluator"] == "clarify_intent"
        assert evaluator_payload["capabilities"]["node_tool_requirements"]["replanner"] == "clarify_intent"

        stale_detail = api_client.get("/api/agent-patterns/simple_rag_agent")
        assert stale_detail.status_code == 404

    def test_internal_custom_agent_pattern_is_not_publicly_exposed(self, api_client):
        async def seed_internal_pattern():
            spec = builtin_router_rag_v2_spec()
            spec["pattern_type"] = "internal_api_hidden_agent"
            await AgentPatternRepository().create_internal_template_version(
                template_id="internal_api_hidden_agent",
                name="Internal API Hidden Agent",
                spec_json=spec,
            )

        asyncio.run(seed_internal_pattern())

        listed = api_client.get("/api/agent-patterns")
        detail = api_client.get("/api/agent-patterns/internal_api_hidden_agent")

        assert listed.status_code == 200
        assert "internal_api_hidden_agent" not in {
            item["id"] for item in listed.json()["agent_patterns"]
        }
        assert detail.status_code == 404

    def test_internal_agent_pattern_endpoint_creates_and_fetches_custom_v2_spec(self, api_client):
        spec = builtin_router_rag_v2_spec()
        spec["pattern_type"] = "internal_api_agent"

        created = api_client.post(
            "/api/internal/agent-patterns",
            json={
                "template_id": "internal_api_agent",
                "name": "Internal API Agent",
                "description": "JSON-authored internal pattern.",
                "changelog": "Initial internal API version.",
                "spec_json": spec,
            },
        )
        fetched = api_client.get("/api/internal/agent-patterns/internal_api_agent")
        public_detail = api_client.get("/api/agent-patterns/internal_api_agent")

        assert created.status_code == 200
        created_payload = created.json()
        assert created_payload["agent_pattern"]["id"] == "internal_api_agent"
        assert created_payload["agent_pattern"]["visibility"] == "internal"
        assert created_payload["version"]["id"] == "internal_api_agent:v1"
        assert created_payload["version"]["schema_version"] == 2
        assert created_payload["version"]["validation"]["valid"] is True
        assert created_payload["version"]["validation_result_json"] == {"valid": True, "errors": []}
        assert fetched.status_code == 200
        assert fetched.json()["current_version"]["id"] == "internal_api_agent:v1"
        assert public_detail.status_code == 404

    def test_internal_agent_pattern_endpoint_rejects_invalid_specs_without_storing(self, api_client):
        invalid_spec = builtin_router_rag_v2_spec()
        invalid_spec["pattern_type"] = "internal_api_invalid_agent"
        invalid_spec["config"]["graph"]["edges"][2].pop("route_fn")

        invalid = api_client.post(
            "/api/internal/agent-patterns",
            json={
                "template_id": "internal_api_invalid_agent",
                "name": "Internal API Invalid Agent",
                "spec_json": invalid_spec,
            },
        )
        fetched = api_client.get("/api/internal/agent-patterns/internal_api_invalid_agent")

        assert invalid.status_code == 400
        assert "must declare route_fn" in invalid.json()["detail"]
        assert fetched.status_code == 404

    def test_internal_agent_pattern_endpoint_rejects_builtin_ids(self, api_client):
        spec = builtin_router_rag_v2_spec()
        rejected = api_client.post(
            "/api/internal/agent-patterns",
            json={
                "template_id": ROUTER_RAG_AGENT_ID,
                "name": "Not Allowed",
                "spec_json": spec,
            },
        )

        assert rejected.status_code == 400
        assert "built-in agent pattern templates cannot be authored" in rejected.json()["detail"]

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

    def test_validate_thread_agent_config_endpoint_resolves_without_running_chat(self, api_client, sample_thread, monkeypatch):
        async def fake_get_thread_settings(_thread_id):
            return {
                "agent_pattern": {"template_id": EVALUATOR_REPLANNER_RAG_AGENT_ID},
                "hitl_web_approval": True,
            }

        monkeypatch.setattr("app.api.agent_patterns.get_thread_settings", fake_get_thread_settings)

        response = api_client.post(
            f"/api/threads/{sample_thread.id}/agent-config/validate",
            json={"overrides": {"use_web_search": True, "replans": 2}},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is True
        assert payload["template_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert payload["template_version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert payload["validation"]["valid"] is True
        assert payload["resolved_spec_json"]["config"]["use_web_search"] is True
        assert payload["resolved_spec_json"]["config"]["replans"] == 2
        gates = payload["resolved_spec_json"]["config"]["hitl_policy"]["gates"]
        assert gates["web_approval_gate"]["target"] == {"node_id": "web_worker", "node_type": "web_worker"}

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

    @pytest.mark.asyncio
    async def test_list_thread_agent_runs_can_filter_awaiting_human(self, api_client, engine, sample_thread):
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
            running = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            awaiting = await repo.create_run(
                thread_id=sample_thread.id,
                template_id=template.id,
                template_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            await repo.mark_run_awaiting_human(
                awaiting.id,
                {
                    "interrupt_id": "api-list-interrupt",
                    "allowed_actions": ["approve", "reject"],
                    "title": "Approve web search?",
                },
            )

        response = api_client.get(f"/api/threads/{sample_thread.id}/agent-runs?status=awaiting_human")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "awaiting_human"
        assert [run["id"] for run in payload["agent_runs"]] == [awaiting.id]
        assert running.id not in [run["id"] for run in payload["agent_runs"]]
        assert payload["agent_runs"][0]["pending_interrupt"]["interrupt_id"] == "api-list-interrupt"

    def test_list_thread_agent_runs_returns_404_for_missing_thread(self, api_client):
        response = api_client.get(f"/api/threads/{uuid.uuid4()}/agent-runs")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_agent_run_includes_pending_interrupt_and_resume_is_idempotent(self, api_client, engine, sample_thread):
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
            await repo.mark_run_awaiting_human(
                run.id,
                {
                    "interrupt_id": "api-resume-interrupt",
                    "allowed_actions": ["approve", "reject"],
                    "resume_token": "api-token",
                    "resume_version": 2,
                    "title": "Approve final answer?",
                },
            )

        get_response = api_client.get(f"/api/agent-runs/{run.id}?thread_id={sample_thread.id}")
        assert get_response.status_code == 200
        get_payload = get_response.json()["agent_run"]
        assert get_payload["status"] == "awaiting_human"
        assert get_payload["completed_at"] is None
        assert get_payload["pending_interrupt"]["interrupt_id"] == "api-resume-interrupt"

        missing_thread_response = api_client.get(f"/api/agent-runs/{run.id}")
        assert missing_thread_response.status_code == 422

        wrong_thread_response = api_client.get(f"/api/agent-runs/{run.id}?thread_id={uuid.uuid4()}")
        assert wrong_thread_response.status_code == 404

        request_payload = {
            "action": "approve",
            "interrupt_id": "api-resume-interrupt",
            "resume_token": "api-token",
            "resume_version": 2,
            "thread_id": sample_thread.id,
            "client_metadata": {"test": "resume"},
        }
        first = api_client.post(f"/api/agent-runs/{run.id}/resume", json=request_payload)
        second = api_client.post(f"/api/agent-runs/{run.id}/resume", json=request_payload)

        assert first.status_code == 200
        first_payload = first.json()
        assert first_payload["outcome"] == "resumed"
        assert first_payload["duplicate"] is False
        assert first_payload["agent_run"]["status"] == "running"
        assert first_payload["agent_run"]["pending_interrupt"]["status"] == "resumed"
        assert first_payload["agent_run"]["metrics_json"]["interrupt_resolution_count"] == 1
        assert second.status_code == 200
        second_payload = second.json()
        assert second_payload["outcome"] == "resumed"
        assert second_payload["duplicate"] is True
        assert second_payload["agent_run"]["metrics_json"]["interrupt_resolution_count"] == 1

    @pytest.mark.asyncio
    async def test_resume_agent_run_requires_matching_thread_id(self, api_client, engine, sample_thread):
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
            await repo.mark_run_awaiting_human(
                run.id,
                {
                    "interrupt_id": "api-thread-boundary-interrupt",
                    "allowed_actions": ["approve", "reject"],
                    "resume_version": 1,
                },
            )

        missing_thread = api_client.post(
            f"/api/agent-runs/{run.id}/resume",
            json={
                "action": "approve",
                "interrupt_id": "api-thread-boundary-interrupt",
                "resume_version": 1,
            },
        )
        assert missing_thread.status_code == 422

        wrong_thread = api_client.post(
            f"/api/agent-runs/{run.id}/resume",
            json={
                "action": "approve",
                "interrupt_id": "api-thread-boundary-interrupt",
                "resume_version": 1,
                "thread_id": str(uuid.uuid4()),
            },
        )
        assert wrong_thread.status_code == 404

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
        metrics = {
            "duration_ms": 42.0,
            "route": "web",
            "node_event_count": 1,
            "node_elapsed_ms": {"router": 4.5},
            "node_total_elapsed_ms": 4.5,
            "tool_event_count": 1,
            "tool_warning_count": 0,
            "tool_error_count": 0,
            "tool_elapsed_ms": 12.3,
        }

        turn = ChatTurn(
            id=turn_id,
            thread_id=sample_thread.id,
            agent_run_id=run.id,
            agent_run_turn_kind="assistant_final",
            agent_run_sequence=0,
            agent_trace_refs_json={
                "node_ids": ["router"],
                "span_ids": ["node:router:0", "tool:search_web:0"],
                "interrupt_id": None,
            },
            status="completed",
            payload={
                "question": "What happened?",
                "answer": "Answer",
                "metadata": {
                    "agent_route": "web",
                    "agent_route_reason": "Needs live evidence.",
                },
            },
        )
        async with session_factory() as write_session:
            write_session.add(turn)
            await write_session.commit()
        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            completed_run = await repo.complete_run(
                run.id,
                status="completed",
                metrics_json=metrics,
            )
            debug_payload = build_debug_payload(
                run=completed_run,
                chat_turn_id=turn_id,
                node_events=node_telemetry,
                tool_events=tool_telemetry,
                metrics=metrics,
                route="web",
                route_reason="Needs live evidence.",
            )
            await repo.set_run_debug_trace(completed_run.id, debug_payload)

        response = api_client.get(f"/api/agent-runs/{run.id}?thread_id={sample_thread.id}")

        assert response.status_code == 200
        payload = response.json()["agent_run"]
        assert payload["id"] == run.id
        assert "chat_turn_id" not in payload
        assert payload["turns"] == [
            {
                "id": turn.id,
                "kind": "assistant_final",
                "sequence": 0,
                "trace_refs": {
                    "node_ids": ["router"],
                    "span_ids": ["node:router:0", "tool:search_web:0"],
                    "interrupt_id": None,
                },
            }
        ]
        assert payload["metrics_json"]["tool_event_count"] == 1
        assert set(payload["debug"]) == {"version", "trace", "summary", "graph"}
        assert "node_events" not in payload["debug"]
        assert "tool_events" not in payload["debug"]
        assert payload["debug"]["version"] == 1
        assert payload["debug"]["summary"]["route"] == "web"
        assert payload["debug"]["graph"]["selectedRoute"] == "web"
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
    async def test_get_agent_run_returns_null_debug_when_trace_not_captured(self, api_client, engine, sample_thread):
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

        response = api_client.get(f"/api/agent-runs/{run.id}?thread_id={sample_thread.id}")

        assert response.status_code == 200
        payload = response.json()["agent_run"]
        assert payload["status"] == "failed"
        assert payload["debug"] is None

    @pytest.mark.asyncio
    async def test_get_agent_run_returns_null_debug_for_malformed_trace_payload(self, api_client, engine, sample_thread):
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
                status="completed",
                metrics_json={"duration_ms": 1.0, "route": "direct"},
                debug_trace_json={"version": 1, "trace": {"schema_version": 1}},
            )

        response = api_client.get(f"/api/agent-runs/{run.id}?thread_id={sample_thread.id}")

        assert response.status_code == 200
        payload = response.json()["agent_run"]
        assert payload["status"] == "completed"
        assert payload["debug"] is None

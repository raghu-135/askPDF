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

from app.agent.tool_registry import collect_tool_contract_metadata_errors, tool_contracts_by_id
from app.agent_workflows.checkpointing import open_agent_checkpointer
from app.agent_workflows.router_runtime import handle_router_rag_chat
from app.agent_workflows.graph import (
    NodeRegistry,
    WorkflowCompiler,
    _final_context_from_state,
    _llm_result_metadata,
    _route_function_for_edge,
    evaluator_route,
    hitl_gate_route,
    hitl_gate_route_for,
    planner_route,
    router_route,
)
from app.agent_workflows.graph import (
    build_planner_prompt,
    infer_required_plan_steps,
    normalize_execution_plan,
    normalize_evaluator_report,
)
from app.agent_workflows.debug_trace import AgentTraceRecorder, build_debug_payload, build_debug_trace, build_runtime_trace_event
from app.agent_workflows.trace_details import TRACE_DETAIL_SCALAR_LIMIT, sanitize_trace_detail
from app.agent_workflows.trace_payloads import merge_debug_payloads
from app.agent_workflows.metrics import build_run_metrics
from app.agent_workflows.node_catalog import collect_node_catalog_errors, get_node_catalog
from app.agent_workflows.repository import AgentWorkflowRepository, AgentRunInterruptError
from app.agent_workflows.route_registry import collect_route_function_registry_errors, get_route_function_registry
from app.agent_workflows.service import AgentRunService
from app.agent_workflows.execution_stream import AgentExecutionEventSink
from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.agent_workflows.validator import WorkflowResolver, WorkflowValidationError, WorkflowValidator
from app.db import get_thread_settings
from app.db.models_sqlmodel import AgentWorkflow, AgentRun, ChatTurn, Thread
from app.models.llm_server_client import REPLANS_LIMIT
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z, utc_now


SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
TRACE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "agent_debug_trace_v1.schema.json"

ROUTER_RAG_AGENT_ID = "router_rag_agent"
PLAN_EXECUTE_RAG_AGENT_ID = "plan_execute_rag_agent"
EVALUATOR_REPLANNER_RAG_AGENT_ID = "evaluator_replanner_rag_agent"
ROUTER_RAG_AGENT_VERSION = 2
PLAN_EXECUTE_RAG_AGENT_VERSION = 2
EVALUATOR_REPLANNER_RAG_AGENT_VERSION = 2
ROUTER_RAG_AGENT_V2_VERSION = 2
PLAN_EXECUTE_RAG_AGENT_V2_VERSION = 2
EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION = 2


def _builtin_spec(builtin_key: str) -> dict:
    for workflow in load_builtin_workflows():
        if workflow.get("builtin_key") == builtin_key:
            return workflow["spec_json"]
    raise AssertionError(f"Missing builtin workflow fixture: {builtin_key}")


def builtin_router_rag_spec() -> dict:
    return _builtin_spec(ROUTER_RAG_AGENT_ID)


def builtin_router_rag_v2_spec() -> dict:
    return _builtin_spec(ROUTER_RAG_AGENT_ID)


def legacy_builtin_router_rag_v1_spec() -> dict:
    spec = _builtin_spec(ROUTER_RAG_AGENT_ID)
    spec["schema_version"] = 1
    return spec


def builtin_router_rag_hitl_web_spec() -> dict:
    return _builtin_spec(ROUTER_RAG_AGENT_ID)


def builtin_plan_execute_rag_spec() -> dict:
    return _builtin_spec(PLAN_EXECUTE_RAG_AGENT_ID)


def builtin_plan_execute_rag_v2_spec() -> dict:
    return _builtin_spec(PLAN_EXECUTE_RAG_AGENT_ID)


def builtin_evaluator_replanner_rag_spec() -> dict:
    return _builtin_spec(EVALUATOR_REPLANNER_RAG_AGENT_ID)


def builtin_evaluator_replanner_rag_v2_spec() -> dict:
    return _builtin_spec(EVALUATOR_REPLANNER_RAG_AGENT_ID)


def make_trace_recorder(run_id: str, thread_id: str, spec: dict, workflow_id: str = ROUTER_RAG_AGENT_ID) -> AgentTraceRecorder:
    return AgentTraceRecorder(
        SimpleNamespace(
            id=run_id,
            thread_id=thread_id,
            user_id=None,
            workflow_id=workflow_id,
            workflow_version_id=f"{workflow_id}:v1",
            resolved_spec_json=spec,
            status="running",
            started_at=utc_now(),
            completed_at=None,
        )
    )


def test_trace_details_keep_loop_visits_full_reasoning_checkpoints_and_final_answer():
    run = SimpleNamespace(
        id="run-full-details",
        thread_id="thread-full-details",
        user_id=None,
        workflow_id=ROUTER_RAG_AGENT_ID,
        resolved_spec_json=builtin_router_rag_v2_spec(),
        status="completed",
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    recorder = AgentTraceRecorder(run)
    long_answer = "First line.\n" + ("Complete answer content. " * 80)
    reasoning = "Inspect evidence.\nChoose the document route."
    response = SimpleNamespace(
        content=long_answer,
        additional_kwargs={"reasoning_content": reasoning},
        response_metadata={},
        usage_metadata={},
    )

    for visit_index in (1, 2):
        before = {
            "question": "What changed?",
            "authorization": "Bearer secret",
            "evidence": f"evidence before visit {visit_index}",
            "node_events": [{"old": True}],
        }
        recorder.record_node_started(
            node_id="evidence_evaluator",
            node_type="evidence_evaluator",
            visit_index=visit_index,
            state=before,
        )
        recorder.record_llm_detail(
            node_id="evidence_evaluator",
            node_type="evidence_evaluator",
            visit_index=visit_index,
            messages=[SimpleNamespace(type="system", content="Evaluate all evidence."), SimpleNamespace(type="human", content="Full prompt")],
            response=response,
        )
        recorder.record_node_completed(
            node_id="evidence_evaluator",
            node_type="evidence_evaluator",
            visit_index=visit_index,
            state=before,
            update={"evaluator_route": "answer", "final_answer": long_answer},
            status="completed",
            event={"evaluator_route": "answer"},
        )

    payload = recorder.finalize(
        run=run,
        chat_turn_id=None,
        metrics={},
        route="document",
        result={"final_answer": long_answer, "route": "document", "reasoning": reasoning, "reasoning_available": True},
    )

    assert [(detail["node_id"], detail["visit_index"]) for detail in payload["details"]] == [
        ("evidence_evaluator", 1),
        ("evidence_evaluator", 2),
    ]
    assert payload["details"][0]["checkpoint_before"]["authorization"] == "[redacted]"
    assert "checkpoint_before.authorization" in payload["details"][0]["safety"]["redacted_fields"]
    assert "node_events" not in payload["details"][0]["checkpoint_before"]
    assert payload["details"][0]["llm"]["reasoning"] == reasoning
    assert payload["details"][0]["changes"]["added"]["final_answer"] == long_answer
    assert payload["final_output"]["answer"] == long_answer
    assert len(payload["final_output"]["answer"]) > 900


def test_resumed_trace_details_share_one_run_size_limit(monkeypatch):
    import app.agent_workflows.trace_details as trace_details

    monkeypatch.setattr(trace_details, "TRACE_DETAIL_RUN_LIMIT", 900)
    trace = {"schema_version": 1, "spans": [], "metrics": {}}
    base = {
        "version": 1,
        "trace": dict(trace),
        "summary": {},
        "details": [{"node_id": "first", "visit_index": 1, "status": "completed", "output": {"text": "a" * 500}}],
    }
    incoming = {
        "version": 1,
        "trace": dict(trace),
        "summary": {},
        "details": [{"node_id": "second", "visit_index": 1, "status": "completed", "output": {"text": "b" * 500}}],
    }

    merged = merge_debug_payloads(base, incoming, resolved_spec={})

    assert merged["detail_safety"]["size_bytes"] <= 900
    assert merged["detail_safety"]["truncated"] is True
    assert any(detail.get("safety", {}).get("run_limit_reached") for detail in merged["details"])


def test_trace_detail_scalar_limit_is_explicit_and_preserves_normal_whitespace():
    normal = "line one\nline two"
    oversized = "x" * (TRACE_DETAIL_SCALAR_LIMIT + 25)
    value, safety = sanitize_trace_detail({"normal": normal, "oversized": oversized, "api_key": "secret"})

    assert value["normal"] == normal
    assert value["api_key"] == "[redacted]"
    assert value["oversized"].endswith("[truncated]")
    assert safety["truncated"] is True
    assert "oversized" in safety["truncated_fields"]
    assert "api_key" in safety["redacted_fields"]


async def create_agent_run_record(
    session_factory,
    *,
    run_id: str,
    thread_id: str,
    spec: dict,
    workflow_id: str = ROUTER_RAG_AGENT_ID,
) -> AgentRun:
    async with session_factory() as repo_session:
        repo = AgentWorkflowRepository(repo_session)
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(workflow_id)
        run = AgentRun(
            id=run_id,
            thread_id=thread_id,
            workflow_id=workflow.id,
            run_metadata_json={"workflow_version_id": version.id, "workflow_version": version.version},
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
            workflow_id=PLAN_EXECUTE_RAG_AGENT_ID,
            workflow_version_id=f"{PLAN_EXECUTE_RAG_AGENT_ID}:v1",
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
            workflow_id=ROUTER_RAG_AGENT_ID,
            workflow_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
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
            workflow_id=ROUTER_RAG_AGENT_ID,
            workflow_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
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
            workflow_id=PLAN_EXECUTE_RAG_AGENT_ID,
            workflow_version_id=f"{PLAN_EXECUTE_RAG_AGENT_ID}:v1",
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
            workflow_id=ROUTER_RAG_AGENT_ID,
            workflow_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
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
            workflow_id=ROUTER_RAG_AGENT_ID,
            workflow_version_id=f"{ROUTER_RAG_AGENT_ID}:v1",
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


class TestRouterRagWorkflowValidator:
    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda spec: spec["config"].update({"surprise": True}), "unknown config keys: surprise"),
            (lambda spec: spec["config"].update({"allowed_tool_ids": ["not_a_tool"]}), "unknown allowed_tool_ids: not_a_tool"),
            (lambda spec: spec["config"].update({"replans": 999}), "replans must be between"),
        ],
    )
    def test_rejects_invalid_router_rag_specs(self, mutate, expected):
        spec = builtin_router_rag_v2_spec()
        mutate(spec)

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert expected in str(exc.value)

    def test_resolver_freezes_thread_and_request_overrides(self):
        resolved = WorkflowResolver().resolve(
            builtin_router_rag_v2_spec(),
            thread_settings={"replans": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )

        assert "replans" not in resolved["config"]
        assert resolved["config"]["use_reranker"] is False
        assert resolved["config"]["use_web_search"] is True

        evaluator_resolved = WorkflowResolver().resolve(
            builtin_evaluator_replanner_rag_v2_spec(),
            thread_settings={"replans": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )
        assert evaluator_resolved["config"]["replans"] == 3
        assert evaluator_resolved["config"]["loop_policy"]["node_visit_limits"]["replanner"] == 3
        assert evaluator_resolved["config"]["loop_policy"]["node_visit_limits"]["evidence_evaluator"] == 4
        assert evaluator_resolved["config"]["loop_policy"]["max_total_visits"] == 28

    def test_rejects_zero_replan_budget(self):
        spec = builtin_evaluator_replanner_rag_v2_spec()
        spec["config"]["replans"] = 0

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert "replans must be between" in str(exc.value)

    @pytest.mark.parametrize(
        "policy_update, expected",
        [
            ({"final_prompt_assembly": "unsupported"}, "context_policy.final_prompt_assembly must be one of"),
            ({"evidence_compression": "lossy"}, "context_policy.evidence_compression must be one of"),
        ],
    )
    def test_rejects_unsupported_context_policy_modes(self, policy_update, expected):
        spec = builtin_router_rag_v2_spec()
        spec["config"]["context_policy"].update(policy_update)

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert expected in str(exc.value)

    def test_rejects_legacy_v1_builtin_specs(self):
        with pytest.raises(WorkflowValidationError, match="schema_version must be 2"):
            WorkflowValidator().validate(legacy_builtin_router_rag_v1_spec())

    def test_accepts_builtin_router_rag_spec(self):
        result = WorkflowValidator().validate(builtin_router_rag_v2_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_router_rag_hitl_web_spec(self):
        spec = builtin_router_rag_v2_spec()
        spec["config"]["hitl_policy"] = builtin_router_rag_hitl_web_spec()["config"]["hitl_policy"]
        result = WorkflowValidator().validate(spec)

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_plan_execute_rag_spec(self):
        result = WorkflowValidator().validate(builtin_plan_execute_rag_v2_spec())

        assert result == {"valid": True, "errors": []}

    def test_accepts_builtin_evaluator_replanner_rag_spec(self):
        result = WorkflowValidator().validate(builtin_evaluator_replanner_rag_v2_spec())

        assert result == {"valid": True, "errors": []}

    def test_evaluator_replanner_loop_policy_matches_replan_budget(self):
        spec = builtin_evaluator_replanner_rag_v2_spec()
        spec["config"]["replans"] = 2
        spec["config"]["loop_policy"] = {
            "max_total_visits": 22,
            "default_max_node_visits": 1,
            "node_visit_limits": {
                "retrieval_worker": 3,
                "memory_worker": 3,
                "timeline_worker": 3,
                "web_worker": 3,
                "evidence_evaluator": 3,
                "replanner": 2,
            },
        }

        result = WorkflowValidator().validate(spec)

        assert result == {"valid": True, "errors": []}

    def test_node_catalog_exposes_required_authoring_metadata(self):
        catalog = get_node_catalog()

        assert collect_node_catalog_errors(catalog) == []
        retrieval = catalog["retrieval_worker"]
        assert retrieval["state_reads"]
        assert "evidence_packets" in retrieval["state_writes"]
        assert retrieval["prompt_slots"] == []
        assert retrieval["context_policy"]["mode"] == "append_evidence"
        assert retrieval["observability"]["span_kind"] == "tool_worker"
        assert retrieval["max_instances"] >= 1

    def test_node_catalog_shape_validation_reports_bad_metadata(self):
        catalog = get_node_catalog()
        catalog["retrieval_worker"].pop("state_reads")
        catalog["router"]["max_instances"] = 0

        errors = collect_node_catalog_errors(catalog)

        assert "retrieval_worker missing catalog keys: state_reads" in errors
        assert "retrieval_worker.state_reads must be a list of non-empty strings" in errors
        assert "router.max_instances must be a positive integer" in errors

    def test_route_function_registry_shape_validation_reports_bad_metadata(self):
        registry = get_route_function_registry()
        registry["router_route"].pop("allowed_source_types")
        registry["planner_route"]["route_labels"] = ["execute", ""]

        errors = collect_route_function_registry_errors(registry)

        assert "router_route missing registry keys: allowed_source_types" in errors
        assert "router_route.allowed_source_types must be a list of non-empty strings" in errors
        assert "planner_route.route_labels must be null or a list of non-empty strings" in errors

    def test_tool_contract_registry_shape_validation_reports_bad_metadata(self):
        records = [record for records in tool_contracts_by_id().values() for record in records]
        records[0] = dict(records[0])
        records[0]["allowed_node_types"] = []
        records[0]["required_node_capabilities"] = []
        records[1] = dict(records[1])
        records[1]["artifact_keys"] = ["document_sources", ""]

        errors = collect_tool_contract_metadata_errors(records)

        assert any(error.endswith("must declare allowed_node_types or required_node_capabilities") for error in errors)
        assert any(error.endswith("artifact_keys must be a list of non-empty strings") for error in errors)

    @pytest.mark.parametrize(
        "spec_factory, expected",
        [
            (
                builtin_router_rag_v2_spec,
                {
                    "node_ids": ["context_loader", "router", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "direct_answer", "synthesizer", "finalizer"],
                    "node_types": {
                        "context_loader": "context_loader",
                        "router": "router",
                        "retrieval_worker": "retrieval_worker",
                        "memory_worker": "memory_worker",
                        "timeline_worker": "timeline_worker",
                        "web_worker": "web_worker",
                        "direct_answer": "direct_answer",
                        "synthesizer": "synthesizer",
                        "finalizer": "finalizer",
                    },
                    "edges": [
                        ("START", "context_loader"),
                        ("context_loader", "router"),
                        ("retrieval_worker", "synthesizer"),
                        ("memory_worker", "synthesizer"),
                        ("timeline_worker", "synthesizer"),
                        ("web_worker", "synthesizer"),
                        ("direct_answer", "finalizer"),
                        ("synthesizer", "finalizer"),
                        ("finalizer", "END"),
                    ],
                    "conditional_edges": {
                        "router": {
                            "route_fn": "router_route",
                            "routes": {
                                "document": "retrieval_worker",
                                "memory": "memory_worker",
                                "timeline": "timeline_worker",
                                "web": "web_worker",
                                "direct": "direct_answer",
                                "clarify": "finalizer",
                            },
                        }
                    },
                },
            ),
            (
                builtin_plan_execute_rag_v2_spec,
                {
                    "node_ids": ["context_loader", "planner", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "direct_answer", "synthesizer", "finalizer"],
                    "node_types": {
                        "context_loader": "context_loader",
                        "planner": "planner",
                        "retrieval_worker": "retrieval_worker",
                        "memory_worker": "memory_worker",
                        "timeline_worker": "timeline_worker",
                        "web_worker": "web_worker",
                        "direct_answer": "direct_answer",
                        "synthesizer": "synthesizer",
                        "finalizer": "finalizer",
                    },
                    "edges": [
                        ("START", "context_loader"),
                        ("context_loader", "planner"),
                        ("retrieval_worker", "memory_worker"),
                        ("memory_worker", "timeline_worker"),
                        ("timeline_worker", "web_worker"),
                        ("web_worker", "synthesizer"),
                        ("direct_answer", "finalizer"),
                        ("synthesizer", "finalizer"),
                        ("finalizer", "END"),
                    ],
                    "conditional_edges": {
                        "planner": {
                            "route_fn": "planner_route",
                            "routes": {
                                "execute": "retrieval_worker",
                                "direct": "direct_answer",
                                "clarify": "finalizer",
                            },
                        }
                    },
                },
            ),
            (
                builtin_evaluator_replanner_rag_v2_spec,
                {
                    "node_ids": [
                        "context_loader",
                        "planner",
                        "retrieval_worker",
                        "memory_worker",
                        "timeline_worker",
                        "web_worker",
                        "evidence_evaluator",
                        "replanner",
                        "direct_answer",
                        "synthesizer",
                        "finalizer",
                    ],
                    "node_types": {
                        "context_loader": "context_loader",
                        "planner": "planner",
                        "retrieval_worker": "retrieval_worker",
                        "memory_worker": "memory_worker",
                        "timeline_worker": "timeline_worker",
                        "web_worker": "web_worker",
                        "evidence_evaluator": "evidence_evaluator",
                        "replanner": "replanner",
                        "direct_answer": "direct_answer",
                        "synthesizer": "synthesizer",
                        "finalizer": "finalizer",
                    },
                    "edges": [
                        ("START", "context_loader"),
                        ("context_loader", "planner"),
                        ("retrieval_worker", "memory_worker"),
                        ("memory_worker", "timeline_worker"),
                        ("timeline_worker", "web_worker"),
                        ("web_worker", "evidence_evaluator"),
                        ("replanner", "retrieval_worker"),
                        ("direct_answer", "finalizer"),
                        ("synthesizer", "finalizer"),
                        ("finalizer", "END"),
                    ],
                    "conditional_edges": {
                        "planner": {
                            "route_fn": "planner_route",
                            "routes": {
                                "execute": "retrieval_worker",
                                "direct": "direct_answer",
                                "clarify": "finalizer",
                            },
                        },
                        "evidence_evaluator": {
                            "route_fn": "evaluator_route",
                            "routes": {
                                "answer": "synthesizer",
                                "replan": "replanner",
                                "answer_budget_exhausted": "synthesizer",
                            },
                        },
                    },
                },
            ),
        ],
    )
    def test_materialized_builtin_graph_signatures_are_stable(self, spec_factory, expected):
        materialized = WorkflowCompiler().materialize_spec(spec_factory())
        WorkflowValidator().validate(materialized)
        graph_spec = materialized["config"]["graph"]

        assert [node["id"] for node in graph_spec["nodes"]] == expected["node_ids"]
        assert {node["id"]: node["type"] for node in graph_spec["nodes"]} == expected["node_types"]
        assert [(edge.get("from"), edge.get("to")) for edge in graph_spec["edges"] if not edge.get("conditional")] == expected["edges"]
        assert {
            edge["from"]: {"route_fn": edge.get("route_fn"), "routes": edge.get("routes")}
            for edge in graph_spec["edges"]
            if edge.get("conditional")
        } == expected["conditional_edges"]
        assert graph_spec["hitl_compiled"] is True
        assert materialized["config"]["loop_policy"]["max_total_visits"] >= len(graph_spec["nodes"])

        for node in graph_spec["nodes"]:
            assert node["label"] == get_node_catalog()[node["type"]]["display_name"]
            assert node["category"] == get_node_catalog()[node["type"]]["category"]
            assert "observability" in node

        assert WorkflowCompiler().compile(materialized) is not None

    def test_route_helpers_are_stable_for_current_state_keys(self):
        assert router_route({"route": "document"}) == "document"
        assert router_route({"route": "surprise"}) == "document"
        assert planner_route({"route": "direct"}) == "direct"
        assert planner_route({"route": "surprise"}) == "execute"
        assert evaluator_route({"evaluator_route": "answer"}) == "answer"
        assert evaluator_route({"evaluator_route": "surprise"}) == "answer"
        assert hitl_gate_route({"hitl_gate_route": "continue_without"}) == "continue_without"
        assert hitl_gate_route({}) == "continue_without"
        assert hitl_gate_route_for("approval_1")({"hitl_gate_routes": {"approval_1": "approve"}}) == "approve"
        assert hitl_gate_route_for("approval_1")({"hitl_gate_routes": {}}) == "continue_without"

    def test_rejects_router_rag_graph_topology_changes(self):
        spec = builtin_router_rag_v2_spec()
        spec["config"]["graph"]["nodes"].append({"id": "surprise", "type": "retrieval_worker"})

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert "graph contains unreachable nodes: surprise" in str(exc.value)

    def test_rejects_router_rag_specs_missing_required_tools(self):
        spec = builtin_router_rag_v2_spec()
        spec["config"]["allowed_tool_ids"].remove("document_evidence")

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert "missing required allowed_tool_ids: document_evidence" in str(exc.value)

    def test_compiles_builtin_router_rag_spec(self):
        graph = WorkflowCompiler().compile(builtin_router_rag_v2_spec())

        assert graph is not None

    def test_compiler_requires_explicit_route_function_after_materialization(self):
        compiler = WorkflowCompiler()
        materialized = compiler.materialize_spec(builtin_router_rag_v2_spec())
        router_edge = next(
            edge
            for edge in materialized["config"]["graph"]["edges"]
            if edge.get("from") == "router" and edge.get("conditional")
        )

        assert router_edge["route_fn"] == "router_route"
        with pytest.raises(ValueError, match="must declare route_fn"):
            _route_function_for_edge(
                {"from": "router", "conditional": True, "routes": {"direct": "finalizer"}},
                source="router",
                node_types={"router": "router"},
            )

    def test_compiles_builtin_router_rag_hitl_web_spec(self):
        spec = builtin_router_rag_v2_spec()
        spec["config"]["hitl_policy"] = builtin_router_rag_hitl_web_spec()["config"]["hitl_policy"]
        graph = WorkflowCompiler().compile(spec)

        assert graph is not None

    def test_materializes_generic_hitl_gate_overlay_for_action_node(self):
        spec = builtin_router_rag_v2_spec()
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

        WorkflowValidator().validate(spec)
        materialized = WorkflowCompiler().materialize_spec(spec)
        WorkflowValidator().validate(materialized)
        graph_spec = materialized["config"]["graph"]

        review_gate = next(node for node in graph_spec["nodes"] if node.get("id") == "review_before_documents")
        assert review_gate["type"] == "hitl_gate"
        assert review_gate["label"] == "HITL Gate"
        assert review_gate["category"] == "human_review"
        router_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "router")
        assert router_edge["routes"]["document"] == "review_before_documents"
        gate_edge = next(edge for edge in graph_spec["edges"] if edge.get("from") == "review_before_documents")
        assert gate_edge["routes"] == {"approve": "retrieval_worker", "continue_without": "synthesizer"}

    def test_materializes_multi_select_choice_gate_overlay(self):
        spec = builtin_plan_execute_rag_v2_spec()
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

        WorkflowValidator().validate(spec)
        materialized = WorkflowCompiler().materialize_spec(spec)
        WorkflowValidator().validate(materialized)
        graph_spec = materialized["config"]["graph"]

        choice_gate = next(node for node in graph_spec["nodes"] if node.get("id") == "research_source_choice")
        assert choice_gate["type"] == "hitl_gate"
        assert choice_gate["label"] == "HITL Gate"
        assert choice_gate["category"] == "human_review"
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
        spec = builtin_router_rag_v2_spec()
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
        materialized = WorkflowCompiler().materialize_spec(spec)
        WorkflowValidator().validate(materialized)
        config = materialized["config"]
        graph_spec = config["graph"]

        assert config["hitl_policy"]["enabled"] is True
        assert config["hitl_policy"]["gates"]["human_review_gate"]["mode"] == "review"
        review_gate = next(node for node in graph_spec["nodes"] if node.get("id") == "human_review_gate")
        assert review_gate["type"] == "hitl_gate"
        assert review_gate["label"] == "HITL Gate"
        assert review_gate["category"] == "human_review"
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
        graph = WorkflowCompiler().compile(builtin_plan_execute_rag_v2_spec())

        assert graph is not None

    def test_compiles_builtin_evaluator_replanner_rag_spec(self):
        graph = WorkflowCompiler().compile(builtin_evaluator_replanner_rag_v2_spec())

        assert graph is not None

    def test_rejects_plan_execute_graph_topology_changes(self):
        spec = builtin_plan_execute_rag_v2_spec()
        spec["config"]["graph"]["edges"].append({"from": "planner", "to": "synthesizer"})

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert "node planner type planner cannot connect to synthesizer type synthesizer" in str(exc.value)

    def test_rejects_evaluator_replanner_graph_topology_changes(self):
        spec = builtin_evaluator_replanner_rag_v2_spec()
        spec["config"]["graph"]["edges"].append({"from": "evidence_evaluator", "to": "finalizer"})

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

        assert "node evidence_evaluator type evidence_evaluator cannot connect to finalizer type finalizer" in str(exc.value)

    def test_rejects_evaluator_replanner_unbounded_replans(self):
        spec = builtin_evaluator_replanner_rag_v2_spec()
        spec["config"]["replans"] = REPLANS_LIMIT + 1

        with pytest.raises(WorkflowValidationError) as exc:
            WorkflowValidator().validate(spec)

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

    def test_normalize_execution_plan_coerces_clarification_options_to_strings(self):
        normalized = normalize_execution_plan(
            {
                "route": "clarify",
                "execution_plan": [],
                "reason": "ambiguous",
                "clarification_options": [
                    {"text": "Which uploaded document?"},
                    "Which previous answer?",
                    {"label": "not a supported option shape"},
                ],
            },
            use_web_search=False,
            question="Which one?",
        )

        assert normalized["route"] == "clarify"
        assert normalized["clarification_options"] == [
            "Which uploaded document?",
            "Which previous answer?",
            "not a supported option shape",
        ]

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
        from app.agent_workflows.graph import _tool_config

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
            "workflow_id": "custom_rag_agent",
            "runtime": builtin_router_rag_v2_spec()["runtime"],
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

        assert WorkflowValidator().validate(spec)["valid"] is True
        assert WorkflowCompiler().compile(spec) is not None

    def test_v2_custom_graph_rejects_incompatible_node_catalog(self, monkeypatch):
        catalog = get_node_catalog()
        catalog["router"].pop("context_policy")
        monkeypatch.setattr("app.agent_workflows.validator.get_node_catalog", lambda: catalog)

        spec = builtin_router_rag_v2_spec()

        with pytest.raises(WorkflowValidationError, match="node catalog incompatible"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_incompatible_route_function_registry(self, monkeypatch):
        registry = get_route_function_registry()
        registry["router_route"]["route_labels"] = ["document", ""]
        monkeypatch.setattr("app.agent_workflows.validator.get_route_function_registry", lambda: registry)

        spec = builtin_router_rag_v2_spec()

        with pytest.raises(WorkflowValidationError, match="route function registry incompatible"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_catalog_route_registry_mismatch(self, monkeypatch):
        registry = get_route_function_registry()
        registry["router_route"]["allowed_source_types"] = ["planner"]
        monkeypatch.setattr("app.agent_workflows.validator.get_route_function_registry", lambda: registry)

        spec = builtin_router_rag_v2_spec()

        with pytest.raises(WorkflowValidationError, match="node catalog type router allows route_fn router_route"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_incompatible_tool_contract_registry(self, monkeypatch):
        contracts = tool_contracts_by_id()
        contracts["document_evidence"] = [dict(contracts["document_evidence"][0])]
        contracts["document_evidence"][0]["artifact_keys"] = ["document_sources", ""]
        monkeypatch.setattr("app.agent_workflows.validator.tool_contracts_by_id", lambda: contracts)

        spec = builtin_router_rag_v2_spec()

        with pytest.raises(WorkflowValidationError, match="tool contract registry incompatible"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_catalog_tool_contract_mismatch(self, monkeypatch):
        contracts = tool_contracts_by_id()
        contracts["document_evidence"] = [dict(contracts["document_evidence"][0])]
        contracts["document_evidence"][0]["allowed_node_types"] = ["memory_worker"]
        contracts["document_evidence"][0]["required_node_capabilities"] = ["retrieval.memory"]
        monkeypatch.setattr("app.agent_workflows.validator.tool_contracts_by_id", lambda: contracts)

        spec = builtin_router_rag_v2_spec()

        with pytest.raises(
            WorkflowValidationError,
            match="node catalog type retrieval_worker allows tool contract document_evidence",
        ):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_node_type_instance_limit_overflow(self):
        spec = {
            "schema_version": 2,
            "workflow_id": "custom_rag_agent",
            "runtime": builtin_evaluator_replanner_rag_v2_spec()["runtime"],
            "config": {
                "allowed_tool_ids": ["thread_shape", "document_evidence", "clarify_intent"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "context_2", "type": "context_loader"},
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
                            "routes": {"document": "retrieval_1", "clarify": "final_1"},
                        },
                        {"from": "retrieval_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        errors = WorkflowValidator().collect_errors(spec)

        assert "graph has 2 nodes of type context_loader; maximum allowed is 1" in errors

    def test_v2_custom_graph_rejects_node_contract_metadata_not_allowed_by_catalog(self):
        spec = {
            "schema_version": 2,
            "workflow_id": "custom_rag_agent",
            "runtime": builtin_evaluator_replanner_rag_v2_spec()["runtime"],
            "config": {
                "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "router_1", "type": "router"},
                        {
                            "id": "retrieval_1",
                            "type": "retrieval_worker",
                            "state_writes": ["final_answer"],
                            "prompt_slots": ["router"],
                            "context_policy": {"mode": "assemble_answer"},
                            "observability": {"span_kind": "answer", "summary_fields": ["answer_chars"]},
                        },
                        {"id": "final_1", "type": "finalizer"},
                    ],
                    "edges": [
                        {"from": "START", "to": "context_1"},
                        {"from": "context_1", "to": "router_1"},
                        {
                            "from": "router_1",
                            "conditional": True,
                            "route_fn": "router_route",
                            "routes": {"document": "retrieval_1", "clarify": "final_1"},
                        },
                        {"from": "retrieval_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        errors = WorkflowValidator().collect_errors(spec)

        assert (
            "graph node retrieval_1.state_writes includes unsupported values for type retrieval_worker: final_answer"
            in errors
        )
        assert "graph node retrieval_1.prompt_slots includes unsupported values for type retrieval_worker: router" in errors
        assert (
            "graph node retrieval_1.context_policy.mode must match catalog value append_evidence for type retrieval_worker"
            in errors
        )
        assert (
            "graph node retrieval_1.observability.span_kind must match catalog value tool_worker for type retrieval_worker"
            in errors
        )
        assert (
            "graph node retrieval_1.observability.summary_fields includes unsupported values for type retrieval_worker: answer_chars"
            in errors
        )

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
            "workflow_id": "custom_rag_agent",
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

        with pytest.raises(WorkflowValidationError, match=match):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_tool_ids_not_supported_by_graph_nodes(self):
        spec = {
            "schema_version": 2,
            "workflow_id": "custom_rag_agent",
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

        with pytest.raises(WorkflowValidationError, match="not supported by any node"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_rejects_unbounded_cycles(self):
        spec = {
            "schema_version": 2,
            "workflow_id": "custom_rag_agent",
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

        with pytest.raises(WorkflowValidationError, match="requires loop_policy"):
            WorkflowValidator().validate(spec)

    def test_v2_custom_graph_accepts_bounded_cycles(self):
        spec = {
            "schema_version": 2,
            "workflow_id": "custom_rag_agent",
            "runtime": builtin_evaluator_replanner_rag_v2_spec()["runtime"],
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

        assert WorkflowValidator().validate(spec)["valid"] is True
        assert WorkflowCompiler().compile(spec) is not None

    @pytest.mark.asyncio
    async def test_bound_node_spec_enforces_visit_limits(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {"content": "Document evidence."}

        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeTool())
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
    async def test_hitl_gate_interrupt_limit_is_visit_scoped_in_loops(self, monkeypatch):
        interrupt_payloads = []

        def fake_interrupt(payload):
            interrupt_payloads.append(payload)
            return {"action": "approve"}

        monkeypatch.setattr("app.agent_workflows.graph.interrupt", fake_interrupt)
        bound = NodeRegistry().get_for_spec({"id": "approval_1", "type": "hitl_gate"})
        update = await bound(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "Should web run again?",
                "route": "web",
                "node_events": [],
                "tool_events": [],
                "node_visit_counts": {"approval_1": 1},
                "node_visit_sequence": [{"node": "approval_1", "node_type": "hitl_gate", "visit_index": 1}],
                "hitl_interrupt_counts": {"approval_1": 1, "approval_1:visit:1": 1},
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
                "loop_policy": {
                    "max_total_visits": 6,
                    "default_max_node_visits": 2,
                    "node_visit_limits": {"approval_1": 2},
                },
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert interrupt_payloads[0]["visit_index"] == 2
        assert interrupt_payloads[0]["interrupt_count_key"] == "approval_1:visit:2"
        assert update["hitl_gate_route"] == "approve"
        assert update["hitl_interrupt_counts"]["approval_1"] == 2
        assert update["hitl_interrupt_counts"]["approval_1:visit:1"] == 1
        assert update["hitl_interrupt_counts"]["approval_1:visit:2"] == 1
        assert update["hitl_decisions"][-1]["visit_index"] == 2
        assert update["hitl_decisions"][-1]["interrupt_count_key"] == "approval_1:visit:2"
        assert update["node_events"][-1]["visit_index"] == 2

    @pytest.mark.asyncio
    async def test_bound_node_spec_emits_instance_id_and_node_type(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {
                    "content": "Document evidence.",
                    "artifacts": {"document_sources": [{"file_hash": "file-1"}]},
                }

        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeTool())
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
    async def test_context_policy_bounds_evidence_packets_and_accumulated_evidence(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {
                    "content": " ".join(["new-document-evidence"] * 20),
                    "artifacts": {"document_sources": [{"file_hash": "file-1"}]},
                }

        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeTool())
        bound = NodeRegistry().get_for_spec({"id": "retrieval_1", "type": "retrieval_worker"})
        update = await bound(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What does the document say?",
                "route": "document",
                "evidence": " ".join(["previous-evidence"] * 80),
                "evidence_packets": [
                    {"id": "old-1", "content": "old one"},
                    {"id": "old-2", "content": "old two"},
                ],
                "context_policy": {
                    "evidence_packet_limit": 2,
                    "evidence_packet_content_limit": 24,
                    "final_prompt_assembly": "evidence_packets",
                },
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "allowed_tool_ids": ["document_evidence"],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        assert len(update["evidence_packets"]) == 2
        assert [packet["id"] for packet in update["evidence_packets"]][0] == "old-2"
        assert len(update["evidence_packets"][-1]["content"]) <= 27
        assert len(update["evidence"]) <= 2 * (24 + 128)

    @pytest.mark.asyncio
    async def test_context_policy_dedupes_evidence_packets_by_content_and_refs(self, monkeypatch):
        class FakeTool:
            async def ainvoke(self, _args, config=None):
                return {
                    "content": "duplicate evidence paragraph",
                    "artifacts": {"document_sources": [{"file_hash": "file-1", "page": 1}]},
                }

        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeTool())
        bound = NodeRegistry().get_for_spec({"id": "retrieval_1", "type": "retrieval_worker"})
        update = await bound(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What does the document say?",
                "route": "document",
                "evidence": "",
                "evidence_packets": [
                    {
                        "id": "old-duplicate",
                        "kind": "document",
                        "content": "duplicate evidence paragraph",
                        "refs": {"document_matches": [{"file_hash": "file-1"}]},
                    },
                    {"id": "old-unique", "kind": "document", "content": "unique evidence"},
                ],
                "context_policy": {
                    "evidence_packet_limit": 5,
                    "evidence_packet_content_limit": 200,
                    "evidence_dedupe": True,
                },
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
                "allowed_tool_ids": ["document_evidence"],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        contents = [packet["content"] for packet in update["evidence_packets"]]
        assert contents.count("duplicate evidence paragraph") == 1
        assert "unique evidence" in contents
        assert update["evidence_packets"][-1]["content_hash"]
        assert update["evidence_packets"][-1]["fingerprint"]

    def test_final_context_from_packets_compacts_duplicate_lines_and_bounds_context(self):
        context, source = _final_context_from_state(
            {
                "evidence_packets": [
                    {
                        "id": "packet-1",
                        "producer_node_id": "retrieval_1",
                        "producer_node_type": "retrieval_worker",
                        "kind": "document",
                        "content": "repeat sentence\nrepeat sentence\n" + "\n".join(f"unique line {index}" for index in range(30)),
                    }
                ],
                "context_policy": {
                    "evidence_packet_limit": 3,
                    "evidence_packet_content_limit": 1000,
                    "final_prompt_assembly": "evidence_packets",
                    "evidence_compression": "compact",
                    "final_context_char_limit": 180,
                },
            }
        )

        assert source == "evidence_packets"
        assert len(context) <= 180
        assert context.count("repeat sentence") <= 1

    @pytest.mark.asyncio
    async def test_final_answer_can_assemble_context_from_bounded_evidence_packets(self, monkeypatch):
        captured_messages = []

        class FakeLlm:
            async def ainvoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content="Packet-based answer.")

        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())

        update = await NodeRegistry().synthesizer(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What does the document say?",
                "llm_model": "test-llm",
                "context_window": 8192,
                "evidence": "legacy evidence should not be used",
                "evidence_packets": [
                    {
                        "id": "packet-1",
                        "producer_node_id": "retrieval_1",
                        "producer_node_type": "retrieval_worker",
                        "kind": "document",
                        "content": " ".join(["packet-evidence"] * 20),
                    }
                ],
                "context_policy": {
                    "evidence_packet_limit": 2,
                    "evidence_packet_content_limit": 30,
                    "final_prompt_assembly": "evidence_packets",
                },
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
                "node_events": [],
                "tool_events": [],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )

        human_prompt = captured_messages[-1].content
        assert "document evidence from retrieval_1" in human_prompt
        assert "packet-evidence" in human_prompt
        assert "legacy evidence should not be used" not in human_prompt
        assert update["node_events"][-1]["input_preview"]["context_source"] == "evidence_packets"
        assert update["final_answer"] == "Packet-based answer."

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
            "app.agent_workflows.graph.search_documents",
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
            "app.agent_workflows.graph.search_conversation_history",
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
            "app.agent_workflows.graph.search_thread_timeline",
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
            "app.agent_workflows.graph.search_web",
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

        monkeypatch.setattr("app.agent_workflows.graph.search_conversation_history", ExplodingTool())

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
    async def test_router_uses_only_routes_configured_for_its_graph_instance(self, monkeypatch):
        captured_messages = []

        class FakeLlm:
            async def ainvoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content=json.dumps({"route": "memory", "reason": "Use memory."}))

        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())

        update = await NodeRegistry().router(
            {
                "agent_run_id": "run-1",
                "thread_id": "thread-1",
                "question": "What did the document say?",
                "llm_model": "test-llm",
                "use_web_search": False,
                "context_window": 8192,
                "pre_fetch_bundle": {},
                "node_events": [],
                "tool_events": [],
            },
            {
                "configurable": {
                    "thread_id": "thread-1",
                    "agent_workflow_node_runtime": {
                        "route_labels": ["document", "clarify"],
                    },
                },
            },
        )

        assert update["route"] == "document"
        assert "configured fallback 'document'" in update["route_reason"]
        assert "document, clarify" in captured_messages[-1].content

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

        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())

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

        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())
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

        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())

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
class TestAgentWorkflowRepository:
    @pytest_asyncio.fixture
    async def repo(self, engine):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            yield AgentWorkflowRepository(repo_session)

    @pytest.mark.asyncio
    async def test_seed_builtin_router_rag_workflow_is_idempotent(self, repo):
        await repo.seed_builtin_workflows()
        await repo.seed_builtin_workflows()

        workflows = await repo.list_workflows()
        router_workflow, router_version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        plan_workflow, plan_version = await repo.get_workflow_with_current_version(PLAN_EXECUTE_RAG_AGENT_ID)
        evaluator_workflow, evaluator_version = await repo.get_workflow_with_current_version(EVALUATOR_REPLANNER_RAG_AGENT_ID)

        assert {workflow.id for workflow in workflows} == {
            ROUTER_RAG_AGENT_ID,
            PLAN_EXECUTE_RAG_AGENT_ID,
            EVALUATOR_REPLANNER_RAG_AGENT_ID,
        }
        assert router_workflow.metadata_json["version_id"] == router_version.id
        assert router_version.version == ROUTER_RAG_AGENT_VERSION
        assert router_version.schema_version == 2
        assert router_version.spec_json["schema_version"] == 2
        assert router_version.validation_result_json == {"valid": True, "errors": []}
        assert plan_workflow.metadata_json["version_id"] == plan_version.id
        assert plan_version.version == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert plan_version.schema_version == 2
        assert plan_version.spec_json["schema_version"] == 2
        assert plan_version.validation_result_json == {"valid": True, "errors": []}
        assert evaluator_workflow.metadata_json["version_id"] == evaluator_version.id
        assert evaluator_version.version == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert evaluator_version.schema_version == 2
        assert evaluator_version.spec_json["schema_version"] == 2
        assert evaluator_version.validation_result_json == {"valid": True, "errors": []}

    @pytest.mark.asyncio
    async def test_seed_builtin_current_v2_versions_validate_and_compile(self, repo):
        await repo.seed_builtin_workflows()

        current_specs = [
            (ROUTER_RAG_AGENT_ID, ROUTER_RAG_AGENT_V2_VERSION, builtin_router_rag_v2_spec),
            (PLAN_EXECUTE_RAG_AGENT_ID, PLAN_EXECUTE_RAG_AGENT_V2_VERSION, builtin_plan_execute_rag_v2_spec),
            (
                EVALUATOR_REPLANNER_RAG_AGENT_ID,
                EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION,
                builtin_evaluator_replanner_rag_v2_spec,
            ),
        ]
        for workflow_id, version_number, spec_factory in current_specs:
            workflow, current_version = await repo.get_workflow_version(
                workflow_id,
                version_number,
            )

            assert workflow.id == workflow_id
            assert current_version.version == version_number
            assert current_version.schema_version == 2
            assert current_version.spec_json == spec_factory()
            assert current_version.validation_result_json == {"valid": True, "errors": []}
            WorkflowCompiler().compile(current_version.spec_json)

    @pytest.mark.asyncio
    async def test_db_loaded_invalid_v2_spec_fails_validation(self, repo):
        bad_spec = builtin_router_rag_v2_spec()
        bad_spec["config"]["graph"]["nodes"].append({"id": "unsafe_1", "type": "unsafe_type"})
        bad_spec["config"]["graph"]["edges"].append({"from": "router", "to": "unsafe_1"})

        async with repo._session.begin():
            repo._session.add(
                AgentWorkflow(
                    id="internal_bad_agent",
                    name="Internal Bad Agent",
                    description="Invalid internal test agent.",
                    visibility="internal",
                    is_builtin=False,
                    schema_version=2,
                    spec_json=bad_spec,
                    validation_result_json={},
                    metadata_json={
                        "version": 1,
                        "version_id": "internal_bad_agent:v1",
                        "changelog": "Invalid test spec.",
                    },
                )
            )

        workflow, version = await repo.get_workflow_with_current_version("internal_bad_agent", include_custom=True)

        assert workflow.id == "internal_bad_agent"
        with pytest.raises(WorkflowValidationError, match="unknown type"):
            WorkflowValidator().validate(version.spec_json)

    @pytest.mark.asyncio
    async def test_create_internal_custom_v2_workflow_version_validates_and_stores_current_version(self, repo):
        spec = builtin_router_rag_v2_spec()
        spec["workflow_id"] = "internal_custom_rag_agent"

        workflow, version = await repo.save_internal_workflow_version(
            workflow_id="internal_custom_rag_agent",
            name="Internal Custom RAG Agent",
            description="Internal JSON-authored custom workflow.",
            spec_json=spec,
            changelog="Initial internal custom workflow.",
        )
        public_workflow = await repo.get_workflow("internal_custom_rag_agent")
        loaded_workflow, loaded_version = await repo.get_workflow_with_current_version(
            "internal_custom_rag_agent",
            include_custom=True,
        )

        assert workflow.id == "internal_custom_rag_agent"
        assert workflow.visibility == "internal"
        assert workflow.is_builtin is False
        assert workflow.metadata_json["version_id"] == "internal_custom_rag_agent:v1"
        assert version.schema_version == 2
        assert version.validation_result_json == {"valid": True, "errors": []}
        assert public_workflow is None
        assert loaded_workflow.id == workflow.id
        assert loaded_version.id == version.id
        assert WorkflowCompiler().compile(loaded_version.spec_json) is not None

    @pytest.mark.asyncio
    async def test_create_internal_custom_workflow_rejects_invalid_or_non_v2_specs(self, repo):
        invalid_spec = builtin_router_rag_v2_spec()
        invalid_spec["config"]["graph"]["edges"][2].pop("route_fn")
        with pytest.raises(WorkflowValidationError, match="must declare route_fn"):
            await repo.save_internal_workflow_version(
                workflow_id="internal_invalid_agent",
                name="Internal Invalid Agent",
                spec_json=invalid_spec,
            )

        with pytest.raises(WorkflowValidationError, match="schema_version 2"):
            await repo.save_internal_workflow_version(
                workflow_id="internal_v1_agent",
                name="Internal v1 Agent",
                spec_json=legacy_builtin_router_rag_v1_spec(),
            )

        missing_workflow, missing_version = await repo.get_workflow_with_current_version(
            "internal_invalid_agent",
            include_custom=True,
        )
        assert missing_workflow is None
        assert missing_version is None

    @pytest.mark.asyncio
    async def test_run_lifecycle_persists_resolved_spec(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)

        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID},
        )
        completed = await repo.complete_run(
            run.id,
            status="completed",
            metrics_json={"duration_ms": 12.5},
        )

        assert completed.status == "completed"
        assert completed.metrics_json == {"duration_ms": 12.5}
        assert completed.resolved_spec_json == {"workflow_id": ROUTER_RAG_AGENT_ID}
        assert completed.run_metadata_json == {"workflow_version_id": version.id}
        assert completed.workflow_version_id == version.id

    @pytest.mark.asyncio
    async def test_mark_run_awaiting_human_persists_bounded_pending_interrupt(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            workflow_version=version.version,
            resolved_spec_json=builtin_router_rag_v2_spec(),
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
        resume_guard = paused.pending_interrupt_json["resume_guard"]
        assert resume_guard["spec_schema_version"] == 2
        assert resume_guard["workflow_id"] == ROUTER_RAG_AGENT_ID
        assert resume_guard["workflow_version_id"] == version.id
        assert resume_guard["workflow_version"] == version.version
        assert resume_guard["checkpoint_thread_id"] == run.checkpoint_thread_id
        assert isinstance(resume_guard["resolved_spec_hash"], str)
        assert len(resume_guard["resolved_spec_hash"]) == 64

        awaiting_runs = await repo.list_runs_for_thread(sample_thread.id, status="awaiting_human")
        assert [item.id for item in awaiting_runs] == [run.id]

    @pytest.mark.asyncio
    async def test_resolve_pending_interrupt_resumes_atomically_and_idempotently(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            workflow_version=version.version,
            resolved_spec_json=builtin_router_rag_v2_spec(),
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
    async def test_resolve_pending_interrupt_rejects_stale_resume_guard(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            workflow_version=version.version,
            resolved_spec_json=builtin_router_rag_v2_spec(),
        )
        await repo.mark_run_awaiting_human(
            run.id,
            {
                "interrupt_id": "interrupt-stale",
                "allowed_actions": ["approve", "reject"],
                "resume_version": 1,
            },
        )

        session = await repo._get_session()
        async with session.begin():
            stored_run = await session.get(AgentRun, run.id)
            pending = dict(stored_run.pending_interrupt_json or {})
            resume_guard = dict(pending.get("resume_guard") or {})
            resume_guard["resolved_spec_hash"] = "0" * 64
            pending["resume_guard"] = resume_guard
            stored_run.pending_interrupt_json = pending

        with pytest.raises(AgentRunInterruptError) as exc:
            await repo.resolve_pending_interrupt(
                run.id,
                interrupt_id="interrupt-stale",
                action="approve",
                resume_version=1,
            )

        assert exc.value.code == "interrupt_resume_guard_mismatch"

    @pytest.mark.asyncio
    async def test_resolve_pending_interrupt_validates_selected_options(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            workflow_version=version.version,
            resolved_spec_json=builtin_router_rag_v2_spec(),
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
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            workflow_version=version.version,
            resolved_spec_json=builtin_router_rag_v2_spec(),
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
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        run = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID},
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
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        first = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "n": 1},
        )
        second = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "n": 2},
        )

        runs = await repo.list_runs_for_thread(sample_thread.id, limit=1)

        assert [run.id for run in runs] == [second.id]
        assert first.id != second.id

    @pytest.mark.asyncio
    async def test_prune_runs_before_deletes_only_matching_old_statuses(self, repo, sample_thread):
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        old_completed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "old_completed"},
        )
        old_running = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "old_running"},
        )
        recent_completed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "recent_completed"},
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

        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        old_completed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "old_completed"},
        )
        old_failed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "old_failed"},
        )
        old_awaiting = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "old_awaiting"},
        )
        recent_completed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "recent_completed"},
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
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        stale_running = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "stale_running"},
        )
        recent_running = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "recent_running"},
        )
        stale_completed = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "stale_completed"},
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
        await repo.seed_builtin_workflows()
        workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
        awaiting = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "awaiting"},
        )
        running = await repo.create_run(
            thread_id=sample_thread.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            resolved_spec_json={"workflow_id": ROUTER_RAG_AGENT_ID, "case": "running"},
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
    async def test_unsupported_simple_rag_workflow_is_not_exposed(self, repo):
        await repo.seed_builtin_workflows()

        workflow, version = await repo.get_workflow_with_current_version("simple_rag_agent")

        assert workflow is None
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
                "agent_workflow": {"workflow_id": ROUTER_RAG_AGENT_ID},
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
        monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            workflow_ref = SimpleNamespace(id=workflow.id)
            version_ref = SimpleNamespace(
                id=version.id,
                version=version.version,
                spec_json=builtin_router_rag_hitl_web_spec(),
            )

            async def fake_get_workflow_with_current_version(_workflow_id):
                return workflow_ref, version_ref

            repo.get_workflow_with_current_version = fake_get_workflow_with_current_version
            service = AgentRunService(repository=repo)
            paused = await service.run_thread_chat(sample_thread.id, self._agent_req(), sample_thread.embedding_model)
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": "simple_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        assert result["agent_workflow_version"] == ROUTER_RAG_AGENT_VERSION
        assert captured_context["agent_run_id"] == result["agent_run_id"]
        assert run.status == "completed"
        assert run.metrics_json["document_source_count"] == 1
        assert run.resolved_spec_json["workflow_id"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_defaults_to_router_rag(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )

        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID

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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": ROUTER_RAG_AGENT_ID}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        assert captured_spec["workflow_id"] == ROUTER_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.metrics_json["route"] == "direct"
        assert run.metrics_json["node_event_count"] == 1
        assert run.metrics_json["node_elapsed_ms"] == {"router": 3.5}
        assert run.metrics_json["tool_event_count"] == 1
        assert run.metrics_json["tool_elapsed_ms"] == 9.25
        assert run.debug_trace_json["version"] == 1
        assert run.debug_trace_json["trace"]["run_id"] == run.id
        assert any(node["id"] == "router" for node in run.debug_trace_json["graph"]["nodes"])

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_current_v2_builtin(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {
                    "agent_workflow": {
                        "workflow_id": ROUTER_RAG_AGENT_ID,
                    }
                }

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        router_edge = next(
            edge
            for edge in captured_spec["config"]["graph"]["edges"]
            if edge.get("from") == "router" and edge.get("conditional")
        )
        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        assert result["agent_workflow_version"] == ROUTER_RAG_AGENT_V2_VERSION
        assert run.workflow_version_id == f"{ROUTER_RAG_AGENT_ID}:v{ROUTER_RAG_AGENT_V2_VERSION}"
        assert run.resolved_spec_json["schema_version"] == 2
        assert router_edge["route_fn"] == "router_route"
        assert captured_spec["config"]["loop_policy"]["max_total_visits"] == 9

    @pytest.mark.asyncio
    async def test_run_thread_chat_falls_back_for_custom_db_workflow_without_opt_in(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["workflow_id"] = "internal_custom_rag_agent"
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        assert captured_spec["workflow_id"] == ROUTER_RAG_AGENT_ID
        assert run.workflow_id == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_can_load_custom_db_workflow_when_service_opted_in(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["workflow_id"] = "internal_custom_rag_agent"
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                allow_custom_agent_workflows=True,
            ).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embedding_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == "internal_custom_rag_agent"
        assert captured_spec["workflow_id"] == "internal_custom_rag_agent"
        assert run.workflow_id == "internal_custom_rag_agent"

    @pytest.mark.asyncio
    async def test_run_thread_chat_can_load_custom_db_workflow_when_opted_in(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["workflow_id"] = "internal_custom_rag_agent"
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                allow_custom_agent_workflows=True,
            ).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embedding_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == "internal_custom_rag_agent"
        assert result["agent_workflow_version"] == 1
        assert run.workflow_version_id == "internal_custom_rag_agent:v1"
        assert run.resolved_spec_json["schema_version"] == 2
        assert captured_spec["workflow_id"] == "internal_custom_rag_agent"

    @pytest.mark.asyncio
    async def test_run_thread_chat_falls_back_when_custom_runtime_lacks_service_opt_in(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            custom_spec = builtin_router_rag_v2_spec()
            custom_spec["workflow_id"] = "internal_custom_rag_agent"
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=custom_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": "internal_custom_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        assert run.workflow_id == ROUTER_RAG_AGENT_ID
        assert captured_spec["workflow_id"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_current_custom_db_workflow_version(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            v1_spec = builtin_router_rag_v2_spec()
            v1_spec["workflow_id"] = "internal_custom_rag_agent_v1"
            v2_spec = builtin_router_rag_v2_spec()
            v2_spec["workflow_id"] = "internal_custom_rag_agent_v2"
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=v1_spec,
            )
            await repo.save_internal_workflow_version(
                workflow_id="internal_custom_rag_agent",
                name="Internal Custom RAG Agent",
                spec_json=v2_spec,
            )

            async def fake_get_thread_settings(_thread_id):
                return {
                    "agent_workflow": {
                        "workflow_id": "internal_custom_rag_agent",
                        # Legacy pins are ignored; chat always runs the current workflow version.
                        "workflow_version": 1,
                    }
                }

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "custom current ok",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                allow_custom_agent_workflows=True,
            ).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embedding_model,
            )
            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == "internal_custom_rag_agent"
        assert result["agent_workflow_version"] == 2
        assert run.workflow_version_id == "internal_custom_rag_agent:v2"
        assert captured_spec["workflow_id"] == "internal_custom_rag_agent_v2"

    @pytest.mark.asyncio
    async def test_internal_custom_workflow_create_select_and_run_keeps_instance_node_identity(
        self,
        async_api_client,
        monkeypatch,
    ):
        custom_spec = {
            "schema_version": 2,
            "workflow_id": "internal_e2e_custom_rag_agent",
            "config": {
                "allowed_tool_ids": ["document_evidence"],
                "graph": {
                    "nodes": [
                        {"id": "context_1", "type": "context_loader"},
                        {"id": "router_1", "type": "router"},
                        {"id": "retrieval_1", "type": "retrieval_worker"},
                        {"id": "synth_1", "type": "synthesizer"},
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
                                "direct": "final_1",
                            },
                        },
                        {"from": "retrieval_1", "to": "synth_1"},
                        {"from": "synth_1", "to": "final_1"},
                        {"from": "final_1", "to": "END"},
                    ],
                },
            },
        }

        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, _messages):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content='{"route":"document","reason":"Need document evidence."}')
                return SimpleNamespace(content="Custom graph answer from retrieval_1.")

        class FakeDocumentTool:
            name = "search_documents"

            async def ainvoke(self, tool_input, config=None):
                return {
                    "content": "Document evidence for the custom graph.",
                    "sources": [{"id": "doc-1", "title": "Custom Doc"}],
                    "artifacts": {
                        "document_sources": [
                            {"id": "doc-1", "title": "Custom Doc", "snippet": "Document evidence."}
                        ]
                    },
                    "trace": {
                        "tool_name": "search_documents",
                        "caller_node": (config or {}).get("configurable", {}).get("caller_node"),
                        "caller_node_type": (config or {}).get("configurable", {}).get("caller_node_type"),
                    },
                    "metrics": {"result_chars": 39, "source_count": 1, "warning_count": 0},
                }

        fake_llm = FakeLlm()

        async def fake_prefetch_context(**_kwargs):
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

        async def fake_index_chat_memory_for_thread(**_kwargs):
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeDocumentTool())
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        thread_response = await async_api_client.post(
            "/api/threads",
            json={"name": "Custom Workflow Thread", "embedding_model": "BAAI/bge-m3"},
        )
        other_thread_response = await async_api_client.post(
            "/api/threads",
            json={"name": "Default Workflow Thread", "embedding_model": "BAAI/bge-m3"},
        )
        created = await async_api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": "internal_e2e_custom_rag_agent",
                "name": "Internal E2E Custom RAG Agent",
                "spec_json": custom_spec,
            },
        )

        assert thread_response.status_code == 200
        assert other_thread_response.status_code == 200
        assert created.status_code == 200

        thread_id = thread_response.json()["id"]
        other_thread_id = other_thread_response.json()["id"]
        listed = await async_api_client.get("/api/agent-workflows")
        selected = await async_api_client.put(
            f"/api/threads/{thread_id}/settings",
            json={"agent_workflow": {"workflow_id": "internal_e2e_custom_rag_agent"}},
        )

        assert selected.status_code == 200
        assert listed.status_code == 200
        assert "internal_e2e_custom_rag_agent" in {
            item["id"] for item in listed.json()["agent_workflows"]
        }
        assert selected.json()["agent_workflow"]["workflow_id"] == "internal_e2e_custom_rag_agent"

        service = AgentRunService(allow_custom_agent_workflows=True)
        result = await service.run_thread_chat(
            thread_id,
            self._agent_req("What does the custom document say?"),
            "BAAI/bge-m3",
        )
        fallback_result = await service.run_thread_chat(
            other_thread_id,
            self._agent_req("Should use the default workflow."),
            "BAAI/bge-m3",
        )
        repo = AgentWorkflowRepository()
        run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == "internal_e2e_custom_rag_agent"
        assert fallback_result["agent_workflow_id"] == ROUTER_RAG_AGENT_ID
        retrieval_node = next(
            node
            for node in run.resolved_spec_json["config"]["graph"]["nodes"]
            if node.get("id") == "retrieval_1"
        )
        assert retrieval_node["id"] == "retrieval_1"
        assert retrieval_node["type"] == "retrieval_worker"
        assert retrieval_node["label"] == "Document Retrieval"
        assert retrieval_node["category"] == "retrieval"
        assert retrieval_node["capabilities"] == ["retrieval.document"]
        assert "document_evidence" in retrieval_node["allowed_tool_contract_ids"]
        assert "evidence_packets" in retrieval_node["state_writes"]
        assert retrieval_node["context_policy"]["mode"] == "append_evidence"
        assert retrieval_node["observability"]["span_kind"] == "tool_worker"
        assert retrieval_node["max_instances"] >= 1
        assert any(
            event.get("node") == "retrieval_1" and event.get("node_type") == "retrieval_worker"
            for event in result["node_events"]
        )
        assert any(
            event.get("caller_node") == "retrieval_1"
            and event.get("caller_node_type") == "retrieval_worker"
            for event in result["tool_events"]
        )
        span_attrs = [
            span.get("attributes") or {}
            for span in (run.debug_trace_json or {}).get("trace", {}).get("spans", [])
        ]
        assert any(
            attrs.get("askpdf.node.id") == "retrieval_1"
            and attrs.get("askpdf.node.type") == "retrieval_worker"
            and attrs.get("askpdf.node.name") == "Document Retrieval"
            and attrs.get("askpdf.node.category") == "retrieval"
            and attrs.get("askpdf.node.capabilities") == ["retrieval.document"]
            and attrs.get("askpdf.observability.span_kind") == "tool_worker"
            and attrs.get("askpdf.observability.event_prefix") == "retrieval_worker"
            and attrs.get("askpdf.observability.summary_fields") == [
                "document_source_count",
                "web_source_count",
                "evidence_chars",
            ]
            for attrs in span_attrs
        )
        graph_nodes = (run.debug_trace_json or {}).get("graph", {}).get("nodes", [])
        debug_retrieval_node = next(node for node in graph_nodes if node.get("id") == "retrieval_1")
        assert debug_retrieval_node["label"] == "Document Retrieval"
        assert debug_retrieval_node["category"] == "retrieval"
        assert debug_retrieval_node["capabilities"] == ["retrieval.document"]
        assert debug_retrieval_node["observability"]["span_kind"] == "tool_worker"

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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": PLAN_EXECUTE_RAG_AGENT_ID}}

            async def fake_handle_plan_execute_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr(
                "app.agent_workflows.router_runtime.handle_plan_execute_rag_chat",
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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert result["agent_workflow_version"] == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert captured_spec["workflow_id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.resolved_spec_json["workflow_id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert run.metrics_json["route"] == "execute"
        assert run.metrics_json["node_elapsed_ms"] == {"planner": 2.0}
        assert run.metrics_json["document_source_count"] == 1
        assert run.metrics_json["used_chat_id_count"] == 1
        assert run.debug_trace_json["summary"]["route"] == "execute"
        assert any(node["id"] == "planner" for node in run.debug_trace_json["graph"]["nodes"])

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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": EVALUATOR_REPLANNER_RAG_AGENT_ID}}

            async def fake_handle_evaluator_replanner_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr(
                "app.agent_workflows.router_runtime.handle_evaluator_replanner_rag_chat",
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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_workflow_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert result["agent_workflow_version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert captured_spec["workflow_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.resolved_spec_json["workflow_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert run.metrics_json["route"] == "execute"
        assert run.metrics_json["replan_count"] == 0
        assert run.metrics_json["evaluation_confidence"] == 0.8
        assert run.debug_trace_json["summary"]["evaluatorRoute"] == "answer"
        assert any(node["id"] == "evidence_evaluator" for node in run.debug_trace_json["graph"]["nodes"])

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
                "agent_workflow": {"workflow_id": ROUTER_RAG_AGENT_ID},
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
        monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as repo_session:
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
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

            paused = await service.run_thread_chat(sample_thread.id, req, sample_thread.embedding_model)
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
                "agent_workflow": {"workflow_id": EVALUATOR_REPLANNER_RAG_AGENT_ID},
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
                embedding_model="BAAI/bge-m3",
                settings={},
            )
            setup_session.add(thread)
            await setup_session.commit()
            await setup_session.refresh(thread)
            thread_id = thread.id
            embedding_model = thread.embedding_model

        monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "postgres")
        monkeypatch.setenv("AGENT_CHECKPOINT_DATABASE_URL", test_database_url)
        monkeypatch.delenv("ASKPDF_AGENT_CHECKPOINTER_ALLOW_MEMORY_FALLBACK", raising=False)
        monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.graph.search_web", fake_web)
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        async with session_factory() as first_session:
            first_repo = AgentWorkflowRepository(first_session)
            await first_repo.seed_builtin_workflows()
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
                embedding_model,
            )
            paused_run = await first_repo.get_run(paused["agent_run_id"])
            pending = paused_run.pending_interrupt_json
            checkpoint_thread_id = paused_run.checkpoint_thread_id

        async with session_factory() as second_session:
            second_repo = AgentWorkflowRepository(second_session)
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
                workflow_version=version.version,
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

            monkeypatch.setattr("app.agent_workflows.router_runtime.resume_compiled_rag_chat", fake_resume_compiled_rag_chat)

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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
                workflow_version=version.version,
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

            monkeypatch.setattr("app.agent_workflows.router_runtime.resume_compiled_rag_chat", fake_resume_compiled_rag_chat)

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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_workflow": {"workflow_id": ROUTER_RAG_AGENT_ID}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embedding_model, *, resolved_spec, agent_run_context, trace_recorder, **_kwargs):
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

            monkeypatch.setattr("app.agent_workflows.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_workflows.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

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
                sample_thread.embedding_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert run.status == "failed"
        assert run.error_json["code"] == "router_rag_execution_failed"
        assert run.metrics_json["route"] == "document"
        assert run.metrics_json["node_event_count"] == 1
        assert run.metrics_json["error_count"] == 1
        assert run.debug_trace_json["trace"]["status"] == "failed"
        assert any(node["id"] == "router" for node in run.debug_trace_json["graph"]["nodes"])


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
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

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

        spec = builtin_router_rag_v2_spec()
        await create_agent_run_record(
            session_factory,
            run_id="run-1",
            thread_id=sample_thread.id,
            spec=spec,
        )
        caplog.set_level(logging.INFO, logger="app.agent_workflows")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embedding_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": "run-1",
                "agent_workflow_id": ROUTER_RAG_AGENT_ID,
                "agent_workflow_version": ROUTER_RAG_AGENT_VERSION,
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

        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FakeTool(document_payload))
        monkeypatch.setattr("app.agent_workflows.graph.search_conversation_history", FakeTool(memory_payload))
        monkeypatch.setattr("app.agent_workflows.graph.search_thread_timeline", FakeTool(timeline_payload))
        monkeypatch.setattr("app.agent_workflows.graph.search_web", FakeTool(web_payload))
        monkeypatch.setattr("app.agent_workflows.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_workflows.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_workflows.router_runtime.increment_qa_stats", fake_increment_qa_stats)

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
        spec = builtin_router_rag_v2_spec()
        await create_agent_run_record(
            session_factory,
            run_id=run_id,
            thread_id=sample_thread.id,
            spec=spec,
        )
        caplog.set_level(logging.INFO, logger="app.agent_workflows")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embedding_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": run_id,
                "agent_workflow_id": ROUTER_RAG_AGENT_ID,
                "agent_workflow_version": ROUTER_RAG_AGENT_VERSION,
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

        monkeypatch.setattr("app.agent_workflows.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_workflows.graph.get_llm", lambda _name: FakeLlm())
        monkeypatch.setattr("app.agent_workflows.graph.search_documents", FailingTool())
        monkeypatch.setattr("app.agent_workflows.router_runtime.create_chat_turn", fake_create_chat_turn)

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

        spec = builtin_router_rag_v2_spec()
        await create_agent_run_record(
            session_factory,
            run_id="run-failed",
            thread_id=sample_thread.id,
            spec=spec,
        )
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embedding_model,
            resolved_spec=spec,
            agent_run_context={
                "agent_run_id": "run-failed",
                "agent_workflow_id": ROUTER_RAG_AGENT_ID,
                "agent_workflow_version": ROUTER_RAG_AGENT_VERSION,
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


@pytest.mark.asyncio
async def test_builder_test_run_uses_resolved_workflow_row_id(monkeypatch):
    import app.api.agent_workflows as agent_workflows_api

    captured = {}

    async def fake_get_thread(_thread_id):
        return SimpleNamespace(embedding_model="test-embedding")

    async def fake_get_workflow(_self, workflow_id, *, include_custom=False):
        assert workflow_id == ROUTER_RAG_AGENT_ID
        assert include_custom is True
        return SimpleNamespace(id="legacy-persisted-router-row")

    async def fake_create_run(_self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="builder-test-run")

    monkeypatch.setattr(agent_workflows_api, "get_thread", fake_get_thread)
    monkeypatch.setattr(AgentWorkflowRepository, "get_workflow", fake_get_workflow)
    monkeypatch.setattr(AgentWorkflowRepository, "create_run", fake_create_run)

    response = await agent_workflows_api.stream_internal_agent_workflow_test(
        SimpleNamespace(
            builder_session_id="builder-session-unsaved",
            base_workflow_id=ROUTER_RAG_AGENT_ID,
            spec=builtin_router_rag_v2_spec(),
            thread_id="thread-1",
            use_web_search=False,
            allow_external_tools=False,
        )
    )

    assert response.media_type == "text/event-stream"
    assert captured["workflow_id"] == "legacy-persisted-router-row"
    assert captured["run_metadata_json"]["base_workflow_id"] == ROUTER_RAG_AGENT_ID
    assert captured["resolved_spec_json"]["workflow_id"] == ROUTER_RAG_AGENT_ID


@pytest.mark.asyncio
async def test_thread_chat_content_negotiation_streams_compact_progress(monkeypatch):
    import app.api.messages as messages_api

    class FakeService:
        async def run_thread_chat(self, thread_id, req, embedding_model, *, execution_event_sink=None):
            assert thread_id == "thread-stream"
            assert embedding_model == "embed-test"
            if execution_event_sink is not None:
                await execution_event_sink.emit("run.started", {"run_id": "run-stream", "workflow_id": "router_rag_agent"})
                await execution_event_sink.emit("node.completed", {
                    "node_id": "router",
                    "visit_index": 1,
                    "route": "document",
                    "detail": {"checkpoint_before": {"secret": "must-not-stream"}},
                    "reasoning": "must-not-stream",
                })
            return {
                "answer": "done",
                "status": "completed",
                "agent_run_id": "run-stream",
                "user_message_id": "turn:user",
                "assistant_message_id": "turn:assistant",
                "used_chat_ids": [],
                "document_sources": [],
            }

    async def fake_get_thread(_thread_id):
        return SimpleNamespace(embedding_model="embed-test")

    async def fake_get_settings(_thread_id):
        return {}

    async def fake_supports_replans(_settings):
        return False

    monkeypatch.setattr(messages_api, "get_thread", fake_get_thread)
    monkeypatch.setattr(messages_api, "get_thread_settings", fake_get_settings)
    monkeypatch.setattr(messages_api, "_settings_workflow_supports_replans", fake_supports_replans)
    monkeypatch.setattr(messages_api, "AgentRunService", FakeService)
    request = SimpleNamespace(
        thread_id="thread-stream",
        question="question",
        llm_model="model",
        replans=None,
        system_role_override="",
        tool_instructions_override={},
        custom_instructions_override="",
    )

    response = await messages_api.thread_chat_endpoint("thread-stream", request, accept="text/event-stream")
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    payload = "".join(chunks)

    assert response.media_type == "text/event-stream"
    assert "event: run.started" in payload
    assert "event: node.completed" in payload
    assert "event: run.completed" in payload
    assert '"answer": "done"' in payload
    assert "must-not-stream" not in payload


@pytest.mark.asyncio
async def test_compact_execution_sink_omits_full_invocation_fields():
    sink = AgentExecutionEventSink(include_details=False)
    await sink.emit("node.completed", {
        "node_id": "router",
        "visit_index": 1,
        "detail": {"checkpoint_after": {"answer": "large"}},
        "prompt": "private prompt",
        "reasoning": "private reasoning",
        "output_preview": {"route": "document"},
    })
    event = await sink.queue.get()
    assert event["data"] == {
        "node_id": "router",
        "visit_index": 1,
        "output_preview": {"route": "document"},
    }


@pytest.mark.asyncio
async def test_agent_run_resume_content_negotiation_streams_progress(monkeypatch):
    import app.api.agent_workflows as agent_workflows_api

    run = SimpleNamespace(
        id="run-resume-stream",
        thread_id="thread-resume-stream",
        workflow_id="router_rag_agent",
        status="completed",
        pending_interrupt_json=None,
    )

    class FakeService:
        async def resume_agent_run(self, run_id, **kwargs):
            sink = kwargs.get("execution_event_sink")
            await sink.emit("run.started", {"run_id": run_id, "resumed": True})
            await sink.emit("node.completed", {"node_id": "finalizer", "visit_index": 2, "detail": {"prompt": "hidden"}})
            return SimpleNamespace(run=run, interrupt={"interrupt_id": "interrupt-1"}, outcome="resumed", duplicate=False)

    async def fake_get_thread(_thread_id):
        return SimpleNamespace(id="thread-resume-stream")

    monkeypatch.setattr(agent_workflows_api, "get_thread", fake_get_thread)
    monkeypatch.setattr(agent_workflows_api, "AgentRunService", FakeService)
    request = SimpleNamespace(
        thread_id="thread-resume-stream",
        interrupt_id="interrupt-1",
        action="approve",
        edited_payload=None,
        client_metadata={},
        selected_option_ids=None,
        resume_token="token",
        resume_version=1,
    )

    response = await agent_workflows_api.resume_agent_run("run-resume-stream", request, accept="text/event-stream")
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    payload = "".join(chunks)

    assert "event: run.started" in payload
    assert "event: node.completed" in payload
    assert "event: run.completed" in payload
    assert '"outcome": "resumed"' in payload
    assert "hidden" not in payload


@pytest.mark.asyncio
async def test_agent_run_detail_endpoint_returns_one_loop_visit(monkeypatch):
    import app.api.agent_workflows as agent_workflows_api

    details = [
        {"node_id": "evidence_evaluator", "node_type": "evidence_evaluator", "visit_index": 1, "status": "completed", "checkpoint_after": {"replan_count": 0}},
        {"node_id": "evidence_evaluator", "node_type": "evidence_evaluator", "visit_index": 2, "status": "completed", "checkpoint_after": {"replan_count": 1}},
    ]

    async def fake_get_run(_self, run_id):
        assert run_id == "run-loop-details"
        return SimpleNamespace(id=run_id, thread_id="thread-loop-details", debug_trace_json={"version": 1, "details": details})

    async def fake_get_thread(thread_id):
        return SimpleNamespace(id=thread_id) if thread_id == "thread-loop-details" else None

    monkeypatch.setattr(AgentWorkflowRepository, "get_run", fake_get_run)
    monkeypatch.setattr(agent_workflows_api, "get_thread", fake_get_thread)

    response = await agent_workflows_api.get_agent_run_node_details(
        "run-loop-details",
        node_id="evidence_evaluator",
        visit_index=2,
        thread_id="thread-loop-details",
    )

    assert response["detail"]["visit_index"] == 2
    assert response["detail"]["checkpoint_after"]["replan_count"] == 1


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentWorkflowApi:
    def test_list_and_get_builtin_agent_workflow(self, api_client):
        listed = api_client.get("/api/agent-workflows")
        assert listed.status_code == 200
        assert {item["id"] for item in listed.json()["agent_workflows"]} == {
            ROUTER_RAG_AGENT_ID,
            PLAN_EXECUTE_RAG_AGENT_ID,
            EVALUATOR_REPLANNER_RAG_AGENT_ID,
        }

        detail = api_client.get(f"/api/agent-workflows/{ROUTER_RAG_AGENT_ID}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["agent_workflow"]["id"] == ROUTER_RAG_AGENT_ID
        assert payload["current_version"]["version"] == ROUTER_RAG_AGENT_VERSION
        assert payload["current_version"]["validation"]["valid"] is True
        assert "document_evidence" in payload["capabilities"]["required_tool_ids"]
        assert payload["capabilities"]["node_tool_requirements"]["retrieval_worker"] == "document_evidence"

        plan_detail = api_client.get(f"/api/agent-workflows/{PLAN_EXECUTE_RAG_AGENT_ID}")
        assert plan_detail.status_code == 200
        plan_payload = plan_detail.json()
        assert plan_payload["agent_workflow"]["id"] == PLAN_EXECUTE_RAG_AGENT_ID
        assert plan_payload["current_version"]["version"] == PLAN_EXECUTE_RAG_AGENT_VERSION
        assert plan_payload["current_version"]["validation"]["valid"] is True
        assert plan_payload["capabilities"]["node_tool_requirements"]["planner"] == "clarify_intent"

        evaluator_detail = api_client.get(f"/api/agent-workflows/{EVALUATOR_REPLANNER_RAG_AGENT_ID}")
        assert evaluator_detail.status_code == 200
        evaluator_payload = evaluator_detail.json()
        assert evaluator_payload["agent_workflow"]["id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert evaluator_payload["current_version"]["version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
        assert evaluator_payload["current_version"]["validation"]["valid"] is True
        assert evaluator_payload["capabilities"]["node_tool_requirements"]["evidence_evaluator"] == "clarify_intent"
        assert evaluator_payload["capabilities"]["node_tool_requirements"]["replanner"] == "clarify_intent"

        stale_detail = api_client.get("/api/agent-workflows/simple_rag_agent")
        assert stale_detail.status_code == 404

    def test_internal_custom_agent_workflow_is_globally_listed(self, api_client):
        async def seed_internal_workflow():
            spec = builtin_router_rag_v2_spec()
            spec["workflow_id"] = "internal_api_global_agent"
            await AgentWorkflowRepository().save_internal_workflow_version(
                workflow_id="internal_api_global_agent",
                name="Internal API Global Agent",
                spec_json=spec,
            )

        asyncio.run(seed_internal_workflow())

        listed = api_client.get("/api/agent-workflows")
        detail = api_client.get("/api/agent-workflows/internal_api_global_agent")

        assert listed.status_code == 200
        assert "internal_api_global_agent" in {
            item["id"] for item in listed.json()["agent_workflows"]
        }
        assert detail.status_code == 200
        assert detail.json()["agent_workflow"]["id"] == "internal_api_global_agent"

    def test_internal_agent_workflow_endpoint_creates_and_fetches_custom_v2_spec(self, api_client):
        spec = builtin_router_rag_v2_spec()
        spec["workflow_id"] = "internal_api_agent"

        created = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": "internal_api_agent",
                "name": "Internal API Agent",
                "description": "JSON-authored internal workflow.",
                "changelog": "Initial internal API version.",
                "spec_json": spec,
            },
        )
        fetched = api_client.get("/api/internal/agent-workflows/internal_api_agent")
        public_detail = api_client.get("/api/agent-workflows/internal_api_agent")

        assert created.status_code == 200
        created_payload = created.json()
        assert created_payload["agent_workflow"]["id"] == "internal_api_agent"
        assert created_payload["agent_workflow"]["visibility"] == "internal"
        assert created_payload["version"]["id"] == "internal_api_agent:v1"
        assert created_payload["version"]["schema_version"] == 2
        assert created_payload["version"]["validation"]["valid"] is True
        assert created_payload["version"]["validation_result_json"] == {"valid": True, "errors": []}
        assert fetched.status_code == 200
        assert fetched.json()["current_version"]["id"] == "internal_api_agent:v1"
        assert public_detail.status_code == 200
        assert public_detail.json()["current_version"]["id"] == "internal_api_agent:v1"

    def test_internal_agent_workflow_endpoint_generates_ids_and_updates_latest_spec(self, api_client):
        spec = builtin_router_rag_v2_spec()
        spec["workflow_id"] = "client_side_placeholder"

        created = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "name": "Generated ID Agent",
                "description": "Created without a caller-owned ID.",
                "spec_json": spec,
            },
        )
        assert created.status_code == 200
        created_payload = created.json()
        workflow_id = created_payload["agent_workflow"]["id"]
        assert workflow_id.startswith("custom_workflow_")
        assert created_payload["version"]["id"] == f"{workflow_id}:v1"
        assert created_payload["version"]["version"] == 1
        assert created_payload["version"]["spec_json"]["workflow_id"] == workflow_id

        updated_spec = builtin_router_rag_v2_spec()
        updated_spec["workflow_id"] = "another_placeholder"
        updated_spec["config"]["context_policy"]["evidence_packet_limit"] = 4
        updated = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": workflow_id,
                "name": "Renamed Generated ID Agent",
                "description": "Updated in place.",
                "spec_json": updated_spec,
            },
        )
        fetched = api_client.get(f"/api/agent-workflows/{workflow_id}")

        assert updated.status_code == 200
        assert updated.json()["version"]["id"] == f"{workflow_id}:v1"
        assert updated.json()["version"]["version"] == 1
        assert fetched.status_code == 200
        assert fetched.json()["agent_workflow"]["name"] == "Renamed Generated ID Agent"
        assert fetched.json()["current_version"]["spec_json"]["workflow_id"] == workflow_id
        assert fetched.json()["current_version"]["spec_json"]["config"]["context_policy"]["evidence_packet_limit"] == 4

    def test_internal_agent_workflow_delete_hides_custom_workflow(self, api_client):
        spec = builtin_router_rag_v2_spec()
        spec["workflow_id"] = "internal_api_delete_agent"

        created = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": "internal_api_delete_agent",
                "name": "Internal API Delete Agent",
                "spec_json": spec,
            },
        )
        deleted = api_client.delete("/api/internal/agent-workflows/internal_api_delete_agent")
        listed = api_client.get("/api/agent-workflows")
        public_detail = api_client.get("/api/agent-workflows/internal_api_delete_agent")
        internal_detail = api_client.get("/api/internal/agent-workflows/internal_api_delete_agent")

        assert created.status_code == 200
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "deleted"
        assert deleted.json()["agent_workflow"]["visibility"] == "deleted"
        assert "internal_api_delete_agent" not in {
            item["id"] for item in listed.json()["agent_workflows"]
        }
        assert public_detail.status_code == 404
        assert internal_detail.status_code == 404

    def test_internal_agent_workflow_delete_rejects_builtin_ids(self, api_client):
        deleted = api_client.delete(f"/api/internal/agent-workflows/{ROUTER_RAG_AGENT_ID}")

        assert deleted.status_code == 400
        assert "built-in agent workflows cannot be deleted" in deleted.json()["detail"]

    def test_internal_agent_workflow_endpoint_rejects_invalid_specs_without_storing(self, api_client):
        invalid_spec = builtin_router_rag_v2_spec()
        invalid_spec["workflow_id"] = "internal_api_invalid_agent"
        invalid_spec["config"]["graph"]["edges"][2].pop("route_fn")

        invalid = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": "internal_api_invalid_agent",
                "name": "Internal API Invalid Agent",
                "spec_json": invalid_spec,
            },
        )
        fetched = api_client.get("/api/internal/agent-workflows/internal_api_invalid_agent")

        assert invalid.status_code == 400
        assert "must declare route_fn" in invalid.json()["detail"]
        assert fetched.status_code == 404

    def test_internal_agent_workflow_endpoint_rejects_builtin_ids(self, api_client):
        spec = builtin_router_rag_v2_spec()
        rejected = api_client.post(
            "/api/internal/agent-workflows",
            json={
                "workflow_id": ROUTER_RAG_AGENT_ID,
                "name": "Not Allowed",
                "spec_json": spec,
            },
        )

        assert rejected.status_code == 400
        assert "built-in agent workflows cannot be authored" in rejected.json()["detail"]

    def test_internal_agent_workflow_catalog_endpoint_exposes_safe_authoring_metadata(self, api_client):
        response = api_client.get("/api/internal/agent-workflows/catalog")

        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == 2
        assert payload["spec_schema_version"] == 2
        assert payload["graph_spec"]["requires_explicit_route_fn"] is True
        assert payload["graph_spec"]["reserved_node_ids"] == ["START", "END"]

        node_catalog = payload["node_catalog"]
        assert node_catalog["retrieval_worker"]["display_name"] == "Document Retrieval"
        assert "document_evidence" in node_catalog["retrieval_worker"]["allowed_tool_contract_ids"]
        assert "router_route" in node_catalog["router"]["allowed_route_functions"]
        assert node_catalog["hitl_gate"]["category"] == "human_review"
        assert node_catalog["retrieval_worker"]["context_policy"]["mode"] == "append_evidence"
        assert node_catalog["retrieval_worker"]["observability"]["span_kind"] == "tool_worker"
        assert "evidence_packets" in node_catalog["retrieval_worker"]["state_writes"]
        assert node_catalog["retrieval_worker"]["max_instances"] >= 1
        assert "implementation" not in node_catalog["retrieval_worker"]
        assert "callable" not in node_catalog["retrieval_worker"]
        assert node_catalog["retrieval_worker"]["ui"]["summary"]
        assert node_catalog["retrieval_worker"]["ui"]["use_when"]
        assert node_catalog["retrieval_worker"]["ui"]["field_guidance"]["tools"]

        route_functions = payload["route_functions"]
        assert route_functions["router_route"]["allowed_source_types"] == ["router"]
        assert "document" in route_functions["router_route"]["route_labels"]
        assert route_functions["router_route"]["route_options"]["document"]["display_name"]
        assert route_functions["planner_route"]["route_labels"] == ["execute", "direct", "clarify"]
        assert route_functions["evaluator_route"]["allowed_source_types"] == ["evidence_evaluator"]
        assert route_functions["hitl_gate_route"]["route_labels"] is None

        tool_contracts = payload["tool_contracts"]
        document_contract = tool_contracts["document_evidence"]
        assert "search_documents" in document_contract["canonical_tools"]
        assert document_contract["allowed_node_types"] == ["retrieval_worker"]
        assert document_contract["required_node_capabilities"] == ["retrieval.document"]
        assert "document_sources" in document_contract["artifact_keys"]
        assert "allowed_caller_nodes" not in document_contract
        assert "default_prompt" not in document_contract
        assert "tool_name" not in document_contract

    def test_builder_test_stream_requires_external_tool_confirmation(self, api_client, sample_thread):
        api_client.get("/api/agent-workflows")
        response = api_client.post(
            "/api/internal/agent-workflows/test-runs/stream",
            json={
                "builder_session_id": "builder-session-confirmation",
                "base_workflow_id": ROUTER_RAG_AGENT_ID,
                "spec": builtin_router_rag_v2_spec(),
                "thread_id": sample_thread.id,
                "question": "What changed today?",
                "llm_model": "test-model",
                "use_web_search": True,
                "allow_external_tools": False,
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "external_tool_confirmation_required"

    def test_builder_latest_test_returns_not_found_for_new_session(self, api_client):
        response = api_client.get(
            "/api/internal/agent-workflows/test-runs/latest",
            params={"builder_session_id": "builder-session-with-no-runs"},
        )

        assert response.status_code == 404

    def test_internal_thread_agent_workflow_selection_endpoint_is_removed(self, api_client, sample_thread):
        response = api_client.post(
            f"/api/internal/threads/{sample_thread.id}/agent-workflow",
            json={"workflow_id": "any_internal_agent"},
        )

        assert response.status_code == 404

    def test_validate_agent_workflow_endpoint(self, api_client):
        valid = api_client.post(
            "/api/agent-workflows/validate",
            json={"spec": builtin_router_rag_v2_spec()},
        )
        invalid_spec = builtin_router_rag_v2_spec()
        invalid_spec["config"]["allowed_tool_ids"] = ["mystery_tool"]
        stale_spec = legacy_builtin_router_rag_v1_spec()
        stale_spec["workflow_id"] = "simple_rag_agent"
        invalid = api_client.post(
            "/api/agent-workflows/validate",
            json={"spec": invalid_spec},
        )
        stale = api_client.post(
            "/api/agent-workflows/validate",
            json={"spec": stale_spec},
        )

        assert valid.status_code == 200
        valid_payload = valid.json()
        assert valid_payload["valid"] is True
        assert valid_payload["errors"] == []
        assert valid_payload["workflow_id"] == ROUTER_RAG_AGENT_ID
        assert "document_evidence" in valid_payload["required_tool_ids"]
        assert invalid.status_code == 200
        invalid_payload = invalid.json()
        assert invalid_payload["valid"] is False
        assert invalid_payload["issues"]
        assert all({"code", "severity", "message"} <= set(issue) for issue in invalid_payload["issues"])
        assert invalid_payload["unknown_allowed_tool_ids"] == ["mystery_tool"]
        assert "document_evidence" in invalid_payload["missing_required_tool_ids"]
        assert stale.status_code == 200
        assert stale.json()["valid"] is False

    def test_validate_thread_agent_config_endpoint_resolves_without_running_chat(self, api_client, sample_thread, monkeypatch):
        async def fake_get_thread_settings(_thread_id):
            return {
                "agent_workflow": {"workflow_id": EVALUATOR_REPLANNER_RAG_AGENT_ID},
                "hitl_web_approval": True,
            }

        monkeypatch.setattr("app.api.agent_workflows.get_thread_settings", fake_get_thread_settings)

        response = api_client.post(
            f"/api/threads/{sample_thread.id}/agent-config/validate",
            json={"overrides": {"use_web_search": True, "replans": 2}},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is True
        assert payload["workflow_id"] == EVALUATOR_REPLANNER_RAG_AGENT_ID
        assert payload["workflow_version"] == EVALUATOR_REPLANNER_RAG_AGENT_VERSION
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
        assert payload["workflow_id"] == ROUTER_RAG_AGENT_ID
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            first = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            await repo.complete_run(
                first.id,
                status="completed",
                metrics_json={"duration_ms": 10.0, "route": "direct", "node_event_count": 2},
            )
            second = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            running = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
                resolved_spec_json=builtin_router_rag_spec(),
            )
            awaiting = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
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
            stored_graph = {
                "nodes": [{"id": "stored_only", "label": "Stored Graph"}],
                "edges": [],
                "selectedRoute": "stored",
            }
            debug_payload["graph"] = stored_graph
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
        assert set(payload["debug"]) == {"version", "trace", "summary", "graph", "detail_manifest", "detail_safety"}
        assert payload["debug"]["detail_manifest"] == []
        assert "node_events" not in payload["debug"]
        assert "tool_events" not in payload["debug"]
        assert payload["debug"]["version"] == 1
        assert payload["debug"]["summary"]["route"] == "web"
        assert payload["debug"]["graph"] == stored_graph
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
        assert trace["workflow_id"] == ROUTER_RAG_AGENT_ID
        assert trace["workflow_id"] == ROUTER_RAG_AGENT_ID
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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
            repo = AgentWorkflowRepository(repo_session)
            await repo.seed_builtin_workflows()
            workflow, version = await repo.get_workflow_with_current_version(ROUTER_RAG_AGENT_ID)
            run = await repo.create_run(
                thread_id=sample_thread.id,
                workflow_id=workflow.id,
                workflow_version_id=version.id,
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

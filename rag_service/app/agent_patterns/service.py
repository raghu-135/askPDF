from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from app.agent_patterns.checkpointing import open_agent_checkpointer
from app.agent_patterns.debug_trace import AgentTraceRecorder, merge_debug_payloads
from app.agent_patterns.metrics import build_run_metrics
from app.agent_patterns.repository import AgentPatternRepository, InterruptResolutionResult
from app.agent_patterns.templates import (
    EVALUATOR_REPLANNER_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_AGENT_ID,
    ROUTER_RAG_AGENT_ID,
    SUPPORTED_BUILTIN_TEMPLATE_IDS,
)
from app.agent_patterns.validator import TemplateResolver
from app.db import get_thread_settings


logger = logging.getLogger(__name__)


class AgentRunService:
    """Runs the selected agent pattern, defaulting to the compiled Router RAG graph."""

    def __init__(
        self,
        repository: Optional[AgentPatternRepository] = None,
        resolver: Optional[TemplateResolver] = None,
    ):
        self.repository = repository or AgentPatternRepository()
        self.resolver = resolver or TemplateResolver()

    async def run_thread_chat(self, thread_id: str, req: Any, embed_model: str) -> Dict[str, Any]:
        thread_settings = await get_thread_settings(thread_id)
        agent_settings = thread_settings.get("agent_pattern") if isinstance(thread_settings, dict) else None
        agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
        template_id = agent_settings.get("template_id") or ROUTER_RAG_AGENT_ID
        if template_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
            logger.warning(
                "Unsupported agent pattern requested for thread %s | requested_template=%s fallback_template=%s",
                thread_id,
                template_id,
                ROUTER_RAG_AGENT_ID,
            )
            template_id = ROUTER_RAG_AGENT_ID
        logger.info("Resolving agent pattern for thread %s | requested_template=%s", thread_id, template_id)

        template, version = await self.repository.get_template_with_current_version(template_id)
        if template is None or version is None:
            await self.repository.seed_builtin_templates()
            template, version = await self.repository.get_template_with_current_version(template_id)
        if template is None or version is None:
            raise RuntimeError("No agent pattern is available")
        logger.info(
            "Selected agent pattern for thread %s | template=%s version=%s",
            thread_id,
            template.id,
            version.version,
        )

        request_overrides = {
            "use_web_search": getattr(req, "use_web_search", None),
            "use_reranker": getattr(req, "use_reranker", None),
            "max_iterations": getattr(req, "max_iterations", None),
            "system_role": getattr(req, "system_role_override", None),
            "tool_instructions": getattr(req, "tool_instructions_override", None),
            "custom_instructions": getattr(req, "custom_instructions_override", None),
        }
        resolved_spec = self.resolver.resolve(
            version.spec_json,
            thread_settings=thread_settings,
            request_overrides=request_overrides,
        )
        from app.agent_patterns.graph import TemplateCompiler, normalize_hitl_policy_for_thread_settings

        resolved_config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), dict) else {}
        resolved_config["hitl_policy"] = normalize_hitl_policy_for_thread_settings(
            resolved_config.get("hitl_policy"),
            thread_settings,
        )
        resolved_spec["config"] = resolved_config
        stored_resolved_spec = TemplateCompiler().materialize_spec(
            resolved_spec,
        )

        run = await self.repository.create_run(
            thread_id=thread_id,
            template_id=template.id,
            template_version_id=version.id,
            template_version=version.version,
            resolved_spec_json=stored_resolved_spec,
        )

        started = time.perf_counter()
        trace_recorder = AgentTraceRecorder(run)
        context = {
            "agent_run_id": run.id,
            "agent_pattern_id": template.id,
            "agent_pattern_version": version.version,
            "agent_pattern_template_version_id": version.id,
            "checkpoint_thread_id": run.checkpoint_thread_id,
        }

        try:
            logger.info("Invoking compiled agent pattern for thread %s | template=%s", thread_id, template.id)
            from app.agent_patterns.router_runtime import (
                handle_evaluator_replanner_rag_chat,
                handle_plan_execute_rag_chat,
                handle_router_rag_chat,
            )

            if template.id == EVALUATOR_REPLANNER_RAG_AGENT_ID:
                handler = handle_evaluator_replanner_rag_chat
            elif template.id == PLAN_EXECUTE_RAG_AGENT_ID:
                handler = handle_plan_execute_rag_chat
            else:
                handler = handle_router_rag_chat
            async with open_agent_checkpointer() as checkpointer:
                result = await handler(
                    thread_id,
                    req,
                    embed_model,
                    resolved_spec=stored_resolved_spec,
                    agent_run_context=context,
                    trace_recorder=trace_recorder,
                    checkpointer=checkpointer,
                )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = result.get("status") if isinstance(result.get("status"), str) else None
            if status is None:
                status = "failed" if error_json else "clarification" if result.get("clarification_options") else "completed"
            metrics = build_run_metrics(result, duration_ms=duration_ms)
            if status == "awaiting_human":
                if hasattr(trace_recorder, "record_runtime_event"):
                    trace_recorder.record_runtime_event(
                        "checkpoint.created",
                        attributes={
                            "askpdf.run.id": run.id,
                            "askpdf.thread.id": thread_id,
                            "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                            "askpdf.status": "awaiting_human",
                        },
                        output_data={
                            "interrupt_id": (result.get("pending_interrupt") or {}).get("interrupt_id"),
                            "route": result.get("route"),
                        },
                    )
                debug_payload = trace_recorder.finalize(
                    run=run,
                    chat_turn_id=None,
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                )
                paused_run = await self.repository.mark_run_awaiting_human(
                    run.id,
                    result.get("pending_interrupt") or {},
                    metrics_json=metrics,
                    debug_trace_json=debug_payload,
                )
                if paused_run is not None:
                    result["pending_interrupt"] = paused_run.pending_interrupt_json
                result.update(context)
                return result
            completed_run = await self.repository.complete_run(
                run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=result.get("chat_turn_id"),
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            result.update(context)
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = {
                "code": "agent_run_failed",
                "raw_message": str(exc),
                "retryable": True,
            }
            metrics = build_run_metrics({"agent_error": error_json}, duration_ms=duration_ms)
            completed_run = await self.repository.complete_run(
                run.id,
                status="failed",
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=None,
                    metrics=metrics,
                    error=error_json,
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            raise

    async def resume_agent_run(
        self,
        run_id: str,
        *,
        interrupt_id: str,
        action: str,
        edited_payload: Optional[Dict[str, Any]] = None,
        client_metadata: Optional[Dict[str, Any]] = None,
        selected_option_ids: Optional[list[str]] = None,
        resume_token: Optional[str] = None,
        resume_version: Optional[int] = None,
        expected_thread_id: Optional[str] = None,
    ) -> Optional[InterruptResolutionResult]:
        resolution = await self.repository.resolve_pending_interrupt(
            run_id,
            interrupt_id=interrupt_id,
            action=action,
            edited_payload=edited_payload,
            client_metadata=client_metadata,
            selected_option_ids=selected_option_ids,
            resume_token=resume_token,
            resume_version=resume_version,
            expected_thread_id=expected_thread_id,
        )
        if resolution is None:
            return None
        if (
            resolution.duplicate
            or resolution.outcome != "resumed"
            or not isinstance(resolution.interrupt, dict)
            or resolution.interrupt.get("checkpoint_resume") is not True
        ):
            return resolution

        try:
            from app.agent_patterns.router_runtime import resume_compiled_rag_chat

            resume_trace_recorder = AgentTraceRecorder(resolution.run)
            async with open_agent_checkpointer() as checkpointer:
                result = await resume_compiled_rag_chat(
                    resolution.run,
                    interrupt=resolution.interrupt,
                    checkpointer=checkpointer,
                    trace_recorder=resume_trace_recorder,
                )
            prior_metrics = dict(resolution.run.metrics_json or {})
            metrics = {
                **prior_metrics,
                **build_run_metrics(result, duration_ms=float(result.get("duration_ms") or 0)),
            }
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = result.get("status") if isinstance(result.get("status"), str) else "completed"
            if status == "awaiting_human":
                pending_interrupt = result.get("pending_interrupt") or {}
                if hasattr(resume_trace_recorder, "record_runtime_event"):
                    resume_trace_recorder.record_runtime_event(
                        "checkpoint.created",
                        attributes={
                            "askpdf.run.id": resolution.run.id,
                            "askpdf.thread.id": resolution.run.thread_id,
                            "askpdf.checkpoint.thread_id": resolution.run.checkpoint_thread_id,
                            "askpdf.status": "awaiting_human",
                        },
                        output_data={
                            "interrupt_id": pending_interrupt.get("interrupt_id"),
                            "route": result.get("route"),
                        },
                    )
                resume_debug_payload = resume_trace_recorder.finalize(
                    run=resolution.run,
                    chat_turn_id=None,
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                )
                debug_payload = resume_debug_payload
                if isinstance(resolution.run.debug_trace_json, dict):
                    debug_payload = merge_debug_payloads(
                        resolution.run.debug_trace_json,
                        resume_debug_payload,
                        resolved_spec=resolution.run.resolved_spec_json if isinstance(resolution.run.resolved_spec_json, dict) else {},
                        run_status=status,
                        completed_at=None,
                        chat_turn_id=None,
                        metrics=metrics,
                    )
                paused_run = await self.repository.mark_run_awaiting_human(
                    resolution.run.id,
                    pending_interrupt,
                    metrics_json=metrics,
                    debug_trace_json=debug_payload,
                )
                if paused_run is not None:
                    return InterruptResolutionResult(
                        run=paused_run,
                        outcome=resolution.outcome,
                        interrupt=paused_run.pending_interrupt_json or resolution.interrupt,
                        duplicate=False,
                    )
                return resolution

            completed_run = await self.repository.complete_run(
                resolution.run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is None:
                return resolution
            resume_debug_payload = resume_trace_recorder.finalize(
                run=completed_run,
                chat_turn_id=result.get("chat_turn_id"),
                metrics=metrics,
                route=result.get("route"),
                route_reason=result.get("route_reason"),
                error=error_json,
            )
            if isinstance(completed_run.debug_trace_json, dict):
                debug_payload = merge_debug_payloads(
                    completed_run.debug_trace_json,
                    resume_debug_payload,
                    resolved_spec=completed_run.resolved_spec_json if isinstance(completed_run.resolved_spec_json, dict) else {},
                    run_status=status,
                    completed_at=completed_run.completed_at,
                    chat_turn_id=result.get("chat_turn_id"),
                    metrics=metrics,
                )
                completed_run = await self.repository.set_run_debug_trace(completed_run.id, debug_payload) or completed_run
            else:
                completed_run = await self.repository.set_run_debug_trace(completed_run.id, resume_debug_payload) or completed_run
            return InterruptResolutionResult(
                run=completed_run,
                outcome=resolution.outcome,
                interrupt=resolution.interrupt,
                duplicate=False,
            )
        except Exception as exc:
            prior_metrics = dict(resolution.run.metrics_json or {})
            prior_metrics["error_count"] = max(int(prior_metrics.get("error_count") or 0), 1)
            await self.repository.complete_run(
                resolution.run.id,
                status="failed",
                metrics_json=prior_metrics,
                error_json={
                    "code": "agent_run_resume_failed",
                    "raw_message": str(exc),
                    "retryable": True,
                },
            )
            raise

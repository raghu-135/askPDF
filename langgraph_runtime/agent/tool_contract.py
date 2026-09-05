from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.runnables import RunnableConfig

from langgraph_runtime.workflows.trace import artifact_summary, compact_preview, refs_from_artifacts
from runtime_protocol.tool_contract import (
    ToolError as _CanonicalToolError,
    ToolErrorCode as _CanonicalToolErrorCode,
    ToolMetrics as _CanonicalToolMetrics,
    ToolResult as _CanonicalToolResult,
    ToolTrace as _CanonicalToolTrace,
    ToolWarningCode as _CanonicalToolWarningCode,
    normalize_tool_result as _canonical_normalize_tool_result,
)
from langgraph_runtime.rag.enums import TimelineSourceType
from langgraph_runtime.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)
DEFAULT_DOCUMENT_SOURCE_TYPE = "pdf"


class AskPdfTool(Protocol):
    """Protocol for LangChain-compatible askPDF tools."""

    name: str

    async def ainvoke(self, input: Any, config: RunnableConfig = None) -> Any:
        ...


def _wire_string(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def tool_started() -> float:
    return time.perf_counter()


def tool_trace(
    tool_name: str,
    config: RunnableConfig = None,
    *,
    context: Any = None,
) -> ToolTrace:
    """Build a trace from a neutral invocation context or framework config.

    ``config`` remains for workflow compatibility, but framework-neutral
    handlers should pass ``context`` directly and never construct a
    LangChain-shaped configuration dictionary.
    """
    if context is not None:
        return ToolTrace(
            tool_name=tool_name,
            caller_node=getattr(context, "caller_node", None),
            caller_node_type=getattr(context, "caller_node_type", None),
            agent_run_id=getattr(context, "run_id", None),
            thread_id=getattr(context, "thread_id", None),
            route=getattr(context, "route", None),
            tool_call_id=getattr(context, "tool_call_id", None),
            mcp_request_id=getattr(context, "mcp_request_id", None),
        )
    conf = config.get("configurable", {}) if config else {}
    return ToolTrace(
        tool_name=tool_name,
        caller_node=conf.get("caller_node"),
        caller_node_type=conf.get("caller_node_type"),
        agent_run_id=conf.get("agent_run_id"),
        thread_id=conf.get("app_thread_id") or conf.get("thread_id"),
        route=conf.get("route"),
        tool_call_id=conf.get("tool_call_id"),
        mcp_request_id=conf.get("mcp_request_id"),
    )


def make_tool_result(
    *,
    tool_name: str,
    content: str,
    config: RunnableConfig = None,
    context: Any = None,
    started: Optional[float] = None,
    ok: bool = True,
    sources: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[ToolError] = None,
) -> ToolResult:
    source_items = [item for item in (sources or []) if isinstance(item, dict)]
    warning_items = [_wire_string(item) for item in (warnings or []) if item]
    elapsed_ms = (time.perf_counter() - started) * 1000 if started is not None else 0.0
    completed_at = utc_now()
    trace = tool_trace(tool_name, config, context=context)
    if started is not None:
        trace.start_time = iso_utc_z(completed_at - timedelta(milliseconds=elapsed_ms))
        trace.end_time = iso_utc_z(completed_at)
    result = ToolResult(
        ok=ok,
        content=str(content or ""),
        sources=source_items,
        artifacts=artifacts or {},
        warnings=warning_items,
        error=error,
        metrics=ToolMetrics(
            elapsed_ms=round(elapsed_ms, 2),
            result_chars=len(str(content or "")),
            source_count=len(source_items),
            warning_count=len(warning_items),
        ),
        trace=trace,
    )
    logger.info(
        "Tool completed | tool=%s tool_call_id=%s caller_node=%s run_id=%s thread_id=%s ok=%s elapsed_ms=%.1f result_chars=%s sources=%s warnings=%s",
        result.trace.tool_name,
        result.trace.tool_call_id,
        result.trace.caller_node,
        result.trace.agent_run_id,
        result.trace.thread_id,
        result.ok,
        result.metrics.elapsed_ms,
        result.metrics.result_chars,
        result.metrics.source_count,
        result.metrics.warning_count,
    )
    if warning_items:
        logger.warning(
            "Tool completed with warnings | tool=%s warnings=%s",
            result.trace.tool_name,
            warning_items,
        )
    return result


def make_tool_error_result(
    *,
    tool_name: str,
    error: Exception,
    config: RunnableConfig = None,
    context: Any = None,
    started: Optional[float] = None,
    user_message: Optional[str] = None,
    code: Optional[str] = None,
    evidence_gap: bool = False,
) -> ToolResult:
    tool_error = ToolError(
        code=code or ToolErrorCode.failed(tool_name),
        message=str(error),
        type=type(error).__name__,
        retryable=True,
        evidence_gap=evidence_gap,
    )
    result = make_tool_result(
        tool_name=tool_name,
        content=user_message or f"{tool_name} failed: {error}",
        config=config,
        context=context,
        started=started,
        ok=False,
        warnings=[tool_error.code],
        error=tool_error,
    )
    logger.error(
        "Tool failed | tool=%s caller_node=%s run_id=%s thread_id=%s error_type=%s error=%s",
        result.trace.tool_name,
        result.trace.caller_node,
        result.trace.agent_run_id,
        result.trace.thread_id,
        tool_error.type,
        tool_error.message,
        exc_info=(type(error), error, error.__traceback__),
    )
    return result


def _safe_json_object(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def compact_tool_event(payload: Dict[str, Any], *, tool_input: Any = None) -> Dict[str, Any]:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    event = {
        "tool_name": trace.get("tool_name"),
        "tool_call_id": trace.get("tool_call_id"),
        "caller_node": trace.get("caller_node"),
        "caller_node_type": trace.get("caller_node_type"),
        "ok": bool(payload.get("ok", True)),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "result_chars": metrics.get("result_chars"),
        "source_count": metrics.get("source_count"),
        "warnings": list(payload.get("warnings") or []),
        "error": payload.get("error"),
        "dispatch_id": payload.get("dispatch_id"),
        "work_id": payload.get("work_id"),
        "ordinal": payload.get("ordinal"),
        "attempt": payload.get("attempt"),
        "approval_ref": payload.get("approval_ref"),
        "argument_hash": payload.get("argument_hash"),
        "transport": payload.get("transport"),
        "mcp_mode": payload.get("mcp_mode"),
        "mcp_server": payload.get("mcp_server"),
        "mcp_contract_version": payload.get("mcp_contract_version"),
        "mcp_request_id": payload.get("mcp_request_id"),
    }
    if not event["mcp_request_id"]:
        event["mcp_request_id"] = trace.get("mcp_request_id")
    if trace.get("start_time"):
        event["start_time"] = trace.get("start_time")
    if trace.get("end_time"):
        event["end_time"] = trace.get("end_time")
    if tool_input is not None:
        event["tool_input"] = tool_input
    if payload.get("content"):
        event["result_preview"] = compact_preview(payload.get("content"))
    refs = refs_from_artifacts(artifacts)
    if refs:
        event["artifact_refs"] = refs
    summary = artifact_summary(artifacts)
    if summary:
        event["artifact_summary"] = summary
    memory_debug_artifacts = {
        key: artifacts[key]
        for key in ("memory_scopes", "memory_scope_policy")
        if key in artifacts
    }
    if memory_debug_artifacts:
        event["artifacts"] = memory_debug_artifacts
    return event


def collect_tool_sources(
    content: str,
    document_sources: list,
    web_sources: list,
    used_chat_ids: list,
) -> None:
    """Collect source artifacts from the canonical tool result envelope."""

    if not isinstance(content, str):
        return
    data = normalize_tool_result(content)
    artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    document_sources.extend(artifacts.get("document_sources") or [])
    web_sources.extend(artifacts.get("web_sources") or [])
    used_chat_ids.extend(artifacts.get("used_chat_ids") or [])
    for event in artifacts.get("timeline_events") or []:
        if not isinstance(event, dict):
            continue
        source_type = event.get("source_type")
        if source_type == TimelineSourceType.CONVERSATION.value and event.get("message_id"):
            used_chat_ids.append(event["message_id"])
        elif source_type == TimelineSourceType.DOCUMENT.value:
            document_sources.append({
                "text": event.get("excerpt", ""),
                "file_hash": event.get("file_hash"),
                "file_name": event.get("file_name"),
                "source_type": event.get("document_source_type", DEFAULT_DOCUMENT_SOURCE_TYPE),
                "document_available_in_thread_at": event.get("document_available_in_thread_at"),
                "timeline_event_at": event.get("timeline_event_at"),
                "timeline_event_type": event.get("timeline_event_type"),
                "page_count": event.get("page_count"),
                "word_count": event.get("word_count"),
                "sentence_count": event.get("sentence_count"),
                "languages": event.get("languages"),
                "filetype": event.get("filetype"),
                "element_types": event.get("element_types"),
            })
        elif source_type == TimelineSourceType.WEB_CACHE.value:
            web_sources.append({
                "text": event.get("excerpt", ""),
                "url": event.get("url"),
                "title": event.get("title"),
                "web_search_performed_at": event.get("web_search_performed_at"),
                "timeline_event_at": event.get("timeline_event_at"),
                "timeline_event_type": event.get("timeline_event_type"),
                "score": event.get("score"),
            })


# LangGraph-specific tracing remains here; the neutral result wire contract is
# shared with the control plane and cannot drift between services.
ToolWarningCode = _CanonicalToolWarningCode
ToolErrorCode = _CanonicalToolErrorCode
ToolError = _CanonicalToolError
ToolMetrics = _CanonicalToolMetrics
ToolTrace = _CanonicalToolTrace
ToolResult = _CanonicalToolResult
normalize_tool_result = _canonical_normalize_tool_result

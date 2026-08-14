from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, time as datetime_time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from app.agent_workflows.trace import artifact_summary, compact_preview, refs_from_artifacts
from app.db import FileSourceType
from app.rag.enums import TimelineSourceType
from app.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)


def _compat_json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    return str(value)


class AskPdfTool(Protocol):
    """Protocol for LangChain-compatible askPDF tools."""

    name: str

    async def ainvoke(self, input: Any, config: RunnableConfig = None) -> Any:
        ...


class ToolWarningCode(str, Enum):
    EMPTY_EXTERNAL_TOOL_RESULT = "empty_external_tool_result"
    MISSING_DOCUMENT_VECTORS = "missing_document_vectors"
    MISSING_THREAD_CONTEXT = "missing_thread_context"
    MISSING_THREAD_ID = "missing_thread_id"
    NO_RELEVANT_CONTENT = "no_relevant_content"
    NO_RELEVANT_CONVERSATION_HISTORY = "no_relevant_conversation_history"
    NO_RELEVANT_MEMORY = "no_relevant_memory"
    NO_THREAD_DOCUMENTS = "no_thread_documents"
    NO_TIMELINE_EVENTS = "no_timeline_events"
    NO_USABLE_WEB_RESULTS = "no_usable_web_results"
    TOOL_OUTPUT_CONTENT_COERCED = "tool_output_content_coerced"
    TOOL_OUTPUT_MISSING_CONTENT = "tool_output_missing_content"
    TOOL_OUTPUT_SOURCES_INVALID = "tool_output_sources_invalid"
    WEB_SEARCH_DISABLED = "web_search_disabled"


class ToolErrorCode(str, Enum):
    TOOL_FAILED_SUFFIX = "failed"

    @staticmethod
    def failed(tool_name: str) -> str:
        return f"{tool_name}_{ToolErrorCode.TOOL_FAILED_SUFFIX.value}"


def _wire_string(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


class ToolError(BaseModel):
    code: str
    message: str
    type: str = "ToolError"
    retryable: bool = True


class ToolMetrics(BaseModel):
    elapsed_ms: float = 0.0
    result_chars: int = 0
    source_count: int = 0
    warning_count: int = 0


class ToolTrace(BaseModel):
    tool_name: str = ""
    caller_node: Optional[str] = None
    caller_node_type: Optional[str] = None
    agent_run_id: Optional[str] = None
    thread_id: Optional[str] = None
    route: Optional[str] = None
    tool_call_id: Optional[str] = None
    mcp_request_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ToolError] = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    trace: ToolTrace = Field(default_factory=ToolTrace)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_constructor_fields(cls, value: Any) -> Any:
        """Keep old first-party helpers source-compatible during migration.

        The canonical wire shape remains this model; ``text`` and
        ``structured_content`` are accepted only as input compatibility aliases.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "content" not in data and "text" in data:
            data["content"] = data.pop("text")
        structured = data.pop("structured_content", None)
        if isinstance(structured, dict):
            artifacts = dict(data.get("artifacts") or {})
            for key in ("document_sources", "web_sources", "used_chat_ids", "timeline_events", "memory_refs", "evidence_segments", "thread_shape"):
                if key in structured and key not in artifacts:
                    artifacts[key] = structured[key]
            data["artifacts"] = artifacts
            if "sources" not in data and isinstance(structured.get("sources"), list):
                data["sources"] = structured["sources"]
            if "warnings" not in data and isinstance(structured.get("warnings"), list):
                data["warnings"] = structured["warnings"]
        if isinstance(data.get("error"), dict):
            error = dict(data["error"])
            error.setdefault("message", error.get("code", "Tool failed"))
            data["error"] = error
        return data

    @property
    def text(self) -> str:
        return self.content

    def structured(self, *, contract_id: str, contract_version: str = "1") -> Dict[str, Any]:
        value = json.loads(json.dumps(self.model_dump(mode="python", exclude_none=True), default=_compat_json_default))
        value["contract_id"] = contract_id
        value["contract_version"] = contract_version
        return value

    def legacy_payload(self, *, contract_id: str, contract_version: str = "1") -> str:
        structured = self.structured(contract_id=contract_id, contract_version=contract_version)
        payload: Dict[str, Any] = {"content": self.content}
        for key, legacy_key in {
            "document_sources": "__document_sources__",
            "web_sources": "__web_sources__",
            "used_chat_ids": "__used_chat_ids__",
            "timeline_events": "__timeline_events__",
        }.items():
            if structured.get("artifacts", {}).get(key):
                payload[legacy_key] = structured["artifacts"][key]
        if self.warnings:
            payload["__warnings__"] = list(self.warnings)
        if self.artifacts:
            payload["__artifacts__"] = dict(self.artifacts)
        if not self.ok:
            payload["ok"] = False
        if self.error is not None:
            payload["error"] = self.error.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False)

    def to_payload(self, *, legacy_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        if legacy_fields:
            payload.update(legacy_fields)
        return payload

    def to_json(self, *, legacy_fields: Optional[Dict[str, Any]] = None) -> str:
        return json.dumps(self.to_payload(legacy_fields=legacy_fields))


def tool_started() -> float:
    return time.perf_counter()


def tool_trace(
    tool_name: str,
    config: RunnableConfig = None,
    *,
    context: Any = None,
) -> ToolTrace:
    """Build a trace from a neutral invocation context or legacy config.

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
) -> ToolResult:
    tool_error = ToolError(
        code=code or ToolErrorCode.failed(tool_name),
        message=str(error),
        type=type(error).__name__,
        retryable=True,
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


def normalize_tool_result(raw: Any, *, tool_name: str = "unknown_tool", config: RunnableConfig = None) -> Dict[str, Any]:
    """Normalize legacy tool strings/dicts and new ToolResult envelopes."""

    if isinstance(raw, ToolResult):
        payload = raw.to_payload()
    elif isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, str):
        payload = _safe_json_object(raw) or {"content": raw}
    else:
        payload = {"content": str(raw or "")}

    warnings = [_wire_string(item) for item in list(payload.get("warnings") or []) if item]
    content = payload.get("content")
    if content is None:
        content = ""
        warnings.append(ToolWarningCode.TOOL_OUTPUT_MISSING_CONTENT.value)
    if not isinstance(content, str):
        content = str(content)
        warnings.append(ToolWarningCode.TOOL_OUTPUT_CONTENT_COERCED.value)

    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    legacy_artifacts = {
        "document_sources": payload.get("__document_sources__", []),
        "web_sources": payload.get("__web_sources__", []),
        "used_chat_ids": payload.get("__used_chat_ids__", []),
        "timeline_events": payload.get("__timeline_events__", []),
    }
    for key, value in legacy_artifacts.items():
        if value and key not in artifacts:
            artifacts[key] = value

    raw_sources = payload.get("sources")
    if raw_sources is None:
        sources = []
    elif isinstance(raw_sources, list):
        sources = raw_sources
    else:
        sources = []
        warnings.append(ToolWarningCode.TOOL_OUTPUT_SOURCES_INVALID.value)

    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else tool_trace(tool_name, config).model_dump(mode="json")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics.setdefault("result_chars", len(content))
    metrics.setdefault("source_count", len(sources))
    metrics.setdefault("warning_count", len(warnings))

    normalized = {
        **payload,
        "ok": bool(payload.get("ok", True)),
        "content": content,
        "sources": sources,
        "artifacts": artifacts,
        "warnings": warnings,
        "metrics": metrics,
        "trace": trace,
        "__document_sources__": artifacts.get("document_sources", payload.get("__document_sources__", [])),
        "__web_sources__": artifacts.get("web_sources", payload.get("__web_sources__", [])),
        "__used_chat_ids__": artifacts.get("used_chat_ids", payload.get("__used_chat_ids__", [])),
        "__timeline_events__": artifacts.get("timeline_events", payload.get("__timeline_events__", [])),
    }
    if warnings:
        logger.warning(
            "Normalized tool output with warnings | tool=%s warnings=%s",
            tool_name,
            warnings,
        )
    return normalized


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
    """Collect source artifacts from a tool result envelope or legacy payload."""

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
                "source_type": event.get("document_source_type", FileSourceType.PDF.value),
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

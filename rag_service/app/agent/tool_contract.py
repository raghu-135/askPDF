from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AskPdfTool(Protocol):
    """Protocol for LangChain-compatible askPDF tools."""

    name: str

    async def ainvoke(self, input: Any, config: RunnableConfig = None) -> Any:
        ...


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
    tool_name: str
    caller_node: Optional[str] = None
    agent_run_id: Optional[str] = None
    thread_id: Optional[str] = None
    route: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ToolError] = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    trace: ToolTrace

    def to_payload(self, *, legacy_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        if legacy_fields:
            payload.update(legacy_fields)
        return payload

    def to_json(self, *, legacy_fields: Optional[Dict[str, Any]] = None) -> str:
        return json.dumps(self.to_payload(legacy_fields=legacy_fields))


def tool_started() -> float:
    return time.perf_counter()


def tool_trace(tool_name: str, config: RunnableConfig = None) -> ToolTrace:
    conf = config.get("configurable", {}) if config else {}
    return ToolTrace(
        tool_name=tool_name,
        caller_node=conf.get("caller_node"),
        agent_run_id=conf.get("agent_run_id"),
        thread_id=conf.get("thread_id"),
        route=conf.get("route"),
        tool_call_id=conf.get("tool_call_id"),
    )


def make_tool_result(
    *,
    tool_name: str,
    content: str,
    config: RunnableConfig = None,
    started: Optional[float] = None,
    ok: bool = True,
    sources: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[ToolError] = None,
) -> ToolResult:
    source_items = [item for item in (sources or []) if isinstance(item, dict)]
    warning_items = [str(item) for item in (warnings or []) if item]
    elapsed_ms = (time.perf_counter() - started) * 1000 if started is not None else 0.0
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
        trace=tool_trace(tool_name, config),
    )
    logger.info(
        "Tool completed | tool=%s caller_node=%s run_id=%s thread_id=%s ok=%s elapsed_ms=%.1f result_chars=%s sources=%s warnings=%s",
        result.trace.tool_name,
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
    started: Optional[float] = None,
    user_message: Optional[str] = None,
    code: Optional[str] = None,
) -> ToolResult:
    tool_error = ToolError(
        code=code or f"{tool_name}_failed",
        message=str(error),
        type=type(error).__name__,
        retryable=True,
    )
    result = make_tool_result(
        tool_name=tool_name,
        content=user_message or f"{tool_name} failed: {error}",
        config=config,
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

    warnings = list(payload.get("warnings") or [])
    content = payload.get("content")
    if content is None:
        content = ""
        warnings.append("tool_output_missing_content")
    if not isinstance(content, str):
        content = str(content)
        warnings.append("tool_output_content_coerced")

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
        warnings.append("tool_output_sources_invalid")

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


def compact_tool_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return {
        "tool_name": trace.get("tool_name"),
        "caller_node": trace.get("caller_node"),
        "ok": bool(payload.get("ok", True)),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "result_chars": metrics.get("result_chars"),
        "source_count": metrics.get("source_count"),
        "warnings": list(payload.get("warnings") or []),
        "error": payload.get("error"),
    }

"""Canonical, framework-neutral tool result wire contract."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


MAX_TOOL_RESULT_BYTES = 256_000
MAX_TOOL_RESULT_STRING_LENGTH = 20_000
MAX_TOOL_RESULT_COLLECTION_ITEMS = 2_000


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
    WEB_SEARCH_FAILED = "search_web_failed"
    RESPONSE_TRUNCATED = "tool_response_truncated"


class ToolErrorCode(str, Enum):
    TOOL_FAILED_SUFFIX = "failed"
    RESULT_TOO_LARGE = "tool_result_size_exceeded"
    PROTOCOL_ERROR = "mcp_protocol_error"

    @staticmethod
    def failed(tool_name: str) -> str:
        return f"{tool_name}_{ToolErrorCode.TOOL_FAILED_SUFFIX.value}"


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    type: str = "ToolError"
    retryable: bool = True
    evidence_gap: bool = False


class ToolMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_ms: float = 0.0
    result_chars: int = 0
    source_count: int = 0
    warning_count: int = 0


class ToolTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    """The only accepted tool result shape at a service boundary."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    content: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ToolError] = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    trace: ToolTrace = Field(default_factory=ToolTrace)

    @property
    def text(self) -> str:
        return self.content

    def to_payload(self) -> Dict[str, Any]:
        value = self.model_dump(mode="json", exclude_none=False)
        validate_tool_result_envelope(value)
        validate_tool_result_payload(value)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)


CANONICAL_TOOL_RESULT_FIELDS = frozenset(
    {"ok", "content", "sources", "artifacts", "warnings", "error", "metrics", "trace"}
)


def validate_tool_result_envelope(value: Dict[str, Any]) -> Dict[str, Any]:
    """Strictly validate the wire shape; trusted factories use ``ToolResult``."""

    if not isinstance(value, dict):
        raise ValueError("tool result must be an object envelope")
    missing = sorted(CANONICAL_TOOL_RESULT_FIELDS - set(value))
    if missing:
        raise ValueError(f"tool result is missing canonical fields: {', '.join(missing)}")
    try:
        validated = ToolResult.model_validate(value, strict=True)
    except Exception as exc:
        raise ValueError("tool result has invalid canonical field types") from exc
    if validated.ok and validated.error is not None:
        raise ValueError("successful tool result cannot contain an error")
    if not validated.ok and validated.error is None:
        raise ValueError("failed tool result must contain a structured error")
    if validated.error is not None and (not validated.error.code or not validated.error.message):
        raise ValueError("tool result error requires code and message")
    return value


def tool_result_size(value: Dict[str, Any]) -> Dict[str, int]:
    """Return the shared response-budget measurements for a canonical payload."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {
        "content_chars": len(value.get("content") or ""),
        "source_count": len(value.get("sources") or []),
        "warning_count": len(value.get("warnings") or []),
        "serialized_bytes": len(encoded.encode("utf-8")),
    }


def validate_tool_result_payload(value: Dict[str, Any]) -> Dict[str, Any]:
    """Reject oversized neutral tool results instead of truncating them."""

    validate_tool_result_envelope(value)

    if len(value.get("content") or "") > MAX_TOOL_RESULT_STRING_LENGTH:
        raise ValueError("tool result content exceeds the maximum length")
    if len(value.get("sources") or []) > MAX_TOOL_RESULT_COLLECTION_ITEMS:
        raise ValueError("tool result contains too many sources")
    if len(value.get("warnings") or []) > MAX_TOOL_RESULT_COLLECTION_ITEMS:
        raise ValueError("tool result contains too many warnings")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("tool result contains non-JSON values") from exc
    if len(encoded.encode("utf-8")) > MAX_TOOL_RESULT_BYTES:
        raise ValueError("tool result exceeds the maximum serialized size")
    return value


def normalize_tool_result(raw: Any, *, tool_name: str = "unknown_tool", config: Any = None) -> Dict[str, Any]:
    """Validate a canonical result; reject legacy and non-object payloads."""

    if isinstance(raw, ToolResult):
        return raw.to_payload()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"tool {tool_name} returned a non-canonical result envelope") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"tool {tool_name} returned a non-canonical result envelope")
    legacy = sorted(key for key in raw if str(key).startswith("__"))
    if legacy:
        raise ValueError(f"tool {tool_name} returned legacy fields: {', '.join(legacy)}")
    value = dict(raw)
    validate_tool_result_envelope(value)
    return ToolResult.model_validate(value, strict=True).to_payload()

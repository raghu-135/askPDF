"""Dependency-light JSON envelope, canonical event, and SSE primitives.

This module is intentionally independent of askPDF application/runtime code so
the control-plane, LangGraph runtime, and Hermes gateway share one wire-level
implementation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping


CANONICAL_RUNTIME_EVENT_KINDS = frozenset({
    "run.queued", "run.started", "run.paused", "run.resumed", "run.completed",
    "run.failed", "run.cancel_requested", "run.cancelled", "output.delta",
    "output.completed", "llm.started", "llm.completed", "llm.failed",
    "reasoning.available", "interrupt.requested", "interrupt.responded",
    "approval.requested", "approval.responded", "tool.started", "tool.progress",
    "tool.completed", "tool.failed", "subagent.started", "subagent.progress",
    "subagent.completed", "subagent.failed", "subagent.cancelled", "artifact.created",
    "artifact.updated", "artifact.completed", "runtime.event", "operation.started",
    "operation.completed", "operation.failed", "operation.skipped", "dispatch.planned",
    "dispatch.started", "dispatch.barrier_reached", "dispatch.cancelled", "worker.queued",
    "worker.started", "worker.progress", "worker.retrying", "worker.completed",
    "worker.skipped", "worker.failed", "worker.timed_out", "worker.cancelled",
    "aggregation.completed", "aggregation.partial",
})
TERMINAL_RUNTIME_EVENT_KINDS = frozenset({"run.completed", "run.failed", "run.cancelled"})


def json_envelope(
    *,
    status: str,
    result: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "result": dict(result or {}),
        "error": dict(error or {}),
        "runtime_metadata": dict(runtime_metadata or {}),
    }


def structured_error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "safe_message": message,
        "retryable": retryable,
        "details": dict(details or {}),
    }


def validate_event_mapping(value: Mapping[str, Any]) -> None:
    required = {"event_id", "run_id", "sequence", "kind"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("runtime event has an incomplete canonical shape")
    if not isinstance(value["event_id"], str) or not value["event_id"].strip():
        raise ValueError("runtime event event_id must be a non-empty string")
    if not isinstance(value["run_id"], str) or not value["run_id"].strip():
        raise ValueError("runtime event run_id must be a non-empty string")
    if not isinstance(value["kind"], str) or value["kind"] not in CANONICAL_RUNTIME_EVENT_KINDS:
        raise ValueError("runtime event kind is not canonical")
    try:
        if int(value["sequence"]) < 1 or int(value.get("attempt") or 1) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime event sequence and attempt must be positive integers") from exc
    if "payload" in value and not isinstance(value["payload"], Mapping):
        raise ValueError("runtime event payload must be an object")
    if "source_metadata" in value and not isinstance(value["source_metadata"], Mapping):
        raise ValueError("runtime event source_metadata must be an object")
    if "terminal" in value and (
        not isinstance(value["terminal"], bool)
        or value["terminal"] != (value["kind"] in TERMINAL_RUNTIME_EVENT_KINDS)
    ):
        raise ValueError("runtime event terminal flag does not match event kind")


def sse_encode(event: Mapping[str, Any] | Any, *, result: Mapping[str, Any] | Any | None = None) -> str:
    event_value = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    result_value = result.to_dict() if hasattr(result, "to_dict") else result
    payload: dict[str, Any] = {"event": event_value}
    if result_value is not None:
        payload["result"] = dict(result_value)
    return f"id: {event_value['event_id']}\nevent: {event_value['kind']}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


async def iter_sse(response: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_id = ""
    event_name = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield event_name, {"event_id": event_id, "data": json.loads("\n".join(data))}
            event_id, event_name, data = "", "message", []
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event_name, {"event_id": event_id, "data": json.loads("\n".join(data))}

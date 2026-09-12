"""Dependency-light JSON envelope, canonical event, and SSE primitives.

This module is intentionally independent of askPDF application/runtime code so
the control-plane, LangGraph runtime, and Hermes gateway share one wire-level
implementation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

from runtime_protocol.contracts import (
    CANONICAL_RUNTIME_EVENT_KINDS,
    TERMINAL_RUNTIME_EVENT_KINDS,
)
from runtime_protocol.validation import validate_runtime_result_for_event


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


def json_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON operation payload without adding negotiation metadata."""

    if not isinstance(value, Mapping):
        raise TypeError("runtime protocol payload must be an object")
    return dict(value)


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
    from runtime_protocol.contracts import AgentRuntimeEvent
    from runtime_protocol.events import validate_runtime_event

    if not isinstance(value, Mapping):
        raise ValueError("runtime event must be an object")
    required = {"event_id", "run_id", "sequence", "kind"}
    if not required.issubset(value):
        raise ValueError("runtime event has an incomplete canonical shape")
    validate_runtime_event(AgentRuntimeEvent(
        event_id=value["event_id"], run_id=value["run_id"],
        sequence=value["sequence"], kind=value["kind"],
        attempt=value.get("attempt", 1), payload=value.get("payload", {}),
        terminal=value.get("terminal", value["kind"] in TERMINAL_RUNTIME_EVENT_KINDS),
        source_metadata=value.get("source_metadata", {}),
    ))


def decode_event_frame(frame: str) -> dict[str, Any]:
    """Validate one persisted canonical SSE frame without repairing its envelope."""
    fields: dict[str, str] = {}
    data: list[str] = []
    for line in frame.splitlines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line.startswith(("id:", "event:")):
            name, value = line.split(":", 1)
            if name in fields:
                raise ValueError("duplicate SSE envelope field")
            fields[name] = value.lstrip()
    body = json.loads("\n".join(data))
    if not isinstance(body, dict) or not isinstance(body.get("event"), dict):
        raise ValueError("SSE frame requires a canonical event")
    event = body["event"]
    validate_event_mapping(event)
    if body.get("result") is not None:
        if not isinstance(body["result"], Mapping):
            raise ValueError("SSE result must be an object")
        validate_runtime_result_for_event(event["kind"], body["result"], terminal=event.get("terminal"), event_payload=event.get("payload"))
    if fields.get("id") != event["event_id"] or fields.get("event") != event["kind"]:
        raise ValueError("SSE envelope does not match canonical event")
    return body


def sse_encode(event: Mapping[str, Any] | Any, *, result: Mapping[str, Any] | Any | None = None) -> str:
    event_value = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    result_value = result.to_dict() if hasattr(result, "to_dict") else result
    validate_event_mapping(event_value)
    payload: dict[str, Any] = {
        "event": event_value,
    }
    if result_value is not None:
        payload["result"] = dict(result_value)
        validate_runtime_result_for_event(event_value["kind"], payload["result"], terminal=event_value.get("terminal"), event_payload=event_value.get("payload"))
    return f"id: {event_value['event_id']}\nevent: {event_value['kind']}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


async def iter_sse(response: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_id = ""
    event_name = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                value = json.loads("\n".join(data))
                yield event_name, {"event_id": event_id, "data": value}
            event_id, event_name, data = "", "message", []
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        value = json.loads("\n".join(data))
        yield event_name, {"event_id": event_id, "data": value}

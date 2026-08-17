"""Wire serialization and SSE helpers for the internal runtime protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)


WIRE_VERSION = 1


def json_envelope(*, status: str, result: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None, request_id: str | None = None, runtime_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "contract_version": WIRE_VERSION,
        "request_id": request_id,
        "status": status,
        "result": dict(result or {}),
        "error": dict(error or {}),
        "runtime_metadata": dict(runtime_metadata or {}),
    }


def _binding(value: Mapping[str, Any] | None) -> ContinuationBinding | None:
    if not value:
        return None
    return ContinuationBinding(
        binding_type=str(value.get("binding_type") or "unknown"),
        payload=dict(value.get("payload") or {}),
        binding_version=int(value.get("binding_version") or 1),
        runtime_version=value.get("runtime_version"),
    )


def request_from_dict(value: Mapping[str, Any]) -> AgentRuntimeRequest:
    return AgentRuntimeRequest(
        run_id=str(value["run_id"]),
        thread_id=str(value["thread_id"]),
        definition_id=str(value["definition_id"]),
        framework=str(value["framework"]),
        builder_id=str(value["builder_id"]),
        input=dict(value.get("input") or {}),
        options=dict(value.get("options") or {}),
        task_id=value.get("task_id"),
        parent_run_id=value.get("parent_run_id"),
        continuation=_binding(value.get("continuation")),
        trace_id=value.get("trace_id"),
        authentication=dict(value.get("authentication") or {}),
        permissions=dict(value.get("permissions") or {}),
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
    )


def definition_from_dict(value: Mapping[str, Any]) -> AgentDefinition:
    return AgentDefinition(
        definition_id=str(value["definition_id"]),
        framework=str(value["framework"]),
        builder_id=str(value["builder_id"]),
        category=value.get("category"),
        display_name=value.get("display_name"),
        capabilities=dict(value.get("capabilities") or {}),
        definition_version=value.get("definition_version"),
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
        runtime_version=value.get("runtime_version"),
    )


def event_from_dict(value: Mapping[str, Any]) -> AgentRuntimeEvent:
    return AgentRuntimeEvent(
        event_id=str(value["event_id"]),
        run_id=str(value["run_id"]),
        sequence=int(value["sequence"]),
        kind=str(value["kind"]),
        payload=dict(value.get("payload") or {}),
        occurred_at=value.get("occurred_at"),
        terminal=bool(value.get("terminal")),
        trace_id=value.get("trace_id"),
        runtime_version=value.get("runtime_version"),
        continuation=_binding(value.get("continuation")),
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
    )


def result_from_dict(value: Mapping[str, Any]) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        status=str(value.get("status") or "failed"),
        output=value.get("output"),
        clarification=dict(value["clarification"]) if isinstance(value.get("clarification"), Mapping) else None,
        interruption=dict(value["interruption"]) if isinstance(value.get("interruption"), Mapping) else None,
        artifacts=tuple(dict(item) for item in value.get("artifacts") or [] if isinstance(item, Mapping)),
        usage=dict(value.get("usage") or {}),
        runtime_metadata=dict(value.get("runtime_metadata") or {}),
        continuation=_binding(value.get("continuation")),
        error=dict(value["error"]) if isinstance(value.get("error"), Mapping) else None,
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
    )


def validation_from_dict(value: Mapping[str, Any]) -> RuntimeValidationResult:
    return RuntimeValidationResult(
        valid=bool(value.get("valid")),
        issues=tuple(
            RuntimeValidationIssue(
                code=str(item.get("code") or "invalid"),
                message=str(item.get("message") or "Invalid workflow"),
                path=item.get("path"),
                severity=str(item.get("severity") or "error"),
                details=dict(item.get("details") or {}),
            )
            for item in value.get("issues") or []
            if isinstance(item, Mapping)
        ),
        normalized_spec=value.get("normalized_spec"),
        runtime_metadata=dict(value.get("runtime_metadata") or {}),
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
    )


def capabilities_from_dict(value: Mapping[str, Any]) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        streaming=bool(value.get("streaming")),
        resume=bool(value.get("resume")),
        cancellation=bool(value.get("cancellation")),
        inspection=bool(value.get("inspection")),
        continuation_cleanup=bool(value.get("continuation_cleanup")),
        task_execution=bool(value.get("task_execution")),
        native_checkpoints=bool(value.get("native_checkpoints")),
        runtime_version=value.get("runtime_version"),
        contract_version=int(value.get("contract_version") or WIRE_VERSION),
    )


@dataclass(frozen=True)
class ServerEnvelope:
    status: str
    request_id: str | None = None
    result: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    runtime_metadata: Mapping[str, Any] | None = None
    contract_version: int = WIRE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "result": dict(self.result or {}),
            "error": dict(self.error or {}),
            "runtime_metadata": dict(self.runtime_metadata or {}),
            "contract_version": self.contract_version,
        }


def sse_encode(event: AgentRuntimeEvent, *, result: AgentRuntimeResult | None = None) -> str:
    payload: dict[str, Any] = {"event": event.to_dict()}
    if result is not None:
        payload["result"] = result.to_dict()
    return f"id: {event.event_id}\nevent: {event.kind}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


async def iter_sse(response: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse SSE without depending on a framework-specific response type."""

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

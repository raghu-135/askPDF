"""Wire serialization and SSE helpers for the internal runtime protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from runtime_protocol.contracts import (
    CONTRACT_VERSION,
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)
from runtime_protocol.errors import ProtocolDecodeError, ProtocolVersionError


WIRE_VERSION = CONTRACT_VERSION


def validate_wire_version(value: Mapping[str, Any], *, default: int = WIRE_VERSION) -> int:
    raw = value.get("contract_version", default)
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolDecodeError("contract_version must be an integer") from exc
    if version != WIRE_VERSION:
        raise ProtocolVersionError(
            f"unsupported runtime contract version {version}; expected {WIRE_VERSION}"
        )
    return version


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
    if not isinstance(value, Mapping):
        raise ProtocolDecodeError("continuation must be an object")
    return ContinuationBinding(
        binding_type=str(value.get("binding_type") or "unknown"),
        payload=dict(value.get("payload") or {}),
        binding_version=int(value.get("binding_version") or 1),
        runtime_version=value.get("runtime_version"),
    )


def request_from_dict(value: Mapping[str, Any]) -> AgentRuntimeRequest:
    version = validate_wire_version(value)
    try:
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
            contract_version=version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProtocolDecodeError):
            raise
        raise ProtocolDecodeError(f"invalid runtime request: {exc}") from exc


def definition_from_dict(value: Mapping[str, Any]) -> AgentDefinition:
    version = validate_wire_version(value)
    try:
        return AgentDefinition(
            definition_id=str(value["definition_id"]),
            framework=str(value["framework"]),
            builder_id=str(value["builder_id"]),
            category=value.get("category"),
            display_name=value.get("display_name"),
            capabilities=dict(value.get("capabilities") or {}),
            definition_version=value.get("definition_version"),
            contract_version=version,
            runtime_version=value.get("runtime_version"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolDecodeError(f"invalid agent definition: {exc}") from exc


def event_from_dict(value: Mapping[str, Any]) -> AgentRuntimeEvent:
    version = validate_wire_version(value)
    try:
        return AgentRuntimeEvent(
            event_id=str(value["event_id"]),
            run_id=str(value["run_id"]),
            sequence=int(value["sequence"]),
            kind=str(value["kind"]),
            attempt=int(value.get("attempt") or 1),
            payload=dict(value.get("payload") or {}),
            occurred_at=value.get("occurred_at"),
            terminal=bool(value.get("terminal")),
            trace_id=value.get("trace_id"),
            runtime_version=value.get("runtime_version"),
            continuation=_binding(value.get("continuation")),
            contract_version=version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolDecodeError(f"invalid runtime event: {exc}") from exc


def result_from_dict(value: Mapping[str, Any]) -> AgentRuntimeResult:
    version = validate_wire_version(value)
    try:
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
            contract_version=version,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolDecodeError(f"invalid runtime result: {exc}") from exc


def validation_from_dict(value: Mapping[str, Any]) -> RuntimeValidationResult:
    version = validate_wire_version(value)
    try:
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
            contract_version=version,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolDecodeError(f"invalid runtime validation result: {exc}") from exc


def capabilities_from_dict(value: Mapping[str, Any]) -> RuntimeCapabilities:
    version = validate_wire_version(value)
    return RuntimeCapabilities(
        streaming=bool(value.get("streaming")),
        resume=bool(value.get("resume")),
        cancellation=bool(value.get("cancellation")),
        inspection=bool(value.get("inspection")),
        continuation_cleanup=bool(value.get("continuation_cleanup")),
        task_execution=bool(value.get("task_execution")),
        native_checkpoints=bool(value.get("native_checkpoints")),
        runtime_version=value.get("runtime_version"),
        contract_version=version,
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


def sse_encode_payload(event: Mapping[str, Any], *, result: Mapping[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"event": dict(event)}
    if result is not None:
        payload["result"] = dict(result)
    return f"id: {event['event_id']}\nevent: {event['kind']}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


def sse_encode(event: AgentRuntimeEvent, *, result: AgentRuntimeResult | None = None) -> str:
    return sse_encode_payload(
        event.to_dict(),
        result=result.to_dict() if result is not None else None,
    )


async def iter_sse(response: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse SSE without depending on a framework-specific response type."""

    event_id = ""
    event_name = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                try:
                    decoded = json.loads("\n".join(data))
                except json.JSONDecodeError as exc:
                    raise ProtocolDecodeError("runtime SSE data is not valid JSON") from exc
                if not isinstance(decoded, Mapping):
                    raise ProtocolDecodeError("runtime SSE data must be an object")
                yield event_name, {"event_id": event_id, "data": dict(decoded)}
            event_id, event_name, data = "", "message", []
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        try:
            decoded = json.loads("\n".join(data))
        except json.JSONDecodeError as exc:
            raise ProtocolDecodeError("runtime SSE data is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProtocolDecodeError("runtime SSE data must be an object")
        yield event_name, {"event_id": event_id, "data": dict(decoded)}

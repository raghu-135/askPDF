"""Normalization helpers for framework-neutral task results."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.contracts import (
    RuntimeArtifact,
    RuntimeCourseCorrectionOutcome,
    RuntimeTaskResult,
    RuntimeTaskResultStatus,
)
from runtime_protocol.validation import validate_runtime_result_envelope


_CANONICAL_TEXT_KEYS = ("text", "summary", "answer")
_TEXT_ALIAS_KEYS = ("output", "content", "result", "message")
_NON_ANSWER_BLOCK_TYPES = frozenset({"reasoning", "thinking", "tool_use", "tool_call"})
_RESULT_CONTROL_KEYS = frozenset({
    "status", "warnings", "gaps", "uncovered_gaps", "error", "usage",
    "framework_details", "artifacts", "structured", "structured_output", "correction_outcomes",
    *_CANONICAL_TEXT_KEYS,
})


def _usable_text(value: Any, *, depth: int = 0) -> str | None:
    """Extract answer text from bounded, framework-neutral response shapes."""

    if depth > 3:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text and text not in {"{}", "[]", "null"} else None
    if isinstance(value, Mapping):
        block_type = str(value.get("type") or "").strip().lower()
        if block_type in _NON_ANSWER_BLOCK_TYPES:
            return None
        for key in (*_CANONICAL_TEXT_KEYS, *_TEXT_ALIAS_KEYS):
            if key in value:
                text = _usable_text(value.get(key), depth=depth + 1)
                if text:
                    return text
        return None
    if isinstance(value, (list, tuple)):
        parts = [
            text for item in value
            if (text := _usable_text(item, depth=depth + 1))
        ]
        return "\n\n".join(parts).strip() or None
    return None


def _has_usable_structured_content(value: Any, *, depth: int = 0) -> bool:
    if depth > 4 or value in (None, "", [], {}):
        return False
    if isinstance(value, Mapping):
        if str(value.get("type") or "").strip().lower() in _NON_ANSWER_BLOCK_TYPES:
            return False
        return any(
            _has_usable_structured_content(item, depth=depth + 1)
            for key, item in value.items() if key != "type"
        )
    if isinstance(value, (list, tuple)):
        return any(_has_usable_structured_content(item, depth=depth + 1) for item in value)
    return True


def _objects(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def normalize_runtime_task_result(
    value: Any,
    *,
    artifacts: tuple[RuntimeArtifact, ...] = (),
    usage: Mapping[str, Any] | None = None,
    framework_details: Mapping[str, Any] | None = None,
    structured_output_requested: bool = False,
    structured_validation_error: Exception | None = None,
) -> RuntimeTaskResult:
    """Normalize flexible model/runtime output while preserving useful text.

    A malformed optional structured projection is a warning, not a failure.
    Empty output remains a retryable failure because there is nothing to show
    or review.
    """

    data = dict(value) if isinstance(value, Mapping) else {}
    if isinstance(value, Mapping):
        # Raw strings remain supported as internal model projections. Every
        # runtime result envelope, however, must carry an explicit status.
        validate_runtime_result_envelope({"status": value.get("status"), "task_result": value})
    raw_status = str(data["status"]) if isinstance(value, Mapping) else "completed"
    canonical_text_value = next(
        (data.get(key) for key in _CANONICAL_TEXT_KEYS if data.get(key) is not None),
        value if isinstance(value, str) else None,
    )
    text = _usable_text(canonical_text_value)
    extracted_from_alias = False
    if text is None and data:
        for key in _TEXT_ALIAS_KEYS:
            if key not in data:
                continue
            text = _usable_text(data.get(key))
            if text:
                extracted_from_alias = True
                break
    structured = data.get("structured_output")
    if structured is None and isinstance(data.get("structured"), Mapping):
        structured = data["structured"]
    warnings = list(_objects(data.get("warnings")))
    if extracted_from_alias:
        warnings.append({
            "code": "task_result_envelope_noncanonical",
            "message": "Usable output was preserved from a noncanonical result field.",
        })
    if structured is None and text is None and data:
        extensions = {
            key: item for key, item in data.items()
            if key not in _RESULT_CONTROL_KEYS and _has_usable_structured_content(item)
        }
        if extensions:
            structured = extensions
            warnings.append({
                "code": "task_result_envelope_noncanonical",
                "message": "Usable structured output was preserved from noncanonical result fields.",
            })
    gaps = _strings(data.get("gaps", data.get("uncovered_gaps")))
    error = dict(data["error"]) if isinstance(data.get("error"), Mapping) else None

    if structured_validation_error is not None:
        warnings.append({
            "code": "structured_output_invalid",
            "message": "Structured output could not be normalized; usable text was preserved.",
            "details": {"error_type": type(structured_validation_error).__name__},
        })
        structured = None
    elif structured_output_requested and structured is None:
        warnings.append({
            "code": "structured_output_missing",
            "message": "The requested structured output was not returned; usable text was preserved.",
        })

    status = RuntimeTaskResultStatus(raw_status)

    usable = bool(text or structured or artifacts)
    if status in {RuntimeTaskResultStatus.FAILED, RuntimeTaskResultStatus.TIMED_OUT, RuntimeTaskResultStatus.CANCELLED} and usable:
        # Preserve the runtime outcome. A caller may still expose the provisional
        # output, but a failure is not silently promoted to success.
        pass
    elif not usable:
        status = RuntimeTaskResultStatus.FAILED
        error = error or {"code": "task_result_empty", "retryable": True}
    elif warnings or gaps:
        status = RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS

    return RuntimeTaskResult(
        status=status,
        text=text,
        structured_output=dict(structured) if isinstance(structured, Mapping) else None,
        artifacts=artifacts,
        warnings=tuple(warnings),
        gaps=gaps,
        usage=dict(usage or data.get("usage") or {}),
        error=error,
        framework_details=dict(framework_details or data.get("framework_details") or {}),
        correction_outcomes=tuple(
            RuntimeCourseCorrectionOutcome(
                correction_id=str(item.get("correction_id") or ""),
                operation_id=str(item.get("operation_id") or ""),
                state=str(item.get("state") or "unresolved"),
                runtime_plan_revision=int(item["runtime_plan_revision"]) if item.get("runtime_plan_revision") else None,
                linked_run_id=str(item["linked_run_id"]) if item.get("linked_run_id") else None,
                todo_ids=tuple(str(value) for value in item.get("todo_ids") or []),
                artifact_ids=tuple(str(value) for value in item.get("artifact_ids") or []),
                explanation=str(item["explanation"]) if item.get("explanation") is not None else None,
                unresolved_reason=str(item["unresolved_reason"]) if item.get("unresolved_reason") is not None else None,
            )
            for item in data.get("correction_outcomes") or [] if isinstance(item, Mapping)
        ),
    )


def runtime_task_result_summary(result: RuntimeTaskResult) -> dict[str, Any]:
    if result.structured_output is not None:
        output_shape = "structured"
    elif (result.text or "").strip():
        output_shape = "text"
    elif result.artifacts:
        output_shape = "artifacts"
    else:
        output_shape = "empty"
    return {
        "outcome": result.status.value,
        "output_shape": output_shape,
        "warning_count": len(result.warnings),
        "gap_count": len(result.gaps),
        "artifact_ids": [item.artifact_id for item in result.artifacts if item.artifact_id],
        "usage": dict(result.usage),
    }

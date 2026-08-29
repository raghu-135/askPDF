"""Normalization helpers for framework-neutral task results."""

from __future__ import annotations

from typing import Any, Mapping

from app.runtime.contracts import (
    RuntimeArtifact,
    RuntimeTaskResult,
    RuntimeTaskResultStatus,
)


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
    raw_status = str(data.get("status") or "completed")
    text_value = data.get("text", data.get("summary", data.get("answer", value if isinstance(value, str) else None)))
    text = str(text_value).strip() if text_value is not None else None
    if text in {"{}", "[]", "null"}:
        text = None
    structured = data.get("structured_output")
    if structured is None and isinstance(data.get("structured"), Mapping):
        structured = data["structured"]
    warnings = list(_objects(data.get("warnings")))
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

    try:
        status = RuntimeTaskResultStatus(raw_status)
    except ValueError:
        status = RuntimeTaskResultStatus.COMPLETED if (text or structured or artifacts) else RuntimeTaskResultStatus.FAILED
        warnings.append({"code": "runtime_status_unknown", "details": {"status": raw_status}})

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

"""Framework-neutral runtime errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class RuntimeError(Exception):
    code: str
    safe_message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.safe_message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def capability_unsupported(
        cls,
        *,
        operation_id: str,
        framework: str,
        builder_id: str,
        explanation: str,
    ) -> "RuntimeError":
        """Build the stable error returned by unsupported adapter defaults."""

        return cls(
            code="runtime_capability_unsupported",
            safe_message="The requested runtime operation is not supported",
            retryable=False,
            details={
                "operation_id": operation_id,
                "framework": framework,
                "builder_id": builder_id,
                "support_level": "unsupported",
                "explanation": explanation,
            },
        )

    @classmethod
    def capability_unavailable(
        cls,
        *,
        operation_id: str,
        framework: str,
        builder_id: str,
        support_level: str,
        disabled_reason: str,
    ) -> "RuntimeError":
        return cls(
            code="runtime_capability_unavailable",
            safe_message="The requested runtime operation is unavailable for this run",
            retryable=False,
            details={
                "operation_id": operation_id,
                "framework": framework,
                "builder_id": builder_id,
                "support_level": support_level,
                "disabled_reason": disabled_reason,
            },
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        code: str = "runtime_error",
        retryable: bool = False,
        safe_message: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "RuntimeError":
        return cls(
            code=code,
            safe_message=safe_message or str(exc) or exc.__class__.__name__,
            retryable=retryable,
            details=dict(details or {}),
        )

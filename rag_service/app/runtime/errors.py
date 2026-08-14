"""Framework-neutral runtime errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RuntimeError:
    code: str
    safe_message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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

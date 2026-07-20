"""Shared string enums for database-backed domain values."""

from enum import Enum


class ProcessStatus(str, Enum):
    """Status values for processing operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def is_completed(cls, status: str) -> bool:
        """Check if status is completed."""
        return status == cls.COMPLETED.value

    @classmethod
    def is_failed(cls, status: str) -> bool:
        """Check if status is failed."""
        return status == cls.FAILED.value

    @classmethod
    def is_running(cls, status: str) -> bool:
        """Check if status is running."""
        return status == cls.RUNNING.value


class MessageRole(str, Enum):
    """Role values for chat messages."""

    USER = "user"
    ASSISTANT = "assistant"


class FileSourceType(str, Enum):
    """Source types for files attached to threads."""

    PDF = "pdf"
    BROWSER = "browser"


class ChatTurnStatus(str, Enum):
    """Persisted chat turn lifecycle statuses."""

    COMPLETED = "completed"
    CLARIFICATION = "clarification"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowVisibility(str, Enum):
    """Visibility values for agent workflows."""

    BUILTIN = "builtin"
    INTERNAL = "internal"
    PUBLIC = "public"
    DELETED = "deleted"


class AgentRunStatus(str, Enum):
    """Persisted agent run lifecycle statuses."""

    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    CLARIFICATION = "clarification"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OperationResultStatus(str, Enum):
    """Simple internal operation result statuses."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class EmbeddingReadinessStatus(str, Enum):
    """Thread/file embedding readiness response statuses."""

    READY = "ready"
    NOT_READY = "not_ready"
    BLOCKED = "blocked"


class ThreadCloneMode(str, Enum):
    """Thread fork source modes."""

    FULL_THREAD = "full_thread"
    FROM_MESSAGE = "from_message"


class FileStatusSection(str, Enum):
    """Top-level file processing status sections."""

    PARSING = "parsing"
    INDEXING = "indexing"


class ReasoningFormat(str, Enum):
    """Stored reasoning trace formats."""

    NONE = "none"
    STRUCTURED = "structured"
    TAGGED_TEXT = "tagged_text"
    MARKDOWN = "markdown"
    RAW = "raw"

"""
models.py - Pydantic models and enums for the RAG service database.

This module contains all data models used throughout the application,
including enums for status types and Pydantic models for database entities.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field


class ProcessStatus(str, Enum):
    """Status values for processing operations (parsing, indexing, etc.)."""
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

    @classmethod
    def is_pending(cls, status: str) -> bool:
        """Check if status is pending."""
        return status == cls.PENDING.value

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is terminal (completed or failed)."""
        return status in (cls.COMPLETED.value, cls.FAILED.value)


class MessageRole(str, Enum):
    """Role values for chat messages."""
    USER = "user"
    ASSISTANT = "assistant"


class Thread(BaseModel):
    """Represents a chat thread."""
    id: str
    name: str
    embed_model: str
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class File(BaseModel):
    """Represents a file (PDF or web source)."""
    file_hash: str
    file_name: str
    file_path: Optional[str] = None
    source_type: str = "pdf"  # 'pdf' or 'web'


class ThreadFile(BaseModel):
    """Represents the association between a thread and a file."""
    thread_id: str
    file_hash: str


class ThreadFileAnnotation(BaseModel):
    """Represents annotation data for a thread-file pair."""
    thread_id: str
    file_hash: str
    annotations_json: str
    created_at: datetime
    updated_at: datetime


class Message(BaseModel):
    """Represents a chat message."""
    id: str
    thread_id: str
    role: MessageRole
    content: str
    context_compact: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_available: bool = False
    reasoning_format: str = "none"
    web_sources: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

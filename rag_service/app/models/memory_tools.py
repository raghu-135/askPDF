"""Framework-neutral contracts for app-owned durable memory tools."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


MEMORY_READ_EFFECTIVE = "memory:read_effective"
MEMORY_READ_STORED = "memory:read_stored"
MEMORY_PROPOSE = "memory:propose"
MEMORY_APPLY_CONFIRMED = "memory:apply_confirmed"


class MemoryToolScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_type: Literal["user", "project", "thread"]
    scope_id: str = Field(min_length=1)


class MemoryToolContext(BaseModel):
    """Trusted server-created memory workspace and its granted capabilities."""

    model_config = ConfigDict(extra="forbid")

    selected_scope: MemoryToolScope
    visible_scopes: List[MemoryToolScope] = Field(default_factory=list, max_length=3)
    capabilities: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None
    project_id: Optional[str] = None


class MemorySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="", max_length=12000)
    view: Literal["effective", "stored"] = "effective"
    scope_types: Optional[List[Literal["user", "project", "thread"]]] = None
    max_results: int = Field(default=10, ge=1, le=500)
    selected_memory_id: Optional[str] = None


class MemoryGetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_ids: List[str] = Field(min_length=1, max_length=40)


class MemoryChangeIntent(BaseModel):
    """Untrusted semantic change requested by a curator model."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["create", "update", "delete", "move", "set_overrides", "noop"]
    memory_id: Optional[str] = None
    scope_type: Optional[Literal["user", "project", "thread"]] = None
    target_scope_type: Optional[Literal["user", "project", "thread"]] = None
    content: Optional[str] = Field(default=None, max_length=12000)
    override_target_ids: Optional[List[str]] = Field(default=None, max_length=20)


class MemoryPrepareChangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: List[MemoryChangeIntent] = Field(default_factory=list, max_length=20)


class MemoryOperationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_group_id: str
    action: Literal["create", "update", "delete", "move", "set_overrides"]
    label: str
    content: Optional[str] = None
    source_memory_id: Optional[str] = None
    source_scope: Optional[MemoryToolScope] = None
    destination_memory_id: Optional[str] = None
    destination_scope: Optional[MemoryToolScope] = None
    override_target_ids: List[str] = Field(default_factory=list)
    removed_incoming_override_count: int = 0


class MemoryChangeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_group_id: str
    action: Literal["create", "update", "delete", "move", "set_overrides"]
    source_memory_id: Optional[str] = None
    result_memory_id: Optional[str] = None
    source_scope: Optional[MemoryToolScope] = None
    destination_scope: Optional[MemoryToolScope] = None
    deleted_memory_ids: List[str] = Field(default_factory=list)
    override_target_ids: List[str] = Field(default_factory=list)
    removed_incoming_override_count: int = 0
    index_status: Optional[str] = None
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

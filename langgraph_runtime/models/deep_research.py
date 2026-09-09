from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEEP_RESEARCH_WORKFLOW_ID = "deep_research_agent"


class AgentTaskStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    CANCELLING = "cancelling"
    RECOVERY_REQUIRED = "recovery_required"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class AgentTaskTodoStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class AgentTaskInterventionKind(str, Enum):
    BUDGET_REVIEW = "budget_review"
    COURSE_CORRECTION = "course_correction"


class AgentTaskBudgetReviewDecision(str, Enum):
    CONTINUE = "continue"
    ACCEPT_PARTIAL = "accept_partial"
    STEER = "steer"


class AgentTaskBudgetBoundaryStrategy(str, Enum):
    SAFE_ATOMIC_BOUNDARY = "safe_atomic_boundary"


class AgentTaskContinuationSemantics(str, Enum):
    CHECKPOINT_SAME_RUN = "checkpoint_same_run"
    LINKED_RUN = "linked_run"


class AgentTaskBudgetCounters(BaseModel):
    model_calls: int = Field(default=0, ge=0)
    model_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    elapsed_active_ms: int = Field(default=0, ge=0)


class AgentTaskBudgetUsage(BaseModel):
    tranche_index: int = Field(ge=1)
    tranche_limits: AgentTaskBudgetCounters
    tranche_usage: AgentTaskBudgetCounters
    lifetime_usage: AgentTaskBudgetCounters
    exhausted_dimensions: List[Literal["model_calls", "model_tokens", "tool_calls", "elapsed_active_ms"]] = Field(default_factory=list)
    boundary_strategy: AgentTaskBudgetBoundaryStrategy = AgentTaskBudgetBoundaryStrategy.SAFE_ATOMIC_BOUNDARY


class SubagentProfileId(str, Enum):
    DOCUMENT = "document_researcher"
    WEB = "web_researcher"
    MEMORY = "memory_researcher"
    CRITIC = "evidence_critic"


class DeepResearchLimits(BaseModel):
    max_todos: int = Field(default=50, ge=1, le=50)
    max_plan_revisions: int = Field(default=8, ge=1, le=8)
    max_replans: int = Field(default=5, ge=0, le=5)
    max_concurrency: int = Field(default=4, ge=1, le=4)
    max_fanout: int = Field(default=4, ge=1, le=4)
    max_attempts_per_todo: int = Field(default=2, ge=1, le=2)
    subagent_timeout_ms: int = Field(default=180_000, ge=1_000, le=180_000)
    max_active_runtime_ms: int = Field(default=3_600_000, ge=60_000, le=86_400_000)
    max_model_calls: int = Field(default=10_000, ge=1, le=100_000)
    max_tool_calls: int = Field(default=100, ge=1, le=100)
    max_model_tokens: int = Field(default=500_000, ge=1_000, le=500_000)
    max_artifacts: int = Field(default=200, ge=1, le=200)
    max_artifact_bytes: int = Field(default=104_857_600, ge=1_024, le=104_857_600)
    max_single_artifact_bytes: int = Field(default=10_485_760, ge=1_024, le=10_485_760)


class AgentTaskCreateRequest(BaseModel):
    definition_id: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=20_000)
    llm_model: str = Field(min_length=1, max_length=300)
    context_window: int = Field(default=32_768, ge=2_048, le=2_000_000)
    web_search_mode: Literal["off", "ask", "on"] = "off"
    enabled_profiles: List[SubagentProfileId] = Field(default_factory=lambda: [
        SubagentProfileId.DOCUMENT,
        SubagentProfileId.MEMORY,
        SubagentProfileId.CRITIC,
    ])
    limits: Optional[DeepResearchLimits] = None

    @model_validator(mode="after")
    def validate_profiles(self):
        unique = list(dict.fromkeys(self.enabled_profiles))
        if self.web_search_mode != "off" and SubagentProfileId.WEB not in unique:
            unique.append(SubagentProfileId.WEB)
        if self.web_search_mode == "off" and SubagentProfileId.WEB in unique:
            raise ValueError("web_researcher requires web_search_mode ask or on")
        if not unique:
            raise ValueError("at least one subagent profile must be enabled")
        if not any(profile != SubagentProfileId.CRITIC for profile in unique):
            raise ValueError("at least one research profile must be enabled")
        self.enabled_profiles = unique
        return self


class AgentTaskCommandRequest(BaseModel):
    expected_version: int = Field(ge=1)


class AgentTaskResultReviewRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)
    interrupt_id: str = Field(min_length=1, max_length=300)
    expected_version: int = Field(ge=1)
    decision: Literal["accept", "retry_with_input"]
    followup_input: Optional[str] = Field(default=None, max_length=20_000)

    @model_validator(mode="after")
    def validate_followup(self):
        if self.decision == "retry_with_input" and not str(self.followup_input or "").strip():
            raise ValueError("retry_with_input requires followup_input")
        if self.followup_input is not None:
            self.followup_input = " ".join(self.followup_input.split()).strip()
        return self


class AgentTaskBudgetReviewRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)
    interrupt_id: str = Field(min_length=1, max_length=300)
    expected_version: int = Field(ge=1)
    decision: AgentTaskBudgetReviewDecision
    guidance: Optional[str] = Field(default=None, max_length=20_000)

    @model_validator(mode="after")
    def validate_guidance(self):
        if self.decision is AgentTaskBudgetReviewDecision.STEER and not str(self.guidance or "").strip():
            raise ValueError("steer requires guidance")
        self.guidance = " ".join(str(self.guidance or "").split()).strip() or None
        return self


class AgentTaskCourseCorrectionRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)
    expected_version: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=20_000)
    scope: Literal["remaining_work"] = "remaining_work"

    @field_validator("instruction")
    @classmethod
    def normalize_instruction(cls, value: str) -> str:
        return " ".join(value.split()).strip()


class DeepResearchTodoProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=4_000)
    completion_criteria: str = Field(min_length=1, max_length=2_000)
    dependency_ids: List[str] = Field(default_factory=list, max_length=50)
    priority: int = Field(default=50, ge=0, le=100)
    required: bool = True
    profile_id: SubagentProfileId
    evidence_expectations: List[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def normalize_id(self):
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.id.strip()).strip("-")
        if not normalized:
            raise ValueError("todo id must contain a letter or number")
        self.id = normalized
        self.dependency_ids = list(dict.fromkeys(self.dependency_ids))
        if self.id in self.dependency_ids:
            raise ValueError("todo cannot depend on itself")
        return self


class DeepResearchPlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1, max_length=20_000)
    success_criteria: List[str] = Field(min_length=1, max_length=20)
    assumptions: List[str] = Field(default_factory=list, max_length=20)
    constraints: List[str] = Field(default_factory=list, max_length=20)
    todos: List[DeepResearchTodoProposal] = Field(min_length=1, max_length=50)
    incorporated_correction_ids: List[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_dag(self):
        by_id = {todo.id: todo for todo in self.todos}
        if len(by_id) != len(self.todos):
            raise ValueError("todo ids must be unique")
        for todo in self.todos:
            missing = sorted(set(todo.dependency_ids) - set(by_id))
            if missing:
                raise ValueError(f"todo {todo.id} has unknown dependencies: {', '.join(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(todo_id: str) -> None:
            if todo_id in visiting:
                raise ValueError("todo dependencies must be acyclic")
            if todo_id in visited:
                return
            visiting.add(todo_id)
            for dependency_id in by_id[todo_id].dependency_ids:
                visit(dependency_id)
            visiting.remove(todo_id)
            visited.add(todo_id)

        for todo_id in by_id:
            visit(todo_id)
        return self

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DeepResearchSubagentResult(BaseModel):
    status: Literal["completed", "failed", "timed_out", "cancelled"]
    summary: str = Field(default="", max_length=12_000)
    claims: List[Dict[str, Any]] = Field(default_factory=list, max_length=100)
    source_refs: List[Dict[str, Any]] = Field(default_factory=list, max_length=100)
    uncovered_gaps: List[str] = Field(default_factory=list, max_length=20)
    retryable: bool = False
    usage: Dict[str, int] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    @field_validator("uncovered_gaps", mode="before")
    @classmethod
    def normalize_uncovered_gaps(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def normalize_usage(cls, value: Any) -> Dict[str, int]:
        if not isinstance(value, dict):
            return {}
        allowed = {
            "model_calls", "tool_calls", "total_tokens",
            "prompt_tokens", "completion_tokens",
        }
        normalized: Dict[str, int] = {}
        for key, raw in value.items():
            if key not in allowed or isinstance(raw, bool):
                continue
            try:
                normalized[key] = max(0, int(raw))
            except (TypeError, ValueError):
                continue
        return normalized

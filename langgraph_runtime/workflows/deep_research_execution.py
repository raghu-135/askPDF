"""Typed execution ports shared by product and external Deep Research runs."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, TypeVar

from langgraph_runtime.runtime_support.cancellation import race_with_cancellation


T = TypeVar("T")
Compactor = Callable[[str], Awaitable[tuple[str, Mapping[str, Any]]]]


@dataclass(frozen=True)
class PlanRevisionRecord:
    revision: int


@dataclass
class TodoRecord:
    id: str
    title: str = ""
    description: str = ""
    completion_criteria: str = ""
    status: str = "pending"
    priority: int = 0
    required: bool = True
    profile_id: str = ""
    attempt: int = 1
    max_attempts: int = 2
    progress: int = 0
    result_summary: str | None = None
    version: int = 1
    dependency_ids_json: list[str] = field(default_factory=list)
    artifact_ids_json: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], **overrides: Any) -> "TodoRecord":
        data = {**dict(value), **overrides}
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            completion_criteria=str(data.get("completion_criteria") or ""),
            status=str(data.get("status") or "pending"),
            priority=int(data.get("priority") or 0),
            required=bool(data.get("required", True)),
            profile_id=str(data.get("profile_id") or ""),
            attempt=int(data.get("attempt") or 1),
            max_attempts=int(data.get("max_attempts") or 2),
            progress=int(data.get("progress") or 0),
            result_summary=data.get("result_summary"),
            version=int(data.get("version") or 1),
            dependency_ids_json=list(data.get("dependency_ids") or data.get("dependency_ids_json") or []),
            artifact_ids_json=list(data.get("artifact_ids") or data.get("artifact_ids_json") or []),
        )


@dataclass
class SubagentRecord:
    id: str
    status: str = "running"
    output_artifact_ids_json: list[str] = field(default_factory=list)
    usage_json: dict[str, Any] = field(default_factory=dict)


class CancellationToken(Protocol):
    async def requested(self) -> bool: ...


class EventPort(Protocol):
    async def emit(self, kind: str, payload: Mapping[str, Any]) -> None: ...


class TodoStore(Protocol):
    async def list(self, task_id: str) -> list[Any]: ...


class ArtifactStore(Protocol):
    async def list(self, task_id: str) -> list[Any]: ...


class BudgetController(Protocol):
    async def consume(self, task_id: str, **usage: int) -> Mapping[str, Any]: ...


class MemoryReader(Protocol):
    async def resolve(self, *, thread_id: str, limit: int) -> Mapping[str, Any]: ...


@dataclass
class DeepResearchExecutionServices:
    todos: TodoStore | None
    artifacts: ArtifactStore | None
    budgets: BudgetController | None
    cancellation: CancellationToken
    events: EventPort | None
    memory: MemoryReader | None
    state: Mapping[str, Any] = field(default_factory=dict)
    pause_checker: Any = None

    async def consume_budget(self, task_id: str, **usage: int) -> Mapping[str, Any]:
        if self.budgets is None:
            return {}
        return await self.budgets.consume(task_id, **usage)

    async def resolve_memory(self, *, thread_id: str, limit: int) -> Mapping[str, Any]:
        if self.memory is None:
            return {}
        return await self.memory.resolve(thread_id=thread_id, limit=limit)

    async def persist_plan(self, task_id: str, proposal: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        raise NotImplementedError

    async def schedule_ready(self, task_id: str, *, limit: int) -> list[TodoRecord]:
        raise NotImplementedError

    async def list_todos(self, task_id: str) -> list[TodoRecord]:
        raise NotImplementedError

    async def block_todos(self, task_id: str, todo_ids: list[str], *, reason: str) -> None:
        raise NotImplementedError

    async def start_subagent(self, **kwargs: Any) -> tuple[SubagentRecord, bool]:
        raise NotImplementedError

    async def persist_artifact(self, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError

    async def record_result_packets(self, packets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def persist_web_access(self, status: str, *, run_id: str, interrupt_id: str) -> None:
        raise NotImplementedError

    async def pause_requested(self) -> bool:
        if bool(self.state.get("task_pause_requested")):
            return True
        if self.pause_checker is None:
            return False
        value = self.pause_checker()
        if hasattr(value, "__await__"):
            value = await value
        return bool(value)

    async def budget_boundary(self) -> Mapping[str, Any] | None:
        boundary = self.state.get("task_budget_boundary")
        return dict(boundary) if isinstance(boundary, Mapping) else None

    async def pending_course_corrections(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self.state.get("task_course_corrections") or [] if isinstance(value, Mapping)]

    async def mark_course_corrections_applied(self, correction_ids: list[str], *, plan_revision: int) -> None:
        return None

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        raise NotImplementedError

    async def report_contents(self, context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        raise NotImplementedError


class RuntimeExecutionServices(DeepResearchExecutionServices):
    async def persist_plan(self, task_id: str, proposal: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        revision = PlanRevisionRecord(revision=int(self.state.get("task_plan_revision") or 0) + 1)
        limits = self.state.get("task_limits") if isinstance(self.state.get("task_limits"), Mapping) else {}
        todos = [TodoRecord.from_mapping(
            todo.model_dump(mode="json"), status="pending", attempt=1,
            max_attempts=int(limits.get("max_attempts", 2)), progress=0,
            result_summary=None, artifact_ids_json=[], version=1,
            dependency_ids_json=list(todo.dependency_ids),
        ) for todo in proposal.todos]
        return revision, todos

    async def schedule_ready(self, task_id: str, *, limit: int) -> list[TodoRecord]:
        return [
            TodoRecord.from_mapping(todo)
            for todo in self.state.get("task_todos") or []
            if isinstance(todo, Mapping) and todo.get("status") in {"pending", "ready"}
        ][:limit]

    async def list_todos(self, task_id: str) -> list[TodoRecord]:
        return [TodoRecord.from_mapping(todo) for todo in self.state.get("task_todos") or [] if isinstance(todo, Mapping)]

    async def block_todos(self, task_id: str, todo_ids: list[str], *, reason: str) -> None:
        blocked = set(todo_ids)
        for todo in self.state.get("task_todos") or []:
            if isinstance(todo, dict) and str(todo.get("id")) in blocked:
                todo.update({"status": "blocked", "result_summary": reason})

    async def start_subagent(self, **kwargs: Any) -> tuple[SubagentRecord, bool]:
        return SubagentRecord(id=f"runtime:{uuid.uuid4()}"), False

    async def persist_artifact(self, **kwargs: Any) -> Mapping[str, Any]:
        content = str(kwargs.get("content") or "")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        artifact_id = f"runtime:{digest[:24]}"
        artifact = {
            "artifact_id": artifact_id, "id": artifact_id,
            "kind": kwargs.get("kind"), "content": content, "sha256": digest,
            "byte_size": len(content.encode("utf-8")),
            "media_type": kwargs.get("media_type", "text/plain"),
            "todo_id": kwargs.get("todo_id"), "subagent_run_id": kwargs.get("subagent_run_id"),
            "provenance": dict(kwargs.get("provenance") or {}),
            "source_refs": dict(kwargs.get("source_refs") or {}),
        }
        self.state.setdefault("runtime_artifacts", []).append(artifact)  # type: ignore[attr-defined]
        return artifact

    async def record_result_packets(self, packets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        todos = [dict(todo) for todo in self.state.get("task_todos") or [] if isinstance(todo, Mapping)]
        by_id = {str(todo.get("id")): todo for todo in todos}
        for packet in packets:
            todo = by_id.get(str(packet.get("todo_id") or ""))
            if todo is None:
                continue
            todo["status"] = "completed" if packet.get("status") == "completed" else packet.get("status") or "failed"
            todo["result_summary"] = str(packet.get("summary") or "")[:4000]
            todo["artifact_ids"] = list(dict.fromkeys([*(todo.get("artifact_ids") or []), *(packet.get("artifact_ids") or [])]))
            todo["progress"] = 100 if todo["status"] == "completed" else todo.get("progress", 0)
        return todos

    async def persist_web_access(self, status: str, *, run_id: str, interrupt_id: str) -> None:
        return None

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        current_run_id = str(self.state.get("agent_run_id") or "")
        manifests = [dict(value) for value in self.state.get("runtime_artifact_manifest") or self.state.get("task_artifact_manifest") or [] if isinstance(value, Mapping)]
        artifacts = [dict(value) for value in self.state.get("runtime_artifacts") or [] if isinstance(value, Mapping)]
        by_id = {str(value.get("artifact_id") or value.get("id")): value for value in [*artifacts, *manifests]}
        completed_ids = {
            str(artifact_id)
            for todo in self.state.get("task_todos") or []
            if isinstance(todo, Mapping) and todo.get("status") == "completed"
            for artifact_id in todo.get("artifact_ids") or []
        }
        selected = [by_id[value] for value in completed_ids if value in by_id]
        evidence_manifest = [{
            "id": str(value.get("artifact_id") or value.get("id")),
            "kind": value.get("kind"), "sha256": value.get("sha256"),
            "byte_size": value.get("byte_size") or len(str(value.get("content") or "").encode()),
            "summary": value.get("summary") or {}, "todo_id": value.get("todo_id"),
            "plan_revision": int((value.get("provenance") or {}).get("plan_revision") or 0),
            "origin_run_id": current_run_id, "origin_attempt": 1,
            "inherited": False, "validity": "valid",
        } for value in selected if value.get("kind") in {"tool_output", "intermediate_report", "context_summary"}]
        return {
            "runtime_artifacts": artifacts,
            "task_artifact_manifest": evidence_manifest,
            "task_evidence_manifest": [value for value in evidence_manifest if value.get("kind") in {"tool_output", "intermediate_report"}],
            "task_evidence_gaps": [f"{value}:missing" for value in sorted(completed_ids) if value not in by_id],
            "task_context_summary": {
                "source_hash": _stable_hash([(value.get("id"), value.get("sha256")) for value in evidence_manifest]),
                "estimated_chars": sum(int(value.get("byte_size") or 0) for value in evidence_manifest),
                "compaction_required": False, "compaction_forced": False,
                "summary_artifact_id": next((value.get("id") for value in evidence_manifest if value.get("kind") == "context_summary"), None),
                "policy_version": 1,
            },
        }

    async def report_contents(self, context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        artifacts = [value for value in self.state.get("runtime_artifacts") or [] if isinstance(value, Mapping)]
        contents = dict(self.state.get("runtime_artifact_contents") or {})
        contents.update({str(value.get("artifact_id") or value.get("id")): str(value.get("content") or "") for value in artifacts})
        reports = [contents.get(str(value.get("id")), "")[:20_000] for value in context.get("task_evidence_manifest") or []]
        return reports, [str(value) for value in context.get("task_evidence_gaps") or []]


def _todo_dict(todo: Any) -> dict[str, Any]:
    return {
        "id": todo.id, "title": todo.title, "description": todo.description,
        "completion_criteria": todo.completion_criteria, "status": todo.status,
        "priority": todo.priority, "required": todo.required,
        "dependency_ids": list(todo.dependency_ids_json or []), "profile_id": todo.profile_id,
        "attempt": todo.attempt, "max_attempts": todo.max_attempts, "progress": todo.progress,
        "result_summary": todo.result_summary, "artifact_ids": list(todo.artifact_ids_json or []),
        "version": todo.version,
    }


def _stable_hash(value: Any) -> str:
    import json
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CallbackCancellationToken:
    callback: Any

    async def requested(self) -> bool:
        value = self.callback()
        if hasattr(value, "__await__"):
            value = await value
        return bool(value)


def services_from_config(config: Mapping[str, Any] | None, state: Mapping[str, Any]) -> DeepResearchExecutionServices:
    configurable = dict((config or {}).get("configurable") or {})
    factory = configurable.get("deep_research_services_factory")
    if factory is None:
        raise RuntimeError("Deep Research execution services were not configured")
    services = factory(state, configurable)
    if not isinstance(services, DeepResearchExecutionServices):
        raise TypeError("deep_research_services_factory returned an invalid service bundle")
    return services


def _common_services(state: Mapping[str, Any], configurable: Mapping[str, Any]) -> dict[str, Any]:
    checker = configurable.get("cancellation_checker")
    if checker is None:
        raise RuntimeError("Deep Research cancellation checker was not configured")
    token: CancellationToken = CallbackCancellationToken(checker)
    return {
        "todos": None,
        "artifacts": None,
        "cancellation": token,
        "events": configurable.get("execution_event_sink"),
        "state": state,
        "pause_checker": configurable.get("pause_checker"),
    }


def runtime_execution_services_factory(
    state: Mapping[str, Any], configurable: Mapping[str, Any]
) -> DeepResearchExecutionServices:
    return RuntimeExecutionServices(
        budgets=None, memory=None,
        **_common_services(state, configurable),
    )


async def run_cancellable(
    awaitable: Awaitable[T],
    token: CancellationToken,
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float | None = None,
) -> T:
    """Race work against the authoritative cancellation token and clean up both tasks."""

    return await race_with_cancellation(
        awaitable,
        token.requested,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )

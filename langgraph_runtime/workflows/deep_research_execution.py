"""Typed execution ports shared by product and external Deep Research runs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Protocol, TypeVar

from langgraph_runtime.runtime_support.cancellation import race_with_cancellation
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


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
    execution_key: str | None = None

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
            execution_key=str(data["execution_key"]) if data.get("execution_key") else None,
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
    course_correction_reader: Any = None
    course_correction_acknowledger: Any = None
    runtime_budget_meter: "RuntimeBudgetMeter | None" = None

    async def consume_budget(self, task_id: str, **usage: int) -> Mapping[str, Any]:
        if self.budgets is None:
            return {}
        return await self.budgets.consume(task_id, **usage)

    @asynccontextmanager
    async def execution_span(self, *, enabled: bool = True):
        yield

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

    async def budget_snapshot(self) -> Mapping[str, Any]:
        if self.runtime_budget_meter is not None:
            return await self.runtime_budget_meter.snapshot()
        return dict(self.state.get("task_budget_usage") or {})

    async def pending_course_corrections(self) -> list[dict[str, Any]]:
        current = [
            {**dict(value), "id": value.get("id") or value.get("correction_id")}
            for value in self.state.get("task_course_corrections") or []
            if isinstance(value, Mapping) and value.get("status", "accepted") in {"pending", "accepted"}
        ]
        if self.course_correction_reader is not None:
            value = self.course_correction_reader()
            if hasattr(value, "__await__"):
                value = await value
            current.extend(
                {**dict(item), "id": item.get("id") or item.get("correction_id")}
                for item in value or [] if isinstance(item, Mapping)
            )
        return list({str(item.get("correction_id") or item.get("id")): item for item in current}.values())

    async def mark_course_corrections_applied(self, correction_ids: list[str], *, plan_revision: int) -> None:
        if self.course_correction_acknowledger is None:
            return None
        value = self.course_correction_acknowledger(
            correction_ids, plan_revision=plan_revision
        )
        if hasattr(value, "__await__"):
            await value

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        raise NotImplementedError

    async def report_contents(self, context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        raise NotImplementedError


class RuntimeBudgetMeter:
    """Concurrency-safe counters shared by all nodes in one graph invocation.

    Parallel LangGraph nodes receive independent state snapshots. The meter is
    carried in RunnableConfig for the duration of an invocation, while safe
    boundary nodes copy its JSON snapshot back into checkpointed graph state.
    """

    def __init__(self, budget: Mapping[str, Any] | None, limits: Mapping[str, Any] | None):
        source = copy.deepcopy(dict(budget or {}))
        tranche_limits = dict(source.get("tranche_limits") or {})
        required = {"model_calls": "max_model_calls", "model_tokens": "max_model_tokens", "tool_calls": "max_tool_calls", "elapsed_active_ms": "max_active_runtime_ms"}
        if set(tranche_limits) & set(required) != set(required):
            raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget snapshot is missing required tranche limits", details={"missing_dimensions": sorted(set(required) - set(tranche_limits))})
        for dimension in required:
            value = tranche_limits.get(dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value <= 0:
                raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget snapshot contains an invalid tranche limit", details={"dimension": dimension})
        ordinary_limits = dict(limits or {})
        try:
            contradictory = [dimension for dimension, ordinary_name in required.items() if ordinary_name in ordinary_limits and ordinary_limits[ordinary_name] is not None and int(ordinary_limits[ordinary_name]) != int(tranche_limits[dimension])]
        except (TypeError, ValueError):
            raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget configuration contains an invalid ordinary limit", details={"field": "task_limits"}) from None
        if contradictory:
            raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget configuration contains contradictory limits", details={"contradictory_dimensions": contradictory})
        self._limits = tranche_limits
        for field_name in ("tranche_usage", "lifetime_usage"):
            usage = source.get(field_name)
            if not isinstance(usage, Mapping):
                raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget snapshot is missing usage counters", details={"field": field_name})
            for dimension in required:
                value = usage.get(dimension, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 0:
                    raise AgentRuntimeError("budget_snapshot_invalid", "The runtime budget snapshot contains an invalid usage counter", details={"field": field_name, "dimension": dimension})
        self._budget = {
            **source,
            "tranche_index": max(1, int(source.get("tranche_index") or 1)),
            "tranche_limits": tranche_limits,
            "tranche_usage": dict(source.get("tranche_usage") or {}),
            "lifetime_usage": dict(source.get("lifetime_usage") or {}),
        }
        self._lock = asyncio.Lock()
        self._clock_lock = asyncio.Lock()
        self._active_operations = 0
        self._active_started_at: float | None = None

    @asynccontextmanager
    async def execution_span(self):
        """Measure the union of active runtime intervals, once per invocation.

        Parallel LangGraph workers share this meter. A reference count keeps
        overlapping operations from charging their durations twice, while the
        checkpointed budget remains the only persisted clock state.
        """
        async with self._clock_lock:
            if self._active_operations == 0:
                self._active_started_at = time.perf_counter()
            self._active_operations += 1
        try:
            yield
        finally:
            elapsed_ms = 0
            async with self._clock_lock:
                self._active_operations -= 1
                if self._active_operations == 0 and self._active_started_at is not None:
                    elapsed_ms = max(0, int((time.perf_counter() - self._active_started_at) * 1000))
                    self._active_started_at = None
            if elapsed_ms:
                await self.consume(elapsed_active_ms=elapsed_ms)

    async def consume(self, **usage: int) -> Mapping[str, Any]:
        async with self._lock:
            tranche_usage = dict(self._budget.get("tranche_usage") or {})
            lifetime_usage = dict(self._budget.get("lifetime_usage") or {})
            limit_names = {
                "model_calls": "max_model_calls",
                "model_tokens": "max_model_tokens",
                "tool_calls": "max_tool_calls",
                "elapsed_active_ms": "max_active_runtime_ms",
            }
            for key, amount in usage.items():
                increment = max(0, int(amount or 0))
                if key in limit_names:
                    tranche_usage[key] = int(tranche_usage.get(key) or 0) + increment
                lifetime_usage[key] = int(lifetime_usage.get(key) or 0) + increment
            self._budget.update({
                "tranche_usage": tranche_usage,
                "lifetime_usage": lifetime_usage,
            })
            exhausted = [
                key for key, limit_name in limit_names.items()
                if int(tranche_usage.get(key) or 0)
                >= max(
                    1,
                    int(
                        self._limits[key]
                    ),
                )
            ]
            boundary = self._budget.get("boundary")
            if exhausted and not (isinstance(boundary, Mapping) and boundary):
                self._budget["boundary"] = {
                    "status": "requested",
                    "dimensions": exhausted,
                    "tranche_index": self._budget["tranche_index"],
                }
            return copy.deepcopy(self._budget)

    async def snapshot(self) -> Mapping[str, Any]:
        async with self._lock:
            return copy.deepcopy(self._budget)

    async def boundary(self) -> Mapping[str, Any] | None:
        snapshot = await self.snapshot()
        boundary = snapshot.get("boundary")
        return dict(boundary) if isinstance(boundary, Mapping) and boundary else None


class RuntimeExecutionServices(DeepResearchExecutionServices):
    _SUCCESS_STATUSES = {"completed", "skipped"}
    _FAILED_STATUSES = {"failed", "blocked", "cancelled", "timed_out"}

    def _task_todos_state(self) -> list[dict[str, Any]]:
        if not isinstance(self.state, MutableMapping):
            raise AgentRuntimeError(
                "runtime_task_state_not_mutable",
                "The runtime scheduler requires mutable checkpoint state.",
            )
        values = self.state.get("task_todos") or []
        if not isinstance(values, list):
            raise AgentRuntimeError(
                "runtime_task_state_invalid",
                "Checkpoint task_todos must be a list.",
            )
        todos = [dict(value) for value in values if isinstance(value, Mapping)]
        if len(todos) != len(values):
            raise AgentRuntimeError(
                "runtime_task_state_invalid",
                "Every checkpoint task todo must be an object.",
            )
        ids = [str(todo.get("id") or "") for todo in todos]
        if any(not todo_id for todo_id in ids) or len(ids) != len(set(ids)):
            raise AgentRuntimeError(
                "runtime_task_state_invalid",
                "Checkpoint task todos must have unique non-empty IDs.",
            )
        return todos

    def _write_task_todos(self, todos: list[dict[str, Any]]) -> None:
        if not isinstance(self.state, MutableMapping):
            raise AgentRuntimeError(
                "runtime_task_state_not_mutable",
                "The runtime scheduler requires mutable checkpoint state.",
            )
        self.state["task_todos"] = todos

    @classmethod
    def _validate_dependencies(cls, todos: list[dict[str, Any]]) -> dict[str, list[str]]:
        by_id = {str(todo["id"]): todo for todo in todos}
        dependencies: dict[str, list[str]] = {}
        for todo in todos:
            todo_id = str(todo["id"])
            raw = todo.get("dependency_ids") or todo.get("dependency_ids_json") or []
            if not isinstance(raw, list):
                raise AgentRuntimeError(
                    "runtime_dependency_state_invalid",
                    "Todo dependency_ids must be a list.",
                    details={"todo_id": todo_id},
                )
            dependency_ids = [str(value) for value in raw]
            unknown = sorted(set(dependency_ids) - set(by_id))
            if unknown:
                raise AgentRuntimeError(
                    "runtime_dependency_missing",
                    "A todo references an unknown dependency.",
                    details={"todo_id": todo_id, "dependency_ids": unknown},
                )
            dependencies[todo_id] = dependency_ids

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(todo_id: str) -> None:
            if todo_id in visiting:
                raise AgentRuntimeError(
                    "runtime_dependency_cycle",
                    "Todo dependencies must form an acyclic graph.",
                    details={"todo_id": todo_id},
                )
            if todo_id in visited:
                return
            visiting.add(todo_id)
            for dependency_id in dependencies[todo_id]:
                visit(dependency_id)
            visiting.remove(todo_id)
            visited.add(todo_id)

        for todo_id in dependencies:
            visit(todo_id)
        return dependencies

    @asynccontextmanager
    async def execution_span(self, *, enabled: bool = True):
        if not enabled:
            yield
            return
        if self.runtime_budget_meter is None:
            self.runtime_budget_meter = RuntimeBudgetMeter(
                self.state.get("task_budget_usage") if isinstance(self.state, Mapping) else None,
                self.state.get("task_limits") if isinstance(self.state, Mapping) else None,
            )
        async with self.runtime_budget_meter.execution_span():
            yield

    async def consume_budget(self, task_id: str, **usage: int) -> Mapping[str, Any]:
        if self.runtime_budget_meter is None:
            self.runtime_budget_meter = RuntimeBudgetMeter(
                self.state.get("task_budget_usage") if isinstance(self.state, Mapping) else None,
                self.state.get("task_limits") if isinstance(self.state, Mapping) else None,
            )
        return await self.runtime_budget_meter.consume(**usage)

    async def budget_boundary(self) -> Mapping[str, Any] | None:
        if self.runtime_budget_meter is not None:
            boundary = await self.runtime_budget_meter.boundary()
            if boundary:
                return boundary
        return await super().budget_boundary()

    async def persist_plan(self, task_id: str, proposal: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        revision = PlanRevisionRecord(revision=int(self.state.get("task_plan_revision") or 0) + 1)
        limits = self.state.get("task_limits") if isinstance(self.state.get("task_limits"), Mapping) else {}
        prior_todos = [dict(value) for value in self.state.get("task_todos") or [] if isinstance(value, Mapping)]
        completed_by_id = {
            str(value.get("id")): value for value in prior_todos
            if value.get("status") == "completed" and value.get("id")
        }
        todos: list[TodoRecord] = []
        proposed_ids: set[str] = set()
        for todo in proposal.todos:
            todo_id = str(todo.id)
            proposed_ids.add(todo_id)
            if todo_id in completed_by_id:
                todos.append(TodoRecord.from_mapping(completed_by_id[todo_id]))
                continue
            todos.append(TodoRecord.from_mapping(
                todo.model_dump(mode="json"), status="pending", attempt=1,
                max_attempts=int(limits.get("max_attempts", 2)), progress=0,
                result_summary=None, artifact_ids_json=[], version=1,
                dependency_ids_json=list(todo.dependency_ids),
            ))
        todos.extend(
            TodoRecord.from_mapping(value)
            for todo_id, value in completed_by_id.items()
            if todo_id not in proposed_ids
        )
        return revision, todos

    async def schedule_ready(self, task_id: str, *, limit: int) -> list[TodoRecord]:
        todos = self._task_todos_state()
        dependencies = self._validate_dependencies(todos)
        by_id = {str(todo["id"]): todo for todo in todos}

        # Resolve blocked dependents to a fixed point before selecting work.
        # This makes failed prerequisite propagation deterministic even when a
        # chain contains more than one not-yet-evaluated dependent.
        changed = True
        while changed:
            changed = False
            for todo in todos:
                todo_id = str(todo["id"])
                if todo.get("status") not in {"pending", "ready"}:
                    continue
                failed = [
                    dependency_id for dependency_id in dependencies[todo_id]
                    if by_id[dependency_id].get("status") in self._FAILED_STATUSES
                ]
                if failed:
                    todo.update({
                        "status": "blocked",
                        "result_summary": f"dependency_failed:{','.join(sorted(failed))}",
                    })
                    changed = True

        eligible = [
            todo for todo in todos
            if todo.get("status") in {"pending", "ready"}
            and all(by_id[dependency_id].get("status") in self._SUCCESS_STATUSES
                    for dependency_id in dependencies[str(todo["id"])] )
        ]
        eligible.sort(key=lambda todo: (-int(todo.get("priority") or 0), str(todo["id"])))

        ready: list[TodoRecord] = []
        for value in eligible[:max(0, int(limit))]:
            previous_status = str(value.get("status") or "pending")
            attempt = int(value.get("attempt") or 0)
            max_attempts = max(1, int(value.get("max_attempts") or 2))
            if previous_status == "ready":
                if attempt >= max_attempts:
                    value.update({
                        "status": "failed",
                        "result_summary": "max_attempts_exhausted",
                    })
                    continue
                attempt += 1
            else:
                attempt = max(1, attempt)
            value.update({"status": "running", "attempt": attempt})
            ready.append(TodoRecord.from_mapping(value))

        self._write_task_todos(todos)
        return ready

    async def list_todos(self, task_id: str) -> list[TodoRecord]:
        return [TodoRecord.from_mapping(todo) for todo in self._task_todos_state()]

    async def block_todos(self, task_id: str, todo_ids: list[str], *, reason: str) -> None:
        blocked = set(todo_ids)
        todos = self._task_todos_state()
        for todo in todos:
            if str(todo.get("id")) in blocked:
                todo.update({"status": "blocked", "result_summary": reason})
        self._write_task_todos(todos)

    async def start_subagent(self, **kwargs: Any) -> tuple[SubagentRecord, bool]:
        await self.consume_budget(str(kwargs.get("task_id") or ""), subagent_attempts=1)
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
        await self.consume_budget(str(kwargs.get("task_id") or ""), artifact_bytes=len(content.encode("utf-8")))
        return artifact

    async def record_result_packets(self, packets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        todos = self._task_todos_state()
        by_id = {str(todo.get("id")): todo for todo in todos}
        for packet in packets:
            todo_id = str(packet.get("todo_id") or "")
            todo = by_id.get(todo_id)
            if todo is None:
                raise AgentRuntimeError(
                    "runtime_result_unknown_todo",
                    "A result packet references an unknown todo.",
                    details={"todo_id": todo_id},
                )
            if todo.get("status") != "running":
                raise AgentRuntimeError(
                    "runtime_result_stale",
                    "A result packet does not target a running todo.",
                    details={"todo_id": todo_id, "status": todo.get("status")},
                )
            expected_attempt = int(todo.get("attempt") or 0)
            packet_attempt = int(packet.get("attempt") or 0)
            if packet_attempt != expected_attempt:
                raise AgentRuntimeError(
                    "runtime_result_attempt_mismatch",
                    "A result packet targets a stale or invalid attempt.",
                    details={"todo_id": todo_id, "expected_attempt": expected_attempt, "packet_attempt": packet_attempt},
                )
            expected_execution_key = todo.get("execution_key")
            if expected_execution_key and str(packet.get("execution_key") or "") != str(expected_execution_key):
                raise AgentRuntimeError(
                    "runtime_result_execution_mismatch",
                    "A result packet has an invalid execution identity.",
                    details={"todo_id": todo_id},
                )
            if packet.get("status") == "completed":
                todo["status"] = "completed"
            elif bool(packet.get("retryable")) and expected_attempt < int(todo.get("max_attempts") or 2):
                todo["status"] = "ready"
            else:
                todo["status"] = packet.get("status") or "failed"
                if bool(packet.get("retryable")):
                    todo["status"] = "failed"
                    todo["result_summary"] = "max_attempts_exhausted"
            if todo.get("result_summary") != "max_attempts_exhausted":
                todo["result_summary"] = str(packet.get("summary") or "")[:4000]
            todo["artifact_ids"] = list(dict.fromkeys([*(todo.get("artifact_ids") or []), *(packet.get("artifact_ids") or [])]))
            todo["progress"] = 100 if todo["status"] == "completed" else todo.get("progress", 0)
        self._write_task_todos(todos)
        return todos

    async def persist_web_access(self, status: str, *, run_id: str, interrupt_id: str) -> None:
        return None

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        current_run_id = str(self.state.get("agent_run_id") or "")
        manifests = [dict(value) for value in self.state.get("runtime_artifact_manifest") or self.state.get("task_artifact_manifest") or [] if isinstance(value, Mapping)]
        artifacts = [dict(value) for value in self.state.get("runtime_artifacts") or [] if isinstance(value, Mapping)]
        by_id = {str(value.get("artifact_id") or value.get("id")): value for value in [*manifests, *artifacts] if str(value.get("artifact_id") or value.get("id") or "").strip()}
        completed_ids = {
            str(artifact_id)
            for todo in self.state.get("task_todos") or []
            if isinstance(todo, Mapping) and todo.get("status") == "completed"
            for artifact_id in todo.get("artifact_ids") or []
        }
        selected = [by_id[value] for value in completed_ids if value in by_id]
        evidence_manifest = []
        for value in selected:
            artifact_id = str(value.get("artifact_id") or value.get("id") or "").strip()
            if not artifact_id or value.get("kind") not in {"tool_output", "intermediate_report", "context_summary"}:
                continue
            provenance = dict(value.get("provenance") or {}) if isinstance(value.get("provenance"), Mapping) else {}
            inherited = bool(value.get("inherited", provenance.get("inherited", False)))
            evidence_manifest.append({
                **dict(value), "id": artifact_id, "kind": value.get("kind"),
                "sha256": value.get("sha256"), "byte_size": value.get("byte_size"),
                "summary": value.get("summary") or {}, "todo_id": value.get("todo_id"),
                "plan_revision": int(provenance.get("plan_revision") or value.get("plan_revision") or 0),
                "origin_run_id": provenance.get("origin_run_id", value.get("origin_run_id", current_run_id if not inherited else None)),
                "origin_attempt": provenance.get("origin_attempt", value.get("origin_attempt", 1 if not inherited else None)),
                "inherited": inherited,
                "validity": value.get("validity", provenance.get("validity", "valid")),
                "provenance": provenance,
            })
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
        contents.update({str(value.get("artifact_id") or value.get("id")): value.get("content") for value in artifacts if value.get("content") is not None})
        reports: list[str] = []
        for manifest in context.get("task_evidence_manifest") or []:
            artifact_id = str(manifest.get("id") or "").strip()
            content = contents.get(artifact_id)
            if not artifact_id or not isinstance(content, str):
                raise AgentRuntimeError("evidence_content_missing", "Required evidence content is unavailable to the runtime", details={"artifact_id": artifact_id or None})
            encoded_size = len(content.encode("utf-8"))
            expected_digest = str(manifest.get("sha256") or "").strip()
            digest_match = bool(expected_digest) and hashlib.sha256(content.encode("utf-8")).hexdigest() == expected_digest
            try:
                size_match = manifest.get("byte_size") is not None and int(manifest["byte_size"]) == encoded_size
            except (TypeError, ValueError):
                size_match = False
            if not digest_match or not size_match:
                raise AgentRuntimeError("artifact_packet_invalid", "Evidence content failed integrity validation", details={"artifact_id": artifact_id, "digest_match": digest_match, "byte_size_match": size_match})
            reports.append(content[:20_000])
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
        "course_correction_reader": configurable.get("course_correction_reader"),
        "course_correction_acknowledger": configurable.get("course_correction_acknowledger"),
    }


def runtime_execution_services_factory(
    state: Mapping[str, Any], configurable: Mapping[str, Any]
) -> DeepResearchExecutionServices:
    meter = configurable.get("runtime_budget_meter")
    if not isinstance(meter, RuntimeBudgetMeter):
        budget = state.get("task_budget_usage") if isinstance(state, Mapping) else None
        meter = (
            RuntimeBudgetMeter(budget, state.get("task_limits") if isinstance(state, Mapping) else None)
            if budget is not None
            else None
        )
        if isinstance(configurable, dict):
            if meter is not None:
                configurable["runtime_budget_meter"] = meter
    return RuntimeExecutionServices(
        budgets=None, memory=None,
        runtime_budget_meter=meter,
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

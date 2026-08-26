"""Typed execution ports shared by product and external Deep Research runs."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, TypeVar

from app.runtime.cancellation import race_with_cancellation


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
        return bool(self.state.get("task_pause_requested"))

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        raise NotImplementedError

    async def report_contents(self, context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        raise NotImplementedError


class _ProductBudgets:
    async def consume(self, task_id: str, **usage: int) -> Mapping[str, Any]:
        from app.services.agent_task_repository import consume_budget
        return await consume_budget(task_id, **usage)


class _ProductMemory:
    async def resolve(self, *, thread_id: str, limit: int) -> Mapping[str, Any]:
        from app.services.effective_memory_service import resolve_effective_memory_context
        return await resolve_effective_memory_context(thread_id=thread_id, limit=limit)


class ProductExecutionServices(DeepResearchExecutionServices):
    async def persist_plan(self, task_id: str, proposal: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        from app.services.agent_task_repository import persist_plan
        return await persist_plan(task_id, proposal, **kwargs)

    async def schedule_ready(self, task_id: str, *, limit: int) -> list[TodoRecord]:
        from app.services.agent_task_repository import schedule_ready_todos
        return await schedule_ready_todos(task_id, limit=limit)

    async def list_todos(self, task_id: str) -> list[TodoRecord]:
        from app.services.agent_task_repository import list_todos
        return await list_todos(task_id)

    async def block_todos(self, task_id: str, todo_ids: list[str], *, reason: str) -> None:
        from app.services.agent_task_repository import block_todos
        await block_todos(task_id, todo_ids, reason=reason)

    async def start_subagent(self, **kwargs: Any) -> tuple[SubagentRecord, bool]:
        from app.services.agent_task_repository import record_subagent_started
        return await record_subagent_started(**kwargs)

    async def persist_artifact(self, **kwargs: Any) -> Mapping[str, Any]:
        from app.services.task_artifact_service import persist_task_artifact
        artifact = await persist_task_artifact(**kwargs)
        return {"id": artifact.id, "artifact_id": artifact.id}

    async def record_result_packets(self, packets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        from app.services.agent_task_repository import list_todos, record_subagent_result
        task_id = str(self.state.get("agent_task_id") or "")
        for packet in packets:
            await record_subagent_result(
                task_id=str(packet.get("task_id") or task_id),
                todo_id=str(packet.get("todo_id") or ""),
                subagent_run_id=str(packet.get("subagent_run_id") or ""),
                status=str(packet.get("status") or "failed"),
                summary=str(packet.get("summary") or ""),
                artifact_ids=[str(value) for value in packet.get("artifact_ids") or []],
                usage=dict(packet.get("usage") or {}),
                error=packet.get("error") if isinstance(packet.get("error"), dict) else None,
                retryable=bool(packet.get("retryable")),
            )
        return [_todo_dict(todo) for todo in await list_todos(task_id)]

    async def persist_web_access(self, status: str, *, run_id: str, interrupt_id: str) -> None:
        from app.services.agent_task_repository import set_task_web_access
        await set_task_web_access(
            str(self.state.get("agent_task_id") or ""), status,
            agent_run_id=run_id, interrupt_id=interrupt_id,
        )

    async def pause_requested(self) -> bool:
        from app.services.agent_task_repository import get_task
        task = await get_task(str(self.state.get("agent_task_id") or ""))
        return bool(self.state.get("task_pause_requested")) or bool(task and task.status == "pausing")

    async def assemble_artifact_context(self, compact: Compactor) -> dict[str, Any]:
        from app.services.agent_task_repository import invalidate_context_summaries, list_artifacts, list_task_runs
        from app.services.content_store import get_content_store
        from app.services.task_artifact_service import persist_task_artifact

        task_id = str(self.state.get("agent_task_id") or "")
        current_run_id = str(self.state.get("agent_run_id") or "")
        artifacts = await list_artifacts(task_id)
        runs = await list_task_runs(task_id)
        attempts_by_run = {str(run.id): int(run.task_attempt or 0) for run in runs}
        completed_ids = {
            str(artifact_id)
            for todo in self.state.get("task_todos") or []
            if isinstance(todo, Mapping) and todo.get("status") == "completed"
            for artifact_id in todo.get("artifact_ids") or []
        }
        by_id = {str(artifact.id): artifact for artifact in artifacts}
        gaps = [
            f"{artifact_id}:{'missing' if by_id.get(artifact_id) is None else by_id[artifact_id].validity}"
            for artifact_id in sorted(completed_ids)
            if by_id.get(artifact_id) is None or by_id[artifact_id].validity != "valid"
        ]
        sources = [
            artifact for artifact in artifacts
            if artifact.id in completed_ids and artifact.validity == "valid"
            and artifact.kind in {"tool_output", "intermediate_report"}
        ]
        source_hash = _stable_hash([(artifact.id, artifact.sha256) for artifact in sources])
        await invalidate_context_summaries(task_id, source_hash=source_hash)
        valid_summary = next((
            artifact for artifact in reversed(artifacts)
            if artifact.validity == "valid" and artifact.kind == "context_summary"
            and (artifact.provenance_json or {}).get("source_hash") == source_hash
            and (artifact.provenance_json or {}).get("policy_version") == 1
        ), None)

        def manifest_item(artifact: Any) -> dict[str, Any]:
            return {
                "id": artifact.id, "kind": artifact.kind, "sha256": artifact.sha256,
                "byte_size": artifact.byte_size, "summary": artifact.summary_json,
                "todo_id": artifact.todo_id,
                "plan_revision": int((getattr(artifact, "provenance_json", None) or {}).get("plan_revision") or 0),
                "origin_run_id": artifact.agent_run_id,
                "origin_attempt": attempts_by_run.get(str(artifact.agent_run_id), 0),
                "inherited": str(artifact.agent_run_id) != current_run_id,
                "validity": artifact.validity,
            }

        evidence_manifest = [manifest_item(artifact) for artifact in sources]
        manifest = list(evidence_manifest)
        estimated_chars = sum(int(item["byte_size"]) for item in evidence_manifest)
        context_window = max(1, int(self.state.get("context_window") or 8192))
        threshold_chars = int(context_window * 4 * 0.70)
        force_chars = int(context_window * 4 * 0.85)
        if estimated_chars > threshold_chars and valid_summary is None:
            store = get_content_store()
            excerpts: list[str] = []
            for artifact in sources:
                if sum(len(value) for value in excerpts) >= 100_000:
                    break
                body = await store.read(artifact.object_key)
                excerpts.append(body.decode("utf-8", errors="replace")[:16_000])
            text, metadata = await compact(chr(10).join(excerpts))
            valid_summary = await persist_task_artifact(
                task_id=task_id, agent_run_id=current_run_id, kind="context_summary", content=text,
                provenance={
                    "source_hash": source_hash,
                    "source_artifact_hashes": {artifact.id: artifact.sha256 for artifact in sources},
                    "policy_version": 1,
                    "plan_revision": int(self.state.get("task_plan_revision") or 0),
                    "model": dict(metadata),
                },
                source_refs={"artifact_ids": [artifact.id for artifact in sources]},
            )
            manifest.append(manifest_item(valid_summary))
        return {
            "task_artifact_manifest": manifest,
            "task_evidence_manifest": evidence_manifest,
            "task_evidence_gaps": gaps,
            "task_context_summary": {
                "source_hash": source_hash, "estimated_chars": estimated_chars,
                "compaction_required": estimated_chars > threshold_chars,
                "compaction_forced": estimated_chars > force_chars,
                "summary_artifact_id": valid_summary.id if valid_summary is not None else None,
                "policy_version": 1,
            },
        }

    async def report_contents(self, context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        from app.services.agent_task_repository import list_artifacts
        from app.services.content_store import get_content_store

        artifacts = await list_artifacts(str(self.state.get("agent_task_id") or ""))
        by_id = {artifact.id: artifact for artifact in artifacts}
        gaps = [str(value) for value in context.get("task_evidence_gaps") or []]
        summary_id = str((context.get("task_context_summary") or {}).get("summary_artifact_id") or "")
        evidence = [value for value in context.get("task_evidence_manifest") or [] if isinstance(value, Mapping)]
        selected_ids = [summary_id] if summary_id else [str(value.get("id") or "") for value in evidence]
        for artifact_id in selected_ids:
            if artifact_id and artifact_id not in by_id:
                gaps.append(f"{artifact_id}:missing")
        reports: list[str] = []
        store = get_content_store()
        for artifact in [by_id[value] for value in selected_ids if value in by_id]:
            if artifact.validity != "valid" or artifact.kind not in {"intermediate_report", "context_summary"}:
                continue
            if sum(len(value) for value in reports) >= 120_000:
                break
            try:
                stat = await store.stat(artifact.object_key)
                if stat.sha256 != artifact.sha256:
                    gaps.append(f"{artifact.id}:hash_mismatch")
                    continue
                reports.append((await store.read(artifact.object_key)).decode("utf-8", errors="replace")[:20_000])
            except (FileNotFoundError, OSError):
                gaps.append(f"{artifact.id}:missing")
        return reports, gaps


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
    }


def product_execution_services_factory(
    state: Mapping[str, Any], configurable: Mapping[str, Any]
) -> DeepResearchExecutionServices:
    return ProductExecutionServices(
        budgets=_ProductBudgets(), memory=_ProductMemory(),
        **_common_services(state, configurable),
    )


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

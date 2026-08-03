"""Ephemeral LLM-assisted curation over canonical durable memory."""

from __future__ import annotations

import json
import logging
import uuid
import asyncio
from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, or_
from sqlalchemy.future import select

from app.agent_workflows.runtime_invocation import safe_json_object
from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import ChatTurnStatus, MemoryScopeType
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import ChatTurn, Memory, MemoryEvent, MemoryOverride, Project, Thread
from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
from app.db.project_activity import touch_project_activity
from app.db.vector import get_vector_db
from app.models.llm_server_client import (
    check_chat_model_ready,
    check_model_supports_tools,
    get_llm,
)
from app.models.memory_curator_budget import bound_curator_transcript, compute_curator_budget
from app.models.memory_tools import (
    MEMORY_PROPOSE,
    MEMORY_READ_STORED,
    MemoryChangeIntent,
    MemoryGetInput,
    MemoryPrepareChangeInput,
    MemorySearchInput,
)
from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorContext,
    MemoryCuratorOperation,
    MemoryCuratorRespondRequest,
)
from app.models.retry import invoke_with_retry
from app.prompts.loaders import load_prompt
from app.services.embedding_model_service import (
    require_embedding_model_ready,
    resolve_scope_embedding_model,
)
from app.services.memory_policy import (
    LOCAL_USER_MEMORY_SCOPE_ID,
    normalize_project_memory_settings,
    normalize_thread_memory_settings,
)
from app.services.memory_tool_service import (
    MemoryToolError,
    MemoryToolNotFoundError,
    build_memory_tool_context,
    get_memory_tool,
    prepare_memory_change,
    search_memory_tool,
)
from app.services.memory_service import index_memory_record, memory_content_hash
from app.services.web_search_service import WEB_SEARCH_CAPABILITY, format_search_context, search_internet
from app.services.memory_review_service import (
    build_conversation_review_batch,
    build_memory_review_batch,
    build_review_memory_search_query,
    complete_memory_review,
)
from app.services.effective_memory_service import (
    serialize_memories_with_relationships,
)
from app.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)

MAX_CURATOR_TOOL_CALLS = 4
MAX_CURATOR_WEB_CALLS = 2

class MemoryCuratorError(ValueError):
    code = "memory_curator_error"


class MemoryCuratorNotFoundError(MemoryCuratorError):
    code = "memory_not_found"


class MemoryCuratorModelUnavailableError(MemoryCuratorError):
    code = "llm_model_unavailable"


class CuratorWebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    reason: str = Field(default="Verify current external information", max_length=500)


class MemoryChangedError(MemoryCuratorError):
    code = "memory_changed"

    def __init__(self, memories: Sequence[Memory]):
        super().__init__("Memory changed after the curator proposal was prepared")
        self.memories = list(memories)


async def _resolve_visible_scopes(
    context: MemoryCuratorContext,
) -> tuple[List[Dict[str, str]], Thread | None, Project | None]:
    selected_type = context.selected_scope_type
    selected_id = (
        LOCAL_USER_MEMORY_SCOPE_ID
        if selected_type == MemoryScopeType.USER.value
        else context.selected_scope_id.strip()
    )
    async with async_session_maker() as session:
        thread = await session.get(Thread, context.thread_id) if context.thread_id else None
        if context.thread_id and thread is None:
            raise MemoryCuratorNotFoundError("Thread not found")
        project_id = context.project_id or (thread.project_id if thread else None)
        project = await session.get(Project, project_id) if project_id else None
        if project_id and project is None:
            raise MemoryCuratorNotFoundError("Project not found")
        if thread and project and thread.project_id != project.id:
            raise MemoryCuratorError("Thread does not belong to the selected project")

    scopes: List[Dict[str, str]] = []
    if thread is not None:
        scopes.append({"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread.id})
    if project is not None:
        scopes.append({"scope_type": MemoryScopeType.PROJECT.value, "scope_id": project.id})
    scopes.append({
        "scope_type": MemoryScopeType.USER.value,
        "scope_id": LOCAL_USER_MEMORY_SCOPE_ID,
    })
    deduped = {
        (scope["scope_type"], scope["scope_id"]): scope
        for scope in scopes
    }
    if (selected_type, selected_id) not in deduped:
        raise MemoryCuratorError("Selected memory scope is not available in this workspace")
    return list(deduped.values()), thread, project


def _consent_snapshot(thread: Thread | None, project: Project | None) -> Dict[str, Any]:
    thread_memory = normalize_thread_memory_settings(thread.settings if thread else {})
    project_memory = normalize_project_memory_settings(project.settings_json if project else {})
    return {
        "administration_available": True,
        "thread_reads_project_memory": (
            thread_memory["thread_reads_project_memory"] if thread else None
        ),
        "project_reads_user_memory": (
            project_memory["project_reads_user_memory"] if project else None
        ),
        "thread_reads_user_memory": (
            thread_memory["thread_reads_user_memory"] if thread else None
        ),
        "effective_user_recall": (
            project_memory["project_reads_user_memory"]
            and thread_memory["thread_reads_user_memory"]
            if thread and project else None
        ),
    }


def _normalize_choice(raw: Any, index: int) -> Dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label") or "").strip()
    if not label:
        return None
    return {
        "id": str(raw.get("id") or f"choice-{index + 1}"),
        "label": label[:300],
        "description": str(raw.get("description") or "")[:1000],
        "user_message": str(raw.get("user_message") or label)[:2000],
    }


def _permission_only_choices(choices: Sequence[Dict[str, str]]) -> bool:
    """Detect redundant approval prompts that belong to the proposal UI."""

    if not choices or len(choices) > 2:
        return False
    approval_prefixes = (
        "yes",
        "no",
        "confirm",
        "cancel",
        "save",
        "apply",
        "proceed",
        "update it",
        "do it",
    )
    return all(
        any(
            str(choice.get(field) or "").strip().casefold().startswith(approval_prefixes)
            for field in ("label", "user_message")
        )
        for choice in choices
    )


def _intent_from_raw(raw: Any) -> MemoryChangeIntent | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action") or "")
    if action not in {"create", "update", "delete", "move", "set_overrides", "noop"}:
        return None
    raw_targets = raw.get("override_target_ids")
    if raw_targets is None and isinstance(raw.get("override_targets"), list):
        raw_targets = [
            str(item.get("memory_id") or "")
            for item in raw["override_targets"]
            if isinstance(item, dict) and item.get("memory_id")
        ]
    try:
        return MemoryChangeIntent(
            action=action,
            memory_id=str(raw.get("memory_id") or "").strip() or None,
            scope_type=raw.get("scope_type"),
            target_scope_type=raw.get("target_scope_type"),
            content=raw.get("content"),
            override_target_ids=raw_targets,
            web_source_ids=[str(item) for item in (raw.get("web_source_ids") or [])],
        )
    except Exception:
        return None


def _curator_safe_payload(value: Any) -> Any:
    """Hide canonical scope IDs from the model while retaining memory IDs."""

    if isinstance(value, list):
        return [_curator_safe_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _curator_safe_payload(item)
            for key, item in value.items()
            if key != "scope_id"
        }
    return value


def _sanitize_curator_intents(
    intents: Sequence[MemoryChangeIntent],
    *,
    scope_ids: set[str],
) -> tuple[List[MemoryChangeIntent], List[str]]:
    """Remove scope identifiers mistakenly emitted as override memory IDs."""

    sanitized = []
    warnings = []
    for intent in intents:
        targets = intent.override_target_ids
        if targets is None:
            sanitized.append(intent)
            continue
        filtered = [memory_id for memory_id in targets if memory_id not in scope_ids]
        if len(filtered) != len(targets):
            warnings.append("Ignored a workspace scope identifier used as an override memory ID.")
        sanitized.append(intent.model_copy(update={"override_target_ids": filtered}))
    return sanitized, warnings


def _restrict_conversation_review_intents(
    intents: Sequence[MemoryChangeIntent],
) -> List[MemoryChangeIntent]:
    """Limit conversation-derived proposals to new memories in the active thread."""

    restricted = []
    for intent in intents:
        if intent.action == "noop":
            restricted.append(intent)
            continue
        if intent.action != "create":
            raise MemoryToolError("Conversation review can only create new Thread memories")
        if intent.scope_type not in {None, MemoryScopeType.THREAD.value}:
            raise MemoryToolError("Conversation review cannot create Project or Global memory")
        if intent.memory_id or intent.target_scope_type:
            raise MemoryToolError("Conversation review cannot modify or move existing memory")
        restricted.append(intent.model_copy(update={"scope_type": MemoryScopeType.THREAD.value}))
    return restricted


def _memory_review_context_from_request(
    context: MemoryCuratorContext,
) -> tuple[str, str]:
    if context.thread_id:
        return "thread", context.thread_id
    if context.project_id:
        return "project", context.project_id
    if context.selected_scope_type == MemoryScopeType.USER.value:
        return "user", LOCAL_USER_MEMORY_SCOPE_ID
    raise MemoryCuratorError("Memory review requires a user, project, or thread context")


async def respond_to_memory_curator(req: MemoryCuratorRespondRequest) -> Dict[str, Any]:
    try:
        tool_context, thread, project = await build_memory_tool_context(
            selected_scope_type=req.context.selected_scope_type,
            selected_scope_id=req.context.selected_scope_id,
            thread_id=req.context.thread_id,
            project_id=req.context.project_id,
            capabilities=[MEMORY_READ_STORED, MEMORY_PROPOSE, WEB_SEARCH_CAPABILITY],
        )
    except MemoryToolNotFoundError as exc:
        raise MemoryCuratorNotFoundError(str(exc)) from exc
    except MemoryToolError as exc:
        raise MemoryCuratorError(str(exc)) from exc
    scopes = [scope.model_dump() for scope in tool_context.visible_scopes]
    consent = _consent_snapshot(thread, project)
    if not await check_chat_model_ready(req.llm_model):
        raise MemoryCuratorModelUnavailableError(f"Chat model {req.llm_model} is unavailable")
    review = (
        await build_conversation_review_batch(
            thread,
            context_window=req.context_window,
            session_factory=async_session_maker,
        )
        if req.mode == "conversation_review"
        else None
    )
    if req.mode == "conversation_review" and not (review or {}).get("turns"):
        return {
            "message": "There are no new completed conversation turns to review.",
            "state": "no_changes",
            "choices": [],
            "operations": [],
            "review": review,
            "embedding_readiness": [],
            "consent": consent,
        }
    memory_review = None
    if req.mode == "memory_review":
        context_type, context_id = _memory_review_context_from_request(req.context)
        position = req.memory_review_cursor.anchor_position if req.memory_review_cursor else 0
        memory_review = await build_memory_review_batch(
            context_type,
            context_id,
            context_window=req.context_window,
            anchor_position=position,
            snapshot_at=(req.memory_review_cursor.snapshot_at if req.memory_review_cursor else None),
            snapshot_scope_versions=(
                req.memory_review_cursor.snapshot_scope_versions
                if req.memory_review_cursor else None
            ),
        )
        if memory_review.get("blocked"):
            return {
                "message": (
                    f'{memory_review["missing_representation_count"]} Global memory representation(s) '
                    f'are being indexed with {memory_review["embedding_model"]}. Retry the review shortly.'
                ),
                "state": "clarification",
                "choices": [{
                    "id": "retry-memory-review",
                    "label": "Retry review",
                    "description": "Check whether the required Global representations are ready.",
                    "user_message": "Retry the memory consistency review now.",
                }],
                "operations": [],
                "review": None,
                "memory_review": memory_review,
                "embedding_readiness": [{
                    "embedding_model": memory_review["embedding_model"],
                    "ready": False,
                    "reason": "global_representation_warming",
                }],
                "consent": consent,
            }
        if not memory_review["candidate_groups"] and memory_review["remaining_anchor_count"] == 0:
            return {
                "message": "No related memory groups require changes in this review snapshot.",
                "state": "no_changes",
                "choices": [],
                "operations": [],
                "review": None,
                "memory_review": memory_review,
                "embedding_readiness": [{
                    "embedding_model": memory_review["embedding_model"],
                    "ready": not memory_review["representation_pending"],
                    "reason": (
                        "global_representation_warming"
                        if memory_review["representation_pending"] else None
                    ),
                }],
                "consent": consent,
            }

    transcript = bound_curator_transcript(
        [item.model_dump(mode="json", exclude_none=True) for item in req.messages],
        context_window=req.context_window,
    )
    query = build_review_memory_search_query(transcript, review)
    curator_budget = compute_curator_budget(req.context_window)
    search_result = await search_memory_tool(tool_context, MemorySearchInput(
        query=query,
        view="stored",
        max_results=curator_budget["memory_result_limit"],
        selected_memory_id=req.memory_id,
    ))
    readiness = search_result.get("readiness", [])
    context_memories = list(search_result.get("memories", []))
    char_budget = curator_budget["memory_context_chars"]
    used_chars = 0
    bounded_memories = []
    for memory in context_memories:
        encoded = json.dumps(memory, ensure_ascii=True)
        if used_chars + len(encoded) > char_budget:
            break
        bounded_memories.append(memory)
        used_chars += len(encoded)
    context_memories = bounded_memories

    system_prompt = load_prompt("memory_curator/system.md")
    tool_limit_prompt = load_prompt("memory_curator/tool_limit.md")
    permission_retry_prompt = load_prompt("memory_curator/permission_retry.md")
    prompt_payload = {
        "mode": req.mode,
        "selected_scope_type": req.context.selected_scope_type,
        "visible_scope_types": [scope["scope_type"] for scope in scopes],
        "selected_memory_id": req.memory_id,
        "conversation": transcript,
        "review": review,
        "memory_review": memory_review,
        "existing_memories": _curator_safe_payload(context_memories),
        "recall_consent": consent,
        "web_search_mode": req.web_search_mode,
        "web_search_decision": (
            req.web_search_decision.model_dump(mode="json")
            if req.web_search_decision else None
        ),
    }
    latest_prepared: Dict[str, Any] | None = None
    latest_intents: List[MemoryChangeIntent] = []
    tool_call_count = 0
    web_call_count = 0
    pending_web_search: Dict[str, str] | None = None
    available_web_sources: Dict[str, Dict[str, Any]] = {}
    scope_ids = {scope.scope_id for scope in tool_context.visible_scopes}
    last_prepare_error: str | None = None
    selected_choice = next(
        (
            message
            for message in reversed(req.messages)
            if message.role == "user" and message.choice_id
        ),
        None,
    )

    async def run_search(**kwargs):
        result = await search_memory_tool(tool_context, MemorySearchInput(**kwargs))
        return json.dumps(_curator_safe_payload(result), ensure_ascii=True)

    async def run_get(**kwargs):
        result = await get_memory_tool(tool_context, MemoryGetInput(**kwargs))
        return json.dumps(_curator_safe_payload(result), ensure_ascii=True)

    async def run_prepare(**kwargs):
        nonlocal latest_intents, latest_prepared
        prepare_req = MemoryPrepareChangeInput(**kwargs)
        intents, input_warnings = _sanitize_curator_intents(
            prepare_req.intents,
            scope_ids=scope_ids,
        )
        if req.mode == "conversation_review":
            intents = _restrict_conversation_review_intents(intents)
        latest_intents = list(intents)
        result = await prepare_memory_change(
            tool_context,
            MemoryPrepareChangeInput(intents=intents),
        )
        if input_warnings:
            result["input_warnings"] = input_warnings
        latest_prepared = result
        return json.dumps(_curator_safe_payload(result), ensure_ascii=True)

    async def run_web_search(**kwargs):
        nonlocal web_call_count, pending_web_search
        if WEB_SEARCH_CAPABILITY not in tool_context.capabilities:
            return json.dumps({"status": "denied", "message": "Web search capability is unavailable."})
        search_req = CuratorWebSearchInput(**kwargs)
        if req.web_search_mode == "off":
            return json.dumps({"status": "disabled", "message": "Internet search is off."})
        decision = req.web_search_decision
        if req.web_search_mode == "ask" and not (
            decision and decision.approved and decision.query == search_req.query
        ):
            if decision and not decision.approved and decision.query == search_req.query:
                return json.dumps({"status": "denied", "message": "The user declined this search."})
            pending_web_search = {"query": search_req.query, "reason": search_req.reason}
            return json.dumps({"status": "approval_required", **pending_web_search})
        if web_call_count >= MAX_CURATOR_WEB_CALLS:
            return json.dumps({"status": "limit_reached"})
        web_call_count += 1
        result = await search_internet(search_req.query, max_results=6)
        for source in result.get("sources") or []:
            available_web_sources[source["id"]] = source
        return json.dumps(_curator_safe_payload(result), ensure_ascii=True)

    tools = [
        StructuredTool.from_function(
            coroutine=run_search,
            name="memory_search",
            description="Search visible effective or stored memory. Curators should search stored memory.",
            args_schema=MemorySearchInput,
        ),
        StructuredTool.from_function(
            coroutine=run_get,
            name="memory_get",
            description="Get exact visible memory records and override relationships by ID.",
            args_schema=MemoryGetInput,
        ),
        StructuredTool.from_function(
            coroutine=run_prepare,
            name="memory_prepare_change",
            description="Validate semantic memory intents and prepare the one confirmable change set.",
            args_schema=MemoryPrepareChangeInput,
        ),
        StructuredTool.from_function(
            coroutine=run_web_search,
            name="internet_search",
            description=(
                "Search current public internet information only when external facts need verification. "
                "Do not use for preferences, supplied facts, scope changes, or memory conflicts."
            ),
            args_schema=CuratorWebSearchInput,
        ),
    ]
    tools_by_name = {tool.name: tool for tool in tools}

    async def invoke_decision(correction: str | None = None):
        nonlocal latest_intents, latest_prepared, tool_call_count, last_prepare_error
        messages = [SystemMessage(content=system_prompt)]
        if correction:
            messages.append(SystemMessage(content=correction))
        messages.append(HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=True)))
        llm = get_llm(req.llm_model, temperature=0.0)
        supports_tools = await check_model_supports_tools(req.llm_model)
        if supports_tools:
            bound = llm.bind_tools(tools)
            loop_count = 0
            while loop_count < MAX_CURATOR_TOOL_CALLS + MAX_CURATOR_WEB_CALLS:
                loop_count += 1
                response = await invoke_with_retry(bound.ainvoke, messages)
                calls = list(getattr(response, "tool_calls", None) or [])
                if not calls:
                    break
                messages.append(response)
                for call in calls:
                    is_web_call = str(call.get("name") or "") == "internet_search"
                    if not is_web_call and tool_call_count >= MAX_CURATOR_TOOL_CALLS:
                        break
                    if not is_web_call:
                        tool_call_count += 1
                    tool = tools_by_name.get(str(call.get("name") or ""))
                    try:
                        output = (
                            await tool.ainvoke(call.get("args") or {})
                            if tool is not None
                            else json.dumps({"error": "Unknown memory tool"})
                        )
                    except Exception as exc:
                        output = json.dumps({"error": str(exc)[:500]})
                    messages.append(ToolMessage(
                        content=str(output),
                        tool_call_id=str(call.get("id") or f"curator-tool-{loop_count}"),
                    ))
                if tool_call_count >= MAX_CURATOR_TOOL_CALLS:
                    messages.append(SystemMessage(content=tool_limit_prompt))
                    response = await invoke_with_retry(llm.ainvoke, messages)
                    break
        else:
            response = await invoke_with_retry(llm.ainvoke, messages)
        parsed = safe_json_object(str(getattr(response, "content", "") or ""))
        raw_web_search = parsed.get("web_search")
        if not supports_tools and isinstance(raw_web_search, dict):
            await run_web_search(**raw_web_search)
            if available_web_sources and pending_web_search is None:
                web_context = format_search_context({"sources": list(available_web_sources.values())})
                response = await invoke_with_retry(llm.ainvoke, [
                    *messages,
                    SystemMessage(content=(
                        "Use the following approved web evidence to finish the curator decision. "
                        "Reference only source IDs you actually use in web_source_ids.\n\n"
                        f"{web_context}"
                    )),
                ])
                parsed = safe_json_object(str(getattr(response, "content", "") or ""))
        state = str(parsed.get("state") or "")
        if state not in {"clarification", "conflict", "proposal", "no_changes"}:
            state = "clarification"
        choices = [
            choice
            for index, raw in enumerate(parsed.get("choices") or [])
            if (choice := _normalize_choice(raw, index)) is not None
        ][:6]
        if latest_prepared is None:
            intents = [
                intent
                for raw in (parsed.get("intents") or parsed.get("operations") or [])
                if (intent := _intent_from_raw(raw)) is not None
            ][:20]
            if intents:
                try:
                    intents, _input_warnings = _sanitize_curator_intents(
                        intents,
                        scope_ids=scope_ids,
                    )
                    if req.mode == "conversation_review":
                        intents = _restrict_conversation_review_intents(intents)
                    latest_intents = list(intents)
                    latest_prepared = await prepare_memory_change(
                        tool_context,
                        MemoryPrepareChangeInput(intents=intents),
                    )
                except MemoryToolError as exc:
                    last_prepare_error = str(exc)
                    parsed["message"] = str(exc)
                    state = "clarification"
        operations = list((latest_prepared or {}).get("operations", []))
        summaries = list((latest_prepared or {}).get("operation_summaries", []))
        selected_source_ids = {
            source_id
            for intent in latest_intents
            for source_id in intent.web_source_ids
            if source_id in available_web_sources
        }
        selected_sources = [
            {key: source[key] for key in ("id", "title", "url", "query", "searched_at")}
            for source_id, source in available_web_sources.items()
            if source_id in selected_source_ids
        ]
        if selected_sources:
            for operation in operations:
                if operation.get("action") in {"create", "update"}:
                    operation["web_sources"] = selected_sources
        return parsed, state, choices, operations, summaries

    parsed, state, choices, operations, summaries = await invoke_decision()
    if pending_web_search is not None:
        return {
            "message": (
                f'The curator wants to search the internet for "{pending_web_search["query"]}". '
                "Allow this search?"
            ),
            "state": "web_search_approval",
            "choices": [],
            "operations": [],
            "operation_summaries": [],
            "review": review,
            "memory_review": memory_review,
            "embedding_readiness": readiness,
            "context_memory_count": len(context_memories),
            "consent": consent,
            "tool_calls_used": tool_call_count,
            "web_calls_used": web_call_count,
            "pending_web_search": pending_web_search,
            "web_sources": [],
        }
    substantive_operations = [op for op in operations if op["action"] != "noop"]
    if substantive_operations:
        state = "proposal"
        choices = []
    elif state in {"clarification", "conflict"} and _permission_only_choices(choices):
        parsed, state, choices, operations, summaries = await invoke_decision(
            permission_retry_prompt
        )
        substantive_operations = [op for op in operations if op["action"] != "noop"]
        if substantive_operations:
            state = "proposal"
            choices = []
    if (
        req.mode == "memory_review"
        and selected_choice is not None
        and last_prepare_error
        and not substantive_operations
    ):
        parsed, state, choices, operations, summaries = await invoke_decision(
            (
                "The user already selected a memory-review resolution choice. "
                f"choice_id={selected_choice.choice_id}; user_message={selected_choice.content}\n"
                f"The previous proposed operation failed validation: {last_prepare_error}\n"
                "Do not ask the same question again. Return one corrected concrete proposal using "
                "only valid existing memory IDs from candidate_groups, or return no_changes if the "
                "selected outcome is already represented."
            )
        )
        substantive_operations = [op for op in operations if op["action"] != "noop"]
        if substantive_operations:
            state = "proposal"
            choices = []
    if state == "proposal" and not substantive_operations:
        state = "clarification"
    if state in {"clarification", "conflict"} and not choices:
        choices = [{
            "id": "clarify",
            "label": "Describe the intended memory",
            "description": "Tell the curator what should be remembered and at which scope.",
            "user_message": "Please help me clarify the exact memory and scope.",
        }]
    return {
        "message": str(parsed.get("message") or "Please review the proposed memory changes.")[:6000],
        "state": state,
        "choices": choices,
        "operations": operations,
        "operation_summaries": summaries,
        "review": review,
        "memory_review": memory_review,
        "embedding_readiness": readiness,
        "context_memory_count": len(context_memories),
        "consent": consent,
        "tool_calls_used": tool_call_count,
        "web_calls_used": web_call_count,
        "pending_web_search": None,
        "web_sources": list(available_web_sources.values()),
    }


def _operation_scope(operation: MemoryCuratorOperation) -> tuple[str, str]:
    scope_type = str(operation.scope_type or "")
    scope_id = str(operation.scope_id or "").strip()
    if scope_type == MemoryScopeType.USER.value:
        scope_id = LOCAL_USER_MEMORY_SCOPE_ID
    return scope_type, scope_id


def _timestamp_matches(memory: Memory, expected: str | None) -> bool:
    if not expected:
        return False
    return expected == iso_utc_z(memory.updated_at or memory.created_at)


_SCOPE_RANK = {"user": 0, "project": 1, "thread": 2}


def _validate_override_scope(source: Memory, target: Memory) -> None:
    source_rank = _SCOPE_RANK[source.scope_type]
    target_rank = _SCOPE_RANK[target.scope_type]
    if source_rank < target_rank:
        raise MemoryCuratorError("A broader memory cannot override a narrower memory")
    if source_rank == target_rank and source.scope_id != target.scope_id:
        raise MemoryCuratorError("Memory overrides cannot cross peers at the same scope")


def _assert_acyclic(edges: Sequence[tuple[str, str]]) -> None:
    adjacency: Dict[str, List[str]] = {}
    for source_id, target_id in edges:
        adjacency.setdefault(source_id, []).append(target_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(memory_id: str) -> None:
        if memory_id in visiting:
            raise MemoryCuratorError("Memory override relationships cannot contain a cycle")
        if memory_id in visited:
            return
        visiting.add(memory_id)
        for target_id in adjacency.get(memory_id, []):
            visit(target_id)
        visiting.remove(memory_id)
        visited.add(memory_id)

    for source_id in list(adjacency):
        visit(source_id)


async def apply_memory_curator_change_set(req: MemoryCuratorApplyRequest) -> Dict[str, Any]:
    if req.memory_review_cursor:
        expected_type, expected_id = _memory_review_context_from_request(req.context)
        if (
            req.memory_review_cursor.context_type != expected_type
            or req.memory_review_cursor.context_id != expected_id
        ):
            raise MemoryCuratorError("Memory review cursor does not match this workspace")
    scopes, context_thread, _project = await _resolve_visible_scopes(req.context)
    visible_scopes = {(scope["scope_type"], scope["scope_id"]) for scope in scopes}
    operations = [operation for operation in req.operations if operation.action != "noop"]
    if req.review_cursor:
        if context_thread is None or req.review_cursor.thread_id != context_thread.id:
            raise MemoryCuratorError("Conversation review cursor does not match this thread")
        for operation in operations:
            if operation.action != "create":
                raise MemoryCuratorError("Conversation review can only create new Thread memories")
            if _operation_scope(operation) != (MemoryScopeType.THREAD.value, context_thread.id):
                raise MemoryCuratorError("Conversation review can only create memory in this thread")
            if operation.semantic_action not in {None, "create"}:
                raise MemoryCuratorError("Conversation review cannot modify existing memory")
    source_ids = [operation.memory_id for operation in operations if operation.memory_id]
    if len(source_ids) != len(set(source_ids)):
        raise MemoryCuratorError("A memory may only appear once in a change set")

    create_ids = {
        index: str(uuid.uuid4())
        for index, operation in enumerate(operations)
        if operation.action == "create"
    }
    target_specs = [
        target
        for operation in operations
        if operation.action in {"create", "update"}
        for target in operation.override_targets
    ]
    affected_ids = sorted({
        *[memory_id for memory_id in source_ids if memory_id],
        *[target.memory_id for target in target_specs],
    })

    create_models: Dict[int, str] = {}
    for index, operation in enumerate(operations):
        scope = _operation_scope(operation)
        if operation.action == "create":
            if scope not in visible_scopes:
                raise MemoryCuratorError("Create operation targets an unavailable scope")
            if not operation.content:
                raise MemoryCuratorError("Create operation requires content")
            model = await resolve_scope_embedding_model(*scope)
            await require_embedding_model_ready(model)
            create_models[index] = model

    changed: List[Memory] = []
    to_index: List[Memory] = []
    deleted_ids: List[str] = []
    cleanup_targets: List[tuple[str, str]] = []
    cleaned_vector_targets: List[tuple[str, str]] = []
    try:
        async with async_session_maker() as session:
            async with session.begin():
                locked: Dict[str, Memory] = {}
                if affected_ids:
                    result = await session.execute(
                        select(Memory)
                        .where(Memory.id.in_(affected_ids))
                        .order_by(Memory.id)
                        .with_for_update()
                    )
                    locked = {memory.id: memory for memory in result.scalars().all()}
                missing = [memory_id for memory_id in affected_ids if memory_id not in locked]
                if missing:
                    raise MemoryCuratorNotFoundError(f"Memory not found: {missing[0]}")
                expected_timestamps = {
                    operation.memory_id: operation.expected_updated_at
                    for operation in operations if operation.memory_id
                }
                expected_timestamps.update({
                    target.memory_id: target.expected_updated_at for target in target_specs
                })
                stale = [memory for memory_id, memory in locked.items()
                         if not _timestamp_matches(memory, expected_timestamps.get(memory_id))]
                if stale:
                    raise MemoryChangedError(stale)

                source_memory_by_index: Dict[int, Memory] = {}
                for index, operation in enumerate(operations):
                    if operation.action == "create":
                        scope_type, scope_id = _operation_scope(operation)
                        source_memory_by_index[index] = Memory(
                            id=create_ids[index],
                            scope_type=scope_type,
                            scope_id=scope_id,
                            content=str(operation.content or "").strip(),
                            embedding_model=create_models[index],
                            content_hash=memory_content_hash(operation.content or ""),
                        )
                        source = source_memory_by_index[index]
                    else:
                        source = locked[operation.memory_id or ""]
                        if (source.scope_type, source.scope_id) not in visible_scopes:
                            raise MemoryCuratorError("Operation targets memory outside this workspace")
                        if _operation_scope(operation) != (source.scope_type, source.scope_id):
                            raise MemoryCuratorError("Memory scope cannot be changed by update or delete")
                    if operation.action not in {"create", "update"}:
                        continue
                    if not operation.content:
                        raise MemoryCuratorError("Create/update operation requires content")
                    target_ids = [target.memory_id for target in operation.override_targets]
                    if len(target_ids) != len(set(target_ids)):
                        raise MemoryCuratorError("Override targets cannot contain duplicates")
                    for target_spec in operation.override_targets:
                        target = locked[target_spec.memory_id]
                        if (target.scope_type, target.scope_id) not in visible_scopes:
                            raise MemoryCuratorError("Override target is outside this workspace")
                        if source.id == target.id:
                            raise MemoryCuratorError("A memory cannot override itself")
                        _validate_override_scope(source, target)

                existing_edges = list((await session.execute(select(MemoryOverride))).scalars().all())
                replacement_sources = {
                    create_ids[index] if operation.action == "create" else str(operation.memory_id)
                    for index, operation in enumerate(operations)
                    if operation.action in {"create", "update"}
                }
                proposed_edges = [
                    (edge.overriding_memory_id, edge.overridden_memory_id)
                    for edge in existing_edges
                    if edge.overriding_memory_id not in replacement_sources
                ]
                for index, operation in enumerate(operations):
                    if operation.action in {"create", "update"}:
                        source_id = create_ids[index] if operation.action == "create" else str(operation.memory_id)
                        proposed_edges.extend((source_id, target.memory_id) for target in operation.override_targets)
                _assert_acyclic(proposed_edges)

                for operation in operations:
                    if operation.action == "create":
                        continue
                    memory = locked[operation.memory_id or ""]
                    content_changed = (
                        operation.action == "update"
                        and memory.content_hash != memory_content_hash(operation.content or "")
                    )
                    if content_changed:
                        await require_embedding_model_ready(memory.embedding_model)
                    if operation.action == "delete" or content_changed:
                        cleanup_targets.append((memory.id, memory.embedding_model))

                for operation in operations:
                    if operation.action not in {"create", "update"}:
                        continue
                    scope_type, scope_id = _operation_scope(operation)
                    duplicate_query = select(Memory).where(
                        Memory.scope_type == scope_type,
                        Memory.scope_id == scope_id,
                        Memory.content_hash == memory_content_hash(operation.content or ""),
                    )
                    if operation.memory_id:
                        duplicate_query = duplicate_query.where(Memory.id != operation.memory_id)
                    duplicate = (await session.execute(duplicate_query.limit(1))).scalar_one_or_none()
                    if duplicate is not None:
                        raise MemoryCuratorError(
                            f"An identical active memory already exists in this scope: {duplicate.id}"
                        )

                reviewed_turn = None
                review_thread = None
                if req.review_cursor is not None:
                    cursor = req.review_cursor
                    if context_thread is None or cursor.thread_id != context_thread.id:
                        raise MemoryCuratorError("Review cursor does not match the active thread")
                    reviewed_turn = await session.get(ChatTurn, cursor.reviewed_through_turn_id)
                    if (
                        reviewed_turn is None
                        or reviewed_turn.thread_id != cursor.thread_id
                        or reviewed_turn.status != ChatTurnStatus.COMPLETED.value
                        or reviewed_turn.created_at != cursor.reviewed_through_created_at
                    ):
                        raise MemoryCuratorError("Review cursor does not reference an eligible turn")
                    review_thread = await session.get(Thread, cursor.thread_id, with_for_update=True)
                    if review_thread is None:
                        raise MemoryCuratorNotFoundError("Review thread not found")

                for memory_id, embedding_model in cleanup_targets:
                    if not await get_vector_db().delete_memory_vectors(memory_id, embedding_model):
                        raise MemoryCuratorError(f"Failed to clean vectors for memory {memory_id}")
                    cleaned_vector_targets.append((memory_id, embedding_model))

                now = utc_now()
                for index, operation in enumerate(operations):
                    scope_type, scope_id = _operation_scope(operation)
                    source_refs: Dict[str, Any] = {}
                    if req.review_cursor is not None:
                        source_refs.update({
                            "curator_mode": "conversation_review",
                            "source_thread_id": req.review_cursor.thread_id,
                            "reviewed_through_turn_id": req.review_cursor.reviewed_through_turn_id,
                        })
                    if operation.move_source_memory_id:
                        source_refs["curator_move_source_memory_id"] = operation.move_source_memory_id
                    if operation.web_sources:
                        source_refs["web_sources"] = [
                            source.model_dump(mode="json") for source in operation.web_sources
                        ]
                    if operation.action == "create":
                        memory = Memory(
                            id=create_ids[index],
                            scope_type=scope_type,
                            scope_id=scope_id,
                            content=str(operation.content or "").strip(),
                            embedding_model=create_models[index],
                            content_hash=memory_content_hash(operation.content or ""),
                            index_status="pending",
                            index_attempts=0,
                            source_refs_json=source_refs,
                            created_at=now,
                            updated_at=now,
                        )
                        session.add(memory)
                        await session.flush()
                        session.add(MemoryEvent(
                            memory_id=memory.id,
                            event_type="curator_created",
                            actor_id=req.actor_id,
                            payload_json={
                                "mode": "confirmed_change_set",
                                "semantic_action": operation.semantic_action or operation.action,
                                "operation_group_id": operation.operation_group_id,
                            },
                            created_at=now,
                        ))
                        changed.append(memory)
                        to_index.append(memory)
                        await MemoryRepository(session).replace_overrides(
                            memory.id,
                            [target.memory_id for target in operation.override_targets],
                            actor_id=req.actor_id,
                            updated_at=now,
                        )
                    elif operation.action == "update":
                        memory = locked[operation.memory_id or ""]
                        existing_source_refs = dict(memory.source_refs_json or {})
                        existing_source_refs.update(source_refs)
                        source_refs = existing_source_refs
                        content_changed = memory.content_hash != memory_content_hash(operation.content or "")
                        if content_changed:
                            updated = await MemoryRepository(session).update_memory(
                                memory.id,
                                content=str(operation.content or "").strip(),
                                content_hash=memory_content_hash(operation.content or ""),
                                source_refs_json=source_refs,
                                actor_id=req.actor_id,
                                event_type="curator_updated",
                                event_payload={
                                    "mode": "confirmed_change_set",
                                    "semantic_action": operation.semantic_action or operation.action,
                                    "operation_group_id": operation.operation_group_id,
                                },
                                updated_at=now,
                            )
                            if updated is None:
                                raise MemoryCuratorNotFoundError("Memory disappeared during update")
                            memory = updated
                            to_index.append(memory)
                        elif operation.web_sources:
                            memory.source_refs_json = source_refs
                            memory.updated_at = now
                            session.add(MemoryEvent(
                                memory_id=memory.id,
                                event_type="curator_updated",
                                actor_id=req.actor_id,
                                payload_json={
                                    "mode": "confirmed_change_set",
                                    "semantic_action": operation.semantic_action or operation.action,
                                    "operation_group_id": operation.operation_group_id,
                                    "provenance_only": True,
                                },
                                created_at=now,
                            ))
                            await session.flush()
                        await MemoryRepository(session).replace_overrides(
                            memory.id,
                            [target.memory_id for target in operation.override_targets],
                            actor_id=req.actor_id,
                            updated_at=now,
                        )
                        changed.append(memory)
                    elif operation.action == "delete":
                        memory = locked[operation.memory_id or ""]
                        await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
                        await session.delete(memory)
                        deleted_ids.append(memory.id)

                if reviewed_turn is not None and review_thread is not None:
                    metadata = dict(review_thread.thread_metadata or {})
                    metadata["memory_curator"] = {
                        "reviewed_through_turn_id": reviewed_turn.id,
                        "reviewed_through_created_at": iso_utc_z(reviewed_turn.created_at),
                        "reviewed_at": iso_utc_z(now),
                    }
                    replace_jsonb_field(review_thread, "thread_metadata", metadata)

                touched_projects = {
                    scope_id
                    for scope_type, scope_id in [
                        _operation_scope(operation) for operation in operations
                    ]
                    if scope_type == MemoryScopeType.PROJECT.value
                }
                for project_id in touched_projects:
                    await touch_project_activity(session, project_id, occurred_at=now)
                from app.services.memory_review_service import bump_memory_scope_activity
                from app.services.memory_representation_service import invalidate_global_representations
                await bump_memory_scope_activity(
                    [_operation_scope(operation) for operation in operations],
                    session=session,
                )
                for memory in changed:
                    await invalidate_global_representations(memory, session=session)
                await session.flush()
    except Exception:
        for memory_id, _embedding_model in cleaned_vector_targets:
            try:
                async with async_session_maker() as session:
                    current = await session.get(Memory, memory_id)
                if current is not None:
                    await index_memory_record(current)
            except Exception:
                logger.exception(
                    "Failed to restore memory vector after curator rollback | memory_id=%s",
                    memory_id,
                )
        raise

    warnings = []
    refreshed_records = []
    to_index_ids = {memory.id for memory in to_index}
    for memory in changed:
        try:
            if memory.id in to_index_ids:
                await index_memory_record(memory)
        except Exception as exc:
            warnings.append({
                "code": "memory_index_failed",
                "memory_id": memory.id,
                "message": str(exc)[:500],
            })
        async with async_session_maker() as session:
            refreshed = await session.get(Memory, memory.id)
        if refreshed is not None:
            refreshed_records.append(refreshed)
    indexed_records = await serialize_memories_with_relationships(refreshed_records)
    for memory in refreshed_records:
        if memory.scope_type == MemoryScopeType.USER.value:
            async with async_session_maker() as session:
                from app.db.models_sqlmodel import GlobalMemoryRepresentation
                models = list((await session.execute(
                    select(GlobalMemoryRepresentation.embedding_model).where(
                        GlobalMemoryRepresentation.memory_id == memory.id,
                        GlobalMemoryRepresentation.index_status.in_(("pending", "failed")),
                    )
                )).scalars().all())
            from app.services.memory_representation_service import warm_global_representations_for_model
            for model in set(models):
                asyncio.create_task(warm_global_representations_for_model(model))
    memory_review_completed = False
    if req.memory_review_cursor and req.memory_review_cursor.remaining_anchor_count == 0:
        await complete_memory_review(
            req.memory_review_cursor.context_type,
            req.memory_review_cursor.context_id,
            req.memory_review_cursor.snapshot_scope_versions,
        )
        memory_review_completed = True
    return {
        "changed_memories": indexed_records,
        "deleted_memory_ids": deleted_ids,
        "warnings": warnings,
        "review_cursor_advanced": bool(req.review_cursor),
        "memory_review_completed": memory_review_completed,
        "memory_review_cursor": (
            req.memory_review_cursor.model_dump(mode="json")
            if req.memory_review_cursor else None
        ),
    }

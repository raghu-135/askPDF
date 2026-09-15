"""Human permission at the shared tool boundary, independent of runtimes."""

from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.agent.tool_registry import TOOL_CONTRACT_METADATA, CAT_WEB, CAT_EXTERNAL_RESEARCH
from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import AgentRun, ToolApprovalDecision, ToolInvocation
from app.time_utils import utc_now
from app.tools.context import ToolInvocationContext
from runtime_protocol.tool_approval import (
    ApprovalMode, ApprovalScope, ToolApprovalPolicy, decision_scope,
    effective_mode, invocation_digest, tool_approval_request,
)
from runtime_protocol.tool_contract import ToolResult, ToolError, ToolTrace


def invocation_policies(
    config: Mapping[str, Any], *, permissions: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve product settings once using the authoritative tool catalog."""
    permissions = permissions or {}
    hitl = config.get("hitl_policy") or {}
    rules = {
        str(name): ToolApprovalPolicy.from_mapping(value).to_dict()
        for name, value in (hitl.get("tools") or {}).items()
    } if hitl.get("enabled") else {}
    unknown = rules.keys() - TOOL_CONTRACT_METADATA.keys()
    if unknown:
        raise ValueError(f"Unknown tools in approval policy: {', '.join(sorted(unknown))}")
    mode = str(permissions.get("web_search_mode") or config.get("web_search_mode") or (
        "ask" if config.get("hitl_web_approval") else "on" if config.get("use_web_search") else "off"
    ))
    if mode not in {"on", "off", "ask"}:
        raise ValueError("Unknown web search permission mode")
    web_mode = {"on": ApprovalMode.ALLOW, "off": ApprovalMode.DENY, "ask": ApprovalMode.ASK}[mode]
    scope = ApprovalScope.TASK if permissions else ApprovalScope.RUN
    for name, metadata in TOOL_CONTRACT_METADATA.items():
        if metadata.get("category") in {CAT_WEB, CAT_EXTERNAL_RESEARCH}:
            if web_mode is ApprovalMode.DENY or name not in rules:
                rules[name] = ToolApprovalPolicy(web_mode, scope).to_dict()
    if config.get("hitl_canvas_publish") and "publish_canvas" not in rules:
        rules["publish_canvas"] = ToolApprovalPolicy(ApprovalMode.ASK, scope).to_dict()
    return rules


async def record_tool_decision(
    session: AsyncSession, run: AgentRun, interrupt: Mapping[str, Any], action: str,
) -> None:
    proposed = interrupt.get("proposed_tool") or {}
    invocation_id = proposed.get("invocation_id")
    if not invocation_id:
        return  # Graph review and choice interrupts do not authorize tools.
    policy = ToolApprovalPolicy(ApprovalMode.ASK, ApprovalScope(interrupt["approval_scope_kind"]))
    scope = decision_scope(action, policy)
    # Continue without applies to this permission scope, preventing repeated
    # prompts for an explicitly denied tool during replanning.
    scope = scope or policy.scope
    scope_id = run.task_id if scope is ApprovalScope.TASK else run.id
    if scope is ApprovalScope.INVOCATION:
        scope_id = str(invocation_id)
    if not scope_id:
        raise ValueError("Tool approval has no matching permission scope")
    values = dict(
        id=str(uuid4()), created_at=utc_now(),
        run_id=run.id, interrupt_id=str(interrupt["interrupt_id"]),
        tool_name=str(proposed["name"]), invocation_id=str(invocation_id),
        argument_hash=str(proposed["argument_hash"]), scope=scope.value,
        scope_id=scope_id, decision="allowed" if action in {"approve", "approve_for_scope"} else "denied",
    )
    await session.execute(insert(ToolApprovalDecision).values(**values).on_conflict_do_update(
        constraint="uq_tool_approval_interrupt",
        set_={key: value for key, value in values.items() if key not in {"id", "run_id", "interrupt_id"}},
    ))


async def check_tool_approval(
    name: str, arguments: Mapping[str, Any], context: ToolInvocationContext, *, invocation_id: str,
) -> tuple[ApprovalMode, dict[str, Any] | None]:
    """Check server-verified policy and durable decisions before any handler."""
    extensions = context.extensions or {}
    rules = extensions.get("tool_approval_policy") or {}
    if name not in rules:
        return ApprovalMode.ALLOW, None
    policy = ToolApprovalPolicy.from_mapping(rules[name])
    if policy.mode is ApprovalMode.DENY:
        return ApprovalMode.DENY, None
    argument_hash = invocation_digest(name, arguments)
    async with async_session_maker() as session:
        decision = (await session.execute(
            select(ToolApprovalDecision).where(
                ToolApprovalDecision.tool_name == name,
                or_(
                    and_(ToolApprovalDecision.scope == "invocation", ToolApprovalDecision.scope_id == invocation_id,
                         ToolApprovalDecision.run_id == context.run_id, ToolApprovalDecision.argument_hash == argument_hash),
                    and_(ToolApprovalDecision.scope == "run", ToolApprovalDecision.scope_id == context.run_id),
                    and_(ToolApprovalDecision.scope == "task", ToolApprovalDecision.scope_id == str(extensions.get("task_id") or "")),
                ),
            ).order_by(ToolApprovalDecision.created_at.desc(), ToolApprovalDecision.id.desc()).limit(1)
        )).scalar_one_or_none()
    mode = effective_mode(policy, grant=decision.decision if decision is not None else None)
    if mode is not ApprovalMode.ASK:
        return mode, None
    request = tool_approval_request(
        name, arguments, policy=policy, caller=context.caller_node or "agent",
        response_operation="run.resume",
    )
    request["proposed_tool"]["invocation_id"] = invocation_id
    return ApprovalMode.ASK, request


async def execute_tool_once(
    name: str, arguments: Mapping[str, Any], context: ToolInvocationContext,
    *, invocation_id: str, invoke: Callable[[], Awaitable[ToolResult]],
) -> ToolResult:
    """Replay completed calls; never re-execute a call with an unknown outcome."""
    if "tool_approval_policy" not in (context.extensions or {}):
        return await invoke()
    argument_hash = invocation_digest(name, arguments)
    key = (ToolInvocation.run_id == context.run_id, ToolInvocation.invocation_id == invocation_id)
    async with async_session_maker() as session:
        async with session.begin():
            claimed = (await session.execute(insert(ToolInvocation).values(
                id=str(uuid4()), run_id=context.run_id, invocation_id=invocation_id,
                tool_name=name, argument_hash=argument_hash, status="running", started_at=utc_now(),
            ).on_conflict_do_nothing(constraint="uq_tool_invocation").returning(ToolInvocation.id))).scalar_one_or_none()
            if claimed is None:
                previous = (await session.execute(select(ToolInvocation).where(*key))).scalar_one()
                if previous.tool_name != name or previous.argument_hash != argument_hash:
                    raise ValueError("Tool invocation identity was reused with different arguments")
                if previous.result_json is not None:
                    return ToolResult.model_validate(previous.result_json)
                return ToolResult(
                    ok=False, content="This invocation is already running or its outcome is unknown. It will not be executed again automatically.",
                    error=ToolError(code="tool_invocation_outcome_unknown", message="Inspect the original invocation before retrying with a new identity.", retryable=False),
                    trace=ToolTrace(tool_name=name, agent_run_id=context.run_id),
                )
    # Cancellation or process death deliberately leaves the fence in place.
    # Silently retrying a potentially completed side effect would violate the
    # human's approval of one invocation.
    result = await invoke()
    payload = result.to_payload()
    async with async_session_maker() as session:
        async with session.begin():
            record = (await session.execute(select(ToolInvocation).where(*key).with_for_update())).scalar_one()
            record.result_json = payload
            record.status = "completed" if result.ok else "failed"
            record.completed_at = utc_now()
    return result

"""String enums for agent workflow contracts and runtime states."""

from enum import Enum


class InterruptStatus(str, Enum):
    PENDING = "pending"
    RESUMED = "resumed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AgentRunResumeAction(str, Enum):
    APPROVE = "approve"
    APPROVE_SELECTED = "approve_selected"
    REJECT = "reject"
    EDIT = "edit"
    CONTINUE_WITHOUT = "continue_without"


class HitlMode(str, Enum):
    APPROVAL = "approval"
    CHOICE = "choice"
    REVIEW = "review"


class HitlPhase(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    INSIDE_TOOL = "inside_tool"


class HitlSelectionMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
    SINGLE_OR_MULTI = "single_or_multi"


class RouterRoute(str, Enum):
    DOCUMENT = "document"
    MEMORY = "memory"
    TIMELINE = "timeline"
    WEB = "web"
    DIRECT = "direct"
    CLARIFY = "clarify"


class PlannerRoute(str, Enum):
    EXECUTE = "execute"
    DIRECT = "direct"
    CLARIFY = "clarify"


class EvaluatorRoute(str, Enum):
    ANSWER = "answer"
    REPLAN = "replan"
    ANSWER_BUDGET_EXHAUSTED = "answer_budget_exhausted"


class RouteFunctionId(str, Enum):
    ROUTER = "router_route"
    PLANNER = "planner_route"
    EVALUATOR = "evaluator_route"
    HITL_GATE = "hitl_gate_route"


ROUTER_ROUTES = {route.value for route in RouterRoute}
PLANNER_ROUTES = {route.value for route in PlannerRoute}
EVALUATOR_ROUTES = {route.value for route in EvaluatorRoute}
HITL_ACTIONS = {action.value for action in AgentRunResumeAction}
RESUME_ACTIONS = {
    AgentRunResumeAction.APPROVE.value,
    AgentRunResumeAction.APPROVE_SELECTED.value,
    AgentRunResumeAction.EDIT.value,
    AgentRunResumeAction.CONTINUE_WITHOUT.value,
}
HITL_PHASES = {phase.value for phase in HitlPhase}
HITL_MODES = {mode.value for mode in HitlMode}
HITL_SELECTION_MODES = {mode.value for mode in HitlSelectionMode}
TERMINAL_INTERRUPT_STATUSES = {
    InterruptStatus.RESUMED.value,
    InterruptStatus.REJECTED.value,
    InterruptStatus.EXPIRED.value,
}

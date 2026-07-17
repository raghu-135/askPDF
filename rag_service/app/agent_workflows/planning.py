from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agent_workflows.enums import PlannerRoute, PLANNER_ROUTES, ToolContractId, WorkflowNodeType
from app.agent_workflows.trace import compact_preview


WORKER_NODE_ORDER = [
    WorkflowNodeType.RETRIEVAL_WORKER.value,
    WorkflowNodeType.MEMORY_WORKER.value,
    WorkflowNodeType.TIMELINE_WORKER.value,
    WorkflowNodeType.WEB_WORKER.value,
]


TEMPORAL_PLAN_RE = re.compile(
    r"\b("
    r"latest|most\s+recent|recent|newest|current|first|earliest|oldest|last|"
    r"since|before|after|earlier|later|timeline|chronolog(?:y|ical)|sequence|order|"
    r"when|date|time"
    r")\b",
    re.IGNORECASE,
)

MEMORY_PLAN_RE = re.compile(
    r"\b("
    r"previously|prior|earlier\s+(answer|conversation|chat|discussion)|"
    r"remember|discussed|talked\s+about|said\s+before|you\s+said|we\s+(said|discussed|talked)"
    r")\b",
    re.IGNORECASE,
)

DOCUMENT_PLAN_RE = re.compile(
    r"\b("
    r"document|pdf|paper|uploaded|upload|file|source|page|section|chapter|"
    r"quote|cite|citation|excerpt|summar(?:y|ize)|abstract"
    r")\b",
    re.IGNORECASE,
)


def infer_required_plan_steps(question: Optional[str]) -> List[str]:
    """Return worker nodes that should be present for obvious query intent cues."""

    text = str(question or "")
    required: List[str] = []
    if TEMPORAL_PLAN_RE.search(text):
        required.append(WorkflowNodeType.TIMELINE_WORKER.value)
    if (
        MEMORY_PLAN_RE.search(text)
        and WorkflowNodeType.MEMORY_WORKER.value not in required
        and WorkflowNodeType.TIMELINE_WORKER.value not in required
    ):
        required.append(WorkflowNodeType.MEMORY_WORKER.value)
    if DOCUMENT_PLAN_RE.search(text) and WorkflowNodeType.RETRIEVAL_WORKER.value not in required:
        required.append(WorkflowNodeType.RETRIEVAL_WORKER.value)
    return required


def ordered_plan_steps(steps: List[str]) -> List[str]:
    return [node for node in WORKER_NODE_ORDER if node in steps]


def fallback_clarification_options() -> List[str]:
    return [
        "Do I want an answer based on the uploaded document evidence?",
        "Do I want an answer based on what we discussed earlier in this thread?",
        "Do I want an answer based on the timeline or order of events in this thread?",
    ]


def normalize_execution_plan(
    parsed: Dict[str, Any],
    *,
    use_web_search: bool,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    route = parsed.get("route") if parsed.get("route") in PLANNER_ROUTES else PlannerRoute.EXECUTE.value
    required_steps = infer_required_plan_steps(question)
    normalization_notes: List[str] = []
    if route == PlannerRoute.DIRECT.value and required_steps:
        route = PlannerRoute.EXECUTE.value
        normalization_notes.append("direct_route_clamped_to_execute")
    raw_steps = parsed.get("execution_plan") or parsed.get("steps") or []
    steps: List[str] = []
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if isinstance(step, str):
                node = step
            elif isinstance(step, dict):
                node = step.get("node") or step.get("worker") or step.get("id")
            else:
                continue
            if node in WORKER_NODE_ORDER and node not in steps:
                steps.append(node)
    if not use_web_search and WorkflowNodeType.WEB_WORKER.value in steps:
        steps = [step for step in steps if step != WorkflowNodeType.WEB_WORKER.value]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    if route == PlannerRoute.EXECUTE.value:
        for required_step in required_steps:
            if required_step not in steps:
                steps.append(required_step)
    if route == PlannerRoute.EXECUTE.value and not steps:
        steps = [WorkflowNodeType.RETRIEVAL_WORKER.value]
        normalization_notes.append("empty_execute_plan_defaulted_to_retrieval_worker")
    steps = ordered_plan_steps(steps)
    if route != PlannerRoute.EXECUTE.value:
        steps = []
    clarification_options = parsed.get("clarification_options")
    if route == PlannerRoute.CLARIFY.value:
        clarification_options = bounded_string_list(clarification_options)
        if not clarification_options:
            clarification_options = fallback_clarification_options()
    return {
        "route": route,
        "route_reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "execution_plan": steps,
        "clarification_options": clarification_options if route == PlannerRoute.CLARIFY.value else None,
        "normalization_notes": normalization_notes,
    }


def risk_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"low", "medium", "high"} else "medium"


def bounded_string_list(value: Any, *, limit: int = 5, chars: int = 240) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            item = item.get("text") or item.get("label") or item.get("title") or item.get("question")
        if item is None:
            continue
        text = compact_preview(str(item), limit=chars)
        if text:
            result.append(text)
    return result


def bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def replan_budget(state: Dict[str, Any]) -> int:
    try:
        return max(1, int(state.get("replans", 1)))
    except (TypeError, ValueError):
        return 1


def current_replan_count(state: Dict[str, Any]) -> int:
    try:
        return max(0, int(state.get("replan_count", 0)))
    except (TypeError, ValueError):
        return 0


def normalize_evaluator_report(parsed: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        sufficient = bool(state.get("evidence")) and bool(state.get("document_sources") or state.get("web_sources") or state.get("used_chat_ids"))
    confidence = bounded_confidence(parsed.get("confidence"))
    missing_evidence = bounded_string_list(parsed.get("missing_evidence"))
    recommended_next_steps = bounded_string_list(parsed.get("recommended_next_steps"))
    return {
        "sufficient": sufficient,
        "confidence": confidence,
        "missing_evidence": missing_evidence,
        "citation_risk": risk_level(parsed.get("citation_risk")),
        "contradiction_risk": risk_level(parsed.get("contradiction_risk")),
        "recommended_next_steps": recommended_next_steps,
        "reason": compact_preview(str(parsed.get("reason") or ""), limit=500),
    }


def normalize_replanner_execution_plan(
    parsed: Dict[str, Any],
    *,
    use_web_search: bool,
    allowed_tool_ids: Any,
) -> Dict[str, Any]:
    normalization_notes: List[str] = []
    raw_steps = parsed.get("execution_plan") or parsed.get("steps") or []
    steps: List[str] = []
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if isinstance(step, str):
                node = step
            elif isinstance(step, dict):
                node = step.get("node") or step.get("worker") or step.get("id")
            else:
                continue
            if node in WORKER_NODE_ORDER and node not in steps:
                steps.append(node)
    if not use_web_search and WorkflowNodeType.WEB_WORKER.value in steps:
        steps = [step for step in steps if step != WorkflowNodeType.WEB_WORKER.value]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    allowed_ids = set(allowed_tool_ids if isinstance(allowed_tool_ids, list) else [])
    if ToolContractId.LIVE_WEB_RECON.value not in allowed_ids and WorkflowNodeType.WEB_WORKER.value in steps:
        steps = [step for step in steps if step != WorkflowNodeType.WEB_WORKER.value]
        normalization_notes.append("web_worker_removed_when_tool_disallowed")
    return {
        "execution_plan": ordered_plan_steps(steps),
        "reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "normalization_notes": normalization_notes,
    }

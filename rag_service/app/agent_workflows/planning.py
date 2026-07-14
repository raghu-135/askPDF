from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agent_workflows.trace import compact_preview


WORKER_NODE_ORDER = [
    "retrieval_worker",
    "memory_worker",
    "timeline_worker",
    "web_worker",
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
        required.append("timeline_worker")
    if MEMORY_PLAN_RE.search(text) and "memory_worker" not in required and "timeline_worker" not in required:
        required.append("memory_worker")
    if DOCUMENT_PLAN_RE.search(text) and "retrieval_worker" not in required:
        required.append("retrieval_worker")
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
    allowed_routes = {"execute", "direct", "clarify"}
    route = parsed.get("route") if parsed.get("route") in allowed_routes else "execute"
    required_steps = infer_required_plan_steps(question)
    normalization_notes: List[str] = []
    if route == "direct" and required_steps:
        route = "execute"
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
    if not use_web_search and "web_worker" in steps:
        steps = [step for step in steps if step != "web_worker"]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    if route == "execute":
        for required_step in required_steps:
            if required_step not in steps:
                steps.append(required_step)
    if route == "execute" and not steps:
        steps = ["retrieval_worker"]
        normalization_notes.append("empty_execute_plan_defaulted_to_retrieval_worker")
    steps = ordered_plan_steps(steps)
    if route != "execute":
        steps = []
    clarification_options = parsed.get("clarification_options")
    if route == "clarify" and not isinstance(clarification_options, list):
        clarification_options = fallback_clarification_options()
    return {
        "route": route,
        "route_reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "execution_plan": steps,
        "clarification_options": clarification_options if route == "clarify" else None,
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
    if not use_web_search and "web_worker" in steps:
        steps = [step for step in steps if step != "web_worker"]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    allowed_ids = set(allowed_tool_ids if isinstance(allowed_tool_ids, list) else [])
    if "live_web_recon" not in allowed_ids and "web_worker" in steps:
        steps = [step for step in steps if step != "web_worker"]
        normalization_notes.append("web_worker_removed_when_tool_disallowed")
    return {
        "execution_plan": ordered_plan_steps(steps),
        "reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "normalization_notes": normalization_notes,
    }

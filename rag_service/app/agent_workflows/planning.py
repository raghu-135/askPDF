from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agent_workflows.enums import PlannerRiskLevel, PlannerRoute, PLANNER_ROUTES, ToolContractId, WorkflowNodeType
from app.agent_workflows.trace import compact_preview


WORKER_NODE_ORDER = [
    WorkflowNodeType.RETRIEVAL_WORKER.value,
    WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value,
    WorkflowNodeType.DURABLE_MEMORY_WORKER.value,
    WorkflowNodeType.THREAD_EVENTS_WORKER.value,
    WorkflowNodeType.WEB_WORKER.value,
]

WORKER_NODE_TYPES = set(WORKER_NODE_ORDER)


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

LONG_TERM_MEMORY_PLAN_RE = re.compile(
    r"\b("
    r"remembered|memory|preference|profile|project\s+(memory|context|fact)|"
    r"what\s+(do|did)\s+you\s+know\s+about\s+me|saved\s+(memory|context)"
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

META_CLARIFICATION_RE = re.compile(
    r"^(?:"
    r"(?:did|do|does|are|were|can|could|would|should)\s+you\s+(?:mean|want|intend|refer|ask)"
    r"|(?:do|did|would|should)\s+i\s+(?:mean|want|intend|refer|ask)"
    r"|(?:is|was)\s+your\s+(?:question|intent|request)"
    r"|(?:is|was|does|did)\s+(?:the\s+)?user\b"
    r")",
    re.IGNORECASE,
)


def infer_required_plan_steps(question: Optional[str]) -> List[str]:
    """Return worker nodes that should be present for obvious query intent cues."""

    text = str(question or "")
    required: List[str] = []
    if TEMPORAL_PLAN_RE.search(text):
        required.append(WorkflowNodeType.THREAD_EVENTS_WORKER.value)
    if (
        MEMORY_PLAN_RE.search(text)
        and WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value not in required
        and WorkflowNodeType.THREAD_EVENTS_WORKER.value not in required
    ):
        required.append(WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value)
    if LONG_TERM_MEMORY_PLAN_RE.search(text) and WorkflowNodeType.DURABLE_MEMORY_WORKER.value not in required:
        required.append(WorkflowNodeType.DURABLE_MEMORY_WORKER.value)
    if DOCUMENT_PLAN_RE.search(text) and WorkflowNodeType.RETRIEVAL_WORKER.value not in required:
        required.append(WorkflowNodeType.RETRIEVAL_WORKER.value)
    return required


def worker_nodes_from_spec(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    workers: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str) or node_type not in WORKER_NODE_TYPES:
            continue
        workers.append({
            "id": node_id,
            "type": node_type,
            "label": str(node.get("label") or node_id),
            "tool_contract_ids": [str(item) for item in node.get("tool_contract_ids") or [] if isinstance(item, str)],
        })
    return workers


def _default_worker_nodes() -> List[Dict[str, Any]]:
    return [{"id": node_type, "type": node_type, "label": node_type, "tool_contract_ids": []} for node_type in WORKER_NODE_ORDER]


def _worker_nodes_by_type(worker_nodes: Any) -> Dict[str, List[Dict[str, Any]]]:
    available = _default_worker_nodes() if worker_nodes is None else worker_nodes if isinstance(worker_nodes, list) else []
    by_type: Dict[str, List[Dict[str, Any]]] = {node_type: [] for node_type in WORKER_NODE_ORDER}
    for node in available:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        if isinstance(node_id, str) and node_id and isinstance(node_type, str) and node_type in WORKER_NODE_TYPES:
            by_type.setdefault(node_type, []).append(node)
    return by_type


def available_worker_node_ids(worker_nodes: Any) -> List[str]:
    by_type = _worker_nodes_by_type(worker_nodes)
    ids: List[str] = []
    for node_type in WORKER_NODE_ORDER:
        ids.extend(str(node["id"]) for node in by_type.get(node_type, []))
    return ids


def worker_decision_contract_errors(
    parsed: Dict[str, Any],
    *,
    worker_nodes: Any,
    use_web_search: bool,
    require_route: bool = True,
) -> List[str]:
    """Validate exhaustive typed worker selection without making semantic choices."""

    available = available_worker_node_ids(worker_nodes)
    decisions = parsed.get("worker_decisions")
    errors: List[str] = []
    if require_route and parsed.get("route") not in PLANNER_ROUTES:
        errors.append("route must be one of execute, direct, or clarify")
    if not isinstance(decisions, list):
        return [*errors, "worker_decisions must be an array containing one decision for every available worker"]

    seen: set[str] = set()
    selected = 0
    by_type = _worker_nodes_by_type(worker_nodes)
    web_ids = {str(node["id"]) for node in by_type.get(WorkflowNodeType.WEB_WORKER.value, [])}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"worker_decisions[{index}] must be an object")
            continue
        worker_id = decision.get("worker_node_id")
        if worker_id not in available:
            errors.append(f"worker_decisions[{index}].worker_node_id must be an exact available worker id")
            continue
        if worker_id in seen:
            errors.append(f"worker_decisions contains duplicate worker id {worker_id}")
        seen.add(worker_id)
        if not isinstance(decision.get("selected"), bool):
            errors.append(f"worker decision for {worker_id} must contain boolean selected")
            continue
        if decision["selected"]:
            selected += 1
            if not str(decision.get("query") or "").strip():
                errors.append(f"selected worker {worker_id} must contain a non-empty query")
            if worker_id in web_ids and not use_web_search:
                errors.append(f"worker {worker_id} cannot be selected while live web search is disabled")
        if not str(decision.get("reason") or "").strip():
            errors.append(f"worker decision for {worker_id} must contain a reason")

    missing = [worker_id for worker_id in available if worker_id not in seen]
    if missing:
        errors.append("worker_decisions is missing: " + ", ".join(missing))
    route = parsed.get("route")
    if require_route and route == PlannerRoute.EXECUTE.value and selected == 0:
        errors.append("execute route must select at least one worker")
    if require_route and route in {PlannerRoute.DIRECT.value, PlannerRoute.CLARIFY.value} and selected:
        errors.append(f"{route} route must not select workers")
    return errors


def selected_worker_decisions(parsed: Dict[str, Any], *, worker_nodes: Any) -> List[Dict[str, Any]]:
    decisions = parsed.get("worker_decisions")
    if not isinstance(decisions, list):
        return []
    available = set(available_worker_node_ids(worker_nodes))
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("selected") is not True:
            continue
        worker_id = decision.get("worker_node_id")
        if worker_id not in available or worker_id in seen:
            continue
        seen.add(worker_id)
        selected.append(decision)
    return selected


def worker_decisions_need_coverage_review(parsed: Dict[str, Any], *, require_route: bool = True) -> bool:
    """Review every valid executable plan once; the review remains model-semantic."""

    if require_route and parsed.get("route") != PlannerRoute.EXECUTE.value:
        return False
    decisions = parsed.get("worker_decisions")
    if not isinstance(decisions, list):
        return False
    return bool(decisions)


def _resolve_worker_reference(value: Any, *, by_id: Dict[str, Dict[str, Any]], by_type: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    node = None
    if isinstance(value, str):
        node = value
    elif isinstance(value, dict):
        node = value.get("node") or value.get("worker") or value.get("id")
    if not isinstance(node, str) or not node:
        return None
    if node in by_id:
        return node
    matches = by_type.get(node) or []
    if len(matches) == 1:
        return str(matches[0]["id"])
    return None


def _resolve_required_worker_type(node_type: str, by_type: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    matches = by_type.get(node_type) or []
    if len(matches) == 1:
        return str(matches[0]["id"])
    return None


def ordered_plan_steps(steps: List[str], worker_nodes: Any = None) -> List[str]:
    ordered_ids = set(steps)
    return [node_id for node_id in available_worker_node_ids(worker_nodes) if node_id in ordered_ids]


def fallback_clarification_options(question: Optional[str] = None) -> List[str]:
    original = compact_preview(str(question or "my original question").strip(), limit=180).rstrip(" ?!.,;:")
    return [
        f'Based on the uploaded documents, what is the answer to "{original}"?',
        f'Based on our earlier conversation, what is the answer to "{original}"?',
        f'Based on the thread timeline, what is the answer to "{original}"?',
    ]


def normalize_clarification_options(value: Any, *, limit: int = 4, chars: int = 240) -> List[str]:
    """Return distinct, bounded questions that are not meta-framed clarification prompts."""

    normalized: List[str] = []
    seen: set[str] = set()
    for text in bounded_string_list(value, limit=limit, chars=chars):
        candidate = text.strip()
        key = candidate.casefold()
        if not candidate or key in seen or META_CLARIFICATION_RE.match(candidate):
            continue
        seen.add(key)
        normalized.append(candidate)
    return normalized


def normalize_execution_plan(
    parsed: Dict[str, Any],
    *,
    use_web_search: bool,
    question: Optional[str] = None,
    bypass_clarification: bool = False,
    worker_nodes: Any = None,
) -> Dict[str, Any]:
    route = parsed.get("route") if parsed.get("route") in PLANNER_ROUTES else PlannerRoute.EXECUTE.value
    has_worker_decisions = isinstance(parsed.get("worker_decisions"), list)
    required_steps = [] if has_worker_decisions else infer_required_plan_steps(question)
    normalization_notes: List[str] = []
    if route == PlannerRoute.CLARIFY.value and bypass_clarification:
        route = PlannerRoute.DIRECT.value
        normalization_notes.append("clarify_route_bypassed_by_user")
    if route == PlannerRoute.DIRECT.value and required_steps:
        route = PlannerRoute.EXECUTE.value
        normalization_notes.append("direct_route_clamped_to_execute")
    selected_decisions = selected_worker_decisions(parsed, worker_nodes=worker_nodes)
    raw_steps = (
        [decision.get("worker_node_id") for decision in selected_decisions]
        if has_worker_decisions
        else parsed.get("execution_plan") or parsed.get("steps") or []
    )
    steps: List[str] = []
    by_type = _worker_nodes_by_type(worker_nodes)
    by_id = {str(node["id"]): node for nodes in by_type.values() for node in nodes}
    if isinstance(raw_steps, list):
        for step in raw_steps:
            node_id = _resolve_worker_reference(step, by_id=by_id, by_type=by_type)
            if node_id and node_id not in steps:
                steps.append(node_id)
    web_node_ids = {str(node["id"]) for node in by_type.get(WorkflowNodeType.WEB_WORKER.value, [])}
    if not use_web_search and any(step in web_node_ids for step in steps):
        steps = [step for step in steps if step not in web_node_ids]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    if route == PlannerRoute.EXECUTE.value:
        for required_step in required_steps:
            required_node_id = _resolve_required_worker_type(required_step, by_type)
            if required_node_id and required_node_id not in steps:
                steps.append(required_node_id)
            elif required_node_id is None:
                normalization_notes.append(f"required_{required_step}_ambiguous_or_unavailable")
    if route == PlannerRoute.EXECUTE.value and not steps:
        default_node_id = _resolve_required_worker_type(WorkflowNodeType.RETRIEVAL_WORKER.value, by_type)
        steps = [default_node_id] if default_node_id else []
        normalization_notes.append("empty_execute_plan_defaulted_to_retrieval_worker")
    steps = ordered_plan_steps(steps, worker_nodes)
    if route != PlannerRoute.EXECUTE.value:
        steps = []
    clarification_options = parsed.get("clarification_options")
    if route == PlannerRoute.CLARIFY.value:
        clarification_options = normalize_clarification_options(clarification_options)
        if len(clarification_options) < 2:
            clarification_options = fallback_clarification_options(question)
    return {
        "route": route,
        "route_reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "execution_plan": steps,
        "clarification_options": clarification_options if route == PlannerRoute.CLARIFY.value else None,
        "normalization_notes": normalization_notes,
    }


def risk_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    allowed = {level.value for level in PlannerRiskLevel}
    return text if text in allowed else PlannerRiskLevel.MEDIUM.value


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
    worker_nodes: Any = None,
) -> Dict[str, Any]:
    normalization_notes: List[str] = []
    selected_decisions = selected_worker_decisions(parsed, worker_nodes=worker_nodes)
    raw_steps = (
        [decision.get("worker_node_id") for decision in selected_decisions]
        if isinstance(parsed.get("worker_decisions"), list)
        else parsed.get("execution_plan") or parsed.get("steps") or []
    )
    steps: List[str] = []
    by_type = _worker_nodes_by_type(worker_nodes)
    by_id = {str(node["id"]): node for nodes in by_type.values() for node in nodes}
    if isinstance(raw_steps, list):
        for step in raw_steps:
            node_id = _resolve_worker_reference(step, by_id=by_id, by_type=by_type)
            if node_id and node_id not in steps:
                steps.append(node_id)
    web_node_ids = {str(node["id"]) for node in by_type.get(WorkflowNodeType.WEB_WORKER.value, [])}
    if not use_web_search and any(step in web_node_ids for step in steps):
        steps = [step for step in steps if step not in web_node_ids]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    allowed_ids = set(allowed_tool_ids if isinstance(allowed_tool_ids, list) else [])
    if ToolContractId.LIVE_WEB_RECON.value not in allowed_ids and any(step in web_node_ids for step in steps):
        steps = [step for step in steps if step not in web_node_ids]
        normalization_notes.append("web_worker_removed_when_tool_disallowed")
    return {
        "execution_plan": ordered_plan_steps(steps, worker_nodes),
        "reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "normalization_notes": normalization_notes,
    }

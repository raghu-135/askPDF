from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, Mapping


CORRECTIVE_WORKFLOW_ID = "corrective_self_rag_agent"
CORRECTIVE_SOURCE_STRATEGIES = (
    "focused_document",
    "cross_document",
    "conversation",
    "timeline",
    "memory",
    "web",
)
CORRECTIVE_SOURCE_STRATEGY_RANK = {
    strategy: index for index, strategy in enumerate(CORRECTIVE_SOURCE_STRATEGIES)
}


def stable_corrective_identity(kind: str, **fields: Any) -> str:
    payload = {"kind": kind, **fields}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:24]


class CorrectiveEventName:
    WAVE_STARTED = "corrective.wave_started"
    WAVE_COMPLETED = "corrective.wave_completed"
    RETRIEVAL_GRADED = "corrective.retrieval_graded"
    DECISION = "corrective.decision"
    QUERY_REWRITE = "corrective.query_rewrite"
    SOURCE_EXPANSION = "corrective.source_expansion"
    SUPPORT_VERIFIED = "corrective.support_verified"
    CITATION_VIOLATION = "corrective.citation_violation"
    CONTRADICTION = "corrective.contradiction"
    UNRESOLVED_GAP = "corrective.unresolved_gap"
    BUDGET_EXHAUSTED = "corrective.budget_exhausted"


def corrective_source_strategy(worker_type: str, *, file_hash: str = "") -> str:
    if worker_type == "retrieval_worker":
        return "focused_document" if file_hash else "cross_document"
    return {
        "thread_conversation_history_worker": "conversation",
        "thread_events_worker": "timeline",
        "durable_memory_worker": "memory",
        "web_worker": "web",
    }.get(worker_type, "")

CORRECTIVE_POLICY_FIELDS: Dict[str, Dict[str, Any]] = {
    "minimum_relevance_confidence": {"type": "number", "default": 0.65, "minimum": 0.0, "maximum": 1.0, "step": 0.05, "label": "Minimum relevance confidence"},
    "minimum_supported_claim_ratio": {"type": "number", "default": 1.0, "minimum": 0.8, "maximum": 1.0, "step": 0.05, "label": "Minimum supported claim ratio"},
    "minimum_usefulness_score": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5, "step": 1, "label": "Minimum usefulness score"},
    "max_corrective_waves": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3, "step": 1, "label": "Maximum corrective waves"},
    "max_total_work_items": {"type": "integer", "default": 8, "minimum": 2, "maximum": 16, "step": 1, "label": "Maximum retrieval work items"},
    "max_total_tool_attempts": {"type": "integer", "default": 12, "minimum": 2, "maximum": 24, "step": 1, "label": "Maximum tool attempts"},
    "max_answer_revisions": {"type": "integer", "default": 1, "minimum": 0, "maximum": 2, "step": 1, "label": "Maximum answer revisions"},
    "allow_web_fallback": {"type": "boolean", "default": True, "label": "Allow policy-approved web fallback"},
    "memory_evidence_mode": {"type": "enum", "default": "policy_scoped", "values": ["disabled", "policy_scoped"], "label": "Memory evidence"},
    "insufficient_evidence_mode": {"type": "enum", "default": "verified_only", "values": ["verified_only"], "label": "Insufficient evidence behavior"},
}

DEFAULT_CORRECTIVE_POLICY = {
    key: descriptor["default"] for key, descriptor in CORRECTIVE_POLICY_FIELDS.items()
}


def corrective_policy_catalog() -> Dict[str, Any]:
    return {"defaults": deepcopy(DEFAULT_CORRECTIVE_POLICY), "fields": deepcopy(CORRECTIVE_POLICY_FIELDS)}


def normalized_corrective_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    policy = deepcopy(DEFAULT_CORRECTIVE_POLICY)
    for key, descriptor in CORRECTIVE_POLICY_FIELDS.items():
        candidate = raw.get(key, policy[key])
        if descriptor["type"] == "boolean":
            if isinstance(candidate, bool):
                policy[key] = candidate
        elif descriptor["type"] == "integer":
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                policy[key] = max(descriptor["minimum"], min(candidate, descriptor["maximum"]))
        elif descriptor["type"] == "number":
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                policy[key] = max(descriptor["minimum"], min(float(candidate), descriptor["maximum"]))
        elif candidate in descriptor["values"]:
            policy[key] = candidate
    policy["max_total_tool_attempts"] = max(policy["max_total_work_items"], policy["max_total_tool_attempts"])
    return policy


def collect_corrective_policy_errors(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["corrective_policy must be an object"]
    errors: list[str] = []
    unknown = sorted(set(value) - set(CORRECTIVE_POLICY_FIELDS))
    if unknown:
        errors.append(f"corrective_policy has unknown keys: {', '.join(unknown)}")
    for key, descriptor in CORRECTIVE_POLICY_FIELDS.items():
        if key not in value:
            continue
        candidate = value[key]
        kind = descriptor["type"]
        if kind == "boolean" and not isinstance(candidate, bool):
            errors.append(f"corrective_policy.{key} must be a boolean")
        elif kind == "integer" and (not isinstance(candidate, int) or isinstance(candidate, bool)):
            errors.append(f"corrective_policy.{key} must be an integer")
        elif kind == "number" and (not isinstance(candidate, (int, float)) or isinstance(candidate, bool)):
            errors.append(f"corrective_policy.{key} must be a number")
        elif kind in {"integer", "number"} and isinstance(candidate, (int, float)):
            if candidate < descriptor["minimum"] or candidate > descriptor["maximum"]:
                errors.append(f"corrective_policy.{key} must be between {descriptor['minimum']} and {descriptor['maximum']}")
        elif kind == "enum" and candidate not in descriptor["values"]:
            errors.append(f"corrective_policy.{key} must be one of: {', '.join(descriptor['values'])}")
    work_items = value.get("max_total_work_items", DEFAULT_CORRECTIVE_POLICY["max_total_work_items"])
    attempts = value.get("max_total_tool_attempts", DEFAULT_CORRECTIVE_POLICY["max_total_tool_attempts"])
    if isinstance(work_items, int) and isinstance(attempts, int) and attempts < work_items:
        errors.append("corrective_policy.max_total_tool_attempts cannot be less than max_total_work_items")
    return errors

"""Context-aware durable-memory retrieval budgets and deterministic applicability."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable

from app.models.llm_server_client import CHARS_PER_TOKEN
from app.models.memory_limits import (
    DEFAULT_MEMORY_RELATIVE_SCORE_RATIO,
    MAX_MEMORY_CONTEXT_CHARS,
)
from app.models.memory_tools import normalize_memory_attributes


STANDARD_MEMORY_RATIO = 0.06
EXPANDED_MEMORY_RATIO = 0.12
DEFAULT_MEMORY_SCORE_FLOOR = 0.20
DEFAULT_RELATIVE_SCORE_RATIO = DEFAULT_MEMORY_RELATIVE_SCORE_RATIO
NEAR_DUPLICATE_TOKEN_SIMILARITY = 0.85
AVERAGE_MEMORY_CHARS = 400


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def compute_memory_retrieval_budget(context_window: int, *, expanded: bool = False) -> Dict[str, int | float]:
    usable_chars = int(max(256, int(context_window)) * 0.80 * CHARS_PER_TOKEN)
    ratio = EXPANDED_MEMORY_RATIO if expanded else STANDARD_MEMORY_RATIO
    minimum, maximum, max_candidates = (
        (1600, MAX_MEMORY_CONTEXT_CHARS, 40)
        if expanded else (800, 8000, 20)
    )
    char_budget = _clamp(int(usable_chars * ratio), minimum, maximum)
    candidate_limit = _clamp(math.ceil(char_budget / AVERAGE_MEMORY_CHARS) * 2, 5, max_candidates)
    return {
        "usable_chars": usable_chars,
        "ratio": ratio,
        "char_budget": char_budget,
        "candidate_limit": candidate_limit,
        "per_memory_min_chars": 200,
        "per_memory_max_chars": 1000,
    }


def memory_score_floor(embedding_model: str) -> float:
    """Resolve a centrally configurable score floor with optional model overrides."""
    raw_default = os.getenv("MEMORY_RETRIEVAL_MIN_SCORE", str(DEFAULT_MEMORY_SCORE_FLOOR))
    try:
        default = max(0.0, float(raw_default))
    except (TypeError, ValueError):
        default = DEFAULT_MEMORY_SCORE_FLOOR
    try:
        overrides = json.loads(os.getenv("MEMORY_RETRIEVAL_MODEL_SCORE_FLOORS", "{}"))
    except (TypeError, json.JSONDecodeError):
        overrides = {}
    if isinstance(overrides, dict) and embedding_model in overrides:
        try:
            return max(0.0, float(overrides[embedding_model]))
        except (TypeError, ValueError):
            pass
    return default


_APPLICABILITY_TERMS = {
    "writing": {"write", "rewrite", "draft", "tone", "style", "email", "summarize", "summary"},
    "code": {"code", "python", "typescript", "javascript", "java", "sql", "api", "function", "class", "debug"},
    "research": {"research", "source", "citation", "evidence", "study", "paper", "compare", "analyze"},
    "project": {"project", "architecture", "repository", "repo", "workflow", "implementation", "plan"},
}


def normalized_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def memory_is_applicable(attributes: Any, query: str) -> bool:
    normalized = normalize_memory_attributes(attributes)
    applicability = set(normalized["applicability"])
    if applicability & {"all_answers", "task_specific"}:
        return True
    query_tokens = normalized_tokens(query)
    return any(query_tokens & _APPLICABILITY_TERMS.get(item, set()) for item in applicability)


def token_similarity(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compact_memory_content(content: str, limit: int) -> str:
    text = " ".join(str(content or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def pack_memory_results(memories: Iterable[Dict[str, Any]], char_budget: int) -> tuple[list[Dict[str, Any]], int]:
    candidates = list(memories)
    if not candidates or char_budget <= 0:
        return [], 0
    expected_count = max(1, min(len(candidates), math.ceil(char_budget / AVERAGE_MEMORY_CHARS)))
    per_memory = _clamp(char_budget // expected_count, 200, 1000)
    packed: list[Dict[str, Any]] = []
    used = 0
    for memory in candidates:
        remaining = char_budget - used
        if remaining < 40:
            break
        excerpt = compact_memory_content(str(memory.get("content") or ""), min(per_memory, remaining))
        if not excerpt:
            continue
        packed.append({**memory, "excerpt": excerpt})
        used += len(excerpt)
    return packed, used

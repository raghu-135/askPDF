"""Central backend limits and proportional context budgets for memory curation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from app.models.llm_server_client import AVG_CHUNK_CHARS, CHARS_PER_TOKEN
from app.models.memory_limits import MAX_MEMORY_QUERY_CHARS, MAX_MEMORY_ROWS


MEMORY_MANAGER_CONTEXT_USABLE_RATIO = 0.70
MEMORY_MANAGER_TRANSCRIPT_RATIO = 0.15
MEMORY_MANAGER_REVIEW_CONTEXT_RATIO = 0.40
MEMORY_MANAGER_MEMORY_CONTEXT_RATIO = 0.30

MIN_MEMORY_MANAGER_TRANSCRIPT_CHARS = 1000
MIN_MEMORY_MANAGER_REVIEW_CONTEXT_CHARS = 2000
MIN_MEMORY_MANAGER_MEMORY_CONTEXT_CHARS = 1000

MAX_MEMORY_MANAGER_MEMORY_RESULTS = MAX_MEMORY_ROWS
MAX_MEMORY_MANAGER_REQUEST_MESSAGES = 500
MAX_MEMORY_SEARCH_QUERY_CHARS = MAX_MEMORY_QUERY_CHARS
MAX_REVIEW_FETCH_ROWS = MAX_MEMORY_ROWS


def compute_memory_manager_input_budget(context_window: int) -> Dict[str, int]:
    """Allocate curator inputs while preserving response and framework headroom."""

    usable = int(max(256, context_window) * MEMORY_MANAGER_CONTEXT_USABLE_RATIO * CHARS_PER_TOKEN)
    memory_chars = max(
        MIN_MEMORY_MANAGER_MEMORY_CONTEXT_CHARS,
        int(usable * MEMORY_MANAGER_MEMORY_CONTEXT_RATIO),
    )
    return {
        "transcript_chars": max(
            MIN_MEMORY_MANAGER_TRANSCRIPT_CHARS,
            int(usable * MEMORY_MANAGER_TRANSCRIPT_RATIO),
        ),
        "review_context_chars": max(
            MIN_MEMORY_MANAGER_REVIEW_CONTEXT_CHARS,
            int(usable * MEMORY_MANAGER_REVIEW_CONTEXT_RATIO),
        ),
        "memory_context_chars": memory_chars,
        "memory_result_limit": min(
            MAX_MEMORY_MANAGER_MEMORY_RESULTS,
            max(3, memory_chars // AVG_CHUNK_CHARS),
        ),
    }


def bound_memory_manager_transcript(
    messages: Sequence[Dict[str, Any]],
    *,
    context_window: int,
) -> List[Dict[str, Any]]:
    """Keep the newest complete messages that fit the backend prompt budget."""

    budget = compute_memory_manager_input_budget(context_window)["transcript_chars"]
    selected: List[Dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        size = len(json.dumps(message, ensure_ascii=True))
        if selected and used + size > budget:
            break
        selected.append(message)
        used += size
    return list(reversed(selected))

"""Shared durable-memory limits with one meaning at every call site."""

from runtime_protocol.context_budget import CHARS_PER_TOKEN, load_context_budget_settings

# Administrative request/result and repository page bound.
MAX_MEMORY_ROWS = 500
# Maximum user/model query or proposed-memory text accepted at an API boundary.
MAX_MEMORY_QUERY_CHARS = 12_000
# Largest context window a request may declare. Prompt packs are ratios of the
# window actually selected; this only bounds that declaration.
MAX_CONTEXT_WINDOW_TOKENS = 2_000_000
# Default relative cutoff applied after absolute retrieval scoring.
DEFAULT_MEMORY_RELATIVE_SCORE_RATIO = 0.60


def max_memory_context_chars() -> int:
    """Largest context-window share allowed for one durable-memory pack argument."""
    settings = load_context_budget_settings()
    return max(
        1,
        int(
            int(MAX_CONTEXT_WINDOW_TOKENS * settings.usable_ratio * CHARS_PER_TOKEN)
            * settings.memory_expanded_ratio
        ),
    )

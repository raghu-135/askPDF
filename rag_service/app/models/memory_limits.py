"""Shared durable-memory limits with one meaning at every call site."""

# Administrative request/result and repository page bound.
MAX_MEMORY_ROWS = 500
# Maximum user/model query or proposed-memory text accepted at an API boundary.
MAX_MEMORY_QUERY_CHARS = 12_000
from runtime_protocol.context_budget import CHARS_PER_TOKEN, default_context_budget_settings

# Largest context window a request may declare. Prompt packs are ratios of the
# window actually selected; this only bounds that declaration.
MAX_CONTEXT_WINDOW_TOKENS = 2_000_000
# Acceptance ceiling for one durable-memory pack argument: the expanded
# retrieval share of MAX_CONTEXT_WINDOW_TOKENS. Pack size is
# compute_memory_retrieval_budget, not this constant.
_budget_defaults = default_context_budget_settings()
MAX_MEMORY_CONTEXT_CHARS = max(
    1,
    int(
        int(MAX_CONTEXT_WINDOW_TOKENS * _budget_defaults.usable_ratio * CHARS_PER_TOKEN)
        * _budget_defaults.memory_expanded_ratio
    ),
)
# Default relative cutoff applied after absolute retrieval scoring.
DEFAULT_MEMORY_RELATIVE_SCORE_RATIO = 0.60

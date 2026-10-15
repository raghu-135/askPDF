"""Shared durable-memory limits with one meaning at every call site."""

# Administrative request/result and repository page bound.
MAX_MEMORY_ROWS = 500
# Maximum user/model query or proposed-memory text accepted at an API boundary.
MAX_MEMORY_QUERY_CHARS = 12_000
# Largest context window a request may declare. Prompt packs are ratios of the
# window actually selected; this only bounds that declaration.
MAX_CONTEXT_WINDOW_TOKENS = 2_000_000
# Acceptance ceiling for one durable-memory pack argument: the expanded
# retrieval share of MAX_CONTEXT_WINDOW_TOKENS. Pack size is
# compute_memory_retrieval_budget, not this constant.
_USABLE_CONTEXT_RATIO = 0.80
_CHARS_PER_TOKEN = 4
_EXPANDED_MEMORY_RATIO = 0.12
MAX_MEMORY_CONTEXT_CHARS = max(
    1,
    int(int(MAX_CONTEXT_WINDOW_TOKENS * _USABLE_CONTEXT_RATIO * _CHARS_PER_TOKEN) * _EXPANDED_MEMORY_RATIO),
)
# Default relative cutoff applied after absolute retrieval scoring.
DEFAULT_MEMORY_RELATIVE_SCORE_RATIO = 0.60

"""Shared durable-memory limits with one meaning at every call site."""

# Administrative request/result and repository page bound.
MAX_MEMORY_ROWS = 500
# Maximum user/model query or proposed-memory text accepted at an API boundary.
MAX_MEMORY_QUERY_CHARS = 12_000
# Maximum durable-memory text packed into one prompt context.
MAX_MEMORY_CONTEXT_CHARS = 16_000
# Default relative cutoff applied after absolute retrieval scoring.
DEFAULT_MEMORY_RELATIVE_SCORE_RATIO = 0.60

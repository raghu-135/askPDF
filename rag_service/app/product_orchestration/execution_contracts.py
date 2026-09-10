"""Canonical execution-policy constants shared by Agent v2 workflows."""

from __future__ import annotations

from typing import Final


PREFETCH_MODE_EVIDENCE: Final = "evidence"
PREFETCH_MODE_ROUTING: Final = "routing"
PREFETCH_MODES: Final = frozenset({PREFETCH_MODE_EVIDENCE, PREFETCH_MODE_ROUTING})
DEFAULT_PREFETCH_MODE: Final = PREFETCH_MODE_EVIDENCE

MAX_ANSWER_REVISIONS: Final = 1
MAX_ANSWER_QUALITY_ISSUES: Final = 8

WORKER_TERMINAL_STATUSES: Final = frozenset(
    {"completed", "skipped", "failed", "timed_out", "cancelled"}
)

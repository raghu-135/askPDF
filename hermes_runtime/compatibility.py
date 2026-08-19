"""Pinned Hermes API compatibility contract used by the gateway and tests."""

HERMES_REPOSITORY = "https://github.com/NousResearch/hermes-agent"
HERMES_REVISION = "bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894"
HERMES_RUN_EVENTS = frozenset({
    "message.delta",
    "tool.started",
    "tool.completed",
    "reasoning.available",
    "approval.request",
    "approval.responded",
    "run.steered",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "subagent.start",
    "subagent.complete",
})
HERMES_TERMINAL_EVENTS = frozenset({"run.completed", "run.failed", "run.cancelled"})


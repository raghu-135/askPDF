"""Dependency-free authority for the pinned askPDF/Hermes contract."""

HERMES_REPOSITORY = "https://github.com/NousResearch/hermes-agent"
HERMES_REVISION = "bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894"
HERMES_CONFIG_SCHEMA_VERSION = 37
HERMES_MIN_CONTEXT_LENGTH = 64_000
HERMES_CHAT_PROVIDER = "custom"
HERMES_OFFLINE_PROFILE = "askpdf-deep-offline"
HERMES_EXTERNAL_PROFILE = "askpdf-deep-external"
HERMES_PROFILE_NAMES = frozenset({HERMES_OFFLINE_PROFILE, HERMES_EXTERNAL_PROFILE})
HERMES_RUN_EVENTS = frozenset({
    "message.delta", "tool.started", "tool.completed", "reasoning.available",
    "approval.request", "approval.responded", "run.completed",
    "run.failed", "run.cancelled", "subagent.start", "subagent.complete",
})
HERMES_TERMINAL_EVENTS = frozenset({"run.completed", "run.failed", "run.cancelled"})
HERMES_APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})
HERMES_CONTROLS = frozenset({"stop", "approval"})


def validate_hermes_context_length(context_length: int) -> None:
    if context_length < HERMES_MIN_CONTEXT_LENGTH:
        raise ValueError(f"Hermes context length must be at least {HERMES_MIN_CONTEXT_LENGTH}")

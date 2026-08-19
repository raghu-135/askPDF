from __future__ import annotations

import copy

import pytest

from app.runtime.hermes_profile import resolve_hermes_profile


def _spec() -> dict:
    return {
        "definition_version": 1,
        "config": {
            "system_prompt": "Use evidence.",
            "mcp_server": "askpdf",
            "allowed_tool_ids": ["thread_conversation_history", "document_evidence", "document_evidence"],
            "model": "model-a",
            "provider": "provider-a",
            "skills": ["summarize", "research", "summarize"],
            "allow_subagents": True,
            "allow_persistent_memory": False,
        },
    }


def test_profile_resolution_is_deterministic_and_canonical() -> None:
    first = resolve_hermes_profile(_spec())
    second = resolve_hermes_profile(copy.deepcopy(_spec()))
    assert first == second
    assert first["mcp"]["allowed_tool_ids"] == ["document_evidence", "thread_conversation_history"]
    assert first["skills"]["enabled"] == ["research", "summarize"]
    assert first["delegation"] == {"enabled": True}
    assert len(first["profile_id"]) == 64


@pytest.mark.parametrize("secret", ["api_key", "provider_token", "password", "credentials"])
def test_profile_rejects_persisted_credentials_at_any_depth(secret: str) -> None:
    spec = _spec()
    spec["config"]["provider_config"] = {secret: "do-not-store"}
    with pytest.raises(ValueError, match="cannot persist credentials"):
        resolve_hermes_profile(spec)


def test_profile_version_is_required() -> None:
    spec = _spec()
    spec.pop("definition_version")
    with pytest.raises(ValueError, match="definition_version"):
        resolve_hermes_profile(spec)

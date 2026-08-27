from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.runtime.hermes_profile import (
    HERMES_EXTERNAL_PROFILE,
    HERMES_OFFLINE_PROFILE,
    resolve_hermes_profile,
)


def _spec() -> dict:
    return {
        "config": {
            "system_prompt": "Use evidence.",
            "mcp_server": "askpdf",
            "allowed_tool_ids": ["search_thread_conversation_history", "search_documents", "search_documents"],
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
    assert first["mcp"]["allowed_tool_ids"] == ["search_documents", "search_thread_conversation_history"]
    assert first["mcp"]["runtime_profile"] == HERMES_OFFLINE_PROFILE
    assert first["skills"]["enabled"] == ["research", "summarize"]
    assert first["delegation"] == {"enabled": True}
    assert len(first["profile_id"]) == 64


@pytest.mark.parametrize("secret", ["api_key", "provider_token", "password", "credentials"])
def test_profile_rejects_persisted_credentials_at_any_depth(secret: str) -> None:
    spec = _spec()
    spec["config"]["provider_config"] = {secret: "do-not-store"}
    with pytest.raises(ValueError, match="cannot persist credentials"):
        resolve_hermes_profile(spec)


def test_external_profile_adds_canonical_langgraph_parity_tools() -> None:
    spec = _spec()
    spec["config"].update({
        "use_web_search": True,
        "allowed_tool_ids": ["search_documents", "search_web", "pubmed", "semantic_scholar"],
    })
    profile = resolve_hermes_profile(spec)
    assert profile["mcp"]["runtime_profile"] == HERMES_EXTERNAL_PROFILE
    assert profile["mcp"]["allowed_tool_ids"] == ["pubmed", "search_documents", "search_web", "semantic_scholar"]


def test_offline_profile_removes_external_tools() -> None:
    spec = _spec()
    spec["config"]["allowed_tool_ids"] = ["search_documents", "search_web", "wikipedia"]
    profile = resolve_hermes_profile(spec)
    assert profile["mcp"]["allowed_tool_ids"] == ["search_documents"]


def test_builtin_requires_document_tool_call_before_no_evidence_claim() -> None:
    definition = json.loads(
        (Path(__file__).parents[1] / "app/agent_workflows/builtins/hermes_rag_agent.json").read_text()
    )
    prompt = definition["spec_json"]["config"]["system_prompt"]
    assert "tool_search searches only the deferred tool catalog" in prompt
    assert "tool_search results" in prompt and "do not count as document evidence" in prompt
    assert "only a successful underlying document-retrieval tool_call result does" in prompt
    assert "If a relevant retrieval call fails or returns no evidence after valid attempts" in prompt
    assert {"get_thread_shape", "search_documents", "search_document_by_id"}.issubset(
        definition["spec_json"]["config"]["allowed_tool_ids"]
    )

"""Controlled extraction of chat turns into reviewable memory candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.db import MemoryScopeType, MemoryType, create_memory_candidate, get_chat_turn, get_thread


REMEMBER_RE = re.compile(
    r"(?:^|\b)(?:please\s+)?remember(?:\s+(?:for|in)\s+(?P<scope>this\s+project|project|this\s+thread|thread|me|my\s+profile|global|user))?\s+that\s+(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class MemoryCandidateProposal:
    scope_type: str
    scope_id: str
    memory_type: str
    content: str
    confidence: float
    reason: str


def _clean_memory_content(raw: str) -> str:
    text = " ".join(str(raw or "").strip().split())
    return text.strip(" \t\r\n\"'")


def _scope_from_phrase(scope: Optional[str], *, thread_id: str, project_id: Optional[str], user_id: Optional[str]) -> tuple[str, str]:
    normalized = " ".join(str(scope or "").lower().split())
    if normalized in {"me", "my profile", "global", "user"}:
        return MemoryScopeType.USER.value, str(user_id or "default")
    if normalized in {"this thread", "thread"}:
        return MemoryScopeType.THREAD.value, thread_id
    if project_id:
        return MemoryScopeType.PROJECT.value, project_id
    return MemoryScopeType.THREAD.value, thread_id


def extract_memory_candidates_from_text(
    text: str,
    *,
    thread_id: str,
    project_id: Optional[str],
    user_id: Optional[str] = None,
) -> List[MemoryCandidateProposal]:
    """Extract explicit remember-intent statements into candidate proposals."""

    proposals: List[MemoryCandidateProposal] = []
    for match in REMEMBER_RE.finditer(text or ""):
        content = _clean_memory_content(match.group("content") or "")
        if not content or len(content) < 8:
            continue
        # Stop at common sentence boundaries after capturing the requested memory.
        content = re.split(r"(?<=[.!?])\s+(?:also|and then|then)\b", content, maxsplit=1, flags=re.IGNORECASE)[0]
        content = content.strip()
        if len(content) > 600:
            content = content[:600].rstrip()
        scope_type, scope_id = _scope_from_phrase(
            match.group("scope"),
            thread_id=thread_id,
            project_id=project_id,
            user_id=user_id,
        )
        proposals.append(
            MemoryCandidateProposal(
                scope_type=scope_type,
                scope_id=scope_id,
                memory_type=MemoryType.SEMANTIC.value,
                content=content,
                confidence=0.92,
                reason="Explicit user remember-intent phrase.",
            )
        )
    return proposals


async def extract_candidates_for_completed_turn(
    *,
    thread_id: str,
    turn_id: str,
    agent_run_id: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create pending memory candidates for a completed turn."""

    thread = await get_thread(thread_id)
    turn = await get_chat_turn(turn_id)
    if thread is None or turn is None:
        return []
    if str(getattr(turn, "status", "")) != "completed":
        return []

    payload = turn.payload if isinstance(turn.payload, dict) else {}
    question = str(payload.get("question") or "")
    user_id = (thread.thread_metadata or {}).get("user_id") if isinstance(thread.thread_metadata, dict) else None
    proposals = extract_memory_candidates_from_text(
        question,
        thread_id=thread_id,
        project_id=getattr(thread, "project_id", None),
        user_id=str(user_id) if user_id else None,
    )
    created = []
    for proposal in proposals:
        candidate = await create_memory_candidate(
            proposed_scope_type=proposal.scope_type,
            proposed_scope_id=proposal.scope_id,
            memory_type=proposal.memory_type,
            content=proposal.content,
            source_thread_id=thread_id,
            source_project_id=getattr(thread, "project_id", None),
            source_agent_run_id=agent_run_id,
            source_turn_id=turn_id,
            confidence=proposal.confidence,
            reason=proposal.reason,
            created_by=actor_id or "memory_promotion_service",
        )
        created.append(
            {
                "id": candidate.id,
                "proposed_scope_type": candidate.proposed_scope_type,
                "proposed_scope_id": candidate.proposed_scope_id,
                "memory_type": candidate.memory_type,
                "content": candidate.content,
                "confidence": candidate.confidence,
                "reason": candidate.reason,
                "status": candidate.status,
            }
        )
    return created

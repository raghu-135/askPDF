"""
message_repo_sqlmodel.py - Compatibility message API backed by chat turns.

The database stores one JSONB chat_turns row per user interaction. This
repository expands turns into user/assistant-shaped message objects for the
existing API and frontend contracts.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import ChatTurn, MessageRole
from app.time_utils import utc_now


TURN_USER_SUFFIX = ":user"
TURN_ASSISTANT_SUFFIX = ":assistant"
VISIBLE_TURN_STATUSES = {"completed", "clarification", "failed"}


@dataclass
class ExpandedMessage:
    """UI/API-compatible message projected from a chat turn."""

    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime
    context_compact: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_available: bool = False
    reasoning_format: str = "none"
    web_sources: Optional[List[Dict[str, Any]]] = None
    turn_id: Optional[str] = None
    turn_status: Optional[str] = None


def message_id_for_turn(turn_id: str, role: str) -> str:
    suffix = TURN_USER_SUFFIX if role == MessageRole.USER.value else TURN_ASSISTANT_SUFFIX
    return f"{turn_id}{suffix}"


def turn_id_from_message_id(message_id: str) -> str:
    if message_id.endswith(TURN_USER_SUFFIX):
        return message_id[: -len(TURN_USER_SUFFIX)]
    if message_id.endswith(TURN_ASSISTANT_SUFFIX):
        return message_id[: -len(TURN_ASSISTANT_SUFFIX)]
    return message_id


def role_from_message_id(message_id: str) -> Optional[str]:
    if message_id.endswith(TURN_USER_SUFFIX):
        return MessageRole.USER.value
    if message_id.endswith(TURN_ASSISTANT_SUFFIX):
        return MessageRole.ASSISTANT.value
    return None


def _copy_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def _normalize_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = _copy_payload(payload)
    data.setdefault("question", "")
    data.setdefault("rewritten_question", None)
    data.setdefault("answer", None)
    data.setdefault("reasoning", "")
    data.setdefault("reasoning_available", False)
    data.setdefault("reasoning_format", "none")
    data.setdefault("web_sources", [])
    data.setdefault("document_sources", [])
    data.setdefault("used_chat_ids", [])
    data.setdefault("clarification_options", None)
    data.setdefault("error", None)
    metadata = data.get("metadata")
    data["metadata"] = metadata if isinstance(metadata, dict) else {}
    return data


def _expand_turn(turn: ChatTurn) -> List[ExpandedMessage]:
    payload = _normalize_payload(turn.payload)
    messages: List[ExpandedMessage] = []

    question = payload.get("question")
    if question not in (None, ""):
        metadata = payload.get("metadata") or {}
        messages.append(
            ExpandedMessage(
                id=message_id_for_turn(turn.id, MessageRole.USER.value),
                thread_id=turn.thread_id,
                role=MessageRole.USER.value,
                content=str(question),
                context_compact=payload.get("rewritten_question"),
                created_at=turn.created_at,
                turn_id=turn.id,
                turn_status=turn.status,
            )
        )

    answer = payload.get("answer")
    if answer not in (None, ""):
        created_at = turn.completed_at or turn.created_at
        messages.append(
            ExpandedMessage(
                id=message_id_for_turn(turn.id, MessageRole.ASSISTANT.value),
                thread_id=turn.thread_id,
                role=MessageRole.ASSISTANT.value,
                content=str(answer),
                context_compact=(payload.get("metadata") or {}).get("context_compact"),
                reasoning=payload.get("reasoning") or "",
                reasoning_available=bool(payload.get("reasoning_available")),
                reasoning_format=payload.get("reasoning_format") or "none",
                web_sources=payload.get("web_sources") or None,
                created_at=created_at,
                turn_id=turn.id,
                turn_status=turn.status,
            )
        )

    return messages


class MessageRepository:
    """Repository for chat turns, exposed through legacy message methods."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create_turn(
        self,
        thread_id: str,
        question: str,
        answer: Optional[str] = None,
        rewritten_question: Optional[str] = None,
        status: str = "completed",
        reasoning: Optional[str] = "",
        reasoning_available: bool = False,
        reasoning_format: str = "none",
        web_sources: Optional[List[Dict[str, Any]]] = None,
        document_sources: Optional[List[Dict[str, Any]]] = None,
        used_chat_ids: Optional[List[str]] = None,
        clarification_options: Optional[List[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> ChatTurn:
        """Create one persisted chat turn."""
        now = created_at or utc_now()
        payload = _normalize_payload(
            {
                "question": question,
                "rewritten_question": rewritten_question,
                "answer": answer,
                "reasoning": reasoning or "",
                "reasoning_available": reasoning_available,
                "reasoning_format": reasoning_format or "none",
                "web_sources": web_sources or [],
                "document_sources": document_sources or [],
                "used_chat_ids": used_chat_ids or [],
                "clarification_options": clarification_options,
                "error": error,
                "metadata": metadata or {},
            }
        )
        turn = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            status=status,
            payload=payload,
            created_at=now,
            completed_at=completed_at or (now if answer not in (None, "") else None),
        )

        session = await self._get_session()
        async with session.begin():
            session.add(turn)
            await session.flush()
            await session.refresh(turn)
        return turn

    async def create(
        self,
        thread_id: str,
        role: MessageRole,
        content: str,
        context_compact: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_available: bool = False,
        reasoning_format: str = "none",
        web_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> ExpandedMessage:
        """
        Backward-compatible single-message create.

        New chat code should call create_turn. This method preserves direct tests
        and any incidental callers by creating a one-sided turn.
        """
        role_value = role.value if isinstance(role, MessageRole) else str(role)
        if role_value == MessageRole.USER.value:
            turn = await self.create_turn(
                thread_id=thread_id,
                question=content,
                rewritten_question=context_compact,
                status="completed",
                completed_at=None,
            )
            return _expand_turn(turn)[0]

        turn = await self.create_turn(
            thread_id=thread_id,
            question="",
            answer=content,
            status="completed",
            reasoning=reasoning,
            reasoning_available=reasoning_available,
            reasoning_format=reasoning_format,
            web_sources=web_sources,
        )
        return _expand_turn(turn)[0]

    async def get_turn(self, turn_id: str) -> Optional[ChatTurn]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(ChatTurn).where(ChatTurn.id == turn_id))
            return result.scalar_one_or_none()

    async def get(self, message_id: str) -> Optional[ExpandedMessage]:
        """Get a compatibility message by turn-derived message ID."""
        turn = await self.get_turn(turn_id_from_message_id(message_id))
        if not turn or turn.status == "cancelled":
            return None

        requested_role = role_from_message_id(message_id)
        expanded = _expand_turn(turn)
        if requested_role:
            return next((msg for msg in expanded if msg.role == requested_role), None)
        return expanded[-1] if expanded else None

    async def get_thread_turns(
        self,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        include_cancelled: bool = False,
    ) -> List[ChatTurn]:
        session = await self._get_session()
        async with session.begin():
            query = select(ChatTurn).where(ChatTurn.thread_id == thread_id)
            if not include_cancelled:
                query = query.where(ChatTurn.status != "cancelled")
            result = await session.execute(
                query.order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExpandedMessage]:
        turns = await self.get_thread_turns(thread_id, limit=10000, offset=0)
        messages: List[ExpandedMessage] = []
        for turn in turns:
            messages.extend(_expand_turn(turn))
        return messages[offset : offset + limit]

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> List[ExpandedMessage]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ChatTurn)
                .where(ChatTurn.thread_id == thread_id, ChatTurn.status != "cancelled")
                .order_by(ChatTurn.created_at.desc(), ChatTurn.id.desc())
                .limit(10000)
            )
            turns = list(reversed(result.scalars().all()))

        messages: List[ExpandedMessage] = []
        for turn in turns:
            messages.extend(_expand_turn(turn))
        return messages[-limit:]

    async def update_context_compact(self, message_id: str, context_compact: str) -> bool:
        """Store semantic-memory compact text in payload.metadata.context_compact."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ChatTurn).where(ChatTurn.id == turn_id_from_message_id(message_id))
            )
            turn = result.scalar_one_or_none()
            if not turn:
                return False

            payload = _normalize_payload(turn.payload)
            metadata = dict(payload.get("metadata") or {})
            metadata["context_compact"] = context_compact
            payload["metadata"] = metadata
            replace_jsonb_field(turn, "payload", payload)
            await session.flush()
        return True

    async def update_reasoning(
        self,
        message_id: str,
        reasoning: str,
        reasoning_format: str = "raw",
    ) -> bool:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ChatTurn).where(ChatTurn.id == turn_id_from_message_id(message_id))
            )
            turn = result.scalar_one_or_none()
            if not turn:
                return False

            payload = _normalize_payload(turn.payload)
            payload["reasoning"] = reasoning
            payload["reasoning_available"] = True
            payload["reasoning_format"] = reasoning_format
            replace_jsonb_field(turn, "payload", payload)
            await session.flush()
        return True

    async def update_web_sources(
        self,
        message_id: str,
        web_sources: List[Dict[str, Any]],
    ) -> bool:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ChatTurn).where(ChatTurn.id == turn_id_from_message_id(message_id))
            )
            turn = result.scalar_one_or_none()
            if not turn:
                return False

            payload = _normalize_payload(turn.payload)
            payload["web_sources"] = web_sources or []
            replace_jsonb_field(turn, "payload", payload)
            await session.flush()
        return True

    async def delete(self, message_id: str) -> bool:
        return bool(await self.delete_pair(message_id))

    async def delete_pair(self, message_id: str) -> List[str]:
        """Delete the whole owning turn for any compatibility message ID."""
        turn_id = turn_id_from_message_id(message_id)
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(ChatTurn).where(ChatTurn.id == turn_id))
            turn = result.scalar_one_or_none()
            if not turn:
                return []

            deleted_ids = [msg.id for msg in _expand_turn(turn)]
            await session.delete(turn)
        return deleted_ids

    async def delete_for_thread(self, thread_id: str) -> int:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(ChatTurn).where(ChatTurn.thread_id == thread_id))
            turns = list(result.scalars().all())
            count = len(turns)
            for turn in turns:
                await session.delete(turn)
        return count

    async def get_count(self, thread_id: str) -> int:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(func.count(ChatTurn.id)).where(
                    ChatTurn.thread_id == thread_id,
                    ChatTurn.status != "cancelled",
                )
            )
            return int(result.scalar() or 0)

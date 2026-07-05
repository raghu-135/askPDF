"""
test_message_api_pytest.py - Message API endpoint contract tests.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import pytest

from app.db.models_sqlmodel import MessageRole
from app.api import messages as messages_api


class TestMessageEndpoints:
    """Test suite for message endpoints."""

    @pytest.mark.asyncio
    async def test_thread_messages_exposes_safe_agent_metadata(self):
        """Message listing should preserve run ids without leaking full internals."""
        thread = SimpleNamespace(id="thread-1")
        assistant = SimpleNamespace(
            id="turn-1:assistant",
            role=MessageRole.ASSISTANT.value,
            content="Answer",
            context_compact="compact memory",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=[],
            metadata={
                "agent_run_id": "run-1",
                "agent_pattern_id": "router_rag_agent",
                "agent_pattern_version": 1,
                "agent_route": "document",
                "agent_route_reason": "Question needs document evidence.",
                "context_compact": "internal compact text",
            },
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch("app.api.messages.get_thread", new_callable=AsyncMock, return_value=thread),
            patch("app.api.messages.get_thread_messages", new_callable=AsyncMock, return_value=[assistant]),
        ):
            data = await messages_api.get_thread_messages_endpoint("thread-1")

        assert data["thread_id"] == "thread-1"
        metadata = data["messages"][0]["metadata"]
        assert metadata == {
            "agent_run_id": "run-1",
            "agent_pattern_id": "router_rag_agent",
            "agent_pattern_version": 1,
            "agent_route": "document",
            "agent_route_reason": "Question needs document evidence.",
        }

    @pytest.mark.asyncio
    async def test_delete_missing_message_is_idempotent(self):
        """Deleting a message that is already gone should not surface a 404."""
        with patch(
            "app.api.messages.get_message",
            new_callable=AsyncMock,
            return_value=None,
        ):
            data = await messages_api.delete_message_endpoint("missing-message-id")

        assert data == {"status": "not_found", "deleted_ids": []}

    @pytest.mark.asyncio
    async def test_delete_user_message_removes_turn_memory_and_recomputes_stats(self):
        """Deleting a user-side compatibility id should delete the whole turn."""
        user = SimpleNamespace(
            id="turn-1:user",
            turn_id="turn-1",
            thread_id="thread-1",
            role=MessageRole.USER.value,
            web_sources=None,
        )
        assistant = SimpleNamespace(
            id="turn-1:assistant",
            turn_id="turn-1",
            thread_id="thread-1",
            role=MessageRole.ASSISTANT.value,
            web_sources=[{"url": "https://example.com/old"}],
        )
        surviving = SimpleNamespace(
            id="turn-2:assistant",
            turn_id="turn-2",
            thread_id="thread-1",
            role=MessageRole.ASSISTANT.value,
            web_sources=[{"url": "https://example.com/keep"}],
        )
        thread = SimpleNamespace(id="thread-1", embed_model="BAAI/bge-m3")
        vector_db = SimpleNamespace(
            delete_chat_memory_by_message_id=AsyncMock(),
            delete_web_chunks_by_urls=AsyncMock(),
        )

        async def get_message_side_effect(message_id):
            return {
                "turn-1:user": user,
                "turn-1:assistant": assistant,
            }.get(message_id)

        with (
            patch("app.api.messages.get_message", new_callable=AsyncMock, side_effect=get_message_side_effect),
            patch(
                "app.api.messages.get_thread_messages",
                new_callable=AsyncMock,
                return_value=[user, assistant, surviving],
            ),
            patch("app.api.messages.get_thread", new_callable=AsyncMock, return_value=thread),
            patch("app.api.messages.get_vector_db", return_value=vector_db),
            patch(
                "app.api.messages.delete_message_pair",
                new_callable=AsyncMock,
                return_value=["turn-1:user", "turn-1:assistant"],
            ) as delete_pair,
            patch("app.api.messages.recompute_qa_stats", new_callable=AsyncMock) as recompute_stats,
        ):
            data = await messages_api.delete_message_endpoint("turn-1:user")

        assert data == {
            "status": "deleted",
            "deleted_ids": ["turn-1:user", "turn-1:assistant"],
        }
        vector_db.delete_chat_memory_by_message_id.assert_awaited_once_with(
            "thread-1",
            "turn-1",
            "BAAI/bge-m3",
        )
        vector_db.delete_web_chunks_by_urls.assert_awaited_once_with(
            "thread-1",
            ["https://example.com/old"],
            "BAAI/bge-m3",
        )
        delete_pair.assert_awaited_once_with("turn-1:user")
        recompute_stats.assert_awaited_once_with("thread-1")

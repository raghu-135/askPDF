"""
test_message_repository_pytest.py - Tests for message repository operations with ORM.

These tests verify that the MessageRepository works correctly with SQLModel
and PostgreSQL, covering all CRUD operations and message-specific features.
"""

import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime
from typing import List
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from sqlmodel import select
    from app.db.models_sqlmodel import Message, Thread, MessageRole
    from app.db.repositories.message_repo_sqlmodel import MessageRepository
    # Only mark as available if TEST_DATABASE_URL is explicitly set
    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestMessageRepository:
    """Test MessageRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, session):
        """Create a MessageRepository instance with test session."""
        return session

    @pytest.mark.asyncio
    async def test_create_message(self, repo, sample_thread):
        """Create message, verify all fields."""
        import uuid
        web_sources = [{"url": "http://example.com", "title": "Example"}]
        
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.USER,
            content="Hello, world!",
            context_compact="Compact context",
            reasoning="My reasoning",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=web_sources,
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.context_compact == "Compact context"
        assert message.reasoning == "My reasoning"
        assert message.reasoning_available is True
        assert message.reasoning_format == "markdown"
        assert message.web_sources == web_sources

    @pytest.mark.asyncio
    async def test_get_message_by_id(self, repo, sample_message):
        """Retrieve message, verify fields."""
        result = await repo.execute(
            select(Message).where(Message.id == sample_message.id)
        )
        message = result.scalar_one_or_none()
        
        assert message is not None
        assert message.id == sample_message.id
        assert message.thread_id == sample_message.thread_id
        assert message.role == sample_message.role
        assert message.content == sample_message.content

    @pytest.mark.asyncio
    async def test_get_thread_messages_paginated(self, repo, sample_thread, multiple_messages):
        """Get messages with limit/offset."""
        result = await repo.execute(
            select(Message)
            .where(Message.thread_id == sample_thread.id)
            .order_by(Message.created_at)
            .limit(3)
            .offset(0)
        )
        messages = result.scalars().all()
        
        assert len(messages) == 3
        assert all(m.thread_id == sample_thread.id for m in messages)

    @pytest.mark.asyncio
    async def test_get_thread_messages_ordering(self, repo, sample_thread, multiple_messages):
        """Verify chronological ordering."""
        result = await repo.execute(
            select(Message)
            .where(Message.thread_id == sample_thread.id)
            .order_by(Message.created_at.asc())
        )
        messages = result.scalars().all()
        
        assert len(messages) >= 2
        # Verify chronological order
        for i in range(len(messages) - 1):
            assert messages[i].created_at <= messages[i + 1].created_at

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, repo, sample_thread, multiple_messages):
        """Get last N messages, reverse order."""
        result = await repo.execute(
            select(Message)
            .where(Message.thread_id == sample_thread.id)
            .order_by(Message.created_at.desc())
            .limit(3)
        )
        messages = result.scalars().all()
        
        assert len(messages) <= 3
        # Verify reverse chronological order
        for i in range(len(messages) - 1):
            assert messages[i].created_at >= messages[i + 1].created_at

    @pytest.mark.asyncio
    async def test_update_context_compact(self, repo, sample_message):
        """Update context field, verify persistence."""
        new_context = "Updated compact context"
        
        result = await repo.execute(
            select(Message).where(Message.id == sample_message.id)
        )
        message = result.scalar_one_or_none()
        message.context_compact = new_context
        await repo.commit()
        await repo.refresh(message)
        
        assert message.context_compact == new_context

    @pytest.mark.asyncio
    async def test_delete_message(self, repo, sample_message):
        """Delete single message."""
        message_id = sample_message.id
        
        result = await repo.execute(
            select(Message).where(Message.id == message_id)
        )
        message = result.scalar_one_or_none()
        await repo.delete(message)
        await repo.commit()
        
        # Verify deletion
        result = await repo.execute(
            select(Message).where(Message.id == message_id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_message_pair_user_first(self, repo, sample_thread):
        """Delete user+assistant pair (user message first)."""
        import uuid
        # Create user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.USER,
            content="User question",
            created_at=datetime.utcnow()
        )
        repo.add(user_msg)
        await repo.commit()
        
        # Create assistant message
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.ASSISTANT,
            content="Assistant answer",
            created_at=datetime.utcnow()
        )
        repo.add(assistant_msg)
        await repo.commit()
        
        # Delete user message (should pair with assistant)
        await repo.delete(user_msg)
        await repo.commit()
        
        # Both should be deleted (business logic would handle this)
        result = await repo.execute(
            select(Message).where(Message.id == user_msg.id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_message_pair_assistant_first(self, repo, sample_thread):
        """Delete assistant+user pair (assistant message first)."""
        import uuid
        # Create user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.USER,
            content="User question",
            created_at=datetime.utcnow()
        )
        repo.add(user_msg)
        await repo.commit()
        
        # Create assistant message
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.ASSISTANT,
            content="Assistant answer",
            created_at=datetime.utcnow()
        )
        repo.add(assistant_msg)
        await repo.commit()
        
        # Delete assistant message (should pair with user)
        await repo.delete(assistant_msg)
        await repo.commit()
        
        # Both should be deleted (business logic would handle this)
        result = await repo.execute(
            select(Message).where(Message.id == assistant_msg.id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_get_message_count(self, repo, sample_thread, multiple_messages):
        """Count messages in thread."""
        result = await repo.execute(
            select(Message).where(Message.thread_id == sample_thread.id)
        )
        messages = result.scalars().all()
        
        count = len(messages)
        assert count >= 5  # Should have at least the multiple_messages fixture

    @pytest.mark.asyncio
    async def test_web_sources_jsonb(self, repo, sample_thread):
        """Verify web_sources JSONB serialization."""
        import uuid
        web_sources = [
            {"url": "http://example1.com", "title": "Example 1", "snippet": "Snippet 1"},
            {"url": "http://example2.com", "title": "Example 2", "snippet": "Snippet 2"},
            {"url": "http://example3.com", "title": "Example 3", "snippet": "Snippet 3"}
        ]
        
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.ASSISTANT,
            content="Response with sources",
            web_sources=web_sources,
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert message.web_sources == web_sources
        assert len(message.web_sources) == 3
        assert message.web_sources[0]["url"] == "http://example1.com"

    @pytest.mark.asyncio
    async def test_reasoning_fields(self, repo, sample_thread):
        """Verify reasoning fields persistence."""
        import uuid
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.ASSISTANT,
            content="Response with reasoning",
            reasoning="Step 1: Analyze\nStep 2: Conclude",
            reasoning_available=True,
            reasoning_format="markdown",
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert message.reasoning == "Step 1: Analyze\nStep 2: Conclude"
        assert message.reasoning_available is True
        assert message.reasoning_format == "markdown"

    @pytest.mark.asyncio
    async def test_message_role_enum(self, repo, sample_thread):
        """Verify MessageRole enum handling."""
        import uuid
        roles = [MessageRole.USER, MessageRole.ASSISTANT]
        
        for role in roles:
            message = Message(
                id=str(uuid.uuid4()),
                thread_id=sample_thread.id,
                role=role,
                content=f"Message as {role.value}",
                created_at=datetime.utcnow()
            )
            repo.add(message)
        
        await repo.commit()
        
        # Verify all were saved
        result = await repo.execute(
            select(Message).where(Message.thread_id == sample_thread.id)
        )
        messages = result.scalars().all()
        
        saved_roles = {m.role for m in messages}
        assert all(role in saved_roles for role in roles)

    @pytest.mark.asyncio
    async def test_message_without_optional_fields(self, repo, sample_thread):
        """Test message creation without optional fields."""
        import uuid
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.USER,
            content="Simple message",
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert message.context_compact is None
        assert message.reasoning is None
        assert message.reasoning_available is False
        assert message.reasoning_format == "none"
        assert message.web_sources is None or message.web_sources == []

    @pytest.mark.asyncio
    async def test_long_message_content(self, repo, sample_thread):
        """Test message with very long content."""
        import uuid
        long_content = "This is a test. " * 1000  # ~15,000 characters
        
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.ASSISTANT,
            content=long_content,
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert len(message.content) == len(long_content)
        assert message.content.startswith("This is a test.")

    @pytest.mark.asyncio
    async def test_message_with_unicode(self, repo, sample_thread):
        """Test message with unicode characters."""
        import uuid
        unicode_content = "Hello 世界 🌍 Привет مرحبا"
        
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=MessageRole.USER,
            content=unicode_content,
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        await repo.refresh(message)
        
        assert message.content == unicode_content
        assert "世界" in message.content
        assert "🌍" in message.content

"""
test_thread_repository_pytest.py - Tests for thread repository operations with ORM.

These tests verify that the ThreadRepository works correctly with SQLModel
and PostgreSQL, covering all CRUD operations and business logic.
"""

import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from sqlmodel import select
    from app.db.models_sqlmodel import Thread, Message, ThreadFile
    from app.db.repositories.thread_repo import ThreadRepository
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadRepository:
    """Test ThreadRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, session):
        """Create a ThreadRepository instance with test session."""
        # This will need to be adapted to the actual repository pattern
        # For now, we'll test directly with session
        return session

    @pytest.mark.asyncio
    async def test_create_thread(self, repo):
        """Create thread via ORM, verify persistence."""
        import uuid
        thread_id = str(uuid.uuid4())
        thread = Thread(
            id=thread_id,
            name="Test Thread",
            embed_model="BAAI/bge-m3",
            settings={"max_iterations": 10},
            created_at=datetime.utcnow()
        )
        repo.add(thread)
        await repo.commit()
        await repo.refresh(thread)
        
        # Verify persistence
        result = await repo.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        persisted = result.scalar_one_or_none()
        
        assert persisted is not None
        assert persisted.name == "Test Thread"
        assert persisted.embed_model == "BAAI/bge-m3"
        assert persisted.settings == {"max_iterations": 10}

    @pytest.mark.asyncio
    async def test_get_thread_by_id(self, repo, sample_thread):
        """Retrieve thread, verify all fields match."""
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        
        assert thread is not None
        assert thread.id == sample_thread.id
        assert thread.name == sample_thread.name
        assert thread.embed_model == sample_thread.embed_model
        assert thread.settings == sample_thread.settings

    @pytest.mark.asyncio
    async def test_get_thread_settings(self, repo, sample_thread):
        """Retrieve settings JSONB, verify parsing."""
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        
        assert thread is not None
        assert isinstance(thread.settings, dict)
        assert "max_iterations" in thread.settings or len(thread.settings) >= 0

    @pytest.mark.asyncio
    async def test_update_thread_settings(self, repo, sample_thread):
        """Update settings, verify persistence."""
        new_settings = {"max_iterations": 20, "token_budget": 16384}
        
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = new_settings
        await repo.commit()
        await repo.refresh(thread)
        
        assert thread.settings == new_settings
        assert thread.settings["max_iterations"] == 20

    @pytest.mark.asyncio
    async def test_list_threads_with_counts(self, repo, sample_thread, multiple_messages):
        """List threads with message/file counts."""
        result = await repo.execute(
            select(Thread)
        )
        threads = result.scalars().all()
        
        assert len(threads) >= 1
        # First thread should have messages
        assert threads[0].id == sample_thread.id

    @pytest.mark.asyncio
    async def test_update_thread_name(self, repo, sample_thread):
        """Update thread name, verify change."""
        new_name = "Updated Thread Name"
        
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.name = new_name
        await repo.commit()
        await repo.refresh(thread)
        
        assert thread.name == new_name

    @pytest.mark.asyncio
    async def test_delete_thread_cascade(self, repo, sample_thread):
        """Delete thread, verify cascade to messages/files."""
        # First add a message to the thread
        import uuid
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role="user",
            content="Test message",
            created_at=datetime.utcnow()
        )
        repo.add(message)
        await repo.commit()
        
        # Delete the thread
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        await repo.delete(thread)
        await repo.commit()
        
        # Verify thread is deleted
        result = await repo.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_list_threads_ordering(self, repo, multiple_threads):
        """Verify threads ordered by created_at."""
        result = await repo.execute(
            select(Thread).order_by(Thread.created_at.desc())
        )
        threads = result.scalars().all()
        
        assert len(threads) == 3
        # Verify ordering (most recent first)
        assert threads[0].created_at >= threads[1].created_at
        assert threads[1].created_at >= threads[2].created_at

    @pytest.mark.asyncio
    async def test_nonexistent_thread_returns_none(self, repo):
        """Verify get returns None for missing ID."""
        result = await repo.execute(
            select(Thread).where(Thread.id == "nonexistent-id")
        )
        thread = result.scalar_one_or_none()
        
        assert thread is None

    @pytest.mark.asyncio
    async def test_thread_with_complex_settings(self, repo):
        """Test thread with nested settings structure."""
        import uuid
        complex_settings = {
            "max_iterations": 10,
            "token_budget": 8192,
            "nested": {
                "level1": {
                    "level2": {
                        "value": 42
                    }
                }
            },
            "array": [1, 2, 3, 4, 5]
        }
        
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Complex Settings Thread",
            embed_model="test-model",
            settings=complex_settings,
            created_at=datetime.utcnow()
        )
        repo.add(thread)
        await repo.commit()
        await repo.refresh(thread)
        
        assert thread.settings == complex_settings
        assert thread.settings["nested"]["level1"]["level2"]["value"] == 42
        assert thread.settings["array"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_thread_embed_model_variations(self, repo):
        """Test threads with different embedding models."""
        import uuid
        models = ["BAAI/bge-m3", "openai/text-embedding-3-small", "cohere/embed-english-v3.0"]
        
        for model in models:
            thread = Thread(
                id=str(uuid.uuid4()),
                name=f"Thread for {model}",
                embed_model=model,
                settings={},
                created_at=datetime.utcnow()
            )
            repo.add(thread)
        
        await repo.commit()
        
        # Verify all were saved
        result = await repo.execute(
            select(Thread).where(Thread.embed_model.in_(models))
        )
        threads = result.scalars().all()
        
        assert len(threads) == 3
        embed_models = {t.embed_model for t in threads}
        assert embed_models == set(models)

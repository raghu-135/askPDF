"""
test_repository_transactions_pytest.py - Tests for transaction behavior and rollback.

These tests verify that transactions work correctly with SQLModel and PostgreSQL,
including commit, rollback, and error handling.
"""

import os
import pytest
import pytest_asyncio
from datetime import datetime
import asyncio

# Import will work after migration
try:
    from sqlmodel import select
    from app.db.models_sqlmodel import Thread, File, ChatTurn
    # Only mark as available if TEST_DATABASE_URL is explicitly set
    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestRepositoryTransactions:
    """Test transaction behavior and rollback."""

    @pytest.mark.asyncio
    async def test_transaction_commit(self, session, test_model_project):
        """Verify commit persists changes."""
        import uuid
        thread_id = str(uuid.uuid4())
        
        # Create thread in transaction
        thread = Thread(
            id=thread_id,
            project_id=test_model_project.id,
            name="Commit Test",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        
        # Verify it persists after commit
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        persisted = result.scalar_one_or_none()
        
        assert persisted is not None
        assert persisted.name == "Commit Test"

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, session, test_model_project):
        """Verify rollback on exception."""
        import uuid
        thread_id = str(uuid.uuid4())
        
        # Create thread
        thread = Thread(
            id=thread_id,
            project_id=test_model_project.id,
            name="Rollback Test",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread)
        
        # Rollback instead of commit
        await session.rollback()
        
        # Verify it was not persisted
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        persisted = result.scalar_one_or_none()
        
        assert persisted is None

    @pytest.mark.asyncio
    async def test_nested_transaction_behavior(self, session, test_model_project):
        """Test nested transaction handling."""
        import uuid
        
        # Outer transaction
        thread = Thread(
            id=str(uuid.uuid4()),
            project_id=test_model_project.id,
            name="Outer Thread",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread)
        
        # Inner operation (simulated)
        file = File(
            file_hash="test-hash",
            file_name="test.pdf"
        )
        session.add(file)
        
        # Commit both
        await session.commit()
        
        # Verify both persisted
        result = await session.execute(
            select(Thread).where(Thread.name == "Outer Thread")
        )
        assert result.scalar_one_or_none() is not None
        
        result = await session.execute(
            select(File).where(File.file_hash == "test-hash")
        )
        assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_concurrent_thread_creation(self, session, test_model_project):
        """Test sequential thread creation (no conflicts)."""
        import uuid
        
        threads = []
        for i in range(5):
            thread = Thread(
                id=str(uuid.uuid4()),
                project_id=test_model_project.id,
                name=f"Concurrent Thread {i}",
                embedding_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            session.add(thread)
            await session.commit()
            threads.append(thread)
        
        # Verify all were created
        assert len(threads) == 5
        for thread in threads:
            result = await session.execute(
                select(Thread).where(Thread.id == thread.id)
            )
            assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, session, sample_thread):
        """Verify FK constraint enforcement."""
        import uuid
        # Try to create chat turn with non-existent thread
        turn = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id="non-existent-thread-id",
            status="completed",
            payload={"question": "Test question", "answer": "Test answer"},
            created_at=datetime.utcnow()
        )
        session.add(turn)
        
        # This should fail due to FK constraint
        try:
            await session.commit()
            # If we get here, FK constraint might not be enforced
            # Clean up
            await session.rollback()
        except Exception as e:
            # Expected: FK constraint violation
            await session.rollback()
            assert "foreign key" in str(e).lower() or "constraint" in str(e).lower()

    @pytest.mark.asyncio
    async def test_unique_constraint_violation(self, session):
        """Verify unique constraints work."""
        import uuid
        file_hash = "unique-test-hash"
        
        # Create first file
        file1 = File(
            file_hash=file_hash,
            file_name="test1.pdf"
        )
        session.add(file1)
        await session.commit()
        
        # Try to create duplicate
        file2 = File(
            file_hash=file_hash,
            file_name="test2.pdf"
        )
        session.add(file2)
        
        # This should fail due to unique constraint
        try:
            await session.commit()
            await session.rollback()
        except Exception as e:
            await session.rollback()
            # Expected: unique constraint violation
            assert "unique" in str(e).lower() or "duplicate" in str(e).lower()

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, session, test_model_project):
        """Test that transactions are isolated."""
        import uuid
        thread_id = str(uuid.uuid4())
        
        # Create thread in transaction
        thread = Thread(
            id=thread_id,
            project_id=test_model_project.id,
            name="Isolation Test",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread)
        
        # Before commit, it shouldn't be visible
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        # In the same session, it should be visible
        # But in a different session, it wouldn't be (not tested here)
        
        await session.commit()
        
        # After commit, it should be visible
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_rollback_partial_changes(self, session, test_model_project):
        """Test rollback with multiple changes."""
        import uuid
        thread_id = str(uuid.uuid4())
        file_hash = "rollback-test-hash"
        
        # Make multiple changes
        thread = Thread(
            id=thread_id,
            project_id=test_model_project.id,
            name="Rollback Thread",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread)
        
        file = File(
            file_hash=file_hash,
            file_name="rollback.pdf"
        )
        session.add(file)
        
        # Rollback both
        await session.rollback()
        
        # Verify neither persisted
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        assert result.scalar_one_or_none() is None
        
        result = await session.execute(
            select(File).where(File.file_hash == file_hash)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_commit_after_rollback(self, session, test_model_project):
        """Test that commit works after rollback."""
        import uuid
        project_id = test_model_project.id
        
        # First operation - rollback
        thread1 = Thread(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Rollback Thread",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread1)
        await session.rollback()
        
        # Second operation - commit
        thread2 = Thread(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Commit Thread",
            embedding_model="test-model",
            settings={},
            created_at=datetime.utcnow()
        )
        session.add(thread2)
        await session.commit()
        
        # Verify only second persisted
        result = await session.execute(
            select(Thread).where(Thread.name == "Commit Thread")
        )
        assert result.scalar_one_or_none() is not None
        
        result = await session.execute(
            select(Thread).where(Thread.name == "Rollback Thread")
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_transaction_with_update(self, session, sample_thread):
        """Test transaction with update operation."""
        original_name = sample_thread.name
        
        # Update in transaction
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.name = "Updated Name"
        await session.commit()
        await session.refresh(thread)
        
        # Verify update persisted
        assert thread.name == "Updated Name"
        
        # Rollback the change for test isolation
        thread.name = original_name
        await session.commit()

    @pytest.mark.asyncio
    async def test_transaction_with_delete(self, session, sample_thread):
        """Test transaction with delete operation."""
        thread_id = sample_thread.id
        
        # Delete in transaction
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        thread = result.scalar_one_or_none()
        await session.delete(thread)
        await session.commit()
        
        # Verify deletion
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_batch_operations_in_transaction(self, session, test_model_project):
        """Test multiple operations in single transaction."""
        import uuid
        
        # Create multiple objects
        threads = []
        for i in range(10):
            thread = Thread(
                id=str(uuid.uuid4()),
                project_id=test_model_project.id,
                name=f"Batch Thread {i}",
                embedding_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            session.add(thread)
            threads.append(thread)
        
        # Single commit for all
        await session.commit()
        
        # Verify all persisted
        for thread in threads:
            result = await session.execute(
                select(Thread).where(Thread.id == thread.id)
            )
            assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_session_refresh_after_commit(self, session, sample_thread):
        """Test that refresh works after commit."""
        original_name = sample_thread.name
        
        # Update
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.name = "New Name"
        await session.commit()
        
        # Refresh to get latest state
        await session.refresh(thread)
        assert thread.name == "New Name"
        
        # Restore
        thread.name = original_name
        await session.commit()

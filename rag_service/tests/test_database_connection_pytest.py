"""
test_database_connection_pytest.py - Tests for PostgreSQL connection and session management.

These tests verify that the PostgreSQL connection, async engine, and session management
work correctly with SQLModel and SQLAlchemy async.
"""

import os
import sys
import pytest
import pytest_asyncio

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text
    from sqlmodel import SQLModel, select
    from app.db.connection_sqlmodel import get_session, init_db
    from app.db.models_sqlmodel import Thread
    # Only mark as available if TEST_DATABASE_URL is explicitly set
    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestPostgreSQLConnection:
    """Test PostgreSQL connection and basic operations."""

    @pytest_asyncio.fixture
    async def test_engine(self):
        """Create a test engine for connection tests."""
        test_url = os.getenv(
            "TEST_DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/test_askpdf"
        )
        engine = create_async_engine(test_url, echo=False, future=True)
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        
        yield engine
        
        # Cleanup
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_postgresql_connection_success(self, test_engine):
        """Verify connection to PostgreSQL succeeds."""
        async with test_engine.connect() as conn:
            # Simple query to test connection
            result = await conn.execute(text("SELECT 1"))
            assert result is not None

    @pytest.mark.asyncio
    async def test_async_session_creation(self, test_engine):
        """Verify async session factory works."""
        async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with async_session() as session:
            assert session is not None
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_database_initialization(self, test_engine):
        """Verify tables are created from SQLModel models."""
        # Check if Thread table exists
        async with test_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'threads'"
            ))
            tables = result.fetchall()
            assert len(tables) > 0, "Thread table not created"

    @pytest.mark.asyncio
    async def test_session_context_manager(self, test_engine):
        """Verify session lifecycle (open, commit, close)."""
        async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with async_session() as session:
            # Create a thread
            import uuid
            from datetime import datetime
            thread = Thread(
                id=str(uuid.uuid4()),
                name="Test Thread",
                embed_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            session.add(thread)
            await session.commit()
            await session.refresh(thread)
            
            # Verify it was saved
            assert thread.id is not None
            assert thread.name == "Test Thread"

    @pytest.mark.asyncio
    async def test_connection_pooling(self, test_engine):
        """Verify connection pooling works under load."""
        async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create multiple concurrent sessions
        async def create_and_query():
            async with async_session() as session:
                result = await session.execute(select(Thread))
                return result.scalars().all()
        
        # Run 10 concurrent queries
        results = await asyncio.gather(*[create_and_query() for _ in range(10)])
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_session_rollback(self, test_engine):
        """Verify session rollback works correctly."""
        async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        thread_id = None
        
        # Create and rollback
        async with async_session() as session:
            import uuid
            from datetime import datetime
            thread = Thread(
                id=str(uuid.uuid4()),
                name="Rollback Test",
                embed_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            thread_id = thread.id
            session.add(thread)
            await session.rollback()
        
        # Verify thread was not saved
        async with async_session() as session:
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.first()
            assert thread is None, "Thread should not exist after rollback"

    @pytest.mark.asyncio
    async def test_multiple_operations_in_transaction(self, test_engine):
        """Verify multiple operations in a single transaction."""
        async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        import uuid
        from datetime import datetime
        
        async with async_session() as session:
            # Create multiple threads
            thread_ids = []
            for i in range(5):
                thread = Thread(
                    id=str(uuid.uuid4()),
                    name=f"Thread {i}",
                    embed_model="test-model",
                    settings={},
                    created_at=datetime.utcnow()
                )
                session.add(thread)
                thread_ids.append(thread.id)
            
            await session.commit()
        
        # Verify all were saved
        async with async_session() as session:
            result = await session.execute(
                select(Thread).where(Thread.id.in_(thread_ids))
            )
            threads = result.scalars().all()
            assert len(threads) == 5


# Import asyncio for concurrent tests
import asyncio

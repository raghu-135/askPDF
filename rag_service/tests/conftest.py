"""
conftest.py - Pytest configuration and fixtures for database tests.

This module provides shared fixtures for both SQLite and PostgreSQL database testing,
including connection management, session handling, and test data.
"""

import os
import sys
import asyncio
import tempfile
import uuid
from typing import AsyncGenerator, Generator
from datetime import datetime

import pytest
import pytest_asyncio
from faker import Faker
import aiosqlite

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# These imports will work after migration is complete
# For now, we'll handle the import error gracefully
try:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    from sqlmodel import SQLModel
    from app.db.connection_sqlmodel import get_session, init_db
    from app.db.models_sqlmodel import (
        Thread, File, ThreadFile, ThreadFileAnnotation,
        Message, ThreadStats, ProcessStatus, MessageRole
    )
    # Only mark as available if TEST_DATABASE_URL is explicitly set
    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
except ImportError:
    SQLMODEL_AVAILABLE = False
    # Create stub classes for testing before migration
    SQLModel = object
    AsyncSession = object
    async_session_maker = None

# SQLite imports for DB-agnostic tests
from app.db.config import SCHEMA, MIGRATIONS


# Faker instance for generating test data
fake = Faker()


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Get the test database URL from environment or use default with random database name.
    
    In Docker: Use the postgres service
    Locally: Use localhost postgres
    """
    test_url = os.getenv("TEST_DATABASE_URL")
    if test_url:
        # If TEST_DATABASE_URL is set (by run_tests.sh), use it as-is
        return test_url
    
    # Otherwise, generate a random database name for local testing
    base_url = "postgresql+asyncpg://postgres:postgres@localhost:5432"
    random_db_name = f"test_askpdf_{uuid.uuid4().hex[:12]}"
    return f"{base_url}/{random_db_name}"


@pytest_asyncio.fixture(scope="function")
async def engine(test_database_url: str):
    """
    Create a test database engine.
    
    This fixture is function-scoped to create tables for each test.
    Uses NullPool to avoid connection conflicts between concurrent tests.
    """
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    from sqlalchemy.pool import NullPool
    engine = create_async_engine(
        test_database_url,
        poolclass=NullPool,
        echo=False,
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine
    
    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def session(engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create a test database session with transaction rollback.
    
    Each test gets a clean session that rolls back at the end,
    ensuring tests don't affect each other.
    """
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    
    async with async_session() as session:
        # Begin transaction
        await session.begin()
        
        yield session
        
        # Rollback transaction to keep test database clean
        await session.rollback()


# Test data fixtures for Thread model
@pytest.fixture
def thread_data():
    """Generate sample thread data."""
    return {
        "name": fake.sentence(nb_words=4),
        "embed_model": "BAAI/bge-m3",
        "settings": {"max_iterations": 10, "token_budget": 8192}
    }


@pytest_asyncio.fixture
async def sample_thread(session, thread_data):
    """Create a sample thread in the database."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import uuid
    thread = Thread(
        id=str(uuid.uuid4()),
        name=thread_data["name"],
        embed_model=thread_data["embed_model"],
        settings=thread_data["settings"],
        created_at=datetime.utcnow()
    )
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    return thread


# Test data fixtures for File model
@pytest.fixture
def file_data():
    """Generate sample file data."""
    return {
        "file_hash": fake.sha256(),
        "file_name": f"{fake.word()}.pdf",
        "file_path": f"/data/{fake.word()}.pdf",
        "source_type": "pdf"
    }


@pytest_asyncio.fixture
async def sample_file(session, file_data):
    """Create a sample file in the database."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    file = File(**file_data)
    session.add(file)
    await session.commit()
    await session.refresh(file)
    return file


# Test data fixtures for Message model
@pytest.fixture
def message_data(sample_thread):
    """Generate sample message data."""
    return {
        "thread_id": sample_thread.id if SQLMODEL_AVAILABLE else "test-thread-id",
        "role": MessageRole.USER,
        "content": fake.paragraph(nb_sentences=3),
        "context_compact": fake.sentence(),
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


@pytest_asyncio.fixture
async def sample_message(session, message_data):
    """Create a sample message in the database."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import uuid
    message = Message(
        id=str(uuid.uuid4()),
        **message_data,
        created_at=datetime.utcnow()
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


# Test data fixtures for ThreadFile association
@pytest_asyncio.fixture
async def sample_thread_file(session, sample_thread, sample_file):
    """Create a sample thread-file association."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    thread_file = ThreadFile(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        added_at=datetime.utcnow()
    )
    session.add(thread_file)
    await session.commit()
    await session.refresh(thread_file)
    return thread_file


# Test data fixtures for ThreadFileAnnotation
@pytest.fixture
def annotation_data():
    """Generate sample annotation data."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": fake.sentence(),
                "label": "important"
            }
        ]
    }


@pytest_asyncio.fixture
async def sample_annotation(session, sample_thread, sample_file, annotation_data):
    """Create a sample thread-file annotation."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import json
    annotation = ThreadFileAnnotation(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        annotations_json=json.dumps(annotation_data["annotations"]),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return annotation


# Test data fixtures for ThreadStats
@pytest_asyncio.fixture
async def sample_thread_stats(session, sample_thread):
    """Create a sample thread stats record."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import json
    stats = ThreadStats(
        thread_id=sample_thread.id,
        total_qa_pairs=5,
        total_qa_chars=1000,
        avg_qa_chars=200.0,
        last_qa_at=datetime.utcnow(),
        documents_meta=json.dumps({}),
        last_updated_at=datetime.utcnow()
    )
    session.add(stats)
    await session.commit()
    await session.refresh(stats)
    return stats


# Fixture for multiple threads
@pytest_asyncio.fixture
async def multiple_threads(session, thread_data):
    """Create multiple sample threads."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import uuid
    threads = []
    for i in range(3):
        thread = Thread(
            id=str(uuid.uuid4()),
            name=f"{thread_data['name']} {i}",
            embed_model=thread_data["embed_model"],
            settings=thread_data["settings"],
            created_at=datetime.utcnow()
        )
        session.add(thread)
        threads.append(thread)
    
    await session.commit()
    for thread in threads:
        await session.refresh(thread)
    
    return threads


# Fixture for multiple messages
@pytest_asyncio.fixture
async def multiple_messages(session, sample_thread):
    """Create multiple sample messages in a thread."""
    if not SQLMODEL_AVAILABLE:
        pytest.skip("SQLModel not available - migration not complete")
    
    import uuid
    messages = []
    for i in range(5):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            role=role,
            content=fake.paragraph(nb_sentences=2),
            created_at=datetime.utcnow()
        )
        session.add(message)
        messages.append(message)
    
    await session.commit()
    for message in messages:
        await session.refresh(message)
    
    return messages


# Fixture for file status JSON
@pytest.fixture
def file_status_data():
    """Generate sample file status data."""
    return {
        "parsing": {
            "status": ProcessStatus.COMPLETED.value,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat()
        },
        "indexing": {
            "status": ProcessStatus.COMPLETED.value,
            "chunk_count": 100,
            "total_chars": 50000
        }
    }


# Fixture for parsed sentences JSON
@pytest.fixture
def parsed_sentences_data():
    """Generate sample parsed sentences data."""
    return {
        "sentences": [
            {
                "id": "1",
                "text": fake.sentence(),
                "page": 1,
                "bbox": [0, 0, 100, 20]
            }
        ]
    }


# ============================================================================
# SQLite Test Fixtures for DB-Agnostic Testing
# ============================================================================

@pytest.fixture(scope="session")
def test_db_path():
    """
    Get the test database path for SQLite with random database name.
    
    Uses a temporary file for test isolation with a random name.
    Can be overridden with TEST_DB_PATH environment variable.
    """
    if os.getenv("TEST_DB_PATH"):
        return os.getenv("TEST_DB_PATH")
    
    # Create a temporary file for the test database with random name
    temp_dir = tempfile.mkdtemp()
    random_db_name = f"test_rag_{uuid.uuid4().hex[:12]}.db"
    return os.path.join(temp_dir, random_db_name)


@pytest_asyncio.fixture(scope="session")
async def init_test_db(test_db_path: str):
    """
    Initialize the test database with schema and migrations.
    
    This fixture is session-scoped to initialize the database once per test session.
    """
    async with aiosqlite.connect(test_db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(SCHEMA)
        
        # Run migrations
        for stmt in MIGRATIONS:
            try:
                await db.execute(stmt)
            except aiosqlite.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
        await db.commit()
    
    yield test_db_path
    
    # Clean up the test database file
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


@pytest_asyncio.fixture(scope="function")
async def test_session(init_test_db: str) -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Create a test database session with transaction rollback.
    
    Each test gets a clean session that rolls back at the end,
    ensuring tests don't affect each other.
    """
    async with aiosqlite.connect(init_test_db) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("BEGIN")
        
        yield db
        
        # Rollback transaction to keep test database clean
        await db.rollback()


@pytest_asyncio.fixture(scope="function")
async def shared_db_connection(init_test_db: str) -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Create a shared database connection for integration tests.
    
    This connection is committed after each operation (no rollback),
    so data is visible across different repository instances.
    """
    async with aiosqlite.connect(init_test_db) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        
        yield db


# SQLite test data fixtures
@pytest.fixture
def sqlite_thread_data():
    """Generate sample thread data for SQLite tests."""
    import uuid
    return {
        "id": str(uuid.uuid4()),
        "name": fake.sentence(nb_words=4),
        "embed_model": "BAAI/bge-m3",
        "settings": '{"max_iterations": 10, "token_budget": 8192}'
    }


@pytest.fixture
def sqlite_file_data():
    """Generate sample file data for SQLite tests."""
    return {
        "file_hash": fake.sha256(),
        "file_name": f"{fake.word()}.pdf",
        "file_path": f"/data/{fake.word()}.pdf",
        "source_type": "pdf",
        "file_status": "{}"
    }


@pytest.fixture
def sqlite_message_data():
    """Generate sample message data for SQLite tests."""
    import uuid
    return {
        "id": str(uuid.uuid4()),
        "thread_id": str(uuid.uuid4()),
        "role": "user",
        "content": fake.paragraph(nb_sentences=3),
        "context_compact": fake.sentence(),
        "reasoning": None,
        "reasoning_available": 0,
        "reasoning_format": "none",
        "web_sources": None
    }

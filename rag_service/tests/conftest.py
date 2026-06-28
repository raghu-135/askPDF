"""
conftest.py - Pytest configuration and fixtures for database tests.

This module provides shared fixtures for PostgreSQL database testing,
including connection management, session handling, and test data.
"""

import os
import sys
import asyncio
import uuid
from typing import AsyncGenerator, Generator
from datetime import datetime

import pytest
import pytest_asyncio
from faker import Faker

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# SQLModel imports - PostgreSQL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from app.db.connection_sqlmodel import get_session, init_db
from app.db.models_sqlmodel import (
    Thread, File, ThreadFile,
    ChatTurn, ProcessStatus, MessageRole
)


collect_ignore = [
    "test_modular_visualization.py",
    "test_parsing_service.py",
]


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
    
    file = File(**file_data)
    session.add(file)
    await session.commit()
    await session.refresh(file)
    return file


# Test data fixtures for ChatTurn-backed message compatibility
@pytest.fixture
def message_data(sample_thread):
    """Generate sample chat turn data."""
    return {
        "thread_id": sample_thread.id,
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
    """Create a sample chat turn in the database."""
    
    import uuid
    message = ChatTurn(
        id=str(uuid.uuid4()),
        thread_id=message_data["thread_id"],
        status="completed",
        payload={
            "question": message_data["content"],
            "rewritten_question": message_data["context_compact"],
            "answer": None,
            "reasoning": message_data["reasoning"],
            "reasoning_available": message_data["reasoning_available"],
            "reasoning_format": message_data["reasoning_format"],
            "web_sources": message_data["web_sources"],
            "metadata": {},
        },
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
    
    thread_file = ThreadFile(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        added_at=datetime.utcnow()
    )
    session.add(thread_file)
    await session.commit()
    await session.refresh(thread_file)
    return thread_file


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
    """Create a sample thread-file association with annotations."""
    annotation = ThreadFile(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        added_at=datetime.utcnow(),
        annotations=annotation_data["annotations"],
        annotations_updated_at=datetime.utcnow()
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return annotation


# Test data fixtures for thread stats fields
@pytest_asyncio.fixture
async def sample_thread_stats(session, sample_thread):
    """Populate sample thread stats fields."""
    sample_thread.total_qa_pairs = 5
    sample_thread.total_qa_chars = 1000
    sample_thread.avg_qa_chars = 200.0
    sample_thread.last_qa_at = datetime.utcnow()
    sample_thread.documents_meta = {}
    sample_thread.stats_last_updated_at = datetime.utcnow()
    session.add(sample_thread)
    await session.commit()
    await session.refresh(sample_thread)
    return sample_thread


# Fixture for multiple threads
@pytest_asyncio.fixture
async def multiple_threads(session, thread_data):
    """Create multiple sample threads."""
    
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


# Fixture for multiple chat turns
@pytest_asyncio.fixture
async def multiple_messages(session, sample_thread):
    """Create multiple sample chat turns in a thread."""
    
    import uuid
    messages = []
    for i in range(5):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        is_user = role == MessageRole.USER
        message = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            status="completed",
            payload={
                "question": fake.paragraph(nb_sentences=2) if is_user else "",
                "answer": fake.paragraph(nb_sentences=2) if not is_user else None,
                "metadata": {},
            },
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

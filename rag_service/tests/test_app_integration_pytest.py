"""
test_app_integration_pytest.py - DB-agnostic integration tests for repositories.

This module provides comprehensive integration tests for all repository operations
using SQLite with a shared database connection for test isolation. These tests are DB-agnostic
and will work with both SQLite (current) and PostgreSQL (post-migration).
"""

import os
import sys
import json
from datetime import datetime

import pytest
import pytest_asyncio
import aiosqlite

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db.repositories.thread_repo import ThreadRepository
from app.db.repositories.file_repo import FileRepository
from app.db.repositories.message_repo import MessageRepository
from app.db.repositories.thread_file_repo import ThreadFileRepository
from app.db.repositories.stats_repo import StatsRepository
from app.db.repositories.base import BaseRepository
from app.db.models import MessageRole


@pytest_asyncio.fixture
async def cleanup_db(init_test_db):
    """
    Fixture that cleans up the database after each test.
    """
    yield
    # Delete all data after test
    async with aiosqlite.connect(init_test_db) as db:
        await db.execute("PRAGMA foreign_keys = OFF")
        await db.execute("DELETE FROM thread_file_annotations")
        await db.execute("DELETE FROM thread_files")
        await db.execute("DELETE FROM messages")
        await db.execute("DELETE FROM thread_stats")
        await db.execute("DELETE FROM files")
        await db.execute("DELETE FROM threads")
        await db.commit()


@pytest_asyncio.fixture
async def repo_cleanup(init_test_db, cleanup_db):
    """
    Fixture that patches DB_PATH to use test database and cleans up after.
    Also disables foreign key constraints for integration tests.
    Uses a shared connection to ensure data visibility across repository calls.
    """
    from app.db import config
    from app.db.repositories.thread_repo import ThreadRepository
    original_path = config.DB_PATH
    config.DB_PATH = init_test_db
    
    # Create a shared connection for the test
    shared_conn = await aiosqlite.connect(init_test_db)
    await shared_conn.execute("PRAGMA foreign_keys = OFF")
    await shared_conn.commit()
    
    # Patch BaseRepository methods to use shared connection
    original_execute = BaseRepository._execute
    original_fetch_one = BaseRepository._fetch_one
    original_fetch_all = BaseRepository._fetch_all
    original_transaction = BaseRepository._transaction
    
    async def patched_execute(self, sql, params=None, commit=True):
        cursor = await shared_conn.execute(sql, params or ())
        if commit:
            await shared_conn.commit()
        # Return a mock cursor with actual rowcount
        class MockCursor:
            def __init__(self, rowcount):
                self.rowcount = rowcount
        return MockCursor(cursor.rowcount)
    
    async def patched_fetch_one(self, sql, params=None):
        shared_conn.row_factory = aiosqlite.Row
        cursor = await shared_conn.execute(sql, params or ())
        row = await cursor.fetchone()
        await shared_conn.commit()
        return row
    
    async def patched_fetch_all(self, sql, params=None):
        shared_conn.row_factory = aiosqlite.Row
        cursor = await shared_conn.execute(sql, params or ())
        rows = await cursor.fetchall()
        await shared_conn.commit()
        return rows
    
    async def patched_transaction(self, callback):
        try:
            result = await callback(shared_conn)
            await shared_conn.commit()
            return result
        except Exception:
            await shared_conn.rollback()
            raise
    
    BaseRepository._execute = patched_execute
    BaseRepository._fetch_one = patched_fetch_one
    BaseRepository._fetch_all = patched_fetch_all
    BaseRepository._transaction = patched_transaction
    
    # Patch ThreadRepository.create to use shared connection
    original_thread_create = ThreadRepository.create
    
    async def patched_thread_create(self, name, embed_model):
        import uuid
        from datetime import datetime
        from app.db.models import Thread
        thread_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        await shared_conn.execute(
            "INSERT INTO threads (id, name, embed_model, settings, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, name, embed_model, "{}", created_at)
        )
        await shared_conn.commit()
        
        return Thread(id=thread_id, name=name, embed_model=embed_model, settings={}, created_at=created_at)
    
    ThreadRepository.create = patched_thread_create
    
    yield init_test_db
    
    BaseRepository._execute = original_execute
    BaseRepository._fetch_one = original_fetch_one
    BaseRepository._fetch_all = original_fetch_all
    BaseRepository._transaction = original_transaction
    ThreadRepository.create = original_thread_create
    config.DB_PATH = original_path
    await shared_conn.close()


class TestThreadRepository:
    """Test suite for ThreadRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, repo_cleanup):
        """Create a ThreadRepository instance with test database path."""
        repo = ThreadRepository()
        repo.db_path = repo_cleanup
        return repo

    @pytest.mark.asyncio
    async def test_create_thread(self, repo):
        """Test creating a new thread."""
        thread = await repo.create("Test Thread", "BAAI/bge-m3")
        assert thread.id is not None
        assert thread.name == "Test Thread"
        assert thread.embed_model == "BAAI/bge-m3"
        assert thread.settings == {}

    @pytest.mark.asyncio
    async def test_get_thread(self, repo):
        """Test getting a thread by ID."""
        created = await repo.create("Test Thread", "BAAI/bge-m3")
        retrieved = await repo.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_thread(self, repo):
        """Test getting a thread that doesn't exist."""
        result = await repo.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_threads_empty(self, repo):
        """Test listing threads when database is empty."""
        threads = await repo.list_all()
        assert threads == []

    @pytest.mark.asyncio
    async def test_list_threads(self, repo):
        """Test listing all threads."""
        await repo.create("Thread 1", "BAAI/bge-m3")
        await repo.create("Thread 2", "BAAI/bge-m3")
        
        threads = await repo.list_all()
        assert len(threads) == 2
        assert all("id" in t for t in threads)
        assert all("name" in t for t in threads)
        assert all("message_count" in t for t in threads)
        assert all("file_count" in t for t in threads)

    @pytest.mark.asyncio
    async def test_update_thread_name(self, repo):
        """Test updating a thread's name."""
        created = await repo.create("Original Name", "BAAI/bge-m3")
        updated = await repo.update(created.id, "Updated Name")
        assert updated.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_thread(self, repo):
        """Test updating a thread that doesn't exist."""
        result = await repo.update("nonexistent-id", "New Name")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_thread(self, repo):
        """Test deleting a thread."""
        created = await repo.create("To Delete", "BAAI/bge-m3")
        result = await repo.delete(created.id)
        assert result is True
        
        retrieved = await repo.get(created.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(self, repo):
        """Test deleting a thread that doesn't exist."""
        result = await repo.delete("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_settings(self, repo):
        """Test getting thread settings."""
        created = await repo.create("Test Thread", "BAAI/bge-m3")
        settings = await repo.get_settings(created.id)
        assert settings == {}

    @pytest.mark.asyncio
    async def test_update_settings(self, repo):
        """Test updating thread settings."""
        created = await repo.create("Test Thread", "BAAI/bge-m3")
        new_settings = {"max_iterations": 20, "token_budget": 16384}
        updated = await repo.update_settings(created.id, new_settings)
        assert updated == new_settings

    @pytest.mark.asyncio
    async def test_update_settings_complex_json(self, repo):
        """Test updating thread settings with complex JSON."""
        created = await repo.create("Test Thread", "BAAI/bge-m3")
        complex_settings = {
            "max_iterations": 10,
            "token_budget": 8192,
            "nested": {
                "key1": "value1",
                "key2": ["a", "b", "c"]
            }
        }
        updated = await repo.update_settings(created.id, complex_settings)
        assert updated == complex_settings


class TestFileRepository:
    """Test suite for FileRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, repo_cleanup):
        """Create a FileRepository instance."""
        repo = FileRepository()
        repo.db_path = repo_cleanup
        return repo

    @pytest.mark.asyncio
    async def test_create_file(self, repo):
        """Test creating a new file."""
        file = await repo.create_or_get(
            "abc123",
            "test.pdf",
            "/data/test.pdf",
            "pdf"
        )
        assert file.file_hash == "abc123"
        assert file.file_name == "test.pdf"
        assert file.file_path == "/data/test.pdf"
        assert file.source_type == "pdf"

    @pytest.mark.asyncio
    async def test_create_or_get_existing(self, repo):
        """Test create_or_get returns existing file."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        # Try to create again with same hash
        file = await repo.create_or_get("abc123", "new_name.pdf", "/data/new.pdf")
        assert file.file_hash == "abc123"
        # Should update with new values
        assert file.file_name == "new_name.pdf"

    @pytest.mark.asyncio
    async def test_get_file(self, repo):
        """Test getting a file by hash."""
        created = await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        retrieved = await repo.get("abc123")
        assert retrieved is not None
        assert retrieved.file_hash == created.file_hash

    @pytest.mark.asyncio
    async def test_get_nonexistent_file(self, repo):
        """Test getting a file that doesn't exist."""
        result = await repo.get("nonexistent-hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_parsed_sentences(self, repo):
        """Test updating parsed sentences."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        parsed_data = {"sentences": [{"id": "1", "text": "Test sentence"}]}
        result = await repo.update_parsed_sentences("abc123", json.dumps(parsed_data))
        assert result is True

    @pytest.mark.asyncio
    async def test_get_parsed_sentences(self, repo):
        """Test getting parsed sentences."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        parsed_data = {"sentences": [{"id": "1", "text": "Test sentence"}]}
        await repo.update_parsed_sentences("abc123", json.dumps(parsed_data))
        
        retrieved = await repo.get_parsed_sentences("abc123")
        assert retrieved is not None
        assert "sentences" in retrieved

    @pytest.mark.asyncio
    async def test_get_file_status(self, repo):
        """Test getting file status."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        status = await repo.get_status("abc123")
        assert status is not None

    @pytest.mark.asyncio
    async def test_update_file_status(self, repo):
        """Test updating file status."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        status_data = {"parsing": {"status": "completed"}}
        result = await repo.update_status("abc123", status_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_parsing_status(self, repo):
        """Test updating parsing status."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        result = await repo.update_parsing_status(
            "abc123",
            "completed",
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:01:00"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_indexing_status(self, repo):
        """Test updating indexing status."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        result = await repo.update_indexing_status(
            "abc123",
            "completed",
            embedding_model="BAAI/bge-m3",
            thread_id="thread-123",
            chunk_count=100,
            total_chars=50000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_file(self, repo):
        """Test deleting a file."""
        await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        result = await repo.delete("abc123")
        assert result is True
        
        retrieved = await repo.get("abc123")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_source_type_default(self, repo):
        """Test that source_type defaults to 'pdf'."""
        file = await repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        assert file.source_type == "pdf"

    @pytest.mark.asyncio
    async def test_source_type_custom(self, repo):
        """Test setting custom source_type."""
        file = await repo.create_or_get("abc123", "test.docx", "/data/test.docx", "docx")
        assert file.source_type == "docx"


class TestMessageRepository:
    """Test suite for MessageRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, repo_cleanup):
        """Create a MessageRepository instance."""
        repo = MessageRepository()
        repo.db_path = repo_cleanup
        return repo

    @pytest.mark.asyncio
    async def test_create_message(self, repo):
        """Test creating a new message."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        message = await repo.create(
            thread.id,
            MessageRole.USER,
            "Test content"
        )
        assert message.id is not None
        assert message.thread_id == thread.id
        assert message.role == MessageRole.USER
        assert message.content == "Test content"

    @pytest.mark.asyncio
    async def test_create_message_with_all_fields(self, repo):
        """Test creating a message with all optional fields."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        message = await repo.create(
            thread.id,
            MessageRole.ASSISTANT,
            "Test content",
            context_compact="Compact context",
            reasoning="Reasoning text",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=[{"url": "https://example.com", "title": "Example"}]
        )
        assert message.context_compact == "Compact context"
        assert message.reasoning == "Reasoning text"
        assert message.reasoning_available is True
        assert message.reasoning_format == "markdown"
        assert message.web_sources is not None
        assert len(message.web_sources) == 1

    @pytest.mark.asyncio
    async def test_get_message(self, repo):
        """Test getting a message by ID."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        created = await repo.create(thread.id, MessageRole.USER, "Test content")
        retrieved = await repo.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == created.content

    @pytest.mark.asyncio
    async def test_get_nonexistent_message(self, repo):
        """Test getting a message that doesn't exist."""
        result = await repo.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_thread_messages(self, repo):
        """Test getting messages for a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        await repo.create(thread.id, MessageRole.USER, "Message 1")
        await repo.create(thread.id, MessageRole.ASSISTANT, "Message 2")
        await repo.create(thread.id, MessageRole.USER, "Message 3")
        
        messages = await repo.get_thread_messages(thread.id)
        assert len(messages) == 3
        assert messages[0].content == "Message 1"

    @pytest.mark.asyncio
    async def test_get_thread_messages_pagination(self, repo):
        """Test getting messages with pagination."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        for i in range(5):
            await repo.create(thread.id, MessageRole.USER, f"Message {i}")
        
        messages = await repo.get_thread_messages(thread.id, limit=2, offset=0)
        assert len(messages) == 2
        
        messages = await repo.get_thread_messages(thread.id, limit=2, offset=2)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, repo):
        """Test getting recent messages."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        for i in range(5):
            await repo.create(thread.id, MessageRole.USER, f"Message {i}")
        
        messages = await repo.get_recent_messages(thread.id, limit=3)
        assert len(messages) == 3
        # Should be in chronological order
        assert messages[0].content == "Message 2"

    @pytest.mark.asyncio
    async def test_update_context_compact(self, repo):
        """Test updating message context_compact."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        created = await repo.create(thread.id, MessageRole.USER, "Test content")
        
        result = await repo.update_context_compact(created.id, "New compact context")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_message(self, repo):
        """Test deleting a message."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        created = await repo.create(thread.id, MessageRole.USER, "Test content")
        
        result = await repo.delete(created.id)
        assert result is True
        
        retrieved = await repo.get(created.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_message_pair(self, repo):
        """Test deleting a message pair (user + assistant)."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        user_msg = await repo.create(thread.id, MessageRole.USER, "Question")
        assistant_msg = await repo.create(thread.id, MessageRole.ASSISTANT, "Answer")
        
        deleted_ids = await repo.delete_pair(assistant_msg.id)
        assert len(deleted_ids) == 2
        assert assistant_msg.id in deleted_ids
        assert user_msg.id in deleted_ids

    @pytest.mark.asyncio
    async def test_get_message_count(self, repo):
        """Test getting message count for a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        assert await repo.get_count(thread.id) == 0
        
        await repo.create(thread.id, MessageRole.USER, "Message 1")
        await repo.create(thread.id, MessageRole.ASSISTANT, "Message 2")
        
        count = await repo.get_count(thread.id)
        assert count == 2


class TestThreadFileRepository:
    """Test suite for ThreadFileRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, repo_cleanup):
        """Create a ThreadFileRepository instance."""
        repo = ThreadFileRepository()
        repo.db_path = repo_cleanup
        return repo

    @pytest.mark.asyncio
    async def test_add_file_to_thread(self, repo):
        """Test adding a file to a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        result = await repo.add(thread.id, file.file_hash)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_thread_files(self, repo):
        """Test getting files for a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        await repo.add(thread.id, file.file_hash)
        
        files = await repo.get_files(thread.id)
        assert len(files) == 1
        assert files[0].file_hash == file.file_hash

    @pytest.mark.asyncio
    async def test_remove_file_from_thread(self, repo):
        """Test removing a file from a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        await repo.add(thread.id, file.file_hash)
        
        result = await repo.remove(thread.id, file.file_hash)
        assert result is True
        
        files = await repo.get_files(thread.id)
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_is_file_in_thread(self, repo):
        """Test checking if file is in thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        assert await repo.is_file_in_thread(thread.id, file.file_hash) is False
        
        await repo.add(thread.id, file.file_hash)
        
        assert await repo.is_file_in_thread(thread.id, file.file_hash) is True

    @pytest.mark.asyncio
    async def test_count_threads_with_file(self, repo):
        """Test counting threads with a file."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        await repo.add(thread.id, file.file_hash)
        
        count = await repo.count_threads_with_file(file.file_hash)
        assert count == 1

    @pytest.mark.asyncio
    async def test_upsert_annotations(self, repo):
        """Test upserting annotations."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        annotations = [
            {"page": 1, "bbox": [100, 200, 300, 400], "text": "Test", "label": "important"}
        ]
        
        result = await repo.upsert_annotations(thread.id, file.file_hash, annotations)
        assert result is not None
        assert "annotations" in result
        assert len(result["annotations"]) == 1

    @pytest.mark.asyncio
    async def test_get_annotations(self, repo):
        """Test getting annotations."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        annotations = [{"page": 1, "text": "Test"}]
        await repo.upsert_annotations(thread.id, file.file_hash, annotations)
        
        result = await repo.get_annotations(thread.id, file.file_hash)
        assert result is not None
        assert len(result["annotations"]) == 1

    @pytest.mark.asyncio
    async def test_delete_annotations(self, repo):
        """Test deleting annotations."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        annotations = [{"page": 1, "text": "Test"}]
        await repo.upsert_annotations(thread.id, file.file_hash, annotations)
        
        count = await repo.delete_annotations(thread.id, file.file_hash)
        assert count > 0
        
        result = await repo.get_annotations(thread.id, file.file_hash)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_annotations_thread_only(self, repo):
        """Test deleting all annotations for a thread."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.file_repo import FileRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        file_repo = FileRepository()
        file_repo.db_path = repo.db_path
        file = await file_repo.create_or_get("abc123", "test.pdf", "/data/test.pdf")
        
        annotations = [{"page": 1, "text": "Test"}]
        await repo.upsert_annotations(thread.id, file.file_hash, annotations)
        
        count = await repo.delete_annotations(thread.id)
        assert count > 0


class TestStatsRepository:
    """Test suite for StatsRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, repo_cleanup):
        """Create a StatsRepository instance."""
        repo = StatsRepository()
        repo.db_path = repo_cleanup
        return repo

    @pytest.mark.asyncio
    async def test_get_thread_shape_empty(self, repo):
        """Test getting thread shape when empty."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        shape = await repo.get_thread_shape(thread.id)
        assert shape["total_qa_pairs"] == 0
        assert shape["total_qa_chars"] == 0
        assert shape["avg_qa_chars"] == 0.0
        assert shape["last_qa_at"] is None
        assert shape["documents"] == {}

    @pytest.mark.asyncio
    async def test_upsert_document_in_stats(self, repo):
        """Test upserting document metadata."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        meta = {
            "file_name": "test.pdf",
            "chunk_count": 100,
            "total_chars": 50000
        }
        
        await repo.upsert_document_in_stats(thread.id, "abc123", meta)
        
        shape = await repo.get_thread_shape(thread.id)
        assert "abc123" in shape["documents"]
        assert shape["documents"]["abc123"]["file_name"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_remove_document_from_stats(self, repo):
        """Test removing document from stats."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        meta = {"file_name": "test.pdf"}
        await repo.upsert_document_in_stats(thread.id, "abc123", meta)
        
        await repo.remove_document_from_stats(thread.id, "abc123")
        
        shape = await repo.get_thread_shape(thread.id)
        assert "abc123" not in shape["documents"]

    @pytest.mark.asyncio
    async def test_increment_qa_stats(self, repo):
        """Test incrementing QA stats."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        await repo.increment_qa_stats(thread.id, 100)
        
        shape = await repo.get_thread_shape(thread.id)
        assert shape["total_qa_pairs"] == 1
        assert shape["total_qa_chars"] == 100
        assert shape["avg_qa_chars"] == 100.0
        assert shape["last_qa_at"] is not None

    @pytest.mark.asyncio
    async def test_increment_qa_stats_multiple(self, repo):
        """Test incrementing QA stats multiple times."""
        from app.db.repositories.thread_repo import ThreadRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        await repo.increment_qa_stats(thread.id, 100)
        await repo.increment_qa_stats(thread.id, 200)
        
        shape = await repo.get_thread_shape(thread.id)
        assert shape["total_qa_pairs"] == 2
        assert shape["total_qa_chars"] == 300
        assert shape["avg_qa_chars"] == 150.0

    @pytest.mark.asyncio
    async def test_recompute_qa_stats(self, repo):
        """Test recomputing QA stats from messages."""
        from app.db.repositories.thread_repo import ThreadRepository
        from app.db.repositories.message_repo import MessageRepository
        
        thread_repo = ThreadRepository()
        thread_repo.db_path = repo.db_path
        thread = await thread_repo.create("Test Thread", "BAAI/bge-m3")
        
        msg_repo = MessageRepository()
        msg_repo.db_path = repo.db_path
        
        # Create assistant messages
        await msg_repo.create(thread.id, MessageRole.ASSISTANT, "Answer 1")
        await msg_repo.create(thread.id, MessageRole.ASSISTANT, "Answer 2 with more text")
        
        await repo.recompute_qa_stats(thread.id)
        
        shape = await repo.get_thread_shape(thread.id)
        assert shape["total_qa_pairs"] == 2
        assert shape["total_qa_chars"] > 0

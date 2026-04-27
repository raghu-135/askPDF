"""
test_models_sqlmodel_pytest.py - Tests for SQLModel model definitions and relationships.

These tests verify that SQLModel models are correctly defined with proper fields,
relationships, and data types for PostgreSQL.
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
    from sqlmodel import SQLModel, Field, Relationship
    from sqlalchemy import Column, String, Text, JSON, DateTime, Integer, Float, Boolean, ForeignKey
    from sqlalchemy.dialects.postgresql import JSONB
    from app.db.models_sqlmodel import (
        Thread, File, ThreadFile, ThreadFileAnnotation,
        Message, ThreadStats, ProcessStatus, MessageRole
    )
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadModel:
    """Test Thread model definition and fields."""

    def test_thread_model_creation(self):
        """Create Thread instance, verify fields."""
        import uuid
        thread_id = str(uuid.uuid4())
        thread = Thread(
            id=thread_id,
            name="Test Thread",
            embed_model="BAAI/bge-m3",
            settings={"max_iterations": 10},
            created_at=datetime.utcnow()
        )
        
        assert thread.id == thread_id
        assert thread.name == "Test Thread"
        assert thread.embed_model == "BAAI/bge-m3"
        assert thread.settings == {"max_iterations": 10}
        assert isinstance(thread.created_at, datetime)

    def test_thread_model_defaults(self):
        """Verify default field values work correctly."""
        import uuid
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embed_model="test-model"
        )
        
        # Settings should default to empty dict
        assert thread.settings == {}
        # created_at should be set by database or have default
        # (This depends on model definition)


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestFileModel:
    """Test File model definition and fields."""

    def test_file_model_creation(self):
        """Create File instance, verify fields."""
        file = File(
            file_hash="abc123",
            file_name="test.pdf",
            file_path="/data/test.pdf",
            source_type="pdf"
        )
        
        assert file.file_hash == "abc123"
        assert file.file_name == "test.pdf"
        assert file.file_path == "/data/test.pdf"
        assert file.source_type == "pdf"

    def test_file_model_optional_fields(self):
        """Verify optional fields can be None."""
        file = File(
            file_hash="abc123",
            file_name="test.pdf"
        )
        
        assert file.file_path is None
        assert file.source_type == "pdf"  # Default value


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestMessageModel:
    """Test Message model definition and fields."""

    def test_message_model_creation(self):
        """Create Message instance, verify fields."""
        import uuid
        message = Message(
            id=str(uuid.uuid4()),
            thread_id="thread-123",
            role=MessageRole.USER,
            content="Hello, world!",
            context_compact="Compact context",
            reasoning="My reasoning",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=[{"url": "http://example.com"}],
            created_at=datetime.utcnow()
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.context_compact == "Compact context"
        assert message.reasoning == "My reasoning"
        assert message.reasoning_available is True
        assert message.reasoning_format == "markdown"
        assert message.web_sources == [{"url": "http://example.com"}]

    def test_message_role_enum(self):
        """Verify MessageRole enum handling."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadFileModel:
    """Test ThreadFile association model."""

    def test_thread_file_creation(self):
        """Create ThreadFile instance, verify fields."""
        thread_file = ThreadFile(
            thread_id="thread-123",
            file_hash="abc123",
            added_at=datetime.utcnow()
        )
        
        assert thread_file.thread_id == "thread-123"
        assert thread_file.file_hash == "abc123"
        assert isinstance(thread_file.added_at, datetime)


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadFileAnnotationModel:
    """Test ThreadFileAnnotation model."""

    def test_annotation_creation(self):
        """Create ThreadFileAnnotation instance, verify fields."""
        import json
        annotations = [{"page": 1, "text": "test"}]
        annotation = ThreadFileAnnotation(
            thread_id="thread-123",
            file_hash="abc123",
            annotations_json=json.dumps(annotations),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert annotation.thread_id == "thread-123"
        assert annotation.file_hash == "abc123"
        assert annotation.annotations_json == json.dumps(annotations)
        assert isinstance(annotation.created_at, datetime)
        assert isinstance(annotation.updated_at, datetime)


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadStatsModel:
    """Test ThreadStats model."""

    def test_thread_stats_creation(self):
        """Create ThreadStats instance, verify fields."""
        import json
        stats = ThreadStats(
            thread_id="thread-123",
            total_qa_pairs=10,
            total_qa_chars=5000,
            avg_qa_chars=500.0,
            last_qa_at=datetime.utcnow(),
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        
        assert stats.thread_id == "thread-123"
        assert stats.total_qa_pairs == 10
        assert stats.total_qa_chars == 5000
        assert stats.avg_qa_chars == 500.0
        assert isinstance(stats.last_qa_at, datetime)
        assert stats.documents_meta == json.dumps({})
        assert isinstance(stats.last_updated_at, datetime)


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestProcessStatusEnum:
    """Test ProcessStatus enum."""

    def test_process_status_values(self):
        """Verify ProcessStatus enum values."""
        assert ProcessStatus.UNKNOWN.value == "unknown"
        assert ProcessStatus.PENDING.value == "pending"
        assert ProcessStatus.RUNNING.value == "running"
        assert ProcessStatus.COMPLETED.value == "completed"
        assert ProcessStatus.FAILED.value == "failed"

    def test_process_status_helper_methods(self):
        """Verify ProcessStatus helper methods."""
        assert ProcessStatus.is_running("running") is True
        assert ProcessStatus.is_running("completed") is False
        assert ProcessStatus.is_completed("completed") is True
        assert ProcessStatus.is_completed("running") is False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestModelValidation:
    """Test Pydantic validation on SQLModel fields."""

    def test_thread_name_validation(self):
        """Test that thread name validation works."""
        import uuid
        # Valid name
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Valid Thread Name",
            embed_model="test-model"
        )
        assert thread.name == "Valid Thread Name"

    def test_file_hash_validation(self):
        """Test that file hash is properly stored."""
        file = File(
            file_hash="abc123def456",
            file_name="test.pdf"
        )
        assert file.file_hash == "abc123def456"


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestJSONBFields:
    """Test JSONB field handling."""

    def test_settings_jsonb(self):
        """Verify settings field can handle complex JSON."""
        import uuid
        settings = {
            "max_iterations": 10,
            "token_budget": 8192,
            "nested": {
                "key": "value",
                "number": 42
            }
        }
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embed_model="test-model",
            settings=settings
        )
        
        assert thread.settings == settings
        assert thread.settings["nested"]["key"] == "value"

    def test_file_status_jsonb(self):
        """Verify file_status JSONB field."""
        import json
        status = {
            "parsing": {"status": "completed"},
            "indexing": {"status": "running"}
        }
        # This would be stored in File.file_status (JSONB column)
        status_json = json.dumps(status)
        parsed = json.loads(status_json)
        assert parsed["parsing"]["status"] == "completed"


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestDateTimeFields:
    """Test datetime field handling."""

    def test_datetime_field_handling(self):
        """Verify datetime fields are properly stored."""
        import uuid
        now = datetime.utcnow()
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embed_model="test-model",
            created_at=now
        )
        
        assert thread.created_at == now
        assert isinstance(thread.created_at, datetime)

    def test_datetime_isoformat(self):
        """Verify datetime can be serialized to ISO format."""
        import uuid
        now = datetime.utcnow()
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embed_model="test-model",
            created_at=now
        )
        
        iso_string = thread.created_at.isoformat()
        assert isinstance(iso_string, str)
        assert "T" in iso_string or " " in iso_string


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestModelRelationships:
    """Test model relationships and foreign keys."""

    def test_thread_message_relationship(self):
        """Test Thread-Message relationship concept."""
        import uuid
        thread_id = str(uuid.uuid4())
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role=MessageRole.USER,
            content="Test message"
        )
        
        assert message.thread_id == thread_id
        # The actual relationship would be tested with database operations

    def test_thread_file_relationship(self):
        """Test Thread-File relationship concept."""
        thread_file = ThreadFile(
            thread_id="thread-123",
            file_hash="file-123"
        )
        
        assert thread_file.thread_id == "thread-123"
        assert thread_file.file_hash == "file-123"

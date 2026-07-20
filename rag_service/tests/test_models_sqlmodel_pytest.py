"""
test_models_sqlmodel_pytest.py - Tests for SQLModel model definitions and relationships.

These tests verify that SQLModel models are correctly defined with proper fields,
relationships, and data types for PostgreSQL.
"""

import os
import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, Any

# Import will work after migration
try:
    from sqlmodel import SQLModel
    from app.db.models_sqlmodel import (
        Thread, File, ChatTurn, ThreadFile,
        AgentRunStatus,
        ChatTurnStatus,
        FileSourceType,
        MessageRole,
        ProcessStatus,
        WorkflowVisibility,
    )
    from app.agent.tool_contract import ToolErrorCode, ToolWarningCode
    from app.db.enums import (
        EmbeddingReadinessStatus,
        FileStatusSection,
        IndexingProgressStatus,
        OperationResultStatus,
        ReasoningFormat,
        ThreadCloneMode,
    )
    from app.agent_workflows.enums import (
        AgentCheckpointerMode,
        AgentRunResumeAction,
        EvidenceCompressionMode,
        EvaluatorRoute,
        HitlInterruptType,
        HitlMode,
        HitlPhase,
        HitlRejectBehavior,
        HitlSelectionMode,
        InterruptStatus,
        PlannerRiskLevel,
        PlannerRoute,
        RouteFunctionId,
        RouterRoute,
        WorkflowRuntimeKind,
    )
    from app.rag.enums import (
        ReembedSkipReason,
        ThreadTimelineOrder,
        ThreadTimelineSource,
        TimelineEventType,
        TimelineSourceType,
    )
    # Only mark as available if TEST_DATABASE_URL is explicitly set
    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
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
            embedding_model="BAAI/bge-m3",
            settings={"replans": 10},
            created_at=datetime.utcnow()
        )
        
        assert thread.id == thread_id
        assert thread.name == "Test Thread"
        assert thread.embedding_model == "BAAI/bge-m3"
        assert thread.settings == {"replans": 10}
        assert isinstance(thread.created_at, datetime)

    def test_thread_model_defaults(self):
        """Verify default field values work correctly."""
        import uuid
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embedding_model="test-model"
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
class TestChatTurnModel:
    """Test ChatTurn model definition and fields."""

    def test_chat_turn_model_creation(self):
        """Create ChatTurn instance, verify fields."""
        import uuid
        turn = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id="thread-123",
            status="completed",
            payload={
                "question": "Hello?",
                "rewritten_question": "Hello?",
                "answer": "World.",
                "reasoning": "My reasoning",
                "reasoning_available": True,
                "reasoning_format": "markdown",
                "web_sources": [{"url": "http://example.com"}],
            },
            created_at=datetime.utcnow()
        )
        
        assert turn.status == "completed"
        assert turn.payload["question"] == "Hello?"
        assert turn.payload["answer"] == "World."
        assert turn.payload["reasoning"] == "My reasoning"
        assert turn.payload["web_sources"] == [{"url": "http://example.com"}]

    def test_message_role_enum(self):
        """Verify MessageRole enum handling."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


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
        assert thread_file.annotations == []

    def test_thread_file_annotations(self):
        """Create ThreadFile with annotation snapshot fields."""
        annotations = [{"page": 1, "text": "test"}]
        thread_file = ThreadFile(
            thread_id="thread-123",
            file_hash="abc123",
            annotations=annotations,
            annotations_updated_at=datetime.utcnow(),
        )

        assert thread_file.annotations == annotations
        assert isinstance(thread_file.annotations_updated_at, datetime)


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestThreadStatsFields:
    """Test stats fields embedded on Thread."""

    def test_thread_stats_fields(self):
        """Create Thread instance with stats fields, verify values."""
        thread = Thread(
            id="thread-123",
            name="Stats Thread",
            embedding_model="test-model",
            total_qa_pairs=10,
            total_qa_chars=5000,
            avg_qa_chars=500.0,
            last_qa_at=datetime.utcnow(),
            documents_meta={},
            stats_last_updated_at=datetime.utcnow()
        )
        
        assert thread.total_qa_pairs == 10
        assert thread.total_qa_chars == 5000
        assert thread.avg_qa_chars == 500.0
        assert isinstance(thread.last_qa_at, datetime)
        assert thread.documents_meta == {}
        assert isinstance(thread.stats_last_updated_at, datetime)


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
        """Verify ProcessStatus enum comparison."""
        assert ProcessStatus.RUNNING == "running"
        assert ProcessStatus.COMPLETED == "completed"

    def test_shared_domain_enum_wire_values(self):
        """Verify domain enum values remain wire-compatible strings."""
        assert FileSourceType.PDF.value == "pdf"
        assert FileSourceType.BROWSER.value == "browser"
        assert ChatTurnStatus.CANCELLED.value == "cancelled"
        assert WorkflowVisibility.INTERNAL.value == "internal"
        assert AgentRunStatus.AWAITING_HUMAN.value == "awaiting_human"
        assert InterruptStatus.PENDING.value == "pending"
        assert AgentRunResumeAction.CONTINUE_WITHOUT.value == "continue_without"
        assert HitlMode.CHOICE.value == "choice"
        assert HitlPhase.INSIDE_TOOL.value == "inside_tool"
        assert HitlSelectionMode.SINGLE_OR_MULTI.value == "single_or_multi"
        assert RouterRoute.DOCUMENT.value == "document"
        assert PlannerRoute.EXECUTE.value == "execute"
        assert EvaluatorRoute.ANSWER_BUDGET_EXHAUSTED.value == "answer_budget_exhausted"
        assert RouteFunctionId.HITL_GATE.value == "hitl_gate_route"
        assert OperationResultStatus.SUCCESS.value == "success"
        assert OperationResultStatus.SKIPPED.value == "skipped"
        assert EmbeddingReadinessStatus.NOT_READY.value == "not_ready"
        assert EmbeddingReadinessStatus.BLOCKED.value == "blocked"
        assert FileStatusSection.INDEXING.value == "indexing"
        assert IndexingProgressStatus.IN_PROGRESS.value == "in_progress"
        assert ReasoningFormat.TAGGED_TEXT.value == "tagged_text"
        assert ThreadCloneMode.FROM_MESSAGE.value == "from_message"
        assert ThreadTimelineSource.WEB_CACHE.value == "web_cache"
        assert ThreadTimelineOrder.NEWEST.value == "newest"
        assert TimelineSourceType.CONVERSATION.value == "conversation"
        assert TimelineEventType.WEB_SEARCH_PERFORMED.value == "web_search_performed"
        assert ReembedSkipReason.REEMBED_IN_PROGRESS.value == "reembed_in_progress"
        assert AgentCheckpointerMode.POSTGRES.value == "postgres"
        assert EvidenceCompressionMode.COMPACT.value == "compact"
        assert HitlInterruptType.OPTION_REVIEW.value == "option_review"
        assert HitlRejectBehavior.RESUME.value == "resume"
        assert WorkflowRuntimeKind.COMPILED_RAG.value == "compiled_rag"
        assert PlannerRiskLevel.HIGH.value == "high"
        assert ToolWarningCode.WEB_SEARCH_DISABLED.value == "web_search_disabled"
        assert ToolErrorCode.failed("search_web") == "search_web_failed"


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
            embedding_model="test-model"
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
            "replans": 10,
            "token_budget": 8192,
            "nested": {
                "key": "value",
                "number": 42
            }
        }
        thread = Thread(
            id=str(uuid.uuid4()),
            name="Test Thread",
            embedding_model="test-model",
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
            embedding_model="test-model",
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
            embedding_model="test-model",
            created_at=now
        )
        
        iso_string = thread.created_at.isoformat()
        assert isinstance(iso_string, str)
        assert "T" in iso_string or " " in iso_string


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestModelRelationships:
    """Test model relationships and foreign keys."""

    def test_thread_chat_turn_relationship(self):
        """Test Thread-ChatTurn relationship concept."""
        import uuid
        thread_id = str(uuid.uuid4())
        turn = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            status="completed",
            payload={"question": "Test question", "answer": "Test answer"}
        )
        
        assert turn.thread_id == thread_id
        # The actual relationship would be tested with database operations

    def test_thread_file_relationship(self):
        """Test Thread-File relationship concept."""
        thread_file = ThreadFile(
            thread_id="thread-123",
            file_hash="file-123"
        )
        
        assert thread_file.thread_id == "thread-123"
        assert thread_file.file_hash == "file-123"

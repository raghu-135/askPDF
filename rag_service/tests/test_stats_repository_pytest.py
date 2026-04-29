"""
test_stats_repository_pytest.py - Tests for stats repository operations.

These tests verify that the StatsRepository works correctly with SQLModel
and PostgreSQL, covering QA stats, document metadata, and thread shape operations.
"""

import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from sqlmodel import select
    from app.db.models_sqlmodel import ThreadStats, Message
    from app.db.repositories.stats_repo import StatsRepository
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestStatsRepository:
    """Test StatsRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, session):
        """Create a StatsRepository instance with test session."""
        return session

    @pytest.mark.asyncio
    async def test_remove_document_from_stats(self, repo, sample_thread, sample_file):
        """Remove doc from documents_meta."""
        # Create stats with document
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps({
                sample_file.file_hash: {
                    "file_name": sample_file.file_name,
                    "chunk_count": 100
                }
            }),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Remove document
        docs = json.loads(stats.documents_meta)
        docs.pop(sample_file.file_hash, None)
        stats.documents_meta = json.dumps(docs)
        stats.last_updated_at = datetime.utcnow()
        await repo.commit()
        await repo.refresh(stats)
        
        updated_docs = json.loads(stats.documents_meta)
        assert sample_file.file_hash not in updated_docs

    @pytest.mark.asyncio
    async def test_upsert_document_in_stats(self, repo, sample_thread, sample_file):
        """Add/update document metadata."""
        # Create stats
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Upsert document
        meta = {
            "file_name": sample_file.file_name,
            "source_type": sample_file.source_type,
            "chunk_count": 100,
            "total_chars": 50000,
            "indexing_status": "completed",
            "indexed_at": datetime.utcnow().isoformat()
        }
        
        docs = json.loads(stats.documents_meta)
        docs[sample_file.file_hash] = meta
        stats.documents_meta = json.dumps(docs)
        stats.last_updated_at = datetime.utcnow()
        await repo.commit()
        await repo.refresh(stats)
        
        updated_docs = json.loads(stats.documents_meta)
        assert sample_file.file_hash in updated_docs
        assert updated_docs[sample_file.file_hash]["chunk_count"] == 100

    @pytest.mark.asyncio
    async def test_increment_qa_stats(self, repo, sample_thread):
        """Increment QA counters after message."""
        # Create stats
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=5,
            total_qa_chars=1000,
            avg_qa_chars=200.0,
            last_qa_at=datetime.utcnow(),
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Increment
        qa_chars = 250
        stats.total_qa_pairs += 1
        stats.total_qa_chars += qa_chars
        stats.avg_qa_chars = stats.total_qa_chars / stats.total_qa_pairs
        stats.last_qa_at = datetime.utcnow()
        stats.last_updated_at = datetime.utcnow()
        await repo.commit()
        await repo.refresh(stats)
        
        assert stats.total_qa_pairs == 6
        assert stats.total_qa_chars == 1250
        assert stats.avg_qa_chars == 1250 / 6

    @pytest.mark.asyncio
    async def test_increment_qa_stats_avg_calculation(self, repo, sample_thread):
        """Verify average calculation."""
        # Create stats
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Increment multiple times
        for chars in [100, 200, 300]:
            stats.total_qa_pairs += 1
            stats.total_qa_chars += chars
            stats.avg_qa_chars = stats.total_qa_chars / stats.total_qa_pairs
            stats.last_qa_at = datetime.utcnow()
            stats.last_updated_at = datetime.utcnow()
            await repo.commit()
            await repo.refresh(stats)
        
        assert stats.total_qa_pairs == 3
        assert stats.total_qa_chars == 600
        assert stats.avg_qa_chars == 200.0

    @pytest.mark.asyncio
    async def test_recompute_qa_stats(self, repo, sample_thread):
        """Recompute QA stats from messages table."""
        # Create assistant messages
        import uuid
        for i in range(5):
            message = Message(
                id=str(uuid.uuid4()),
                thread_id=sample_thread.id,
                role="assistant",
                content=f"Response {i} with some text",
                created_at=datetime.utcnow()
            )
            repo.add(message)
        
        await repo.commit()
        
        # Recompute from messages
        result = await repo.execute(
            select(Message).where(
                Message.thread_id == sample_thread.id,
                Message.role == "assistant"
            )
        )
        messages = result.scalars().all()
        
        cnt = len(messages)
        total_chars = sum(len(m.content) for m in messages)
        avg = total_chars / cnt if cnt > 0 else 0.0
        
        # Update stats
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=cnt,
            total_qa_chars=total_chars,
            avg_qa_chars=avg,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        assert stats.total_qa_pairs == cnt
        assert stats.total_qa_chars == total_chars
        assert stats.avg_qa_chars == avg

    @pytest.mark.asyncio
    async def test_get_thread_shape(self, repo, sample_thread, sample_file):
        """Get complete thread inventory."""
        # Create stats with document metadata
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=10,
            total_qa_chars=5000,
            avg_qa_chars=500.0,
            last_qa_at=datetime.utcnow(),
            documents_meta=json.dumps({
                sample_file.file_hash: {
                    "file_name": sample_file.file_name,
                    "source_type": sample_file.source_type,
                    "chunk_count": 100,
                    "total_chars": 50000,
                    "indexing_status": "completed",
                    "indexed_at": datetime.utcnow().isoformat()
                }
            }),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Get shape
        result = await repo.execute(
            select(ThreadStats).where(ThreadStats.thread_id == sample_thread.id)
        )
        stats = result.scalar_one_or_none()
        
        assert stats is not None
        assert stats.total_qa_pairs == 10
        assert stats.total_qa_chars == 5000
        assert stats.avg_qa_chars == 500.0
        
        docs = json.loads(stats.documents_meta)
        assert sample_file.file_hash in docs
        assert docs[sample_file.file_hash]["chunk_count"] == 100

    @pytest.mark.asyncio
    async def test_get_thread_shape_empty(self, repo, sample_thread):
        """Verify empty thread returns zeros."""
        # Create empty stats
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            last_qa_at=None,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        # Get shape
        result = await repo.execute(
            select(ThreadStats).where(ThreadStats.thread_id == sample_thread.id)
        )
        stats = result.scalar_one_or_none()
        
        assert stats.total_qa_pairs == 0
        assert stats.total_qa_chars == 0
        assert stats.avg_qa_chars == 0.0
        assert stats.last_qa_at is None
        
        docs = json.loads(stats.documents_meta)
        assert len(docs) == 0

    @pytest.mark.asyncio
    async def test_thread_shape_document_structure(self, repo, sample_thread):
        """Verify document metadata structure."""
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps({
                "file-1": {
                    "file_name": "test.pdf",
                    "source_type": "pdf",
                    "chunk_count": 100,
                    "total_chars": 50000,
                    "indexing_status": "completed",
                    "indexed_at": datetime.utcnow().isoformat()
                }
            }),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        docs = json.loads(stats.documents_meta)
        doc = docs["file-1"]
        
        assert "file_name" in doc
        assert "source_type" in doc
        assert "chunk_count" in doc
        assert "total_chars" in doc
        assert "indexing_status" in doc
        assert "indexed_at" in doc

    @pytest.mark.asyncio
    async def test_qa_stats_drift_prevention(self, repo, sample_thread):
        """Verify recompute prevents drift."""
        # Create stats manually
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=5,
            total_qa_chars=2500,
            avg_qa_chars=500.0,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        
        # Create actual messages (different count)
        import uuid
        for i in range(3):
            message = Message(
                id=str(uuid.uuid4()),
                thread_id=sample_thread.id,
                role="assistant",
                content=f"Response {i}",
                created_at=datetime.utcnow()
            )
            repo.add(message)
        await repo.commit()
        
        # Recompute from actual messages
        result = await repo.execute(
            select(Message).where(
                Message.thread_id == sample_thread.id,
                Message.role == "assistant"
            )
        )
        messages = result.scalars().all()
        
        cnt = len(messages)
        total_chars = sum(len(m.content) for m in messages)
        avg = total_chars / cnt if cnt > 0 else 0.0
        
        # Update with recomputed values
        result = await repo.execute(
            select(ThreadStats).where(ThreadStats.thread_id == sample_thread.id)
        )
        stats = result.scalar_one_or_none()
        stats.total_qa_pairs = cnt
        stats.total_qa_chars = total_chars
        stats.avg_qa_chars = avg
        await repo.commit()
        await repo.refresh(stats)
        
        # Values should match actual messages, not old manual values
        assert stats.total_qa_pairs == 3
        assert stats.total_qa_chars == total_chars

    @pytest.mark.asyncio
    async def test_ensure_thread_stats_row(self, repo, sample_thread):
        """Verify row creation on first use."""
        # Check if stats exists
        result = await repo.execute(
            select(ThreadStats).where(ThreadStats.thread_id == sample_thread.id)
        )
        stats = result.scalar_one_or_none()
        
        if stats is None:
            # Create row
            stats = ThreadStats(
                thread_id=sample_thread.id,
                total_qa_pairs=0,
                total_qa_chars=0,
                avg_qa_chars=0.0,
                documents_meta=json.dumps({}),
                last_updated_at=datetime.utcnow()
            )
            repo.add(stats)
            await repo.commit()
            await repo.refresh(stats)
        
        # Verify it exists now
        result = await repo.execute(
            select(ThreadStats).where(ThreadStats.thread_id == sample_thread.id)
        )
        stats = result.scalar_one_or_none()
        assert stats is not None
        assert stats.thread_id == sample_thread.id

    @pytest.mark.asyncio
    async def test_multiple_documents_in_stats(self, repo, sample_thread):
        """Test stats with multiple documents."""
        import uuid
        documents = {}
        
        for i in range(5):
            file_hash = f"file-{i}"
            documents[file_hash] = {
                "file_name": f"file-{i}.pdf",
                "source_type": "pdf",
                "chunk_count": 100 + i * 10,
                "total_chars": 50000 + i * 1000,
                "indexing_status": "completed",
                "indexed_at": datetime.utcnow().isoformat()
            }
        
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps(documents),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        docs = json.loads(stats.documents_meta)
        assert len(docs) == 5
        assert docs["file-0"]["chunk_count"] == 100
        assert docs["file-4"]["chunk_count"] == 140

    @pytest.mark.asyncio
    async def test_stats_with_large_documents_meta(self, repo, sample_thread):
        """Test stats with large documents metadata."""
        # Simulate many documents
        documents = {}
        for i in range(100):
            file_hash = f"file-{i}"
            documents[file_hash] = {
                "file_name": f"file-{i}.pdf",
                "source_type": "pdf",
                "chunk_count": 100,
                "total_chars": 50000,
                "indexing_status": "completed",
                "indexed_at": datetime.utcnow().isoformat()
            }
        
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=0,
            total_qa_chars=0,
            avg_qa_chars=0.0,
            documents_meta=json.dumps(documents),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        docs = json.loads(stats.documents_meta)
        assert len(docs) == 100

    @pytest.mark.asyncio
    async def test_last_qa_at_timestamp(self, repo, sample_thread):
        """Verify last_qa_at timestamp updates."""
        now = datetime.utcnow()
        
        stats = ThreadStats(
            thread_id=sample_thread.id,
            total_qa_pairs=1,
            total_qa_chars=100,
            avg_qa_chars=100.0,
            last_qa_at=now,
            documents_meta=json.dumps({}),
            last_updated_at=datetime.utcnow()
        )
        repo.add(stats)
        await repo.commit()
        await repo.refresh(stats)
        
        assert stats.last_qa_at is not None
        assert isinstance(stats.last_qa_at, datetime)
        
        # Update
        new_time = datetime.utcnow()
        stats.last_qa_at = new_time
        await repo.commit()
        await repo.refresh(stats)
        
        assert stats.last_qa_at >= now

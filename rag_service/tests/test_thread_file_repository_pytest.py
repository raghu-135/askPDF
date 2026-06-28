"""
test_thread_file_repository_pytest.py - Tests for thread-file association operations.

These tests verify that the ThreadFileRepository works correctly with SQLModel
and PostgreSQL, covering associations, annotations, and counting operations.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import List
import json

from sqlmodel import select
from sqlalchemy.exc import IntegrityError
from app.db.models_sqlmodel import ThreadFile, Thread, File
from app.db.repositories.thread_file_repo_sqlmodel import ThreadFileRepository


class TestThreadFileRepository:
    """Test ThreadFileRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, session):
        """Create a ThreadFileRepository instance with test session."""
        return session

    @pytest.mark.asyncio
    async def test_add_file_to_thread(self, repo, sample_thread, sample_file):
        """Associate file with thread."""
        thread_file = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow()
        )
        repo.add(thread_file)
        await repo.commit()
        await repo.refresh(thread_file)
        
        assert thread_file.thread_id == sample_thread.id
        assert thread_file.file_hash == sample_file.file_hash

    @pytest.mark.asyncio
    async def test_get_thread_files(self, repo, sample_thread, sample_file):
        """Get all files for thread, verify ordering."""
        # Add association
        thread_file = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow()
        )
        repo.add(thread_file)
        await repo.commit()
        
        # Query with join
        result = await repo.execute(
            select(File)
            .join(ThreadFile, File.file_hash == ThreadFile.file_hash)
            .where(ThreadFile.thread_id == sample_thread.id)
            .order_by(ThreadFile.added_at.desc())
        )
        files = result.scalars().all()
        
        assert len(files) >= 1
        assert files[0].file_hash == sample_file.file_hash

    @pytest.mark.asyncio
    async def test_remove_file_from_thread(self, repo, sample_thread, sample_file):
        """Remove association, verify deletion."""
        # Add association
        thread_file = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow()
        )
        repo.add(thread_file)
        await repo.commit()
        
        # Remove association
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        tf = result.scalar_one_or_none()
        await repo.delete(tf)
        await repo.commit()
        
        # Verify deletion
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_is_file_in_thread(self, repo, sample_thread, sample_file):
        """Check association exists."""
        # Add association
        thread_file = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow()
        )
        repo.add(thread_file)
        await repo.commit()
        
        # Check exists
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_count_threads_with_file(self, repo, sample_file):
        """Count thread references."""
        # Create multiple threads with the same file
        import uuid
        for i in range(3):
            thread = Thread(
                id=str(uuid.uuid4()),
                name=f"Thread {i}",
                embed_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            repo.add(thread)
            
            thread_file = ThreadFile(
                thread_id=thread.id,
                file_hash=sample_file.file_hash,
                added_at=datetime.utcnow()
            )
            repo.add(thread_file)
        
        await repo.commit()
        
        # Count
        result = await repo.execute(
            select(ThreadFile).where(ThreadFile.file_hash == sample_file.file_hash)
        )
        count = len(result.scalars().all())
        
        assert count >= 3

    @pytest.mark.asyncio
    async def test_count_threads_with_file_for_model(self, repo, sample_file):
        """Count by embedding model."""
        import uuid
        # Create threads with different models
        models = ["BAAI/bge-m3", "openai/text-embedding-3-small"]
        
        for model in models:
            thread = Thread(
                id=str(uuid.uuid4()),
                name=f"Thread for {model}",
                embed_model=model,
                settings={},
                created_at=datetime.utcnow()
            )
            repo.add(thread)
            
            thread_file = ThreadFile(
                thread_id=thread.id,
                file_hash=sample_file.file_hash,
                added_at=datetime.utcnow()
            )
            repo.add(thread_file)
        
        await repo.commit()
        
        # Count for specific model
        result = await repo.execute(
            select(ThreadFile)
            .join(Thread, Thread.id == ThreadFile.thread_id)
            .where(
                ThreadFile.file_hash == sample_file.file_hash,
                Thread.embed_model == "BAAI/bge-m3"
            )
        )
        count = len(result.scalars().all())
        
        assert count >= 1

    @pytest.mark.asyncio
    async def test_count_threads_exclude_thread(self, repo, sample_file, sample_thread):
        """Test exclude_thread_id parameter."""
        import uuid
        # Create additional threads with the file
        for i in range(2):
            thread = Thread(
                id=str(uuid.uuid4()),
                name=f"Thread {i}",
                embed_model="test-model",
                settings={},
                created_at=datetime.utcnow()
            )
            repo.add(thread)
            
            thread_file = ThreadFile(
                thread_id=thread.id,
                file_hash=sample_file.file_hash,
                added_at=datetime.utcnow()
            )
            repo.add(thread_file)
        
        # Add sample_thread association
        thread_file = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow()
        )
        repo.add(thread_file)
        await repo.commit()
        
        # Count excluding sample_thread
        result = await repo.execute(
            select(ThreadFile)
            .where(
                ThreadFile.file_hash == sample_file.file_hash,
                ThreadFile.thread_id != sample_thread.id
            )
        )
        count = len(result.scalars().all())
        
        assert count >= 2

    @pytest.mark.asyncio
    async def test_get_thread_file_annotations(self, repo, sample_thread, sample_file, annotation_data):
        """Retrieve annotations JSONB."""
        annotation = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow(),
            annotations=annotation_data["annotations"],
            annotations_updated_at=datetime.utcnow()
        )
        repo.add(annotation)
        await repo.commit()
        await repo.refresh(annotation)
        
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        retrieved = result.scalar_one_or_none()
        
        assert retrieved is not None
        annotations = retrieved.annotations
        assert len(annotations) > 0

    @pytest.mark.asyncio
    async def test_upsert_annotations(self, repo, sample_thread, sample_file):
        """Insert/update annotations."""
        # Insert
        annotations = [{"page": 1, "text": "Test"}]
        annotation = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow(),
            annotations=annotations,
            annotations_updated_at=datetime.utcnow()
        )
        repo.add(annotation)
        await repo.commit()
        await repo.refresh(annotation)
        
        # Update
        new_annotations = [{"page": 1, "text": "Updated"}, {"page": 2, "text": "New"}]
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        ann = result.scalar_one_or_none()
        ann.annotations = new_annotations
        ann.annotations_updated_at = datetime.utcnow()
        await repo.commit()
        await repo.refresh(ann)
        
        updated = ann.annotations
        assert len(updated) == 2
        assert updated[0]["text"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_annotations_specific(self, repo, sample_thread, sample_file):
        """Delete specific thread/file annotations."""
        annotation = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow(),
            annotations=[{"id": "a1"}],
            annotations_updated_at=datetime.utcnow()
        )
        repo.add(annotation)
        await repo.commit()
        
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        ann = result.scalar_one_or_none()
        ann.annotations = []
        ann.annotations_updated_at = None
        await repo.commit()
        
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id,
                ThreadFile.file_hash == sample_file.file_hash
            )
        )
        cleared = result.scalar_one_or_none()
        assert cleared is not None
        assert cleared.annotations == []
        assert cleared.annotations_updated_at is None

    @pytest.mark.asyncio
    async def test_delete_annotations_thread_wide(self, repo, sample_thread, sample_file):
        """Delete all annotations for thread."""
        # Create files first to satisfy foreign key constraint
        import uuid
        file_hashes = []
        for i in range(3):
            file_hash = f"file-{uuid.uuid4().hex[:8]}"
            file = File(
                file_hash=file_hash,
                file_name=f"file-{i}.pdf",
                source_type="pdf"
            )
            repo.add(file)
            file_hashes.append(file_hash)
        
        # Create associations with annotations for the thread
        for file_hash in file_hashes:
            annotation = ThreadFile(
                thread_id=sample_thread.id,
                file_hash=file_hash,
                added_at=datetime.utcnow(),
                annotations=[{"id": file_hash}],
                annotations_updated_at=datetime.utcnow()
            )
            repo.add(annotation)
        
        await repo.commit()
        
        # Delete all for thread
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id
            )
        )
        annotations = result.scalars().all()
        
        for ann in annotations:
            ann.annotations = []
            ann.annotations_updated_at = None
        await repo.commit()
        
        result = await repo.execute(
            select(ThreadFile).where(
                ThreadFile.thread_id == sample_thread.id
            )
        )
        assert all(not row.annotations for row in result.scalars().all())

    @pytest.mark.asyncio
    async def test_annotation_updated_at(self, repo, sample_thread, sample_file):
        """Verify annotations_updated_at updates."""
        annotation = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow(),
            annotations=[],
            annotations_updated_at=None
        )
        repo.add(annotation)
        await repo.commit()
        await repo.refresh(annotation)
        
        annotation.annotations = [{"test": "data"}]
        annotation.annotations_updated_at = datetime.utcnow()
        await repo.commit()
        await repo.refresh(annotation)
        
        assert annotation.annotations_updated_at is not None

    @pytest.mark.asyncio
    async def test_multiple_files_per_thread(self, repo, sample_thread):
        """Test thread with multiple files."""
        import uuid
        file_hashes = [f"file-{i}" for i in range(5)]
        
        for i, file_hash in enumerate(file_hashes):
            file = File(
                file_hash=file_hash,
                file_name=f"file-{i}.pdf",
                source_type="pdf"
            )
            repo.add(file)
            
            thread_file = ThreadFile(
                thread_id=sample_thread.id,
                file_hash=file_hash,
                added_at=datetime.utcnow()
            )
            repo.add(thread_file)
        
        await repo.commit()
        
        # Verify all associations
        result = await repo.execute(
            select(ThreadFile).where(ThreadFile.thread_id == sample_thread.id)
        )
        associations = result.scalars().all()
        
        assert len(associations) == 5

    @pytest.mark.asyncio
    async def test_complex_annotation_structure(self, repo, sample_thread, sample_file):
        """Test annotations with complex nested structure."""
        complex_annotations = [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": "Test text",
                "label": "important",
                "metadata": {
                    "confidence": 0.95,
                    "source": "ocr",
                    "language": "en"
                },
                "children": [
                    {"id": "child1", "text": "Nested text"}
                ]
            }
        ]
        
        annotation = ThreadFile(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            added_at=datetime.utcnow(),
            annotations=complex_annotations,
            annotations_updated_at=datetime.utcnow()
        )
        repo.add(annotation)
        await repo.commit()
        await repo.refresh(annotation)
        
        retrieved = annotation.annotations
        assert retrieved[0]["metadata"]["confidence"] == 0.95
        assert len(retrieved[0]["children"]) == 1

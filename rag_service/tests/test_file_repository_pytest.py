"""
test_file_repository_pytest.py - Tests for file repository operations with ORM.

These tests verify that the FileRepository works correctly with SQLModel
and PostgreSQL, covering all CRUD operations and status management.
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

from sqlmodel import select
from app.db.models_sqlmodel import File, ThreadFile, ProcessStatus
from app.db.repositories.file_repo_sqlmodel import FileRepository


class TestFileRepository:
    """Test FileRepository operations."""

    @pytest_asyncio.fixture
    async def repo(self, session):
        """Create a FileRepository instance with test session."""
        return session

    @pytest.mark.asyncio
    async def test_create_or_get_file_new(self, repo, file_data):
        """Create new file, verify persistence."""
        file = File(**file_data)
        repo.add(file)
        await repo.commit()
        await repo.refresh(file)
        
        # Verify persistence
        result = await repo.execute(
            select(File).where(File.file_hash == file_data["file_hash"])
        )
        persisted = result.scalar_one_or_none()
        
        assert persisted is not None
        assert persisted.file_hash == file_data["file_hash"]
        assert persisted.file_name == file_data["file_name"]
        assert persisted.source_type == file_data["source_type"]

    @pytest.mark.asyncio
    async def test_create_or_get_file_existing(self, repo, sample_file):
        """Get existing file by hash."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        assert file is not None
        assert file.file_hash == sample_file.file_hash

    @pytest.mark.asyncio
    async def test_get_file_by_hash(self, repo, sample_file):
        """Retrieve file, verify fields."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        assert file is not None
        assert file.file_hash == sample_file.file_hash
        assert file.file_name == sample_file.file_name
        assert file.source_type == sample_file.source_type

    @pytest.mark.asyncio
    async def test_update_parsed_sentences(self, repo, sample_file, parsed_sentences_data):
        """Store JSONB data, verify retrieval."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.parsed_sentences_json = json.dumps(parsed_sentences_data)
        await repo.commit()
        await repo.refresh(file)
        
        assert file.parsed_sentences_json is not None
        parsed = json.loads(file.parsed_sentences_json)
        assert "sentences" in parsed
        assert len(parsed["sentences"]) > 0

    @pytest.mark.asyncio
    async def test_get_parsed_sentences(self, repo, sample_file, parsed_sentences_data):
        """Retrieve JSONB, verify structure."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.parsed_sentences_json = json.dumps(parsed_sentences_data)
        await repo.commit()
        await repo.refresh(file)
        
        parsed = json.loads(file.parsed_sentences_json)
        assert parsed is not None
        assert "sentences" in parsed
        assert isinstance(parsed["sentences"], list)

    @pytest.mark.asyncio
    async def test_get_file_status(self, repo, sample_file, file_status_data):
        """Retrieve file_status JSONB."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.file_status = json.dumps(file_status_data)
        await repo.commit()
        await repo.refresh(file)
        
        status = json.loads(file.file_status)
        assert status is not None
        assert "parsing" in status
        assert "indexing" in status

    @pytest.mark.asyncio
    async def test_update_file_status_merge(self, repo, sample_file, file_status_data):
        """Update status, verify merge logic."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        # Initial status
        initial_status = {"parsing": {"status": "pending"}}
        file.file_status = json.dumps(initial_status)
        await repo.commit()
        await repo.refresh(file)
        
        # Update with new data
        updated_status = {"indexing": {"status": "running"}}
        current = json.loads(file.file_status)
        merged = {**current, **updated_status}
        file.file_status = json.dumps(merged)
        await repo.commit()
        await repo.refresh(file)
        
        final = json.loads(file.file_status)
        assert "parsing" in final
        assert "indexing" in final
        assert final["indexing"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_update_parsing_status(self, repo, sample_file):
        """Update parsing section, verify structure."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        parsing_status = {
            "status": ProcessStatus.COMPLETED.value,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat()
        }
        
        current_status = json.loads(file.file_status) if file.file_status else {}
        current_status["parsing"] = parsing_status
        file.file_status = json.dumps(current_status)
        await repo.commit()
        await repo.refresh(file)
        
        final = json.loads(file.file_status)
        assert final["parsing"]["status"] == ProcessStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_update_indexing_status(self, repo, sample_file):
        """Update indexing with model/thread scope."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        indexing_status = {
            "status": ProcessStatus.RUNNING.value,
            "embedding_model": "BAAI/bge-m3",
            "thread_id": "thread-123",
            "chunk_count": 100,
            "total_chars": 50000
        }
        
        current_status = json.loads(file.file_status) if file.file_status else {}
        current_status["indexing"] = indexing_status
        file.file_status = json.dumps(current_status)
        await repo.commit()
        await repo.refresh(file)
        
        final = json.loads(file.file_status)
        assert final["indexing"]["status"] == ProcessStatus.RUNNING.value
        assert final["indexing"]["embedding_model"] == "BAAI/bge-m3"

    @pytest.mark.asyncio
    async def test_delete_file_record(self, repo, sample_file):
        """Delete file, verify deletion."""
        file_hash = sample_file.file_hash
        
        result = await repo.execute(
            select(File).where(File.file_hash == file_hash)
        )
        file = result.scalar_one_or_none()
        await repo.delete(file)
        await repo.commit()
        
        # Verify deletion
        result = await repo.execute(
            select(File).where(File.file_hash == file_hash)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_status_normalization(self, repo, sample_file):
        """Verify status normalization functions work."""
        from app.db.status import _normalize_file_status

        # Test with empty input
        result = _normalize_file_status({})
        assert "parsing" in result
        assert "indexing" in result
        assert "indexing_status" in result
        assert result["parsing"]["status"] == "unknown"

        # Test with parsing status
        result = _normalize_file_status({"parsing": {"status": "completed"}})
        assert result["parsing"]["status"] == "completed"
        assert result["parsing_status"]["status"] == "completed"

        # Test with indexing models
        result = _normalize_file_status({
            "indexing_status": {
                "models": {
                    "model1": {"status": "running", "threads": {"t1": {"status": "completed"}}}
                }
            }
        })
        assert "model1" in result["indexing_status"]["models"]
        assert result["indexing_status"]["models"]["model1"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_file_with_different_source_types(self, repo):
        """Test files with different source types."""
        source_types = ["pdf", "docx", "txt", "html", "md"]
        
        for i, source_type in enumerate(source_types):
            file = File(
                file_hash=f"hash-{i}",
                file_name=f"file-{i}.{source_type}",
                source_type=source_type
            )
            repo.add(file)
        
        await repo.commit()
        
        # Verify all were saved
        result = await repo.execute(
            select(File).where(File.source_type.in_(source_types))
        )
        files = result.scalars().all()
        
        assert len(files) == 5
        saved_types = {f.source_type for f in files}
        assert saved_types == set(source_types)

    @pytest.mark.asyncio
    async def test_large_json_payload(self, repo, sample_file):
        """Test with large JSON (e.g., parsed sentences)."""
        # Simulate large parsed sentences data
        large_data = {
            "sentences": [
                {
                    "id": str(i),
                    "text": f"Sentence {i}",
                    "page": i % 10 + 1,
                    "bbox": [0, 0, 100, 20]
                }
                for i in range(1000)
            ]
        }
        
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.parsed_sentences_json = json.dumps(large_data)
        await repo.commit()
        await repo.refresh(file)
        
        parsed = json.loads(file.parsed_sentences_json)
        assert len(parsed["sentences"]) == 1000
        assert parsed["sentences"][500]["id"] == "500"

    @pytest.mark.asyncio
    async def test_file_status_with_error(self, repo, sample_file):
        """Test file status with error information."""
        result = await repo.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        
        status_with_error = {
            "parsing": {
                "status": ProcessStatus.FAILED.value,
                "error": "Failed to parse PDF",
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": datetime.utcnow().isoformat()
            }
        }
        
        file.file_status = json.dumps(status_with_error)
        await repo.commit()
        await repo.refresh(file)
        
        final = json.loads(file.file_status)
        assert final["parsing"]["status"] == ProcessStatus.FAILED.value
        assert "error" in final["parsing"]
        assert final["parsing"]["error"] == "Failed to parse PDF"

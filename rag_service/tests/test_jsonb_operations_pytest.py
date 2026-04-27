"""
test_jsonb_operations_pytest.py - Tests for PostgreSQL JSONB-specific operations.

These tests verify that JSONB fields work correctly with PostgreSQL,
including insert, update, query, and array operations.
"""

import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work after migration
try:
    from sqlmodel import select
    from sqlalchemy import text
    from app.db.models_sqlmodel import Thread, File, ThreadFileAnnotation
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available - migration not complete")
class TestJSONBOperations:
    """Test PostgreSQL JSONB field operations."""

    @pytest.mark.asyncio
    async def test_jsonb_insert_query(self, session):
        """Test JSONB insert operations."""
        import uuid
        settings = {
            "max_iterations": 10,
            "token_budget": 8192,
            "nested": {"key": "value"}
        }
        
        thread = Thread(
            id=str(uuid.uuid4()),
            name="JSONB Insert Test",
            embed_model="test-model",
            settings=settings,
            created_at=datetime.utcnow()
        )
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings == settings
        assert thread.settings["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_jsonb_update_merge(self, session, sample_thread):
        """Test JSONB merge updates."""
        # Initial settings
        initial = {"max_iterations": 10}
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = initial
        await session.commit()
        await session.refresh(thread)
        
        # Merge with new data
        new_data = {"token_budget": 8192}
        merged = {**thread.settings, **new_data}
        thread.settings = merged
        await session.commit()
        await session.refresh(thread)
        
        assert "max_iterations" in thread.settings
        assert "token_budget" in thread.settings
        assert thread.settings["token_budget"] == 8192

    @pytest.mark.asyncio
    async def test_jsonb_query_nested(self, session, sample_thread):
        """Test querying nested JSONB fields."""
        # Set nested settings
        settings = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        # Query and access nested value
        assert thread.settings["level1"]["level2"]["level3"]["value"] == "deep"

    @pytest.mark.asyncio
    async def test_jsonb_array_operations(self, session, sample_thread):
        """Test JSONB array handling."""
        settings = {
            "allowed_models": ["model1", "model2", "model3"],
            "numbers": [1, 2, 3, 4, 5]
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        assert isinstance(thread.settings["allowed_models"], list)
        assert len(thread.settings["allowed_models"]) == 3
        assert thread.settings["numbers"][2] == 3

    @pytest.mark.asyncio
    async def test_large_json_payload(self, session, sample_file):
        """Test with large JSON (e.g., parsed sentences)."""
        # Simulate large parsed sentences
        large_data = {
            "sentences": [
                {
                    "id": str(i),
                    "text": f"Sentence {i} with some content",
                    "page": i % 10 + 1,
                    "bbox": [0, 0, 100, 20],
                    "metadata": {
                        "confidence": 0.95,
                        "font": "Arial",
                        "size": 12
                    }
                }
                for i in range(1000)
            ]
        }
        
        result = await session.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.parsed_sentences_json = json.dumps(large_data)
        await session.commit()
        await session.refresh(file)
        
        parsed = json.loads(file.parsed_sentences_json)
        assert len(parsed["sentences"]) == 1000
        assert parsed["sentences"][500]["id"] == "500"

    @pytest.mark.asyncio
    async def test_jsonb_null_handling(self, session, sample_thread):
        """Verify NULL JSONB handling."""
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        
        # Set to empty dict (not None)
        thread.settings = {}
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings == {}
        assert isinstance(thread.settings, dict)

    @pytest.mark.asyncio
    async def test_jsonb_index_performance(self, session):
        """Basic performance test for JSONB queries."""
        import time
        import uuid
        
        # Create threads with JSONB settings
        start = time.time()
        for i in range(100):
            thread = Thread(
                id=str(uuid.uuid4()),
                name=f"Perf Thread {i}",
                embed_model="test-model",
                settings={"index": i, "data": f"value-{i}"},
                created_at=datetime.utcnow()
            )
            session.add(thread)
        await session.commit()
        insert_time = time.time() - start
        
        # Query by JSONB content
        start = time.time()
        result = await session.execute(
            select(Thread).where(Thread.settings["index"] == 50)
        )
        thread = result.scalar_one_or_none()
        query_time = time.time() - start
        
        assert thread is not None
        assert thread.settings["index"] == 50
        print(f"Insert time: {insert_time:.3f}s, Query time: {query_time:.3f}s")

    @pytest.mark.asyncio
    async def test_jsonb_with_special_characters(self, session, sample_thread):
        """Test JSONB with special characters and unicode."""
        settings = {
            "unicode": "Hello 世界 🌍",
            "special": "Line\nBreak\tTab",
            "quotes": 'Text with "quotes" and \'apostrophes\'',
            "emoji": ["😀", "😎", "🚀"]
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings["unicode"] == "Hello 世界 🌍"
        assert "\n" in thread.settings["special"]
        assert "quotes" in thread.settings["quotes"]
        assert thread.settings["emoji"][0] == "😀"

    @pytest.mark.asyncio
    async def test_jsonb_update_partial(self, session, sample_thread):
        """Test updating part of JSONB without full replacement."""
        # Set initial complex settings
        settings = {
            "section1": {"value": "original"},
            "section2": {"value": "original"},
            "section3": {"value": "original"}
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        # Update only section2
        thread.settings["section2"]["value"] = "updated"
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings["section1"]["value"] == "original"
        assert thread.settings["section2"]["value"] == "updated"
        assert thread.settings["section3"]["value"] == "original"

    @pytest.mark.asyncio
    async def test_jsonb_boolean_values(self, session, sample_thread):
        """Test JSONB with boolean values."""
        settings = {
            "feature1": True,
            "feature2": False,
            "feature3": True,
            "nested": {
                "enabled": True,
                "disabled": False
            }
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings["feature1"] is True
        assert thread.settings["feature2"] is False
        assert thread.settings["nested"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_jsonb_numeric_values(self, session, sample_thread):
        """Test JSONB with various numeric types."""
        settings = {
            "integer": 42,
            "float": 3.14159,
            "negative": -10,
            "zero": 0,
            "large": 1000000,
            "scientific": 1.23e-4
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings["integer"] == 42
        assert abs(thread.settings["float"] - 3.14159) < 0.0001
        assert thread.settings["negative"] == -10
        assert thread.settings["zero"] == 0
        assert thread.settings["large"] == 1000000

    @pytest.mark.asyncio
    async def test_jsonb_mixed_types(self, session, sample_thread):
        """Test JSONB with mixed data types."""
        settings = {
            "string": "text",
            "number": 123,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"key": "value"},
            "mixed": [
                {"id": 1, "name": "item1"},
                {"id": 2, "name": "item2"}
            ]
        }
        result = await session.execute(
            select(Thread).where(Thread.id == sample_thread.id)
        )
        thread = result.scalar_one_or_none()
        thread.settings = settings
        await session.commit()
        await session.refresh(thread)
        
        assert thread.settings["string"] == "text"
        assert thread.settings["number"] == 123
        assert thread.settings["boolean"] is True
        assert thread.settings["null"] is None
        assert thread.settings["array"] == [1, 2, 3]
        assert thread.settings["object"]["key"] == "value"
        assert thread.settings["mixed"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_jsonb_annotation_storage(self, session, sample_thread, sample_file):
        """Test storing complex annotation data in JSONB."""
        annotations = [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": "Sample text",
                "label": "important",
                "confidence": 0.95,
                "metadata": {
                    "source": "ocr",
                    "language": "en",
                    "font": "Arial"
                }
            },
            {
                "page": 2,
                "bbox": [50, 100, 250, 150],
                "text": "Another text",
                "label": "normal",
                "confidence": 0.87,
                "metadata": {
                    "source": "manual",
                    "language": "en"
                }
            }
        ]
        
        annotation = ThreadFileAnnotation(
            thread_id=sample_thread.id,
            file_hash=sample_file.file_hash,
            annotations_json=json.dumps(annotations),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        session.add(annotation)
        await session.commit()
        await session.refresh(annotation)
        
        retrieved = json.loads(annotation.annotations_json)
        assert len(retrieved) == 2
        assert retrieved[0]["label"] == "important"
        assert retrieved[1]["confidence"] == 0.87

    @pytest.mark.asyncio
    async def test_jsonb_file_status_complex(self, session, sample_file):
        """Test complex file status structure in JSONB."""
        file_status = {
            "parsing": {
                "status": "completed",
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": datetime.utcnow().isoformat(),
                "pages_processed": 10,
                "errors": []
            },
            "indexing": {
                "status": "running",
                "embedding_model": "BAAI/bge-m3",
                "chunk_count": 150,
                "total_chars": 75000,
                "models": {
                    "BAAI/bge-m3": {
                        "status": "completed",
                        "threads": {
                            "thread-1": {
                                "status": "completed",
                                "chunk_count": 150
                            }
                        }
                    }
                }
            }
        }
        
        result = await session.execute(
            select(File).where(File.file_hash == sample_file.file_hash)
        )
        file = result.scalar_one_or_none()
        file.file_status = json.dumps(file_status)
        await session.commit()
        await session.refresh(file)
        
        retrieved = json.loads(file.file_status)
        assert retrieved["parsing"]["status"] == "completed"
        assert retrieved["indexing"]["models"]["BAAI/bge-m3"]["threads"]["thread-1"]["status"] == "completed"

"""
test_dimension_mismatch_scenarios.py - Specific tests for dimension mismatch scenarios.

This test suite focuses on:
- Real-world dimension mismatch scenarios
- Cross-model dimension conflicts
- Migration scenarios with dimension changes
- Error handling and recovery
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from app.db.vector.model_registry import EmbeddingModelRegistry
from app.db.vector.collection_manager import ModelAwareCollectionManager
from app.db.vector.adapter import WeaviateAdapter
from app.db.vector.config import VectorDBInsertError


class TestDimensionMismatchScenarios:
    """Test specific dimension mismatch scenarios."""
    
    @pytest.fixture
    def registry(self):
        registry = EmbeddingModelRegistry()
        # Pre-populate with different model dimensions
        registry._model_cache.update({
            "model-384": {"dimensions": 384, "sanitized_name": "model_384", "is_local": False},
            "model-768": {"dimensions": 768, "sanitized_name": "model_768", "is_local": False},
            "model-1536": {"dimensions": 1536, "sanitized_name": "model_1536", "is_local": True},
        })
        registry._dimension_cache.update({
            "model-384": 384,
            "model-768": 768,
            "model-1536": 1536,
        })
        return registry
    
    @pytest.mark.asyncio
    async def test_cross_model_dimension_conflicts(self, registry):
        """Test scenarios where different models have different dimensions."""
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Test vectors for 384-dim model
        vectors_384 = [[0.1] * 384, [0.2] * 384]
        assert await collection_manager.validate_vectors_for_model(vectors_384, "model-384")
        
        # Same vectors should fail for 768-dim model
        assert not await collection_manager.validate_vectors_for_model(vectors_384, "model-768")
        
        # Test vectors for 768-dim model
        vectors_768 = [[0.1] * 768, [0.2] * 768]
        assert await collection_manager.validate_vectors_for_model(vectors_768, "model-768")
        
        # Same vectors should fail for 384-dim model
        assert not await collection_manager.validate_vectors_for_model(vectors_768, "model-384")
        
        # Test vectors for 1536-dim model
        vectors_1536 = [[0.1] * 1536, [0.2] * 1536]
        assert await collection_manager.validate_vectors_for_model(vectors_1536, "model-1536")
        
        # Should fail for all other models
        assert not await collection_manager.validate_vectors_for_model(vectors_1536, "model-384")
        assert not await collection_manager.validate_vectors_for_model(vectors_1536, "model-768")
    
    @pytest.mark.asyncio
    async def test_mixed_dimension_batch_validation(self, registry):
        """Test validation of batches with mixed dimensions."""
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Mixed dimensions in same batch should fail
        mixed_vectors = [
            [0.1] * 384,   # 384-dim
            [0.2] * 768,   # 768-dim - mismatch!
            [0.3] * 384,   # 384-dim
        ]
        
        assert not await collection_manager.validate_vectors_for_model(mixed_vectors, "model-384")
        assert not await collection_manager.validate_vectors_for_model(mixed_vectors, "model-768")
        
        # All vectors with correct dimensions should pass
        correct_384 = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        assert await collection_manager.validate_vectors_for_model(correct_384, "model-384")
        
        correct_768 = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        assert await collection_manager.validate_vectors_for_model(correct_768, "model-768")
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch_error_messages(self, registry):
        """Test that dimension mismatch errors provide helpful information."""
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Test with wrong dimensions
        wrong_vectors = [[0.1] * 768]  # 768-dim vectors for 384-dim model
        
        with patch('app.db.vector.collection_manager.logger') as mock_logger:
            result = await collection_manager.validate_vectors_for_model(wrong_vectors, "model-384")
            
            assert not result
            # Should log specific error about dimension mismatch
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Vector 0 has 768 dimensions, expected 384" in error_call
    
    @pytest.mark.asyncio
    async def test_adapter_dimension_mismatch_prevention(self, registry):
        """Test that adapter prevents dimension mismatches at indexing time."""
        with patch('app.db.vector.collection_manager.ModelAwareCollectionManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.validate_vectors_for_model.return_value = False
            mock_manager.get_collection.return_value = AsyncMock()
            
            adapter = WeaviateAdapter.__new__(WeaviateAdapter)
            adapter.collection_manager = mock_manager
            
            # Test PDF indexing
            with pytest.raises(ValueError, match=r"Vector dimensions do not match expected dimensions for model 'model-384'"):
                await adapter.index_pdf_chunks(
                    thread_id="test-thread",
                    embedding_model_name="model-384",
                    file_hash="test-file",
                    texts=["test chunk"],
                    embeddings=[[0.1] * 768],  # Wrong dimensions
                    metadatas=[{}]
                )
            
            # Test chat memory indexing
            with pytest.raises(ValueError, match=r"Vector dimensions do not match expected dimensions for model 'model-768'"):
                await adapter.index_chat_memory(
                    thread_id="test-thread",
                    message_id="test-message",
                    question="test question",
                    answer="test answer",
                    texts=["test chunk"],
                    embeddings=[[0.1] * 384],  # Wrong dimensions
                    embedding_model_name="model-768"
                )
            
            # Test web search indexing
            with pytest.raises(ValueError, match=r"Vector dimensions do not match expected dimensions for model 'model-1536'"):
                await adapter.index_web_search_chunks(
                    thread_id="test-thread",
                    query="test query",
                    texts=["test chunk"],
                    embeddings=[[0.1] * 768],  # Wrong dimensions
                    embedding_model_name="model-1536"
                )


class TestMigrationScenarios:
    """Test scenarios that might occur during model migration."""
    
    @pytest.mark.asyncio
    async def test_model_dimension_change_migration(self):
        """Test handling when a model's dimensions change (rare but possible)."""
        registry = EmbeddingModelRegistry()
        
        # Simulate initial model info
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            await registry.get_model_info("upgraded-model")
            assert registry._dimension_cache["upgraded-model"] == 384
        
        # Simulate model upgrade with new dimensions
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            # Force refresh by clearing cache
            registry._model_cache.clear()
            registry._dimension_cache.clear()
            
            await registry.get_model_info("upgraded-model")
            assert registry._dimension_cache["upgraded-model"] == 768
        
        # Collection name should reflect new dimensions
        collection_name = registry.get_collection_name("DocumentChunk", "upgraded-model")
        assert collection_name.endswith("_768")
        assert not collection_name.endswith("_384")
    
    @pytest.mark.asyncio
    async def test_collection_compatibility_after_dimension_change(self):
        """Test collection compatibility checks after dimension changes."""
        registry = EmbeddingModelRegistry()
        
        # Setup initial model
        registry._model_cache["model-v1"] = {"dimensions": 384, "sanitized_name": "model_v1", "is_local": False}
        registry._dimension_cache["model-v1"] = 384
        
        # Collection should be compatible
        collection_name = registry.get_collection_name("DocumentChunk", "model-v1")
        assert registry.is_model_compatible(collection_name, "model-v1")
        
        # Simulate model upgrade
        registry._model_cache["model-v1"] = {"dimensions": 768, "sanitized_name": "model_v1", "is_local": False}
        registry._dimension_cache["model-v1"] = 768
        
        # Same collection should now be incompatible
        assert not registry.is_model_compatible(collection_name, "model-v1")
        
        # New collection name should be different
        new_collection_name = registry.get_collection_name("DocumentChunk", "model-v1")
        assert new_collection_name != collection_name
        assert new_collection_name.endswith("_768")
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_with_legacy_collections(self):
        """Test handling of legacy collections without dimension info."""
        registry = EmbeddingModelRegistry()
        
        # Test with collection names that don't follow the new pattern
        legacy_collections = [
            "DocumentChunk",
            "DocumentChunk_legacy",
            "ChatMemoryChunk",
            "WebSearchChunk",
        ]
        
        for collection_name in legacy_collections:
            # Should return None for unknown model
            model = registry.get_model_from_collection(collection_name)
            assert model is None
            
            # Should not be compatible with any model
            assert not registry.is_model_compatible(collection_name, "any-model")


class TestErrorRecoveryScenarios:
    """Test error recovery and resilience scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_batch_insertion_recovery(self):
        """Test recovery from partial batch insertion failures."""
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        
        # Simulate batch failure after some insertions
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return None  # Success for first 2
            else:
                raise Exception("Batch insertion failed")
        
        mock_batch.add_object.side_effect = side_effect
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.dynamic.return_value.__exit__.return_value = None
        
        adapter = WeaviateAdapter.__new__(WeaviateAdapter)
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("Batch failed")
            
            with pytest.raises(VectorDBInsertError):
                await adapter._insert_many_model_aware(
                    collection=mock_collection,
                    points=[{"properties": {}, "vector": [0.1]} for _ in range(5)]
                )
    
    @pytest.mark.asyncio
    async def test_model_registry_recovery_after_failure(self):
        """Test model registry recovery after loading failures."""
        registry = EmbeddingModelRegistry()
        
        # Simulate initial failure
        with patch.object(registry, '_probe_model_dimensions', side_effect=Exception("Model unavailable")):
            with pytest.raises(Exception, match=r"Could not determine dimensions"):
                await registry.get_model_info("failing-model")
        
        # Verify caches are clean after failure
        assert "failing-model" not in registry._model_cache
        assert "failing-model" not in registry._dimension_cache
        
        # Simulate recovery
        with patch.object(registry, '_probe_model_dimensions', return_value=512):
            model_info = await registry.get_model_info("failing-model")
            assert model_info['dimensions'] == 512
            assert "failing-model" in registry._model_cache
            assert "failing-model" in registry._dimension_cache
    
    @pytest.mark.asyncio
    async def test_collection_creation_retry_logic(self):
        """Test retry logic for collection creation failures."""
        mock_client = MagicMock()
        
        # Simulate intermittent failures
        call_count = 0
        def exists_side_effect(collection_name):
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Fail first time, succeed second time
        
        mock_client.collections.exists.side_effect = exists_side_effect
        mock_client.collections.create.return_value = None
        mock_client.collections.use.return_value = MagicMock()
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            # Should succeed on retry
            collection = await collection_manager.get_collection("DocumentChunk", "test-model")
            assert collection is not None
            
            # Should have called exists twice (initial failure + retry)
            assert mock_client.collections.exists.call_count == 2


class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_batch_dimension_validation(self):
        """Test dimension validation with large batches."""
        registry = EmbeddingModelRegistry()
        registry._dimension_cache["large-model"] = 1536
        
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Test with large batch (1000 vectors)
        large_batch = [[0.1] * 1536 for _ in range(1000)]
        
        import time
        start_time = time.time()
        result = await collection_manager.validate_vectors_for_model(large_batch, "large-model")
        end_time = time.time()
        
        assert result  # Should succeed
        assert end_time - start_time < 1.0  # Should complete quickly (< 1 second)
    
    @pytest.mark.asyncio
    async def test_concurrent_dimension_validation(self):
        """Test concurrent dimension validation requests."""
        registry = EmbeddingModelRegistry()
        registry._dimension_cache["concurrent-model"] = 768
        
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Create multiple concurrent validation tasks
        vectors = [[0.1] * 768 for _ in range(100)]
        tasks = []
        
        for i in range(10):
            task = collection_manager.validate_vectors_for_model(vectors, "concurrent-model")
            tasks.append(task)
        
        # All should complete successfully
        results = await asyncio.gather(*tasks)
        assert all(results)  # All should return True
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_many_models(self):
        """Test memory usage with many different models."""
        registry = EmbeddingModelRegistry()
        
        # Add many models to cache
        for i in range(100):
            model_name = f"model-{i}"
            dimensions = 384 + (i % 4) * 384  # Vary dimensions
            registry._model_cache[model_name] = {
                "dimensions": dimensions,
                "sanitized_name": f"model_{i}",
                "is_local": i % 2 == 0
            }
            registry._dimension_cache[model_name] = dimensions
        
        # Test operations with many models
        for i in range(100):
            model_name = f"model-{i}"
            dimensions = registry._dimension_cache[model_name]
            
            # Test collection name generation
            collection_name = registry.get_collection_name("DocumentChunk", model_name)
            assert collection_name.endswith(f"_{dimensions}")
            
            # Test compatibility check
            assert registry.is_model_compatible(collection_name, model_name)
        
        # Cache sizes should be reasonable
        assert len(registry._model_cache) == 100
        assert len(registry._dimension_cache) == 100


if __name__ == "__main__":
    pytest.main([__file__])

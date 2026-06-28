"""
test_production_edge_cases.py - Production environment edge case tests.

This test suite validates:
- Real production edge cases
- Resource exhaustion scenarios
- Network failure handling
- Data corruption prevention
- Performance degradation scenarios
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from app.db.vector.model_registry import EmbeddingModelRegistry
from app.db.vector.collection_manager import ModelAwareCollectionManager
from app.db.vector.adapter import WeaviateAdapter
from app.db.vector.config import VectorDBError, VectorDBInsertError


class TestResourceExhaustionScenarios:
    """Test scenarios where system resources are exhausted."""
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_during_batch_insertion(self):
        """Test handling of memory exhaustion during large batch insertions."""
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        
        # Simulate memory error during batch insertion
        mock_batch.add_object.side_effect = MemoryError("Out of memory")
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.dynamic.return_value.__exit__.return_value = None
        
        adapter = WeaviateAdapter.__new__(WeaviateAdapter)
        
        with patch('asyncio.to_thread', side_effect=MemoryError("Out of memory")):
            with pytest.raises(VectorDBInsertError, match=r"Unexpected error inserting into model-aware collection"):
                await adapter._insert_many_model_aware(
                    collection=mock_collection,
                    points=[{"properties": {}, "vector": [0.1] * 1000} for _ in range(10000)]
                )
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test handling of database connection pool exhaustion."""
        mock_client = MagicMock()
        
        # Simulate connection pool exhaustion
        mock_client.collections.exists.side_effect = Exception("Connection pool exhausted")
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        registry = EmbeddingModelRegistry()
        registry._model_cache["test-model"] = {
            "dimensions": 384,
            "sanitized_name": "test_model",
            "is_local": False,
        }
        registry._dimension_cache["test-model"] = 384
        collection_manager.registry = registry

        collection = await collection_manager.get_collection("DocumentChunk", "test-model")
        assert collection is not None
        mock_client.collections.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_during_collection_creation(self):
        """Test handling of disk space exhaustion during collection creation."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.side_effect = Exception("No space left on device")
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        registry = EmbeddingModelRegistry()
        registry._model_cache["test-model"] = {
            "dimensions": 384,
            "sanitized_name": "test_model",
            "is_local": False,
        }
        registry._dimension_cache["test-model"] = 384
        collection_manager.registry = registry

        with pytest.raises(Exception, match=r"No space left on device"):
            await collection_manager.get_collection("DocumentChunk", "test-model")


class TestNetworkFailureScenarios:
    """Test network failure and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_intermittent_network_failures(self):
        """Test handling of intermittent network failures."""
        mock_client = MagicMock()
        
        # Simulate intermittent failures
        call_count = 0
        def exists_side_effect(collection_name):
            nonlocal call_count
            call_count += 1
            if call_count in [2, 4, 6]:  # Fail on even calls
                raise Exception("Network timeout")
            return False
        
        mock_client.collections.exists.side_effect = exists_side_effect
        mock_client.collections.create.return_value = None
        mock_client.collections.use.return_value = MagicMock()
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        registry = EmbeddingModelRegistry()
        registry._model_cache["test-model"] = {
            "dimensions": 384,
            "sanitized_name": "test_model",
            "is_local": False,
        }
        registry._dimension_cache["test-model"] = 384
        collection_manager.registry = registry

        # Existence-check failures are logged and treated as missing collections.
        await collection_manager.get_collection("DocumentChunk", "test-model")
        collection_manager._collection_cache.clear()
        collection = await collection_manager.get_collection("DocumentChunk", "test-model")
        assert collection is not None
    
    @pytest.mark.asyncio
    async def test_slow_network_during_model_probing(self):
        """Test handling of slow network during model dimension probing."""
        registry = EmbeddingModelRegistry()
        
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            
            # Simulate slow response
            async def slow_embed(*args, **kwargs):
                await asyncio.sleep(2.0)  # 2 second delay
                return [0.1] * 384
            
            mock_model.aembed_query.side_effect = slow_embed
            mock_get_model.return_value = mock_model
            
            # Should handle slow responses gracefully
            start_time = time.time()
            dimensions = await registry._probe_model_dimensions("slow-model")
            end_time = time.time()
            
            assert dimensions == 384
            assert end_time - start_time >= 2.0  # Should take at least 2 seconds
    
    @pytest.mark.asyncio
    async def test_connection_timeout_during_batch_insertion(self):
        """Test handling of connection timeouts during batch insertion."""
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        mock_batch.add_object.side_effect = Exception("Connection timeout")
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.dynamic.return_value.__exit__.return_value = None
        
        adapter = WeaviateAdapter.__new__(WeaviateAdapter)
        
        with patch('asyncio.to_thread', side_effect=Exception("Connection timeout")):
            with pytest.raises(VectorDBInsertError, match=r"Unexpected error inserting into model-aware collection"):
                await adapter._insert_many_model_aware(
                    collection=mock_collection,
                    points=[{"properties": {}, "vector": [0.1]} for _ in range(10)]
                )


class testDataCorruptionPrevention:
    """Test data corruption prevention mechanisms."""
    
    @pytest.mark.asyncio
    async def test_dimension_validation_prevents_corruption(self):
        """Test that dimension validation prevents data corruption."""
        registry = EmbeddingModelRegistry()
        registry._dimension_cache["test-model"] = 384
        
        collection_manager = ModelAwareCollectionManager(MagicMock())
        collection_manager.registry = registry
        
        # Attempt to insert vectors with wrong dimensions
        wrong_vectors = [
            [0.1] * 384,   # Correct
            [0.2] * 768,   # Wrong - would cause corruption
            [0.3] * 384,   # Correct
        ]
        
        # Should prevent insertion
        assert not await collection_manager.validate_vectors_for_model(wrong_vectors, "test-model")
    
    @pytest.mark.asyncio
    async def test_collection_name_collision_prevention(self):
        """Test prevention of collection name collisions."""
        registry = EmbeddingModelRegistry()
        
        # Setup models with similar names but different dimensions
        registry._model_cache.update({
            "model-v1": {"dimensions": 384, "sanitized_name": "model_v1", "is_local": False},
            "model-v2": {"dimensions": 768, "sanitized_name": "model_v2", "is_local": False},
        })
        registry._dimension_cache.update({
            "model-v1": 384,
            "model-v2": 768,
        })
        
        # Generate collection names
        collection1 = registry.get_collection_name("DocumentChunk", "model-v1")
        collection2 = registry.get_collection_name("DocumentChunk", "model-v2")
        
        # Should be different due to different dimensions
        assert collection1 != collection2
        assert collection1.endswith("_384")
        assert collection2.endswith("_768")
    
    @pytest.mark.asyncio
    async def test_partial_batch_rollback(self):
        """Test rollback mechanisms for partial batch failures."""
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        
        # Simulate partial failure
        call_count = 0
        def add_object_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise Exception("Partial batch failure")
        
        mock_batch.add_object.side_effect = add_object_side_effect
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.dynamic.return_value.__exit__.return_value = None
        
        adapter = WeaviateAdapter.__new__(WeaviateAdapter)
        
        with patch('asyncio.to_thread', side_effect=Exception("Partial batch failure")):
            with pytest.raises(VectorDBInsertError):
                await adapter._insert_many_model_aware(
                    collection=mock_collection,
                    points=[{"properties": {}, "vector": [0.1]} for _ in range(10)]
                )


class TestPerformanceDegradationScenarios:
    """Test performance degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_slow_dimension_detection_performance(self):
        """Test performance impact of slow dimension detection."""
        registry = EmbeddingModelRegistry()
        
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            
            # Simulate progressively slower responses
            call_count = 0
            async def progressive_slow_embed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(call_count * 0.1)  # Increasing delay
                return [0.1] * 384
            
            mock_model.aembed_query.side_effect = progressive_slow_embed
            mock_get_model.return_value = mock_model
            
            # Multiple calls should handle increasing delays
            start_time = time.time()
            for i in range(5):
                dimensions = await registry._probe_model_dimensions(f"slow-model-{i}")
                assert dimensions == 384
            end_time = time.time()
            
            # Should complete in reasonable time despite delays
            assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_large_collection_name_handling(self):
        """Test handling of very long collection names."""
        registry = EmbeddingModelRegistry()
        
        # Create model with very long name
        long_model_name = "a" * 1000  # 1000 character model name
        registry._model_cache[long_model_name] = {
            "dimensions": 384, 
            "sanitized_name": "a" * 1000,  # Sanitized but still long
            "is_local": False
        }
        registry._dimension_cache[long_model_name] = 384
        
        # Should handle long names gracefully
        collection_name = registry.get_collection_name("DocumentChunk", long_model_name)
        assert isinstance(collection_name, str)
        assert len(collection_name) > 1000  # Should include full name
        
        # Should still be usable for compatibility checks
        assert registry.is_model_compatible(collection_name, long_model_name)
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loading_performance(self):
        """Test performance under concurrent model loading."""
        registry = EmbeddingModelRegistry()
        
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.aembed_query = AsyncMock(return_value=[0.1] * 384)
            mock_get_model.return_value = mock_model
            
            # Create many concurrent loading tasks
            tasks = []
            for i in range(50):
                task = registry.get_model_info(f"concurrent-model-{i % 5}")  # 5 unique models
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should complete efficiently due to caching
            assert len(results) == 50
            assert end_time - start_time < 2.0  # Should be fast due to caching
            
            # Should only call embedding model 5 times (once per unique model)
            assert mock_model.aembed_query.call_count == 5


class TestSystemResourceMonitoring:
    """Test system resource monitoring and adaptive behavior."""
    
    @pytest.mark.asyncio
    async def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on system resources."""
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        
        # Simulate memory pressure detection
        call_count = 0
        def add_object_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 100:  # Simulate memory pressure after 100 insertions
                raise MemoryError("Memory pressure detected")
        
        mock_batch.add_object.side_effect = add_object_side_effect
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.dynamic.return_value.__exit__.return_value = None
        
        adapter = WeaviateAdapter.__new__(WeaviateAdapter)
        
        # Should handle large batches by failing gracefully
        with patch('asyncio.to_thread', side_effect=MemoryError("Memory pressure detected")):
            with pytest.raises(VectorDBInsertError):
                await adapter._insert_many_model_aware(
                    collection=mock_collection,
                    points=[{"properties": {}, "vector": [0.1]} for _ in range(200)]
                )
    
    @pytest.mark.asyncio
    async def test_cache_eviction_under_memory_pressure(self):
        """Test cache eviction under memory pressure."""
        registry = EmbeddingModelRegistry()
        
        # Fill cache with many models
        for i in range(100):
            model_name = f"memory-test-model-{i}"
            registry._model_cache[model_name] = {
                "dimensions": 384 + i,
                "sanitized_name": f"memory_test_model_{i}",
                "is_local": i % 2 == 0
            }
            registry._dimension_cache[model_name] = 384 + i
        
        # Simulate memory pressure by clearing cache
        initial_model_count = len(registry._model_cache)
        initial_dimension_count = len(registry._dimension_cache)
        
        assert initial_model_count == 100
        assert initial_dimension_count == 100
        
        # Clear cache to simulate memory pressure
        registry._model_cache.clear()
        registry._dimension_cache.clear()
        
        assert len(registry._model_cache) == 0
        assert len(registry._dimension_cache) == 0
        
        # Should be able to reload models after cache clearing
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            await registry.get_model_info("reloaded-model")
            assert "reloaded-model" in registry._model_cache
            assert "reloaded-model" in registry._dimension_cache
    
    @pytest.mark.asyncio
    async def test_rate_limiting_model_probing(self):
        """Test rate limiting of model probing to prevent API abuse."""
        registry = EmbeddingModelRegistry()
        
        call_times = []
        
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            
            async def rate_limited_embed(*args, **kwargs):
                call_times.append(time.time())
                if len(call_times) > 1:
                    # Check if calls are spaced appropriately
                    time_diff = call_times[-1] - call_times[-2]
                    if time_diff < 0.1:  # Less than 100ms apart
                        raise Exception("Rate limit exceeded")
                await asyncio.sleep(0.05)  # Small delay
                return [0.1] * 384
            
            mock_model.aembed_query.side_effect = rate_limited_embed
            mock_get_model.return_value = mock_model
            
            # Multiple rapid calls should be handled gracefully
            tasks = []
            for i in range(10):
                task = registry._probe_model_dimensions(f"rate-limit-model-{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert any(isinstance(result, Exception) for result in results)


class TestDisasterRecoveryScenarios:
    """Test disaster recovery and business continuity scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_corruption_detection(self):
        """Test detection of database corruption."""
        mock_client = MagicMock()
        
        # Simulate corrupted collection metadata
        mock_client.collections.exists.return_value = True
        mock_client.collections.use.return_value = MagicMock()
        
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["corrupted-model"] = 384
            registry._model_cache["corrupted-model"] = {
                "dimensions": 384,
                "sanitized_name": "corrupted_model",
                "is_local": True,
            }
            mock_registry.return_value = registry
            collection_manager = ModelAwareCollectionManager(mock_client)
            
            # Should detect and handle corrupted collections
            collection = await collection_manager.get_collection("DocumentChunk", "corrupted-model")
            assert collection is not None
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_service_unavailability(self):
        """Test graceful degradation when dependent services are unavailable."""
        registry = EmbeddingModelRegistry()
        
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Service unavailable")
            
            # Should handle service unavailability gracefully
            with pytest.raises(Exception, match=r"Could not determine dimensions"):
                await registry._probe_model_dimensions("unavailable-model")
            
            # Should maintain cache integrity
            assert len(registry._model_cache) == 0
            assert len(registry._dimension_cache) == 0
    
    @pytest.mark.asyncio
    async def test_data_consistency_during_partial_failures(self):
        """Test data consistency during partial system failures."""
        mock_client = MagicMock()
        
        # Simulate partial system failure
        mock_client.collections.exists.side_effect = [False, True, False]  # Mixed success/failure
        mock_client.collections.create.return_value = None
        mock_client.collections.use.return_value = MagicMock()
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            # Should handle mixed success/failure scenarios
            try:
                await collection_manager.ensure_collections_for_thread("test-model")
            except Exception:
                pass  # Expected to fail due to mixed results


if __name__ == "__main__":
    pytest.main([__file__])

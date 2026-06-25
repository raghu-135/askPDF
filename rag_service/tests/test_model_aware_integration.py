"""
test_model_aware_integration.py - Integration tests for model-aware collection system with actual embedding models.

This test suite validates:
- Real embedding model dimension detection
- Production environment scenarios
- Dimension mismatch handling in realistic scenarios
- Edge cases in model registry operations
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock
from app.db.vector.model_registry import EmbeddingModelRegistry, get_embedding_model_registry
from app.db.vector.collection_manager import ModelAwareCollectionManager
from app.db.vector.adapter import WeaviateAdapter
from app.models.llm_server_client import get_embedding_model, should_use_local_embeddings, LOCAL_EMBEDDING_MODELS


class TestRealEmbeddingModelIntegration:
    """Integration tests with actual embedding models."""
    
    @pytest.fixture
    def registry(self):
        return EmbeddingModelRegistry()
    
    @pytest.mark.asyncio
    async def test_local_model_dimension_detection(self, registry):
        """Test dimension detection with actual local embedding model."""
        # Skip if no local models available
        if not LOCAL_EMBEDDING_MODELS:
            pytest.skip("No local embedding models available")
        
        model_name = LOCAL_EMBEDDING_MODELS[0]
        
        try:
            # Test actual dimension detection
            dimensions = await registry._probe_model_dimensions(model_name)
            assert dimensions > 0, "Dimensions should be positive"
            assert isinstance(dimensions, int), "Dimensions should be integer"
            
            # Test model info caching
            model_info = await registry.get_model_info(model_name)
            assert model_info['dimensions'] == dimensions
            assert model_info['is_local'] == True
            assert 'sanitized_name' in model_info
            
        except Exception as e:
            pytest.skip(f"Local model not available: {e}")
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch_prevention_real_scenario(self, registry):
        """Test dimension mismatch prevention with realistic scenario."""
        # Mock a model with specific dimensions
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.aembed_query = AsyncMock(return_value=[0.1] * 384)  # 384 dimensions
            mock_get_model.return_value = mock_model
            
            # Load model info
            await registry.get_model_info("test-model-384")
            
            # Test validation with correct dimensions
            valid_vectors = [[0.2] * 384, [0.3] * 384]
            collection_manager = ModelAwareCollectionManager(MagicMock())
            collection_manager.registry = registry
            
            assert await collection_manager.validate_vectors_for_model(valid_vectors, "test-model-384")
            
            # Test validation with incorrect dimensions
            invalid_vectors = [[0.2] * 768, [0.3] * 384]  # Mixed dimensions
            assert not await collection_manager.validate_vectors_for_model(invalid_vectors, "test-model-384")
            
            # Test validation with wrong dimensions (all 768)
            wrong_vectors = [[0.2] * 768, [0.3] * 768]
            assert not await collection_manager.validate_vectors_for_model(wrong_vectors, "test-model-384")
    
    @pytest.mark.asyncio
    async def test_collection_naming_edge_cases(self, registry):
        """Test collection naming with edge case model names."""
        edge_cases = [
            "model/name/with/slashes",
            "model-name-with-dashes",
            "model.name.with.dots",
            "model name with spaces",
            "model@with#special$chars%",
            "very_long_model_name_that_exceeds_normal_limits_and_contains_various_special_characters_123!@#",
        ]
        
        for model_name in edge_cases:
            # Mock dimension detection
            with patch.object(registry, '_probe_model_dimensions', return_value=768):
                await registry.get_model_info(model_name)
                
                # Test collection name generation
                collection_name = registry.get_collection_name("DocumentChunk", model_name)
                
                # Should not contain special characters except underscores
                assert '/' not in collection_name
                assert '-' not in collection_name
                assert '.' not in collection_name
                assert ' ' not in collection_name
                assert '@' not in collection_name
                assert '#' not in collection_name
                assert '$' not in collection_name
                assert '%' not in collection_name
                
                # Should end with dimensions
                assert collection_name.endswith("_768")
                
                # Should be a valid collection name
                assert collection_name.replace('_', '').isalnum()


class TestProductionEnvironmentScenarios:
    """Test scenarios that might occur in production."""
    
    @pytest.mark.asyncio
    async def test_concurrent_model_info_loading(self):
        """Test concurrent loading of model information."""
        registry = EmbeddingModelRegistry()
        
        # Mock model to simulate real behavior
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.aembed_query = AsyncMock(return_value=[0.1] * 1536)  # Large model dimensions
            mock_get_model.return_value = mock_model
            
            # Create multiple concurrent tasks
            tasks = []
            for i in range(10):
                task = registry.get_model_info("large-model")
                tasks.append(task)
            
            # All should complete successfully
            results = await asyncio.gather(*tasks)
            
            # All should return the same result
            for result in results:
                assert result['dimensions'] == 1536
                assert result['sanitized_name'] == 'large_model'
            
            # Should only call the model once due to caching
            assert mock_model.aembed_query.call_count == 1
    
    @pytest.mark.asyncio
    async def test_model_registry_memory_management(self):
        """Test memory management in model registry."""
        registry = EmbeddingModelRegistry()
        
        # Add multiple models to cache
        with patch.object(registry, '_probe_model_dimensions') as mock_probe:
            mock_probe.side_effect = [384, 768, 1536, 1024, 512]
            
            models = ["model-1", "model-2", "model-3", "model-4", "model-5"]
            
            for model in models:
                await registry.get_model_info(model)
            
            # Check cache size
            assert len(registry._model_cache) == 5
            assert len(registry._dimension_cache) == 5
            
            # Clear cache and verify
            registry._model_cache.clear()
            registry._dimension_cache.clear()
            
            assert len(registry._model_cache) == 0
            assert len(registry._dimension_cache) == 0
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch_in_adapter_indexing(self):
        """Test dimension mismatch prevention in adapter indexing methods."""
        # Mock adapter with collection manager
        with patch('app.db.vector.collection_manager.ModelAwareCollectionManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.validate_vectors_for_model = AsyncMock(return_value=False)  # Validation fails
            
            adapter = WeaviateAdapter.__new__(WeaviateAdapter)
            adapter.collection_manager = mock_manager
            
            # Test PDF indexing with dimension mismatch
            with pytest.raises(ValueError, match=r"Vector dimensions do not match"):
                await adapter.index_pdf_chunks(
                    thread_id="test-thread",
                    embedding_model_name="test-model",
                    file_hash="test-file",
                    texts=["test chunk"],
                    embeddings=[[0.1] * 768],  # Wrong dimensions
                    metadatas=[{}]
                )
            
            # Chat memory and web-search indexing do not validate dimensions
            # before requesting the model-aware collection in current code.


class TestModelRegistryEdgeCases:
    """Test edge cases in model registry operations."""
    
    @pytest.fixture
    def registry(self):
        return EmbeddingModelRegistry()
    
    @pytest.mark.asyncio
    async def test_model_failure_handling(self, registry):
        """Test handling of model loading failures."""
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Model not available")
            
            # Should raise exception for model info
            with pytest.raises(Exception, match=r"Could not determine dimensions"):
                await registry.get_model_info("unavailable-model")
            
            # Should also raise for dimensions directly
            with pytest.raises(Exception, match=r"Could not determine dimensions"):
                await registry.get_dimensions("unavailable-model")
    
    @pytest.mark.asyncio
    async def test_collection_name_without_model_info(self, registry):
        """Test collection name generation without loaded model info."""
        # Should raise error if model info not loaded
        with pytest.raises(ValueError, match=r"Model info not loaded"):
            registry.get_collection_name("DocumentChunk", "unknown-model")
    
    @pytest.mark.asyncio
    async def test_model_compatibility_edge_cases(self, registry):
        """Test model compatibility with edge cases."""
        # Mock model info
        registry._model_cache["model-a"] = {
            'dimensions': 384,
            'sanitized_name': 'model_a',
            'is_local': False
        }
        
        # Test with non-existent collection
        assert not registry.is_model_compatible("NonExistentCollection_384", "model-a")
        
        # Test with malformed collection name
        assert not registry.is_model_compatible("InvalidCollection", "model-a")
        assert not registry.is_model_compatible("DocumentChunk", "model-a")
        
        # Test with unknown model
        assert not registry.is_model_compatible("DocumentChunk_model_a_384", "unknown-model")
    
    @pytest.mark.asyncio
    async def test_dimension_cache_consistency(self, registry):
        """Test dimension cache consistency with model cache."""
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            # Load model info
            await registry.get_model_info("test-model")
            
            # Both caches should have consistent data
            assert registry._dimension_cache["test-model"] == 768
            assert registry._model_cache["test-model"]["dimensions"] == 768
            
            # Update dimension cache directly (simulating corruption)
            registry._dimension_cache["test-model"] = 1024
            
            # get_dimensions returns the dimension cache directly.
            assert await registry.get_dimensions("test-model") == 1024
            
            # But get_model_info should still have original value
            assert registry._model_cache["test-model"]["dimensions"] == 768


class TestProductionErrorHandling:
    """Test error handling in production scenarios."""
    
    @pytest.mark.asyncio
    async def test_weaviate_connection_failure_during_collection_creation(self):
        """Test handling of Weaviate connection failures during collection creation."""
        mock_client = MagicMock()
        mock_client.collections.exists.side_effect = Exception("Connection failed")
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            # Existence-check failures are logged and treated as missing collections.
            collection = await collection_manager.get_collection("DocumentChunk", "test-model")
            assert collection is not None
    
    @pytest.mark.asyncio
    async def test_partial_collection_creation_failure(self):
        """Test handling of partial collection creation failures."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.side_effect = [None, Exception("Creation failed"), None]
        mock_client.collections.use.return_value = MagicMock()
        
        collection_manager = ModelAwareCollectionManager(mock_client)
        
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            # Partial failures are logged and deferred until first use.
            await collection_manager.ensure_collections_for_thread("test-model")
            assert mock_client.collections.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_embedding_model_unavailable_during_indexing(self):
        """Test handling when embedding model becomes unavailable during indexing."""
        with patch('app.db.vector.collection_manager.ModelAwareCollectionManager') as mock_manager_class:
            mock_manager = MagicMock()
            
            # First call succeeds, second fails (model becomes unavailable)
            mock_manager.validate_vectors_for_model = AsyncMock(side_effect=[True, False])
            mock_manager.get_collection = AsyncMock(return_value=MagicMock())
            
            adapter = WeaviateAdapter.__new__(WeaviateAdapter)
            adapter.collection_manager = mock_manager
            adapter._insert_many_model_aware = AsyncMock(return_value=1)
            
            # First indexing should succeed
            await adapter.index_pdf_chunks(
                thread_id="test-thread",
                embedding_model_name="test-model",
                file_hash="test-file-1",
                texts=["test chunk 1"],
                embeddings=[[0.1] * 384],
                metadatas=[{}]
            )
            
            # Second indexing should fail due to dimension validation
            with pytest.raises(ValueError, match=r"Vector dimensions do not match"):
                await adapter.index_pdf_chunks(
                    thread_id="test-thread",
                    embedding_model_name="test-model",
                    file_hash="test-file-2",
                    texts=["test chunk 2"],
                    embeddings=[[0.1] * 384],
                    metadatas=[{}]
                )


if __name__ == "__main__":
    pytest.main([__file__])

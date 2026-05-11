"""
test_model_aware_collections.py - Comprehensive tests for model-aware collection system.

This test suite validates:
- Model registry dimension detection
- Collection manager on-demand creation
- Vector validation for dimension mismatches
- Query operations with model-aware collections
- Backward compatibility during migration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.db.vector.model_registry import EmbeddingModelRegistry, get_embedding_model_registry
from app.db.vector.collection_manager import ModelAwareCollectionManager
from app.db.vector.adapter import WeaviateAdapter
from app.models.llm_server_client import get_embedding_model


class TestEmbeddingModelRegistry:
    """Test cases for EmbeddingModelRegistry."""
    
    @pytest.fixture
    def registry(self):
        return EmbeddingModelRegistry()
    
    @pytest.mark.asyncio
    async def test_model_dimension_detection(self, registry):
        """Test dimension detection for different model types."""
        # Mock local model (384 dimensions)
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = AsyncMock()
            mock_model.aembed_query.return_value = [0.1] * 384
            
            mock_get_model.return_value = mock_model
            dimensions = await registry._probe_model_dimensions("local-model")
            
            assert dimensions == 384
            mock_model.aembed_query.assert_called_once_with("test")
    
    @pytest.mark.asyncio
    async def test_collection_naming(self, registry):
        """Test collection naming with model and dimensions."""
        # Mock model info
        registry._model_cache["test-model"] = {
            'dimensions': 768,
            'sanitized_name': 'test_model',
            'is_local': False
        }
        
        collection_name = registry.get_collection_name("DocumentChunk", "test-model")
        assert collection_name == "DocumentChunk_test_model_768"
    
    @pytest.mark.asyncio
    async def test_model_compatibility_check(self, registry):
        """Test model compatibility checking."""
        # Setup compatible model
        registry._model_cache["model-a"] = {
            'dimensions': 384,
            'sanitized_name': 'model_a',
            'is_local': True
        }
        
        # Test compatible collection
        assert registry.is_model_compatible("DocumentChunk_model_a_384", "model-a")
        
        # Test incompatible collection
        assert not registry.is_model_compatible("DocumentChunk_model_b_768", "model-a")
        
        # Test incompatible dimensions
        registry._model_cache["model-b"] = {
            'dimensions': 768,
            'sanitized_name': 'model_b',
            'is_local': True
        }
        assert not registry.is_model_compatible("DocumentChunk_model_a_384", "model-b")


class TestModelAwareCollectionManager:
    """Test cases for ModelAwareCollectionManager."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Weaviate client."""
        client = AsyncMock()
        client.collections.exists.return_value = False
        client.collections.create.return_value = None
        return client
    
    @pytest.fixture
    def collection_manager(self, mock_client):
        """Collection manager with mock client."""
        return ModelAwareCollectionManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_collection_creation_on_demand(self, collection_manager, mock_client):
        """Test on-demand collection creation."""
        # Mock registry to return dimensions
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            
            mock_registry.return_value = registry
            
            # First call should create collection
            collection = await collection_manager.get_collection("DocumentChunk", "test-model")
            
            assert collection is not None
            mock_client.collections.exists.assert_called_once_with("DocumentChunk_test-model_384")
            mock_client.collections.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collection_caching(self, collection_manager, mock_client):
        """Test collection caching."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            mock_client.collections.exists.return_value = True
            
            # First call creates collection
            collection1 = await collection_manager.get_collection("DocumentChunk", "test-model")
            
            # Second call should use cached collection
            collection2 = await collection_manager.get_collection("DocumentChunk", "test-model")
            
            assert collection1 is collection2
            # Should only check existence once
            mock_client.collections.exists.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_validation(self, collection_manager):
        """Test vector dimension validation."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            registry._dimension_cache["test-model"] = 384
            mock_registry.return_value = registry
            
            # Valid vectors should pass
            valid_vectors = [[0.1] * 384, [0.2] * 384]
            assert await collection_manager.validate_vectors_for_model(valid_vectors, "test-model")
            
            # Invalid vectors should fail
            invalid_vectors = [[0.1] * 768, [0.2] * 384]  # Mixed dimensions
            assert not await collection_manager.validate_vectors_for_model(invalid_vectors, "test-model")


class TestWeaviateAdapterIntegration:
    """Integration tests for updated WeaviateAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('app.db.vector.collection_manager.ModelAwareCollectionManager') as mock_manager_class:
            with patch('app.db.vector.model_registry.get_embedding_model_registry') as mock_registry:
                mock_manager = AsyncMock()
                mock_manager.get_collection.return_value = AsyncMock()
                mock_manager_class.return_value = mock_manager
                mock_registry.return_value = AsyncMock()
                
                adapter = WeaviateAdapter()
                adapter.collection_manager = mock_manager
                
                yield adapter
    
    @pytest.mark.asyncio
    async def test_search_knowledge_sources_uses_model_aware_collection(self, adapter):
        """Test that search uses model-aware collections."""
        # Setup
        mock_collection = AsyncMock()
        adapter.collection_manager.get_collection.return_value = mock_collection
        mock_collection.query.fetch_objects.return_value = AsyncMock()
        mock_collection.query.fetch_objects.return_value.objects = []
        
        # Test search
        result = await adapter.search_knowledge_sources(
            thread_id="test-thread",
            query_vector=[0.1] * 384,
            embedding_model_name="test-model",
            limit=5
        )
        
        # Should call get_collection with correct parameters
        adapter.collection_manager.get_collection.assert_called_once_with("DocumentChunk", "test-model")
        
        # Should not filter by embed_model (handled by collection isolation)
        mock_collection.query.fetch_objects.assert_called_once()
        # Check that embed_model filter is not in the call
        call_args = mock_collection.query.fetch_objects.call_args
        filters = call_args[1].get('filters')
        assert 'embed_model' not in str(filters)
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch_prevention(self, adapter):
        """Test that dimension mismatches are prevented."""
        # Mock collection manager to raise validation error
        adapter.collection_manager.validate_vectors_for_model.return_value = False
        
        # Test indexing with mismatched dimensions
        with pytest.raises(ValueError, match=r"Vector dimensions do not match expected dimensions"):
            await adapter.index_pdf_chunks(
                thread_id="test-thread",
                embedding_model_name="test-model",
                file_hash="test-file",
                texts=["test chunk"],
                embeddings=[[0.1] * 768],  # Wrong dimensions
                metadatas=[{}]
            )


class TestBackwardCompatibility:
    """Test backward compatibility during migration."""
    
    @pytest.mark.asyncio
    async def test_legacy_collection_fallback(self, adapter):
        """Test that legacy collections still work during migration."""
        # Mock legacy collection access
        with patch.object(adapter.client.collections, 'use') as mock_use:
            mock_legacy_collection = AsyncMock()
            mock_use.return_value = mock_legacy_collection
            mock_legacy_collection.query.fetch_objects.return_value = AsyncMock()
            mock_legacy_collection.query.fetch_objects.return_value.objects = []
            
            # Test that legacy queries still work
            result = await adapter.search_knowledge_sources(
                thread_id="test-thread",
                query_vector=[0.1] * 384,
                embedding_model_name="legacy-model",
                limit=5
            )
            
            # Should eventually call legacy collection for backward compatibility
            mock_use.assert_called_with("DocumentChunk")


if __name__ == "__main__":
    pytest.main([__file__])

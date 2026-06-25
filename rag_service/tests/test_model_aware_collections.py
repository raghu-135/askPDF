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
from unittest.mock import AsyncMock, MagicMock, patch
from app.db.vector.model_registry import EmbeddingModelRegistry, get_embedding_model_registry
from app.db.vector.collection_manager import ModelAwareCollectionManager
from app.db.vector.adapter import WeaviateAdapter
from app.models.llm_server_client import get_embedding_model


def seed_model(registry, model_name="test-model", dimensions=384):
    registry._model_cache[model_name] = {
        "dimensions": dimensions,
        "sanitized_name": registry.sanitize_model_name(model_name),
        "is_local": False,
    }
    registry._dimension_cache[model_name] = dimensions


@pytest.fixture
def mock_client():
    """Mock Weaviate client with sync collection management methods."""
    client = MagicMock()
    client.collections.exists.return_value = False
    client.collections.create.return_value = None
    client.collections.use.return_value = MagicMock()
    return client


@pytest.fixture
def adapter():
    """Create adapter with mocked dependencies."""
    with patch('app.db.vector.adapter.weaviate.connect_to_custom') as mock_connect:
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_client.collections.use.return_value = MagicMock()
        mock_connect.return_value = mock_client

        adapter = WeaviateAdapter()
        adapter.collection_manager = AsyncMock()
        adapter.collection_manager.validate_vectors_for_model.return_value = True
        adapter.client = mock_client
        yield adapter


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
        """Test collection naming with model and dimensions."""
        # Mock model info for model-a (384 dimensions)
        registry._model_cache["model-a"] = {
            'dimensions': 384,
            'sanitized_name': 'model_a',
            'is_local': False
        }
        
        collection_name = registry.get_collection_name("DocumentChunk", "model-a")
        assert collection_name == "DocumentChunk_model_a_384"
        
        # Test compatible collection
        assert registry.is_model_compatible("DocumentChunk_model_a_384", "model-a")
        
        # Test incompatible collection
        assert not registry.is_model_compatible("DocumentChunk_model_b_768", "model-a")
        
        # Test incompatible dimensions - add model-b with 768 dimensions
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
        client = MagicMock()
        client.collections.exists.return_value = False
        client.collections.create.return_value = None
        client.collections.use.return_value = MagicMock()
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
            seed_model(registry)
            collection_manager.registry = registry
            
            mock_registry.return_value = registry
            
            # First call should create collection
            collection = await collection_manager.get_collection("DocumentChunk", "test-model")
            
            assert collection is not None
            mock_client.collections.exists.assert_called_once_with("DocumentChunk_test_model_384")
            mock_client.collections.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collection_caching(self, collection_manager, mock_client):
        """Test collection caching."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            seed_model(registry)
            collection_manager.registry = registry
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
    async def test_vector_validation(self, collection_manager, mock_client):
        """Test vector dimension validation."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            seed_model(registry)
            collection_manager.registry = registry
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
                
                # Create a proper mock collection with batch behavior
                mock_collection = AsyncMock()
                mock_batch = AsyncMock()
                mock_batch.__aenter__ = AsyncMock(return_value=mock_batch)
                mock_batch.__aexit__ = AsyncMock(return_value=None)
                mock_batch.add_object = AsyncMock()
                mock_collection.batch.dynamic.return_value = mock_batch
                
                mock_manager.get_collection.return_value = mock_collection
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
        
        # Mock of actual query methods used in the code
        mock_response = AsyncMock()
        mock_response.objects = []
        # The asyncio.to_thread will call the mock, so it should return the response directly
        mock_collection.query.near_vector.return_value = mock_response
        mock_collection.query.hybrid.return_value = mock_response
        
        # Test search
        result = await adapter.search_knowledge_sources(
            thread_id="test-thread",
            query_vector=[0.1] * 384,
            embedding_model_name="test-model",
            limit=5
        )
        
        # Should call get_collection with correct parameters
        adapter.collection_manager.get_collection.assert_called_once_with("DocumentChunk", "test-model")
        
        # Should call near_vector since no query_text provided
        mock_collection.query.near_vector.assert_called_once()
        # Check that embed_model filter is not in the call
        call_args = mock_collection.query.near_vector.call_args
        filters = call_args[1].get('filters')
        assert 'embed_model' not in str(filters)

    @pytest.mark.asyncio
    async def test_search_knowledge_sources_file_hash_filter_is_not_thread_scoped(self, adapter):
        """File-filtered document search should find shared chunks indexed by another thread."""
        mock_collection = AsyncMock()
        adapter.collection_manager.get_collection.return_value = mock_collection

        mock_response = AsyncMock()
        mock_response.objects = []
        mock_collection.query.near_vector.return_value = mock_response

        await adapter.search_knowledge_sources(
            thread_id="second-thread",
            query_vector=[0.1] * 384,
            embedding_model_name="test-model",
            limit=5,
            file_hashes=["shared-file-hash"],
        )

        call_args = mock_collection.query.near_vector.call_args
        filters = str(call_args[1].get("filters"))
        assert "shared-file-hash" in filters
        assert "second-thread" not in filters
    
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


class TestProactiveCollectionCreation:
    """Test proactive collection creation for thread loading."""
    
    @pytest.fixture
    def collection_manager(self, mock_client):
        """Collection manager with mock client."""
        return ModelAwareCollectionManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_ensure_collections_for_thread_creates_all_types(self, collection_manager, mock_client):
        """Test that ensure_collections_for_thread creates all three collection types."""
        # Mock registry to return dimensions
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            seed_model(registry)
            collection_manager.registry = registry
            mock_registry.return_value = registry
            
            mock_client.collections.exists.return_value = False
            mock_client.collections.create.return_value = None
            mock_client.collections.use.return_value = AsyncMock()
            
            # Call ensure_collections_for_thread
            await collection_manager.ensure_collections_for_thread("test-model")
            
            # Should attempt to create all three collection types
            expected_calls = [
                "DocumentChunk_test_model_384",
                "ChatMemoryChunk_test_model_384", 
                "WebSearchChunk_test_model_384"
            ]
            
            actual_calls = [call[0][0] for call in mock_client.collections.exists.call_args_list]
            for expected in expected_calls:
                assert expected in actual_calls, f"Expected collection {expected} not created"
    
    @pytest.mark.asyncio
    async def test_ensure_collections_handles_partial_failures(self, collection_manager, mock_client):
        """Test that ensure_collections handles partial collection creation failures."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            seed_model(registry)
            collection_manager.registry = registry
            mock_registry.return_value = registry
            
            # Mock one collection to fail creation
            mock_client.collections.exists.return_value = False
            mock_client.collections.create.side_effect = [None, Exception("Creation failed"), None]
            mock_client.collections.use.return_value = AsyncMock()
            
            # Partial failures are logged and deferred until first use.
            await collection_manager.ensure_collections_for_thread("test-model")
            assert mock_client.collections.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_ensure_collections_skips_existing(self, collection_manager, mock_client):
        """Test that ensure_collections skips already existing collections."""
        with patch('app.db.vector.collection_manager.get_embedding_model_registry') as mock_registry:
            registry = EmbeddingModelRegistry()
            seed_model(registry)
            collection_manager.registry = registry
            mock_registry.return_value = registry
            
            # Mock all collections as existing
            mock_client.collections.exists.return_value = True
            mock_client.collections.use.return_value = AsyncMock()
            
            await collection_manager.ensure_collections_for_thread("test-model")
            
            # Should not attempt to create any collections
            mock_client.collections.create.assert_not_called()
            
            # Should still use all collections
            assert mock_client.collections.use.call_count == 3


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
            
            adapter.collection_manager.get_collection.assert_called_once_with("DocumentChunk", "legacy-model")


if __name__ == "__main__":
    pytest.main([__file__])

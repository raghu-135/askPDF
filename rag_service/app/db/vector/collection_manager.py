"""
vector/collection_manager.py - Model-aware collection manager for on-demand collection creation.

This module provides dynamic collection management with:
- On-demand collection creation based on embedding model
- Dimension validation and compatibility checking
- Collection lifecycle management
"""

import asyncio
import logging
from typing import Dict, Optional
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateBaseError

from app.db.vector.config import (
    CollectionNames,
    VectorDBError,
    VectorDBInsertError,
)
from app.db.vector.model_registry import get_embedding_model_registry
from app.db.vector.helpers import _metadata_json, _parse_metadata

logger = logging.getLogger(__name__)


class ModelAwareCollectionManager:
    """Manager for model-aware collections with on-demand creation."""
    
    def __init__(self, client):
        self.client = client
        self.registry = get_embedding_model_registry()
        self._collection_cache: Dict[str, any] = {}
    
    async def get_collection(self, base_name: str, model_name: str):
        """Get or create collection for base name and model."""
        # Ensure model info is loaded first
        await self.registry.get_model_info(model_name)
        collection_name = self.registry.get_collection_name(base_name, model_name)
        
        if collection_name not in self._collection_cache:
            if not self.client.collections.exists(collection_name):
                # Ensure model info is loaded
                await self.registry.get_model_info(model_name)
                dimensions = self.registry._dimension_cache.get(model_name)
                if not dimensions:
                    raise ValueError(f"Could not determine dimensions for model '{model_name}'")
                
                logger.info(f"Creating collection '{collection_name}' for model '{model_name}' ({dimensions} dimensions)")
                await self._create_model_collection(collection_name, base_name, dimensions)
            
            self._collection_cache[collection_name] = self.client.collections.use(collection_name)
        
        return self._collection_cache[collection_name]
    
    async def _create_model_collection(self, collection_name: str, base_name: str, dimensions: int):
        """Create a collection with proper schema for given dimensions."""
        try:
            # Define properties based on base collection type
            properties = self._get_collection_properties(base_name)
            
            logger.warning(f"Model-aware collection '{collection_name}' is missing, creating it now...")
            logger.info(f"Creating '{base_name}' collection for {dimensions}-dimensional vectors with {len(properties)} properties")
            
            self.client.collections.create(
                name=collection_name,
                vector_config=wvc.config.Configure.Vectors.self_provided(),
                properties=properties
            )
            logger.info(f"Successfully created model-aware collection '{collection_name}' - ready for {base_name} embeddings")
            
        except WeaviateBaseError as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise VectorDBError(f"Could not create collection '{collection_name}'") from e
    
    def _get_collection_properties(self, base_name: str):
        """Get properties for collection type."""
        if base_name == CollectionNames.DOCUMENT:
            return [
                wvc.config.Property(name="thread_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="embed_model", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_kind", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="file_hash", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="metadata_json", data_type=wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.CHAT_MEMORY:
            return [
                wvc.config.Property(name="thread_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="message_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.WEB_SEARCH:
            return [
                wvc.config.Property(name="thread_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="search_query", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            ]
        else:
            raise ValueError(f"Unknown base collection type: {base_name}")
    
    async def validate_vectors_for_model(self, vectors: list, model_name: str) -> bool:
        """Validate that vectors match expected dimensions for model."""
        try:
            expected_dimensions = await self.registry.get_dimensions(model_name)
            for i, vector in enumerate(vectors):
                if len(vector) != expected_dimensions:
                    logger.error(f"Vector {i} has {len(vector)} dimensions, expected {expected_dimensions}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate vectors for model '{model_name}': {e}")
            return False
    
    def get_collection_info(self, base_name: str, model_name: str) -> Dict[str, any]:
        """Get information about collection for base name and model."""
        collection_name = self.registry.get_collection_name(base_name, model_name)
        model_info = self.registry._model_cache.get(model_name, {})
        
        return {
            'collection_name': collection_name,
            'base_name': base_name,
            'model_name': model_name,
            'dimensions': model_info.get('dimensions'),
            'sanitized_name': model_info.get('sanitized_name'),
            'exists': self.client.collections.exists(collection_name),
            'is_local': model_info.get('is_local', False)
        }
    
    async def ensure_collections_for_thread(self, embedding_model_name: str):
        """Ensure all collections exist and are ready for vector embeddings for thread's embedding model.
        
        Creates DocumentChunk, ChatMemory, and WebSearch collections for the given model.
        Validates that collections can accept vectors with the correct dimensions.
        Runs asynchronously to avoid blocking thread loading.
        """
        from app.db.vector.config import CollectionNames
        
        logger.info(f"Proactively ensuring vector collections for thread embedding model '{embedding_model_name}'")
        
        collection_types = {
            CollectionNames.DOCUMENT: "document chunks",
            CollectionNames.CHAT_MEMORY: "chat memory", 
            CollectionNames.WEB_SEARCH: "web search results"
        }
        
        tasks = []
        for base_name, description in collection_types.items():
            task = asyncio.create_task(
                self._ensure_collection_for_vectors(base_name, embedding_model_name, description)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to ensure vector collection: {result}")
                else:
                    successful_count += 1
            
            logger.info(f"Successfully prepared {successful_count}/{len(tasks)} vector collections for embeddings with model '{embedding_model_name}'")
            
            if successful_count < len(tasks):
                logger.warning(f"Some vector collections failed to initialize for model '{embedding_model_name}' - embeddings may be delayed until first use")
    
    async def _ensure_collection_for_vectors(self, base_name: str, embedding_model_name: str, description: str):
        """Ensure a specific collection exists and is ready for vector embeddings."""
        try:
            # Get or create the collection
            collection = await self.get_collection(base_name, embedding_model_name)
            
            # Validate that the collection can accept vectors for this model
            try:
                # Get the expected dimensions for this model
                expected_dimensions = await self.registry.get_dimensions(embedding_model_name)
                # Test with a properly sized vector
                if await self.validate_vectors_for_model([[0.1] * expected_dimensions], embedding_model_name):
                    logger.info(f"✅ {description.capitalize()} collection ready for {embedding_model_name} embeddings ({expected_dimensions}D)")
                    return collection
                else:
                    logger.warning(f"⚠️ {description.capitalize()} collection exists but vector validation failed for {embedding_model_name}")
                    return collection
            except Exception as validation_error:
                logger.warning(f"⚠️ Could not validate {description} collection for {embedding_model_name}: {validation_error}")
                return collection
                
        except Exception as e:
            logger.error(f"❌ Failed to prepare {description} collection for {embedding_model_name}: {e}")
            raise
    
    def clear_cache(self):
        """Clear collection cache (useful for testing or model changes)."""
        self._collection_cache.clear()
        logger.info("Collection manager cache cleared")

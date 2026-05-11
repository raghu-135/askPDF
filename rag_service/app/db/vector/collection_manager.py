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
            
            self.client.collections.create(
                name=collection_name,
                vector_config=wvc.config.Configure.Vectors.self_provided(),
                properties=properties
            )
            logger.info(f"Successfully created collection '{collection_name}'")
            
        except WeaviateBaseError as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise VectorDBError(f"Could not create collection '{collection_name}'") from e
    
    def _get_collection_properties(self, base_name: str):
        """Get properties for collection type."""
        if base_name == CollectionNames.DOCUMENT:
            return [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("embed_model", wvc.config.DataType.TEXT),
                ("source_kind", wvc.config.DataType.TEXT),
                ("file_hash", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("text", wvc.config.DataType.TEXT),
                ("url", wvc.config.DataType.TEXT),
                ("title", wvc.config.DataType.TEXT),
                ("metadata_json", wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.CHAT_MEMORY:
            return [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("message_id", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("question", wvc.config.DataType.TEXT),
                ("answer", wvc.config.DataType.TEXT),
                ("text", wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.WEB_SEARCH:
            return [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("search_query", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("text", wvc.config.DataType.TEXT),
                ("url", wvc.config.DataType.TEXT),
                ("title", wvc.config.DataType.TEXT),
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
    
    async def ensure_collections_for_thread(self, thread_models: Dict[str, str]):
        """Ensure collections exist for all models used in a thread."""
        tasks = []
        for base_name, model_name in thread_models.items():
            task = asyncio.create_task(self.get_collection(base_name, model_name))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Ensured collections for {len(tasks)} model combinations")
    
    def clear_cache(self):
        """Clear collection cache (useful for testing or model changes)."""
        self._collection_cache.clear()
        logger.info("Collection manager cache cleared")

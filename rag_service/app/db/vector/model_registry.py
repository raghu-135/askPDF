"""
vector/model_registry.py - Embedding model registry for dimension detection and collection naming.

This module provides centralized management of embedding models including:
- Dimension detection for local and remote models
- Sanitized collection naming
- Model metadata caching
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Tuple
from app.models.llm_server_client import (
    get_embedding_model, 
    should_use_local_embeddings,
    LOCAL_EMBEDDING_MODELS
)

logger = logging.getLogger(__name__)


class EmbeddingModelRegistry:
    """Registry for embedding model information with dimension detection and collection naming."""
    
    def __init__(self):
        self._model_cache: Dict[str, Dict[str, any]] = {}
        self._dimension_cache: Dict[str, int] = {}
    
    def sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for safe collection naming."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', model_name.replace('/', '_').replace('-', '_').replace('.', '_'))
    
    async def _probe_model_dimensions(self, model_name: str) -> int:
        """Probe embedding model to determine vector dimensions."""
        try:
            embed_model = get_embedding_model(model_name)
            # Use a simple test text to get dimensions
            test_embedding = await embed_model.aembed_query("test")
            dimensions = len(test_embedding)
            logger.info(f"Detected {dimensions} dimensions for model '{model_name}'")
            return dimensions
        except Exception as e:
            logger.error(f"Failed to probe dimensions for model '{model_name}': {e}")
            raise ValueError(f"Could not determine dimensions for model '{model_name}'")
    
    async def get_model_info(self, model_name: str) -> Dict[str, any]:
        """Get cached or probe model information."""
        if model_name not in self._model_cache:
            try:
                dimensions = await self._probe_model_dimensions(model_name)
                sanitized_name = self.sanitize_model_name(model_name)
                
                self._model_cache[model_name] = {
                    'dimensions': dimensions,
                    'sanitized_name': sanitized_name,
                    'is_local': should_use_local_embeddings(model_name)
                }
                self._dimension_cache[model_name] = dimensions
            except Exception as e:
                logger.error(f"Failed to get model info for '{model_name}': {e}")
                raise
        
        return self._model_cache[model_name]
    
    async def get_dimensions(self, model_name: str) -> int:
        """Get dimensions for a model name."""
        if model_name not in self._dimension_cache:
            await self.get_model_info(model_name)
        return self._dimension_cache[model_name]
    
    def get_collection_name(self, base_name: str, model_name: str) -> str:
        """Generate collection name with model and dimensions."""
        info = self._model_cache.get(model_name)
        if not info:
            raise ValueError(f"Model info not loaded for '{model_name}'. Call get_model_info first.")
        
        return f"{base_name}_{info['sanitized_name']}_{info['dimensions']}"
    
    def get_base_collection_name(self, collection_name: str) -> str:
        """Extract base collection name from full collection name."""
        # Extract base name before model info
        parts = collection_name.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:-2])
        return collection_name
    
    def get_model_from_collection(self, collection_name: str) -> Optional[str]:
        """Extract model name from collection name."""
        parts = collection_name.split('_')
        if len(parts) >= 3:
            # Reconstruct model name (this is approximate)
            model_part = '_'.join(parts[-2])
            # Try to find matching model in cache
            for cached_name, info in self._model_cache.items():
                if info['sanitized_name'] == model_part:
                    return cached_name
        return None
    
    def is_model_compatible(self, collection_name: str, model_name: str) -> bool:
        """Check if collection is compatible with given model."""
        try:
            collection_model = self.get_model_from_collection(collection_name)
            if not collection_model:
                return False
            
            collection_info = self._model_cache[collection_model]
            model_info = self._model_cache.get(model_name)
            
            if not model_info:
                return False
            
            return (collection_info['dimensions'] == model_info['dimensions'] and
                   collection_info['sanitized_name'] == model_info['sanitized_name'])
        except Exception:
            return False


# Global registry instance
_registry: Optional[EmbeddingModelRegistry] = None


def get_embedding_model_registry() -> EmbeddingModelRegistry:
    """Get or create global embedding model registry."""
    global _registry
    if _registry is None:
        _registry = EmbeddingModelRegistry()
    return _registry


async def ensure_model_info(model_name: str) -> Dict[str, any]:
    """Ensure model info is loaded for the given model."""
    registry = get_embedding_model_registry()
    return await registry.get_model_info(model_name)


def get_collection_name(base_name: str, model_name: str) -> str:
    """Get collection name for base collection and model."""
    registry = get_embedding_model_registry()
    return registry.get_collection_name(base_name, model_name)

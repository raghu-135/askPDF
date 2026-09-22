"""
vector/collection_manager.py - Model-aware collection manager for on-demand collection creation.

This module provides dynamic collection management with:
- On-demand collection creation based on embedding model
- Dimension validation and compatibility checking
- Collection lifecycle management
"""

import asyncio
import logging
import re
from typing import Dict, Mapping
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

_SIMILAR_CLASS_RE = re.compile(r'found similar class "([^"]+)"', re.IGNORECASE)
_COLLECTION_DIMENSION_SUFFIX_RE = re.compile(r"_\d+$")


def similar_weaviate_class_name(exc: BaseException) -> str | None:
    """Parse Weaviate's case-insensitive duplicate-class 422 message."""
    match = _SIMILAR_CLASS_RE.search(str(exc))
    if match:
        return match.group(1)
    return None


class ModelAwareCollectionManager:
    """Manager for model-aware collections with on-demand creation."""
    
    def __init__(self, client):
        self.client = client
        self.registry = get_embedding_model_registry()
        self._collection_cache: Dict[str, any] = {}

    def listed_collection_names(self) -> list[str]:
        list_all = getattr(self.client.collections, "list_all", None)
        if list_all is None:
            return []
        listed = list_all()
        if isinstance(listed, Mapping):
            return [str(name) for name in listed.keys()]
        if not isinstance(listed, (list, tuple, set)):
            return []
        names: list[str] = []
        for item in listed:
            if isinstance(item, str):
                names.append(item)
                continue
            name = getattr(item, "name", None)
            if name:
                names.append(str(name))
        return names

    def existing_collection_name(self, collection_name: str) -> str | None:
        """Return the stored Weaviate class, matching case-insensitively."""
        try:
            exists = self.client.collections.exists(collection_name)
            if exists is True:
                return collection_name
        except Exception as exc:
            logger.warning("Failed to check collection existence for '%s': %s", collection_name, exc)
        folded = collection_name.casefold()
        for name in self.listed_collection_names():
            if name.casefold() == folded:
                return name
        return None

    def _collection_prefix(self, base_name: str, model_name: str) -> str:
        sanitized = self.registry.sanitize_model_name(model_name)
        return f"{base_name}_{sanitized}_"

    def find_existing_collection_names(self, base_name: str, model_name: str) -> list[str]:
        """Return stored Weaviate classes for a base/model pair without probing embeddings."""
        prefix = self._collection_prefix(base_name, model_name).casefold()
        matches: list[str] = []
        for name in self.listed_collection_names():
            folded = name.casefold()
            if folded.startswith(prefix) and _COLLECTION_DIMENSION_SUFFIX_RE.search(folded):
                matches.append(name)
        return matches

    def find_existing_collection(self, base_name: str, model_name: str) -> str | None:
        """Return one stored Weaviate class when an unambiguous match exists."""
        matches = self.find_existing_collection_names(base_name, model_name)
        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "Multiple collections match base '%s' and model '%s': %s",
                base_name,
                model_name,
                matches,
            )
        return matches[0]

    async def get_existing_collections(self, base_name: str, model_name: str) -> list:
        """Return existing Weaviate collections for cleanup without live embedding probes."""
        names = self.find_existing_collection_names(base_name, model_name)
        if len(names) > 1:
            logger.warning(
                "Deleting from all %s collections matching base '%s' and model '%s': %s",
                len(names),
                base_name,
                model_name,
                names,
            )
        collections = []
        for stored_name in names:
            if stored_name not in self._collection_cache:
                collection = self.client.collections.use(stored_name)
                self._collection_cache[stored_name] = collection
            collections.append(self._collection_cache[stored_name])
        return collections

    async def get_existing_collection(self, base_name: str, model_name: str):
        """Return one existing collection, or None when no class is stored yet."""
        collections = await self.get_existing_collections(base_name, model_name)
        return collections[0] if collections else None
    
    async def get_collection(self, base_name: str, model_name: str):
        """Get or create collection for base name and model."""
        await self.registry.get_model_info(model_name)
        collection_name = self.registry.get_collection_name(base_name, model_name)
        
        if collection_name not in self._collection_cache:
            stored_name = self.existing_collection_name(collection_name)
            if stored_name is None:
                dimensions = self.registry._dimension_cache.get(model_name)
                if not dimensions:
                    raise ValueError(f"Could not determine dimensions for model '{model_name}'")
                logger.info(
                    "Creating collection '%s' for model '%s' (%s dimensions)",
                    collection_name,
                    model_name,
                    dimensions,
                )
                stored_name = await self._create_model_collection(collection_name, base_name, dimensions)
            else:
                self._ensure_collection_properties(stored_name, self._get_collection_properties(base_name))
            
            collection = self.client.collections.use(stored_name)
            self._collection_cache[collection_name] = collection
            if stored_name != collection_name:
                self._collection_cache[stored_name] = collection
        
        return self._collection_cache[collection_name]
    
    async def _create_model_collection(self, collection_name: str, base_name: str, dimensions: int) -> str:
        """Create a collection, or reuse a case-equivalent class Weaviate already has."""
        properties = self._get_collection_properties(base_name)
        logger.warning("Model-aware collection '%s' is missing, creating it now...", collection_name)
        logger.info(
            "Creating '%s' collection for %s-dimensional vectors with %s properties",
            base_name,
            dimensions,
            len(properties),
        )
        try:
            self.client.collections.create(
                name=collection_name,
                vector_config=wvc.config.Configure.Vectors.self_provided(),
                properties=properties,
            )
            logger.info("Successfully created model-aware collection '%s'", collection_name)
            return collection_name
        except WeaviateBaseError as exc:
            stored_name = similar_weaviate_class_name(exc) or self.existing_collection_name(collection_name)
            if stored_name:
                logger.info(
                    "Reusing existing Weaviate class '%s' for canonical name '%s'",
                    stored_name,
                    collection_name,
                )
                self._ensure_collection_properties(stored_name, properties)
                return stored_name
            logger.error("Failed to create collection '%s': %s", collection_name, exc)
            raise VectorDBError(f"Could not create collection '{collection_name}'") from exc
    
    def _get_collection_properties(self, base_name: str):
        """Get properties for collection type."""
        if base_name == CollectionNames.DOCUMENT:
            return [
                wvc.config.Property(name="thread_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="embedding_model", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_kind", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="file_hash", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="manifest_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="generation", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="chunk_identity", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="section_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="table_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="page_start", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="page_end", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="pages", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="metadata_json", data_type=wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.CHAT_MEMORY:
            return [
                wvc.config.Property(name="thread_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="embedding_model", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="message_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="message_created_at", data_type=wvc.config.DataType.TEXT),
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
                wvc.config.Property(name="web_search_performed_at", data_type=wvc.config.DataType.TEXT),
            ]
        elif base_name == CollectionNames.MEMORY:
            return [
                wvc.config.Property(name="memory_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="scope_type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="scope_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="metadata_json", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="created_at", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="updated_at", data_type=wvc.config.DataType.TEXT),
            ]
        else:
            raise ValueError(f"Unknown base collection type: {base_name}")

    def _ensure_collection_properties(self, collection_name: str, properties: list) -> None:
        """Add missing scalar properties to an existing model-aware collection."""
        try:
            collection = self.client.collections.use(collection_name)
            config = collection.config.get()
            existing_props = getattr(config, "properties", []) or []
            existing_names = {
                getattr(prop, "name", None) or (prop.get("name") if isinstance(prop, dict) else None)
                for prop in existing_props
            }
            for prop in properties:
                prop_name = getattr(prop, "name", None)
                if prop_name in existing_names:
                    continue
                collection.config.add_property(prop)
                logger.info("Added missing property '%s' to model-aware collection '%s'", prop_name, collection_name)
        except Exception as exc:
            logger.warning("Could not verify/update properties for collection '%s': %s", collection_name, exc)
    
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
            'exists': self.existing_collection_name(collection_name) is not None,
            'is_local': model_info.get('is_local', False)
        }
    
    async def ensure_collections_for_thread(self, embedding_model: str):
        """Ensure all collections exist and are ready for vector embeddings for thread's embedding model.
        
        Creates DocumentChunk, ChatMemory, and WebSearch collections for the given model.
        Validates that collections can accept vectors with the correct dimensions.
        Runs asynchronously to avoid blocking thread loading.
        """
        from app.db.vector.config import CollectionNames
        
        logger.info(f"Proactively ensuring vector collections for thread embedding model '{embedding_model}'")
        
        collection_types = {
            CollectionNames.DOCUMENT: "document chunks",
            CollectionNames.CHAT_MEMORY: "chat memory", 
            CollectionNames.WEB_SEARCH: "web search results",
            CollectionNames.MEMORY: "durable memory",
        }
        
        tasks = []
        for base_name, description in collection_types.items():
            task = asyncio.create_task(
                self._ensure_collection_for_vectors(base_name, embedding_model, description)
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
            
            logger.info(f"Successfully prepared {successful_count}/{len(tasks)} vector collections for embeddings with model '{embedding_model}'")
            
            if successful_count < len(tasks):
                logger.warning(f"Some vector collections failed to initialize for model '{embedding_model}' - embeddings may be delayed until first use")
    
    async def _ensure_collection_for_vectors(self, base_name: str, embedding_model: str, description: str):
        """Ensure a specific collection exists and is ready for vector embeddings."""
        try:
            # Get collection name for this model
            collection_name = self.registry.get_collection_name(base_name, embedding_model)
            
            stored_name = self.existing_collection_name(collection_name)
            if stored_name is not None:
                logger.debug(
                    "Model-aware collection '%s' already exists for %s",
                    stored_name,
                    description,
                )
                collection = self.client.collections.use(stored_name)
            else:
                logger.info("Creating new model-aware collection '%s' for %s", collection_name, description)
                collection = await self.get_collection(base_name, embedding_model)
            
            # Validate that the collection can accept vectors for this model
            try:
                # Get the expected dimensions for this model
                expected_dimensions = await self.registry.get_dimensions(embedding_model)
                # Test with a properly sized vector
                if await self.validate_vectors_for_model([[0.1] * expected_dimensions], embedding_model):
                    logger.info(f"✅ {description.capitalize()} collection ready for {embedding_model} embeddings ({expected_dimensions}D)")
                    return collection
                else:
                    logger.warning(f"⚠️ {description.capitalize()} collection exists but vector validation failed for {embedding_model}")
                    return collection
            except Exception as validation_error:
                logger.warning(f"⚠️ Could not validate {description} collection for {embedding_model}: {validation_error}")
                return collection
                
        except Exception as e:
            logger.error(f"❌ Failed to prepare {description} collection for {embedding_model}: {e}")
            raise
    
    def clear_cache(self):
        """Clear collection cache (useful for testing or model changes)."""
        self._collection_cache.clear()
        logger.info("Collection manager cache cleared")

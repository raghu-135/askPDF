"""
app.db.vector - Public API for Weaviate vector database operations.

This module provides a clean public API for vector database operations,
re-exporting functions from the modular structure.
"""

from typing import Optional

from app.db.vector.config import (
    CollectionNames,
    VectorDBError,
    VectorDBConnectionError,
    VectorDBInsertError,
    VectorDBQueryError,
)
from app.db.vector.adapter import WeaviateAdapter

_singleton_instance: Optional[WeaviateAdapter] = None


def get_vector_db() -> WeaviateAdapter:
    """Return a shared Weaviate adapter singleton.
    
    Returns:
        WeaviateAdapter: The singleton instance.
    """
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = WeaviateAdapter()
    return _singleton_instance


__all__ = [
    # Config
    "CollectionNames",
    "VectorDBError",
    "VectorDBConnectionError",
    "VectorDBInsertError",
    "VectorDBQueryError",
    # Adapter
    "WeaviateAdapter",
    # Public API
    "get_vector_db",
]

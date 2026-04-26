"""
vector/config.py - Configuration and exceptions for Weaviate vector database.
"""


class CollectionNames:
    """Centralized collection name constants."""
    DOCUMENT = "DocumentChunk"
    CHAT_MEMORY = "ChatMemoryChunk"
    WEB_SEARCH = "WebSearchChunk"


class VectorDBError(Exception):
    """Base exception for vector database operations."""
    pass


class VectorDBConnectionError(VectorDBError):
    """Exception raised for connection failures."""
    pass


class VectorDBInsertError(VectorDBError):
    """Exception raised for insert failures."""
    pass


class VectorDBQueryError(VectorDBError):
    """Exception raised for query failures."""
    pass

"""
status_fixtures.py - Sample status data for testing.

This module provides sample status data structures for use in tests.
"""

from datetime import datetime


def sample_file_status_data():
    """Generate sample file status data."""
    return {
        "parsing": {
            "status": "completed",
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "pages_processed": 10,
            "errors": []
        },
        "indexing": {
            "status": "running",
            "embedding_model": "BAAI/bge-m3",
            "chunk_count": 150,
            "total_chars": 75000
        }
    }


def sample_parsing_status():
    """Generate sample parsing status."""
    return {
        "status": "completed",
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": datetime.utcnow().isoformat(),
        "pages_processed": 10,
        "errors": []
    }


def sample_indexing_status():
    """Generate sample indexing status."""
    return {
        "status": "running",
        "embedding_model": "BAAI/bge-m3",
        "chunk_count": 150,
        "total_chars": 75000,
        "threads": {
            "thread-1": {
                "status": "completed",
                "chunk_count": 150,
                "total_chars": 75000
            }
        }
    }


def sample_status_with_error():
    """Generate status with error information."""
    return {
        "parsing": {
            "status": "failed",
            "error": "Failed to parse PDF: corrupted file",
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat()
        },
        "indexing": {
            "status": "unknown"
        }
    }


def sample_status_variations():
    """Generate status with various states."""
    return [
        {
            "parsing": {"status": "pending"},
            "indexing": {"status": "unknown"}
        },
        {
            "parsing": {"status": "running", "started_at": datetime.utcnow().isoformat()},
            "indexing": {"status": "unknown"}
        },
        {
            "parsing": {"status": "completed", "started_at": datetime.utcnow().isoformat(), "finished_at": datetime.utcnow().isoformat()},
            "indexing": {"status": "pending"}
        },
        {
            "parsing": {"status": "completed", "started_at": datetime.utcnow().isoformat(), "finished_at": datetime.utcnow().isoformat()},
            "indexing": {"status": "running", "embedding_model": "BAAI/bge-m3"}
        },
        {
            "parsing": {"status": "completed", "started_at": datetime.utcnow().isoformat(), "finished_at": datetime.utcnow().isoformat()},
            "indexing": {"status": "completed", "embedding_model": "BAAI/bge-m3", "chunk_count": 100}
        }
    ]


def sample_complex_indexing_status():
    """Generate complex indexing status with multiple models."""
    return {
        "status": "running",
        "models": {
            "BAAI/bge-m3": {
                "status": "completed",
                "chunk_count": 150,
                "total_chars": 75000,
                "threads": {
                    "thread-1": {
                        "status": "completed",
                        "chunk_count": 150
                    },
                    "thread-2": {
                        "status": "running",
                        "chunk_count": 50
                    }
                }
            },
            "openai/text-embedding-3-small": {
                "status": "pending",
                "threads": {}
            }
        }
    }


def sample_status_with_reused_embeddings():
    """Generate status indicating reused embeddings."""
    return {
        "indexing": {
            "status": "completed",
            "embedding_model": "BAAI/bge-m3",
            "chunk_count": 150,
            "total_chars": 75000,
            "reused_existing_embeddings": True,
            "threads": {
                "thread-1": {
                    "status": "completed",
                    "chunk_count": 150,
                    "reused_existing_embeddings": True
                }
            }
        }
    }


def sample_status_partial_completion():
    """Generate status with partial completion."""
    return {
        "parsing": {
            "status": "completed",
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "pages_processed": 8,
            "total_pages": 10,
            "errors": [
                {"page": 9, "error": "Failed to extract text"},
                {"page": 10, "error": "Corrupted page"}
            ]
        },
        "indexing": {
            "status": "partial",
            "embedding_model": "BAAI/bge-m3",
            "chunk_count": 120,
            "total_chars": 60000,
            "total_expected_chunks": 150
        }
    }

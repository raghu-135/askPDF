"""
file_fixtures.py - Sample file data for testing.

This module provides sample file data structures for use in tests.
"""


def sample_file_data():
    """Generate sample file data."""
    return {
        "file_hash": "abc123def456",
        "file_name": "test_document.pdf",
        "file_path": "/data/test_document.pdf",
        "source_type": "pdf"
    }


def sample_file_with_different_types():
    """Generate files with different source types."""
    return [
        {
            "file_hash": "pdf-hash-1",
            "file_name": "document.pdf",
            "file_path": "/data/document.pdf",
            "source_type": "pdf"
        },
        {
            "file_hash": "docx-hash-1",
            "file_name": "document.docx",
            "file_path": "/data/document.docx",
            "source_type": "docx"
        },
        {
            "file_hash": "txt-hash-1",
            "file_name": "document.txt",
            "file_path": "/data/document.txt",
            "source_type": "txt"
        },
        {
            "file_hash": "html-hash-1",
            "file_name": "document.html",
            "file_path": "/data/document.html",
            "source_type": "html"
        },
        {
            "file_hash": "md-hash-1",
            "file_name": "document.md",
            "file_path": "/data/document.md",
            "source_type": "md"
        }
    ]


def sample_files_list(count=5):
    """Generate a list of sample files."""
    files = []
    for i in range(count):
        files.append({
            "file_hash": f"file-hash-{i}",
            "file_name": f"document_{i}.pdf",
            "file_path": f"/data/document_{i}.pdf",
            "source_type": "pdf"
        })
    return files


def sample_file_status_data():
    """Generate sample file status data."""
    return {
        "parsing": {
            "status": "completed",
            "started_at": "2024-01-01T00:00:00",
            "finished_at": "2024-01-01T00:01:00",
            "pages_processed": 10,
            "errors": []
        },
        "indexing": {
            "status": "running",
            "embedding_model": "BAAI/bge-m3",
            "chunk_count": 150,
            "total_chars": 75000,
            "models": {
                "BAAI/bge-m3": {
                    "status": "completed",
                    "threads": {
                        "thread-1": {
                            "status": "completed",
                            "chunk_count": 150
                        }
                    }
                }
            }
        }
    }


def sample_file_status_with_error():
    """Generate file status with error information."""
    return {
        "parsing": {
            "status": "failed",
            "error": "Failed to parse PDF: corrupted file",
            "started_at": "2024-01-01T00:00:00",
            "finished_at": "2024-01-01T00:00:05"
        },
        "indexing": {
            "status": "unknown"
        }
    }


def sample_parsed_sentences_data():
    """Generate sample parsed sentences data."""
    return {
        "sentences": [
            {
                "id": "1",
                "text": "This is the first sentence.",
                "page": 1,
                "bbox": [0, 0, 100, 20],
                "font": "Arial",
                "size": 12
            },
            {
                "id": "2",
                "text": "This is the second sentence.",
                "page": 1,
                "bbox": [0, 25, 150, 45],
                "font": "Arial",
                "size": 12
            },
            {
                "id": "3",
                "text": "This is the third sentence.",
                "page": 2,
                "bbox": [0, 0, 120, 20],
                "font": "Times New Roman",
                "size": 11
            }
        ]
    }


def sample_large_parsed_sentences(count=1000):
    """Generate large parsed sentences data for performance testing."""
    sentences = []
    for i in range(count):
        sentences.append({
            "id": str(i),
            "text": f"Sentence {i} with some content for testing.",
            "page": i % 10 + 1,
            "bbox": [0, 0, 100, 20],
            "font": "Arial",
            "size": 12
        })
    return {"sentences": sentences}

"""
fixtures/__init__.py - Test fixtures package.

This package contains sample data fixtures for testing the PostgreSQL migration.
"""

from .thread_fixtures import (
    sample_thread_data,
    sample_thread_with_complex_settings,
    sample_threads_list,
    sample_thread_settings_variations
)

from .file_fixtures import (
    sample_file_data,
    sample_file_with_different_types,
    sample_files_list,
    sample_file_status_data,
    sample_file_status_with_error,
    sample_parsed_sentences_data,
    sample_large_parsed_sentences
)

from .annotation_fixtures import (
    sample_annotation_data,
    sample_complex_annotations,
    sample_annotations_list,
    sample_annotation_with_multiple_pages,
    sample_annotation_labels,
    sample_large_annotations
)

from .status_fixtures import (
    sample_file_status_data,
    sample_parsing_status,
    sample_indexing_status,
    sample_status_with_error,
    sample_status_variations,
    sample_complex_indexing_status,
    sample_status_with_reused_embeddings,
    sample_status_partial_completion
)

__all__ = [
    # Thread fixtures
    "sample_thread_data",
    "sample_thread_with_complex_settings",
    "sample_threads_list",
    "sample_thread_settings_variations",
    # File fixtures
    "sample_file_data",
    "sample_file_with_different_types",
    "sample_files_list",
    "sample_file_status_data",
    "sample_file_status_with_error",
    "sample_parsed_sentences_data",
    "sample_large_parsed_sentences",
    # Annotation fixtures
    "sample_annotation_data",
    "sample_complex_annotations",
    "sample_annotations_list",
    "sample_annotation_with_multiple_pages",
    "sample_annotation_labels",
    "sample_large_annotations",
    # Status fixtures
    "sample_file_status_data",
    "sample_parsing_status",
    "sample_indexing_status",
    "sample_status_with_error",
    "sample_status_variations",
    "sample_complex_indexing_status",
    "sample_status_with_reused_embeddings",
    "sample_status_partial_completion",
]

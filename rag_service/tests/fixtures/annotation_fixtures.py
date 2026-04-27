"""
annotation_fixtures.py - Sample annotation data for testing.

This module provides sample annotation data structures for use in tests.
"""

from datetime import datetime


def sample_annotation_data():
    """Generate sample annotation data."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": "Sample text",
                "label": "important"
            }
        ]
    }


def sample_complex_annotations():
    """Generate complex annotation data with nested structures."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": "Important text",
                "label": "important",
                "metadata": {
                    "confidence": 0.95,
                    "source": "manual",
                    "created_by": "user-123"
                },
                "children": [
                    {
                        "id": "child-1",
                        "text": "Nested annotation",
                        "label": "note"
                    }
                ]
            },
            {
                "page": 2,
                "bbox": [50, 100, 250, 150],
                "text": "Another annotation",
                "label": "normal",
                "metadata": {
                    "confidence": 0.87,
                    "source": "ocr"
                }
            }
        ]
    }


def sample_annotations_list(count=5):
    """Generate a list of sample annotations."""
    annotations = []
    for i in range(count):
        annotations.append({
            "page": i % 3 + 1,
            "bbox": [i * 50, i * 50, (i + 3) * 50, (i + 3) * 50],
            "text": f"Annotation {i}",
            "label": "normal" if i % 2 == 0 else "important"
        })
    return {"annotations": annotations}


def sample_annotation_with_multiple_pages():
    """Generate annotations spanning multiple pages."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [0, 0, 500, 50],
                "text": "Header on page 1",
                "label": "header"
            },
            {
                "page": 2,
                "bbox": [0, 0, 500, 50],
                "text": "Header on page 2",
                "label": "header"
            },
            {
                "page": 3,
                "bbox": [0, 0, 500, 50],
                "text": "Header on page 3",
                "label": "header"
            }
        ]
    }


def sample_annotation_labels():
    """Generate annotations with various labels."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": "Important text",
                "label": "important"
            },
            {
                "page": 1,
                "bbox": [50, 100, 250, 150],
                "text": "Normal text",
                "label": "normal"
            },
            {
                "page": 1,
                "bbox": [200, 300, 400, 500],
                "text": "Table data",
                "label": "table"
            },
            {
                "page": 1,
                "bbox": [300, 400, 500, 600],
                "text": "Figure caption",
                "label": "figure"
            },
            {
                "page": 1,
                "bbox": [400, 500, 600, 700],
                "text": "Footnote",
                "label": "footnote"
            }
        ]
    }


def sample_large_annotations(count=100):
    """Generate large annotations dataset for performance testing."""
    annotations = []
    for i in range(count):
        annotations.append({
            "page": i % 10 + 1,
            "bbox": [i * 10, i * 10, (i + 2) * 10, (i + 2) * 10],
            "text": f"Annotation text {i}",
            "label": "normal",
            "metadata": {
                "confidence": 0.9 + (i % 10) * 0.01,
                "source": "ocr"
            }
        })
    return {"annotations": annotations}

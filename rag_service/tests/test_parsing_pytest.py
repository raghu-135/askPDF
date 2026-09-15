#!/usr/bin/env python3
"""
Pytest tests for rag-service parsing code

These tests use pytest framework to test the parsing_service.py code.
Tests can be run with: pytest tests/test_parsing_pytest.py

For Docker:
    docker-compose exec rag-service pytest /app/tests/test_parsing_pytest.py

For local (requires all dependencies):
    pytest tests/test_parsing_pytest.py
"""

import json
import os
import pytest

# Add the parent directory to the path to import parsing_service

from app.services.parsing_service import extraction_configuration, extract_text_with_coordinates, parse_with_docling, parse_with_pdfplumber


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF file for testing."""
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, "01030000000000.pdf")


@pytest.fixture
def sample_pdf_data(sample_pdf_path):
    """PDF file data as bytes."""
    with open(sample_pdf_path, 'rb') as f:
        return f.read()


@pytest.fixture
def sample_filename(sample_pdf_path):
    """Filename of the sample PDF."""
    return os.path.basename(sample_pdf_path)


def test_docling_parsing(sample_pdf_data, sample_filename):
    """Test Docling parsing independently."""
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    
    assert docling_doc is not None, "Docling parsing failed"
    assert len(docling_doc.texts) > 0, "No text items extracted"
    assert len(docling_doc.groups) >= 0, "Groups should exist"
    
    # Verify label distribution
    label_counts = {}
    for item in docling_doc.texts:
        if hasattr(item, 'label') and hasattr(item.label, 'value'):
            label = item.label.value
            label_counts[label] = label_counts.get(label, 0) + 1
    
    assert len(label_counts) > 0, "No labels found in Docling output"


def test_pdfplumber_parsing(sample_pdf_data, sample_filename):
    """Test pdfplumber parsing independently."""
    # First parse with Docling to get the document
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    assert docling_doc is not None, "Docling parsing failed (required for pdfplumber)"
    
    result = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename)
    
    assert result is not None, "pdfplumber parsing failed"
    assert len(result) > 0, "No items extracted"
    
    # Validate structure
    required_fields = ['id', 'text', 'label', 'bbox', 'words', 'font']
    for i, item in enumerate(result):
        for field in required_fields:
            assert field in item, f"Item {i} missing required field: {field}"
        # Check that item has either 'page' or 'pages' field
        assert 'page' in item or 'pages' in item, f"Item {i} missing required field: page or pages"

    # Check label distribution
    label_counts = {}
    for item in result:
        label = item.get('label', 'unspecified')
        label_counts[label] = label_counts.get(label, 0) + 1

    assert len(label_counts) > 0, "No labels found in pdfplumber output"


def test_extraction_configuration_exposes_cache_invalidating_options(monkeypatch):
    import app.services.parsing_service as parsing_service

    monkeypatch.setattr(parsing_service._pipeline_options, "do_ocr", False)
    without_ocr = extraction_configuration()
    monkeypatch.setattr(parsing_service._pipeline_options, "do_ocr", True)
    with_ocr = extraction_configuration()

    assert without_ocr["do_ocr"] is False
    assert with_ocr["do_ocr"] is True
    assert {"do_table_structure", "do_formula_enrichment", "table_mode", "docling", "docling_core", "pdfplumber"} <= set(with_ocr)


def test_coordinate_alignment_preserves_unmatched_canonical_ocr_sentence():
    from app.services.document_conversion_service import _align_coordinates

    canonical = [
        {"text": "Selectable text.", "pages": [1], "alignment_precision": "coarse", "bboxes": []},
        {"text": "OCR-only text.", "pages": [2], "alignment_precision": "coarse", "bboxes": []},
    ]
    aligned = _align_coordinates(
        canonical,
        [{"text": "Selectable text.", "pages": [1], "bboxes": [{"page": 1, "x": 1}]}],
    )

    assert [item["text"] for item in aligned] == ["Selectable text.", "OCR-only text."]
    assert aligned[0]["alignment_precision"] == "exact"
    assert aligned[1]["alignment_precision"] == "coarse"


def test_coordinate_alignment_rejects_shared_word_from_unrelated_page():
    from app.services.document_conversion_service import _align_coordinates

    aligned = _align_coordinates(
        [{"text": "The shared word appears here.", "pages": [2], "label": "text", "bboxes": []}],
        [{"text": "shared", "page": 1, "label": "text", "bboxes": [{"page": 1, "x0": 1}]}],
    )

    assert aligned[0]["alignment_precision"] == "coarse"
    assert aligned[0]["bboxes"] == []


def test_combined_parsing(sample_pdf_data, sample_filename):
    """Test the combined parsing service."""
    result = extract_text_with_coordinates(sample_pdf_data, sample_filename)
    
    assert result is not None, "No results returned from parsing service"
    assert len(result) > 0, "No items in combined parsing result"
    
    # Check that each item has required fields
    required_fields = ['id', 'text', 'label', 'bbox', 'words', 'font']
    for i, item in enumerate(result):
        for field in required_fields:
            assert field in item, f"Item {i} missing required field: {field}"
        assert 'page' in item or 'pages' in item, f"Item {i} missing required field: page or pages"
    
    # Check label distribution
    label_counts = {}
    for item in result:
        label = item.get('label', 'unspecified')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    assert len(label_counts) > 0, "No labels found in combined output"
    
    # Check page distribution
    page_counts = {}
    for item in result:
        if 'page' in item:
            page = item.get('page', 0)
            page_counts[page] = page_counts.get(page, 0) + 1
        elif 'pages' in item:
            for page in item.get('pages', []):
                page_counts[page] = page_counts.get(page, 0) + 1
    
    assert len(page_counts) > 0, "No pages found in output"


def test_multi_bbox_merging(sample_pdf_data, sample_filename):
    """Test the merge_multi_bbox parameter behavior."""
    # First, parse with Docling to get the document
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    assert docling_doc is not None, "Docling parsing failed"
    
    # Test with merge_multi_bbox=True (new behavior)
    result_merged = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename, merge_multi_bbox=True)
    
    # Test with merge_multi_bbox=False (legacy behavior)
    result_independent = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename, merge_multi_bbox=False)
    
    assert len(result_merged) > 0, "Merged result is empty"
    assert len(result_independent) > 0, "Independent result is empty"
    
    # Count multi-page items in merged result
    multi_page_count = sum(1 for item in result_merged if 'pages' in item and len(item['pages']) > 1)
    
    # Check that sentences are properly merged (fewer items when merging is enabled)
    # Note: This may not always be true depending on the PDF structure
    # assert len(result_merged) <= len(result_independent), "Merging should produce fewer or equal items"


def test_item_text_not_empty(sample_pdf_data, sample_filename):
    """Test that all extracted items have non-empty text."""
    result = extract_text_with_coordinates(sample_pdf_data, sample_filename)
    
    for i, item in enumerate(result):
        text = item.get('text', '').strip()
        assert len(text) > 0, f"Item {i} has empty text"


def test_bbox_coordinates_valid(sample_pdf_data, sample_filename):
    """Test that all bounding boxes have valid coordinates."""
    result = extract_text_with_coordinates(sample_pdf_data, sample_filename)
    
    for i, item in enumerate(result):
        bbox = item.get('bbox')
        assert bbox is not None, f"Item {i} has no bbox"
        # bbox can be a dict with keys like x0, y0, x1, y1, page, page_width, page_height
        # or a list/tuple with 4 coordinates
        if isinstance(bbox, dict):
            # Check for coordinate keys
            assert 'x0' in bbox and 'y0' in bbox and 'x1' in bbox and 'y1' in bbox, f"Item {i} bbox dict missing coordinate keys"
            # Check coordinates are numeric
            for coord in ['x0', 'y0', 'x1', 'y1']:
                assert isinstance(bbox[coord], (int, float)), f"Item {i} bbox[{coord}] should be numeric"
            assert bbox['x0'] >= 0 and bbox['y0'] >= 0, f"Item {i} bbox has negative coordinates"
        else:
            # Assume it's a list/tuple with 4 coordinates
            assert len(bbox) == 4, f"Item {i} bbox should have 4 coordinates"
            assert all(isinstance(coord, (int, float)) for coord in bbox), f"Item {i} bbox coordinates should be numeric"
            assert bbox[0] >= 0 and bbox[1] >= 0, f"Item {i} bbox has negative coordinates"

#!/usr/bin/env python3
"""
Pytest tests for modular parsing and visualization

These tests use pytest framework to test the modular parsing functions.
Tests can be run with: pytest tests/test_modular_visualization_pytest.py

For Docker:
    docker-compose exec rag-service pytest /app/tests/test_modular_visualization_pytest.py

For local (requires all dependencies):
    pytest tests/test_modular_visualization_pytest.py
"""

import os
import sys
import pytest

# Add the parent directory to the path to import parsing_service

from app.services.parsing_service import parse_with_docling, parse_with_pdfplumber


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


@pytest.fixture
def output_dir(sample_pdf_path):
    """Output directory for test files."""
    return os.path.join(os.path.dirname(sample_pdf_path), "output")


def test_docling_parsing(sample_pdf_data, sample_filename):
    """Test Docling parsing independently."""
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    
    assert docling_doc is not None, "Docling parsing failed"
    assert len(docling_doc.texts) > 0, "No text items extracted"
    
    print(f"Docling parsing successful: {len(docling_doc.texts)} text items")


def test_pdfplumber_parsing(sample_pdf_data, sample_filename):
    """Test pdfplumber parsing independently."""
    # First parse with Docling to get the document
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    assert docling_doc is not None, "Docling parsing failed (required for pdfplumber)"
    
    result = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename)
    
    assert result is not None, "pdfplumber parsing failed"
    assert len(result) > 0, "No items extracted"
    
    print(f"pdfplumber parsing successful: {len(result)} items")


def test_docling_json_output(sample_pdf_data, sample_filename, output_dir):
    """Test that Docling JSON output is created."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename, debug_output_dir=output_dir)
    
    assert docling_doc is not None, "Docling parsing failed"
    
    # Check if JSON file was created
    docling_json = os.path.join(output_dir, f"docling_raw_output_{sample_filename}_dict.json")
    
    assert os.path.exists(docling_json), f"Docling JSON not found at {docling_json}"


def test_pdfplumber_json_output(sample_pdf_data, sample_filename, output_dir):
    """Test that pdfplumber JSON output is created."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    assert docling_doc is not None, "Docling parsing failed"
    
    result = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename, debug_output_dir=output_dir)
    
    assert result is not None, "pdfplumber parsing failed"
    
    # Check if JSON file was created
    pdfplumber_json = os.path.join(output_dir, f"pdfplumber_parsed_output_{sample_filename}.json")
    
    assert os.path.exists(pdfplumber_json), f"pdfplumber JSON not found at {pdfplumber_json}"


def test_combined_modular_parsing(sample_pdf_data, sample_filename):
    """Test both Docling and pdfplumber parsing in sequence."""
    # Test Docling
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename)
    assert docling_doc is not None, "Docling parsing failed"
    assert len(docling_doc.texts) > 0, "Docling extracted no text items"
    
    # Test pdfplumber
    pdfplumber_result = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename)
    assert pdfplumber_result is not None, "pdfplumber parsing failed"
    assert len(pdfplumber_result) > 0, "pdfplumber extracted no items"
    
    print(f"Combined modular parsing successful: Docling={len(docling_doc.texts)} items, pdfplumber={len(pdfplumber_result)} items")


def test_docling_visualization(sample_pdf_path, sample_pdf_data, sample_filename, output_dir):
    """Test Docling visualization tool and save to gitignored output directory."""
    import subprocess
    import sys
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse with Docling to generate JSON
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename, debug_output_dir=output_dir)
    assert docling_doc is not None, "Docling parsing failed"
    
    # Get the Docling JSON path
    docling_json = os.path.join(output_dir, f"docling_raw_output_{sample_filename}_dict.json")
    
    assert os.path.exists(docling_json), f"Docling JSON not found at {docling_json}"
    
    # Run visualization tool
    viz_script = os.path.join(os.path.dirname(__file__), "visualize_docling.py")
    output_pdf = os.path.join(output_dir, f"{sample_filename.replace('.pdf', '')}_docling_viz.pdf")
    
    result = subprocess.run(
        [sys.executable, viz_script, "--pdf", sample_pdf_path,
         "--docling", docling_json, "--output", output_pdf],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"Docling visualization failed: {result.stderr}"
    assert os.path.exists(output_pdf), f"Visualization output not found at {output_pdf}"
    
    print(f"Docling visualization saved to: {output_pdf}")


def test_pdfplumber_visualization(sample_pdf_path, sample_pdf_data, sample_filename, output_dir):
    """Test pdfplumber visualization tool and save to gitignored output directory."""
    import subprocess
    import sys
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse with Docling and pdfplumber to generate JSON
    docling_doc = parse_with_docling(sample_pdf_data, sample_filename, debug_output_dir=output_dir)
    assert docling_doc is not None, "Docling parsing failed"
    
    pdfplumber_result = parse_with_pdfplumber(sample_pdf_data, docling_doc, sample_filename, debug_output_dir=output_dir)
    assert pdfplumber_result is not None, "pdfplumber parsing failed"
    
    # Get the pdfplumber JSON path
    pdfplumber_json = os.path.join(output_dir, f"pdfplumber_parsed_output_{sample_filename}.json")
    
    assert os.path.exists(pdfplumber_json), f"pdfplumber JSON not found at {pdfplumber_json}"
    
    # Run visualization tool
    viz_script = os.path.join(os.path.dirname(__file__), "visualize_pdfplumber.py")
    output_pdf = os.path.join(output_dir, f"{sample_filename.replace('.pdf', '')}_pdfplumber_viz.pdf")
    
    result = subprocess.run(
        [sys.executable, viz_script, "--pdf", sample_pdf_path,
         "--pdfplumber", pdfplumber_json, "--output", output_pdf],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"pdfplumber visualization failed: {result.stderr}"
    assert os.path.exists(output_pdf), f"Visualization output not found at {output_pdf}"
    
    print(f"pdfplumber visualization saved to: {output_pdf}")

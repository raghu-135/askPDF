#!/usr/bin/env python3
"""
Modular Visualization Test Script

This script tests the modular parsing functions and generates visualizations:
1. Uses parse_with_docling() to get Docling output and visualize it
2. Uses parse_with_pdfplumber() to get pdfplumber output and visualize it

NOTE: This script requires all rag-service dependencies (spacy, docling, pdfplumber, etc.)
It's recommended to run this inside the Docker environment where dependencies are already installed.

Usage in Docker:
    docker-compose exec rag-service python /app/tests/test_modular_visualization.py --pdf /app/tests/01030000000000.pdf

Usage locally (requires all dependencies):
    python test_modular_visualization.py --pdf 01030000000000.pdf
"""

import argparse
import sys
import os

# Add the parent directory to the path to import parsing_service

from app.services.parsing_service import parse_with_docling, parse_with_pdfplumber


def test_docling_visualization(pdf_path, output_dir=None):
    """Test Docling parsing and visualization."""
    print(f"\n{'='*60}")
    print("Testing Docling Parsing and Visualization")
    print(f"{'='*60}")
    
    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Get the filename
    filename = os.path.basename(pdf_path)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse with Docling
    print(f"Parsing with Docling...")
    docling_doc = parse_with_docling(pdf_data, filename, debug_output_dir=output_dir)
    
    if not docling_doc:
        print("ERROR: Docling parsing failed")
        return False
    
    print(f"✓ Docling parsing successful")
    print(f"  Document has {len(docling_doc.texts)} text items")
    
    # Generate visualization
    print(f"\nGenerating Docling visualization...")
    
    # Get the Docling JSON path
    docling_json = os.path.join(output_dir, f"docling_raw_output_{filename}_dict.json")
    
    if not os.path.exists(docling_json):
        print(f"ERROR: Docling JSON not found at {docling_json}")
        return False
    
    # Generate output filename
    output_pdf = os.path.join(output_dir, filename.replace('.pdf', '_docling_modular.pdf'))
    
    # Call visualization tool as subprocess
    try:
        import subprocess
        viz_script = os.path.join(os.path.dirname(__file__), "visualize_docling.py")
        result = subprocess.run(
            ["python3", viz_script, "--pdf", pdf_path, "--docling", docling_json, "--output", output_pdf],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"✓ Docling visualization saved to: {output_pdf}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"ERROR in Docling visualization: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR in Docling visualization: {e}")
        return False
    
    return True


def test_pdfplumber_visualization(pdf_path, output_dir=None):
    """Test pdfplumber parsing and visualization."""
    print(f"\n{'='*60}")
    print("Testing pdfplumber Parsing and Visualization")
    print(f"{'='*60}")
    
    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Get the filename
    filename = os.path.basename(pdf_path)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First parse with Docling to get the document
    print(f"Parsing with Docling (required for pdfplumber)...")
    docling_doc = parse_with_docling(pdf_data, filename, debug_output_dir=output_dir)
    
    if not docling_doc:
        print("ERROR: Docling parsing failed (required for pdfplumber)")
        return False
    
    # Parse with pdfplumber
    print(f"Parsing with pdfplumber...")
    result = parse_with_pdfplumber(pdf_data, docling_doc, filename, debug_output_dir=output_dir)
    
    if not result:
        print("ERROR: pdfplumber parsing failed")
        return False
    
    print(f"✓ pdfplumber parsing successful")
    print(f"  Extracted {len(result)} items")
    
    # Generate visualization
    print(f"\nGenerating pdfplumber visualization...")
    
    # Get the pdfplumber JSON path
    pdfplumber_json = os.path.join(output_dir, f"pdfplumber_parsed_output_{filename}.json")
    
    if not os.path.exists(pdfplumber_json):
        print(f"ERROR: pdfplumber JSON not found at {pdfplumber_json}")
        return False
    
    # Generate output filename
    output_pdf = os.path.join(output_dir, filename.replace('.pdf', '_pdfplumber_modular.pdf'))
    
    # Call visualization tool as subprocess
    try:
        import subprocess
        viz_script = os.path.join(os.path.dirname(__file__), "visualize_pdfplumber.py")
        result = subprocess.run(
            ["python3", viz_script, "--pdf", pdf_path, "--pdfplumber", pdfplumber_json, "--output", output_pdf],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"✓ pdfplumber visualization saved to: {output_pdf}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"ERROR in pdfplumber visualization: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR in pdfplumber visualization: {e}")
        return False
    
    return True


def test_combined_visualization(pdf_path, output_dir=None):
    """Test both Docling and pdfplumber parsing and visualization."""
    print(f"Testing modular visualization on: {pdf_path}")
    
    # Test Docling
    docling_success = test_docling_visualization(pdf_path, output_dir=output_dir)
    
    # Test pdfplumber
    pdfplumber_success = test_pdfplumber_visualization(pdf_path, output_dir=output_dir)
    
    if docling_success and pdfplumber_success:
        print(f"\n{'='*60}")
        print("✓ All modular visualization tests completed successfully")
        print(f"{'='*60}")
        return True
    else:
        print(f"\n{'='*60}")
        print("✗ Some tests failed")
        print(f"{'='*60}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test modular parsing and visualization")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to test")
    parser.add_argument("--output-dir", help="Directory for debug output files (default: tests/output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    success = test_combined_visualization(args.pdf, args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

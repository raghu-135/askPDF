#!/usr/bin/env python3
"""
Test script for rag-service parsing code

This script tests the actual parsing_service.py code by:
1. Calling extract_text_with_coordinates() on a PDF (combined approach)
2. Testing parse_with_docling() independently
3. Testing parse_with_pdfplumber() independently
4. Validating the output structure and content
5. Optionally visualizing the results

NOTE: This script requires all rag-service dependencies (spacy, docling, pdfplumber, etc.)
It's recommended to run this inside the Docker environment where dependencies are already installed.

Usage in Docker:
    docker-compose exec rag-service python /app/tests/test_parsing_service.py --pdf /app/tests/01030000000000.pdf

Usage locally (requires all dependencies):
    python test_parsing_service.py --pdf 01030000000000.pdf
"""

import argparse
import json
import sys
import os

# Add the parent directory to the path to import parsing_service

from app.services.parsing_service import extract_text_with_coordinates, parse_with_docling, parse_with_pdfplumber


def test_docling_parsing(pdf_data, filename, output_dir=None):
    """Test Docling parsing independently."""
    print(f"\n{'='*60}")
    print("Testing Docling parsing independently")
    print(f"{'='*60}")
    
    docling_doc = parse_with_docling(pdf_data, filename, debug_output_dir=output_dir)
    
    if not docling_doc:
        print("ERROR: Docling parsing failed")
        return None
    
    print(f"✓ Docling parsing successful")
    print(f"  Document has {len(docling_doc.texts)} text items")
    print(f"  Document has {len(docling_doc.groups)} groups")
    
    # Get label distribution from Docling
    label_counts = {}
    for item in docling_doc.texts:
        if hasattr(item, 'label') and hasattr(item.label, 'value'):
            label = item.label.value
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nDocling label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    return docling_doc


def test_pdfplumber_parsing(pdf_data, docling_doc, filename, output_dir=None):
    """Test pdfplumber parsing independently."""
    print(f"\n{'='*60}")
    print("Testing pdfplumber parsing independently")
    print(f"{'='*60}")
    
    if not docling_doc:
        print("ERROR: No Docling document provided")
        return None
    
    result = parse_with_pdfplumber(pdf_data, docling_doc, filename, debug_output_dir=output_dir)
    
    if not result:
        print("ERROR: pdfplumber parsing failed")
        return None
    
    print(f"✓ pdfplumber parsing successful")
    print(f"  Extracted {len(result)} items")
    
    # Validate structure
    required_fields = ['id', 'text', 'label', 'bbox', 'words', 'font']
    for i, item in enumerate(result):
        for field in required_fields:
            if field not in item:
                print(f"ERROR: Item {i} missing required field: {field}")
                return None
        # Check that item has either 'page' or 'pages' field
        if 'page' not in item and 'pages' not in item:
            print(f"ERROR: Item {i} missing required field: page or pages")
            return None
    
    print("✓ All items have required fields")
    
    # Check label distribution
    label_counts = {}
    for item in result:
        label = item.get('label', 'unspecified')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\npdfplumber label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    return result


def test_multi_bbox_merging(pdf_data, filename):
    """Test the merge_multi_bbox parameter behavior."""
    print(f"\n{'='*60}")
    print("Testing multi-bbox merging behavior")
    print(f"{'='*60}")
    
    # First, parse with Docling to get the document
    docling_doc = parse_with_docling(pdf_data, filename)
    
    if not docling_doc:
        print("ERROR: Docling parsing failed")
        return False
    
    # Test with merge_multi_bbox=True (new behavior)
    print("\nTesting with merge_multi_bbox=True (merging enabled)...")
    result_merged = parse_with_pdfplumber(pdf_data, docling_doc, filename, merge_multi_bbox=True)
    
    # Test with merge_multi_bbox=False (legacy behavior)
    print("Testing with merge_multi_bbox=False (independent processing)...")
    result_independent = parse_with_pdfplumber(pdf_data, docling_doc, filename, merge_multi_bbox=False)
    
    print(f"\nResults:")
    print(f"  Merged: {len(result_merged)} items")
    print(f"  Independent: {len(result_independent)} items")
    
    # Count multi-page items in merged result
    multi_page_count = sum(1 for item in result_merged if 'pages' in item and len(item['pages']) > 1)
    print(f"  Multi-page items (merged): {multi_page_count}")
    
    # Validate that merged results have pages field for multi-bbox items
    has_pages_field = any('pages' in item for item in result_merged)
    print(f"  Has 'pages' field: {has_pages_field}")
    
    # Check that sentences are properly merged (fewer items when merging is enabled)
    if len(result_merged) <= len(result_independent):
        print("✓ Merging produces fewer or equal items (sentences properly combined)")
    else:
        print("⚠ Merging produced more items than independent processing")
    
    # Sample comparison of first few items
    print(f"\nSample comparison (first 2 items):")
    for i in range(min(2, len(result_merged), len(result_independent))):
        merged_item = result_merged[i]
        independent_item = result_independent[i]
        print(f"\n  Item {i}:")
        print(f"    Merged text: {merged_item.get('text')[:80]}...")
        print(f"    Independent text: {independent_item.get('text')[:80]}...")
        if 'pages' in merged_item:
            print(f"    Merged pages: {merged_item['pages']}")
        else:
            print(f"    Merged page: {merged_item.get('page')}")
        print(f"    Independent page: {independent_item.get('page')}")
    
    return True


def test_combined_parsing(pdf_path, visualize=False, compare_existing=False, test_merge=False, output_dir=None):
    """Test the combined parsing service on a PDF file."""
    print(f"Testing combined parsing service on: {pdf_path}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Get the filename for the parsing service
    filename = os.path.basename(pdf_path)
    
    # Test Docling independently
    docling_doc = test_docling_parsing(pdf_data, filename, output_dir=output_dir)
    
    if not docling_doc:
        return False
    
    # Test pdfplumber independently
    pdfplumber_result = test_pdfplumber_parsing(pdf_data, docling_doc, filename, output_dir=output_dir)
    
    if not pdfplumber_result:
        return False
    
    # Test combined approach
    print(f"\n{'='*60}")
    print("Testing combined extract_text_with_coordinates()")
    print(f"{'='*60}")
    
    print(f"Calling extract_text_with_coordinates()...")
    result = extract_text_with_coordinates(pdf_data, filename)
    
    # Test multi-bbox merging if requested
    if test_merge:
        test_multi_bbox_merging(pdf_data, filename)
    
    print(f"\nCombined parsing service returned {len(result)} items")
    
    # Validate the result structure
    if not result:
        print("ERROR: No results returned from parsing service")
        return False
    
    # Check that each item has required fields
    required_fields = ['id', 'text', 'label', 'bbox', 'words', 'font']
    for i, item in enumerate(result):
        for field in required_fields:
            if field not in item:
                print(f"ERROR: Item {i} missing required field: {field}")
                return False
        # Check that item has either 'page' or 'pages' field
        if 'page' not in item and 'pages' not in item:
            print(f"ERROR: Item {i} missing required field: page or pages")
            return False
    
    print("✓ All items have required fields")
    
    # Check label distribution
    label_counts = {}
    for item in result:
        label = item.get('label', 'unspecified')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nCombined label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Check page distribution
    page_counts = {}
    for item in result:
        if 'page' in item:
            page = item.get('page', 0)
            page_counts[page] = page_counts.get(page, 0) + 1
        elif 'pages' in item:
            # For multi-page items, count each page separately
            for page in item.get('pages', []):
                page_counts[page] = page_counts.get(page, 0) + 1
    
    print(f"\nPage distribution:")
    for page, count in sorted(page_counts.items()):
        print(f"  Page {page}: {count} items")
    
    # Sample first few items
    print(f"\nSample items (first 3):")
    for i, item in enumerate(result[:3]):
        print(f"\n  Item {i}:")
        print(f"    Label: {item.get('label')}")
        print(f"    Text: {item.get('text')[:100]}...")
        print(f"    Page: {item.get('page')}")
        print(f"    BBox: {item.get('bbox')}")
        print(f"    Words: {len(item.get('words', []))} words")
    
    # Compare with existing output if requested
    if compare_existing:
        print(f"\nComparing with existing output...")
        # Build the path to existing pdfplumber output
        test_dir = os.path.dirname(pdf_path)
        filename = os.path.basename(pdf_path)
        existing_output_path = os.path.join(test_dir, f"pdfplumber_parsed_output_{filename}.json")
        
        if os.path.exists(existing_output_path):
            with open(existing_output_path, 'r') as f:
                existing_data = json.load(f)
            
            print(f"Existing output has {len(existing_data)} items")
            print(f"New output has {len(result)} items")
            
            if len(result) == len(existing_data):
                print("✓ Item count matches")
            else:
                print("⚠ Item count differs")
                
            # Compare labels
            existing_labels = set(item.get('label') for item in existing_data)
            new_labels = set(item.get('label') for item in result)
            
            if existing_labels == new_labels:
                print("✓ Label types match")
            else:
                print(f"⚠ Label types differ:")
                print(f"  Existing: {existing_labels}")
                print(f"  New: {new_labels}")
        else:
            print(f"Existing output not found at: {existing_output_path}")
    
    # Visualize results if requested
    if visualize:
        print(f"\nGenerating visualizations...")
        
        # Import visualization tools
        try:
            import visualize_bboxes
            import visualize_docling
            import visualize_pdfplumber
        except ImportError:
            print("ERROR: Visualization tools not found")
            return False
        
        # The parsing service writes debug output to output_dir
        docling_json = os.path.join(output_dir, f"docling_raw_output_{filename}_dict.json")
        pdfplumber_json = os.path.join(output_dir, f"pdfplumber_parsed_output_{filename}.json")
        
        # Check if the files were created by the parsing service
        if os.path.exists(docling_json) and os.path.exists(pdfplumber_json):
            output_base = os.path.join(output_dir, filename.replace('.pdf', '_modular_test'))
            
            print(f"Running combined visualization...")
            try:
                visualize_bboxes.visualize_bboxes(
                    pdf_path, docling_json, pdfplumber_json,
                    f"{output_base}_combined.pdf"
                )
                print(f"  ✓ Saved to: {output_base}_combined.pdf")
            except Exception as e:
                print(f"ERROR in combined visualization: {e}")
            
            print(f"Running Docling-only visualization...")
            try:
                visualize_docling.visualize_docling_bboxes(
                    pdf_path, docling_json,
                    f"{output_base}_docling.pdf"
                )
                print(f"  ✓ Saved to: {output_base}_docling.pdf")
            except Exception as e:
                print(f"ERROR in Docling visualization: {e}")
            
            print(f"Running pdfplumber-only visualization...")
            try:
                visualize_pdfplumber.visualize_pdfplumber_bboxes(
                    pdf_path, pdfplumber_json,
                    f"{output_base}_pdfplumber.pdf"
                )
                print(f"  ✓ Saved to: {output_base}_pdfplumber.pdf")
            except Exception as e:
                print(f"ERROR in pdfplumber visualization: {e}")
        else:
            print(f"Output files not found")
            print(f"Docling JSON: {docling_json}")
            print(f"pdfplumber JSON: {pdfplumber_json}")
    
    print("\n✓ All parsing service tests completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test rag-service parsing code")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to test")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of the output")
    parser.add_argument("--compare-existing", action="store_true", help="Compare with existing output files")
    parser.add_argument("--test-merge", action="store_true", help="Test multi-bbox merging behavior")
    parser.add_argument("--output-dir", help="Directory for debug output files (default: tests/output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    success = test_combined_parsing(args.pdf, args.visualize, args.compare_existing, args.test_merge, args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

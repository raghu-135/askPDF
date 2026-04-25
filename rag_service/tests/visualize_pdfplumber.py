#!/usr/bin/env python3
"""
pdfplumber Bounding Box Visualization Tool

This script visualizes pdfplumber bounding boxes on a PDF file with color-coding by label type.
Each label type (section_header, text, caption, etc.) is drawn with a different color.

Usage:
    python visualize_pdfplumber.py --pdf <pdf_path> --pdfplumber <pdfplumber_json_path> --output <output_path>
"""

import argparse
import json
import fitz  # PyMuPDF


# Color scheme for different pdfplumber label types
LABEL_COLORS = {
    "section_header": (1, 0, 0),      # Red
    "text": (0, 0, 1),                # Blue
    "caption": (0, 1, 0),             # Green
    "footnote": (1, 0.5, 0),          # Orange
    "page_header": (0.5, 0, 1),       # Purple
    "page_footer": (1, 1, 0),         # Yellow
    "title": (0.5, 0, 0),             # Dark Red
    "subtitle": (0, 0, 0.5),          # Dark Blue
    "list_item": (0, 0.5, 0),         # Dark Green
    "code": (0.5, 0.5, 0.5),          # Gray
    "unspecified": (0.5, 0.5, 0.5),   # Gray (default)
}


def get_color_for_label(label):
    """Get color for a given label type."""
    return LABEL_COLORS.get(label, LABEL_COLORS["unspecified"])


def draw_pdfplumber_boxes(page, pdfplumber_data, page_num):
    """Draw pdfplumber bounding boxes on the page with color-coding by label."""
    label_counts = {}
    
    for item in pdfplumber_data:
        # Check if item has a 'pages' field (multi-page sentence)
        if 'pages' in item:
            # Only process if this page is in the pages list
            if (page_num + 1) not in item['pages']:
                continue
        elif item.get("page") != page_num + 1:  # pdfplumber uses 1-indexed
            continue
        
        label = item.get("label", "unspecified")
        if label not in label_counts:
            label_counts[label] = 0
        
        # Use the bboxes array if available (contains all line-level boxes)
        # Otherwise fall back to the primary bbox
        bboxes_to_draw = item.get("bboxes", [])
        if not bboxes_to_draw:
            # Fall back to primary bbox if bboxes array is empty
            primary_bbox = item.get("bbox")
            if primary_bbox:
                bboxes_to_draw = [primary_bbox]
        
        # Get color for this label type
        color = get_color_for_label(label)
        
        # Draw all bounding boxes for this item
        for bbox in bboxes_to_draw:
            # Skip if bbox doesn't match current page
            if isinstance(bbox, dict) and bbox.get("page") != page_num + 1:
                continue
            
            # Handle bbox as dict or list
            if isinstance(bbox, dict):
                x0 = bbox.get("x0")
                y0 = bbox.get("y0")
                x1 = bbox.get("x1")
                y1 = bbox.get("y1")
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
            else:
                continue
            
            # Draw rectangle
            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(
                rect,
                color=color,
                width=2,
                fill=None
            )
            
            # Add label text annotation near the box (only for the first bbox)
            # Only add label text if the box is large enough
            if (x1 - x0) > 50 and (y1 - y0) > 15:
                label_text = label
                try:
                    # Try to insert the label text at the top-left of the box
                    point = fitz.Point(x0, y0 - 2)
                    page.insert_text(
                        point,
                        label_text,
                        fontsize=8,
                        color=color,
                        fontname="helv"
                    )
                except:
                    pass  # Skip if text insertion fails
        
        label_counts[label] += 1
    
    return label_counts


def visualize_pdfplumber_bboxes(pdf_path, pdfplumber_json_path, output_path):
    """Main function to visualize pdfplumber bounding boxes on PDF."""
    # Load pdfplumber JSON
    with open(pdfplumber_json_path, 'r') as f:
        pdfplumber_data = json.load(f)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    total_label_counts = {}
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Draw pdfplumber boxes
        label_counts = draw_pdfplumber_boxes(page, pdfplumber_data, page_num)
        
        # Aggregate counts
        for label, count in label_counts.items():
            if label not in total_label_counts:
                total_label_counts[label] = 0
            total_label_counts[label] += count
    
    # Save the annotated PDF
    doc.save(output_path)
    doc.close()
    
    print(f"pdfplumber annotated PDF saved to: {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(total_label_counts.items()):
        color = LABEL_COLORS.get(label, LABEL_COLORS["unspecified"])
        print(f"  {label}: {count} boxes (RGB: {color})")
    
    print(f"\nTotal boxes drawn: {sum(total_label_counts.values())}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pdfplumber bounding boxes on a PDF with color-coding by label")
    parser.add_argument("--pdf", required=True, help="Path to the original PDF file")
    parser.add_argument("--pdfplumber", required=True, help="Path to pdfplumber output JSON file")
    parser.add_argument("--output", required=True, help="Path for the output annotated PDF file")
    
    args = parser.parse_args()
    
    visualize_pdfplumber_bboxes(args.pdf, args.pdfplumber, args.output)


if __name__ == "__main__":
    main()

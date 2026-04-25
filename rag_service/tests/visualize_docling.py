#!/usr/bin/env python3
"""
Docling Bounding Box Visualization Tool

This script visualizes Docling bounding boxes on a PDF file with color-coding by label type.
Each label type (section_header, text, caption, etc.) is drawn with a different color.

Usage:
    python visualize_docling.py --pdf <pdf_path> --docling <docling_json_path> --output <output_path>
"""

import argparse
import json
import fitz  # PyMuPDF


# Color scheme for different Docling label types
LABEL_COLORS = {
    "section_header": (1, 0, 0),      # Red
    "text": (0, 0, 1),                # Blue
    "caption": (0, 1, 0),             # Green
    "footnote": (1, 0.5, 0),          # Orange
    "page_header": (0.5, 0, 1),       # Purple
    "page_footer": (1, 1, 0),         # Yellow
    "picture": (1, 0, 1),             # Pink
    "table": (0, 1, 1),               # Cyan
    "title": (0.5, 0, 0),             # Dark Red
    "subtitle": (0, 0, 0.5),          # Dark Blue
    "list_item": (0, 0.5, 0),         # Dark Green
    "code": (0.5, 0.5, 0.5),          # Gray
    "formula": (1, 0.65, 0),          # Gold
    "unspecified": (0.5, 0.5, 0.5),   # Gray (default)
}


def convert_bottomleft_to_topleft(bbox, page_height):
    """Convert Docling BOTTOMLEFT bbox to TOPLEFT."""
    l, t, r, b = bbox
    return (l, page_height - t, r, page_height - b)


def get_color_for_label(label):
    """Get color for a given label type."""
    return LABEL_COLORS.get(label, LABEL_COLORS["unspecified"])


def draw_docling_boxes(page, docling_texts, page_num, page_height):
    """Draw Docling bounding boxes on the page with color-coding by label."""
    label_counts = {}
    
    for text_item in docling_texts:
        if not text_item.get("prov"):
            continue
        
        label = text_item.get("label", "unspecified")
        if label not in label_counts:
            label_counts[label] = 0
        
        # Extract additional metadata
        content_layer = text_item.get("content_layer", "")
        self_ref = text_item.get("self_ref", "")
        parent = text_item.get("parent", {})
        parent_ref = parent.get("$ref", "") if isinstance(parent, dict) else str(parent)
        
        for prov in text_item.get("prov", []):
            if prov.get("page_no") != page_num + 1:  # Docling uses 1-indexed
                continue
            
            bbox_data = prov.get("bbox", {})
            if not bbox_data:
                continue
            
            # Convert from BOTTOMLEFT to TOPLEFT
            l, t, r, b = bbox_data["l"], bbox_data["t"], bbox_data["r"], bbox_data["b"]
            x0, y0, x1, y1 = convert_bottomleft_to_topleft((l, t, r, b), page_height)
            
            # Get color for this label type
            color = get_color_for_label(label)
            
            # Draw rectangle
            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(
                rect,
                color=color,
                width=2,
                fill=None
            )
            
            # Add metadata annotation near the box
            # Only add text if the box is large enough
            if (x1 - x0) > 50 and (y1 - y0) > 15:
                # Build annotation text with metadata (excluding orig and text)
                annotation_parts = [label]
                if content_layer:
                    annotation_parts.append(f"layer:{content_layer}")
                if self_ref:
                    annotation_parts.append(f"self:{self_ref.split('/')[-1]}")  # Show only the last part of self_ref
                if parent_ref:
                    annotation_parts.append(f"parent:{parent_ref.split('/')[-1]}")  # Show only the last part of parent_ref
                
                annotation_text = " | ".join(annotation_parts)
                
                try:
                    # Try to insert the annotation text at the top-left of the box
                    point = fitz.Point(x0, y0 - 2)
                    page.insert_text(
                        point,
                        annotation_text,
                        fontsize=7,
                        color=color,
                        fontname="helv"
                    )
                except:
                    pass  # Skip if text insertion fails
            
            label_counts[label] += 1
    
    return label_counts


def visualize_docling_bboxes(pdf_path, docling_json_path, output_path):
    """Main function to visualize Docling bounding boxes on PDF."""
    # Load Docling JSON
    with open(docling_json_path, 'r') as f:
        docling_data = json.load(f)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    total_label_counts = {}
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        
        # Draw Docling boxes
        docling_texts = docling_data.get("texts", [])
        label_counts = draw_docling_boxes(page, docling_texts, page_num, page_height)
        
        # Aggregate counts
        for label, count in label_counts.items():
            if label not in total_label_counts:
                total_label_counts[label] = 0
            total_label_counts[label] += count
    
    # Save the annotated PDF
    doc.save(output_path)
    doc.close()
    
    print(f"Docling annotated PDF saved to: {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(total_label_counts.items()):
        color = LABEL_COLORS.get(label, LABEL_COLORS["unspecified"])
        print(f"  {label}: {count} boxes (RGB: {color})")
    
    print(f"\nTotal boxes drawn: {sum(total_label_counts.values())}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Docling bounding boxes on a PDF with color-coding by label")
    parser.add_argument("--pdf", required=True, help="Path to the original PDF file")
    parser.add_argument("--docling", required=True, help="Path to Docling output JSON file")
    parser.add_argument("--output", required=True, help="Path for the output annotated PDF file")
    
    args = parser.parse_args()
    
    visualize_docling_bboxes(args.pdf, args.docling, args.output)


if __name__ == "__main__":
    main()

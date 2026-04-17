import io
import os
import logging
import json
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# OpenDataLoader configuration via environment variables
OPEN_DATALOADER_FORMAT = os.environ.get("OPEN_DATALOADER_FORMAT", "json")

def _get_table_bboxes(page):
    """Identify bounding boxes for all tables detected on a PDF page."""
    tables = page.find_tables()
    return [fitz.Rect(t.bbox) for t in tables]

def _get_image_bboxes(page):
    """Identify bounding boxes for all images detected on a PDF page."""
    image_bboxes = []
    for img in page.get_images():
        rects = page.get_image_rects(img[0])
        image_bboxes.extend(rects)
    return image_bboxes

def _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
    """
    Determine if a text block should be filtered out (e.g., headers, footers, 
    items inside tables or overlapping images).
    """
    if bbox.y1 < header_height or bbox.y0 > footer_y:
        return True
    block_center = fitz.Point((bbox.x0 + bbox.x1)/2, (bbox.y0 + bbox.y1)/2)
    if any(block_center in t_bbox for t_bbox in table_bboxes):
        return True
    if any(block_center in i_bbox for i_bbox in image_bboxes):
        return True
    return False

def _process_line(line, page_num, page_height, page_width):
    """
    Process a single line of text from PyMuPDF, extracting character-level 
    coordinates and font information.
    """
    line_text = ""
    line_chars = []
    processed_fonts = []
    
    spans = sorted(line.get("spans", []), key=lambda s: s["bbox"][0])
    for span in spans:
        font_name = span.get("font", "")
        font_size = span.get("size", 0)
        processed_fonts.append({"name": font_name, "size": font_size})
        
        chars = span.get("chars", [])
        for char_info in chars:
            c = char_info.get("c", "")
            if not c:
                continue
            c_bbox = char_info.get("bbox", [0,0,0,0])
            c_height = c_bbox[3] - c_bbox[1]
            c_y_bottom = page_height - c_bbox[1] - c_height
            line_text += c
            line_chars.append({
                "page": page_num + 1,
                "x": c_bbox[0],
                "y": c_y_bottom,
                "width": c_bbox[2] - c_bbox[0],
                "height": c_height,
                "page_height": page_height,
                "page_width": page_width,
                "font": font_name,
                "size": font_size
            })
    
    line_font = processed_fonts[0] if processed_fonts else {"name": "", "size": 0}
    return line_text, line_chars, line_font

def _process_block(block, page_num, page_height, page_width):
    """
    Process a text block containing multiple lines, grouping them into 
    segments with consistent font styles.
    """
    segments = []
    current_text = ""
    current_chars = []
    current_font = None
    
    NO_SPACE_ENDINGS = ("/", "@", "-", ".", "_", ":")
    
    for line in block.get("lines", []):
        line_text, line_chars, line_font = _process_line(line, page_num, page_height, page_width)
        if not line_text:
            continue
            
        if current_font is not None:
            font_changed = (line_font["name"] != current_font["name"] or 
                          abs(line_font["size"] - current_font["size"]) > 0.5)
            
            if font_changed:
                if current_text.strip():
                    segments.append({
                        "text": current_text,
                        "chars": current_chars,
                        "font": current_font
                    })
                current_text = ""
                current_chars = []
                current_font = line_font
            
        if current_font is None:
            current_font = line_font
            
        if current_text:
            prev_trimmed = current_text.strip()
            needs_space = not (prev_trimmed.endswith(NO_SPACE_ENDINGS) or line_text.startswith(tuple(NO_SPACE_ENDINGS) + (" ",)))
            
            if needs_space:
                current_text += " "
                if current_chars:
                    last = current_chars[-1]
                    current_chars.append({
                        "page": last["page"],
                        "x": last["x"] + last["width"],
                        "y": last["y"],
                        "width": 0,
                        "height": last["height"],
                        "page_height": last["page_height"],
                        "page_width": last["page_width"],
                        "is_space": True,
                        "font": last.get("font", ""),
                        "size": last.get("size", 0)
                    })
        
        current_text += line_text
        current_chars.extend(line_chars)
        
    if current_text.strip():
        segments.append({
            "text": current_text,
            "chars": current_chars,
            "font": current_font
        })
        
    return segments


def _flatten_odl_elements(element: dict, flat_list: list) -> None:
    """
    Recursively flatten the hierarchical OpenDataLoader structure into a flat list.
    Preserves reading order as elements appear in the 'kids' array.
    """
    # Add the current element if it has the required fields
    if 'type' in element and 'page number' in element:
        flat_list.append(element)
    
    # Recursively process nested kids
    if 'kids' in element and isinstance(element['kids'], list):
        for kid in element['kids']:
            _flatten_odl_elements(kid, flat_list)


def _load_opendataloader_elements(data: bytes) -> list:
    """
    Use OpenDataLoader to extract structural elements with bounding boxes.
    Returns a flat list of elements with type, content, page number, and bounding box.
    """
    try:
        import opendataloader_pdf
        
        # Create a temporary file for OpenDataLoader
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name
        
        try:
            # Convert PDF to JSON format
            result = opendataloader_pdf.convert(
                input_path=[tmp_path],
                output_dir=None,  # Don't write to disk, use return value
                format="json"
            )
            
            # Parse the JSON output - it's a dict with 'kids' array
            if result and len(result) > 0:
                json_output = result[0].get('json', '{}')
                odl_doc = json.loads(json_output) if isinstance(json_output, str) else json_output

                logger.info(f"ODL doc keys: {odl_doc.keys() if isinstance(odl_doc, dict) else 'not a dict'}")
                
                # Flatten the hierarchical structure while preserving reading order
                flat_elements = []
                _flatten_odl_elements(odl_doc, flat_elements)
                logger.info(f"Flattened {len(flat_elements)} elements from OpenDataLoader")
                return flat_elements
            else:
                logger.warning(f"OpenDataLoader returned no result or empty result")
                return []
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"OpenDataLoader conversion failed: {e}")
        return []


def _map_odl_type_to_label(element: dict) -> str:
    """
    Map OpenDataLoader element type to our label system.
    """
    element_type = element.get('type', 'text').lower()
    
    type_mapping = {
        'heading': 'heading',
        'title': 'heading',
        'sectionheader': 'heading',
        'table': 'table',
        'figure': 'picture',
        'picture': 'picture',
        'image': 'picture',
        'caption': 'text',
        'footnote': 'text',
        'text': 'text',
        'paragraph': 'text',
        'list': 'text',
        'listitem': 'text',
        'code': 'text',
        'equation': 'text',
        'formula': 'text',
    }
    
    return type_mapping.get(element_type, 'text')


def extract_text_with_coordinates(data: bytes, filename: str):
    """
    Extract high-fidelity text items from a PDF using a hybrid approach:
    OpenDataLoader for structural segmentation (labels like 'table', 'heading') 
    combined with PyMuPDF for precise character-level coordinates.
    """
    if not filename:
        return []

    doc = fitz.open(stream=data, filetype="pdf")
    items = []
    
    # Load structural elements from OpenDataLoader
    logger.info("Loading OpenDataLoader elements")
    odl_elements = _load_opendataloader_elements(data)
    logger.info(f"Loaded {len(odl_elements)} odl_elements from OpenDataLoader")

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        table_bboxes = _get_table_bboxes(page)
        image_bboxes = _get_image_bboxes(page)
        header_height = page_height * 0.05
        footer_y = page_height * 0.95
        
        text_page = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        all_blocks = [b for b in text_page.get("blocks", []) if b.get("type") == 0]
        processed_indices = set()

        if odl_elements:
            # Filter elements for current page (OpenDataLoader uses 1-based page numbers)
            page_elements = [
                elem for elem in odl_elements 
                if elem.get('page number', 0) == (page_num + 1)
            ]
            
            for elem in page_elements:
                item_label = _map_odl_type_to_label(elem)
                
                # Get bounding box from OpenDataLoader [x1, y1, x2, y2]
                bbox = elem.get('bounding box', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    # OpenDataLoader bbox is [left, top, right, bottom] in PDF coordinates
                    # where (0,0) is bottom-left, so we need to flip Y
                    rect = fitz.Rect(
                        bbox[0],
                        page_height - bbox[3],  # Convert top to y0 (bottom-up)
                        bbox[2],
                        page_height - bbox[1]   # Convert bottom to y1 (bottom-up)
                    )
                else:
                    continue
                
                item_blocks = []
                for i, b in enumerate(all_blocks):
                    if i in processed_indices:
                        continue
                    b_rect = fitz.Rect(b["bbox"])
                    
                    if _is_block_filtered(b_rect, header_height, footer_y, table_bboxes, image_bboxes):
                        processed_indices.add(i)
                        continue
                        
                    intersection = rect.intersect(b_rect)
                    b_center = fitz.Point((b_rect.x0 + b_rect.x1)/2, (b_rect.y0 + b_rect.y1)/2)
                    
                    if (intersection.get_area() / b_rect.get_area() > 0.4 or 
                        b_center in rect or 
                        rect.contains(b_rect)):
                        item_blocks.append(b)
                        processed_indices.add(i)
                
                if item_blocks:
                    item_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
                    for b in item_blocks:
                        segments = _process_block(b, page_num, page_height, page_width)
                        for seg in segments:
                            seg_text = seg["text"]
                            seg_chars = seg["chars"]
                            seg_font = seg["font"]
                            
                            while seg_text and seg_text[-1].isspace():
                                seg_text = seg_text[:-1]
                                if seg_chars:
                                    seg_chars.pop()
                            
                            if seg_text.strip():
                                items.append({
                                    "text": seg_text,
                                    "label": item_label,
                                    "char_map": seg_chars,
                                    "page": page_num + 1,
                                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                                    "font": seg_font
                                })

        remaining_blocks = [all_blocks[i] for i in range(len(all_blocks)) if i not in processed_indices]
        sorted_remaining = sorted(remaining_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        for block in sorted_remaining:
            bbox = fitz.Rect(block["bbox"])
            if _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
                continue
            
            segments = _process_block(block, page_num, page_height, page_width)
            for seg in segments:
                seg_text = seg["text"]
                seg_chars = seg["chars"]
                seg_font = seg["font"]
                
                while seg_text and seg_text[-1].isspace():
                    seg_text = seg_text[:-1]
                    if seg_chars:
                        seg_chars.pop()
                
                if seg_text.strip():
                    items.append({
                        "text": seg_text,
                        "label": "text",
                        "char_map": seg_chars,
                        "page": page_num + 1,
                        "bbox": list(bbox),
                        "font": seg_font
                    })
            
    doc.close()
    return items

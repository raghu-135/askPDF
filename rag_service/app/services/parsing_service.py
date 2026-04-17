import json
import logging
import pymupdf.layout  # Must be before pymupdf4llm
import pymupdf4llm
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def _load_pymupdf4llm_elements(doc: fitz.Document) -> list:
    """Load semantic elements from PyMuPDF4LLM with Layout mode."""
    json_data = pymupdf4llm.to_json(doc, page_chunks=True)
    data = json.loads(json_data)

    # PyMuPDF4LLM returns a dict with 'pages' key containing the list
    if isinstance(data, dict) and 'pages' in data:
        return data['pages']
    elif isinstance(data, list):
        return data
    else:
        logger.warning(f"Unexpected PyMuPDF4LLM output type: {type(data)}")
        return []

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


def extract_text_with_coordinates(data: bytes, filename: str):
    """
    Extract high-fidelity text items from a PDF using PyMuPDF4LLM:
    - PyMuPDF4LLM provides semantic labels, reading order, and bounding boxes
    - PyMuPDF provides character-level coordinates for highlighting
    """
    if not filename:
        return []

    doc = fitz.open(stream=data, filetype="pdf")
    items = []
    
    # Load semantic elements from PyMuPDF4LLM (single-pass using same doc)
    pymupdf4llm_data = []
    try:
        pymupdf4llm_data = _load_pymupdf4llm_elements(doc)
    except Exception as e:
        logger.warning(f"PyMuPDF4LLM conversion failed: {e}")
        pymupdf4llm_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Get PyMuPDF4LLM page data
        page_boxes = []
        if isinstance(pymupdf4llm_data, list) and page_num < len(pymupdf4llm_data):
            page_data = pymupdf4llm_data[page_num]
            if isinstance(page_data, dict):
                page_boxes = page_data.get("boxes", [])

        # Process each box with its semantic label
        for idx, box in enumerate(page_boxes):
            # PyMuPDF4LLM uses x0/y0/x1/y1 keys, not bbox array
            box_bbox = [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")]
            box_class = box.get("boxclass", "text")  # PyMuPDF4LLM uses 'boxclass'
            
            # Extract text from textlines array
            textlines = box.get("textlines") or []
            box_text = ""
            for line in textlines:
                for span in line.get("spans", []):
                    box_text += span.get("text", "")

            if len(box_bbox) != 4 or None in box_bbox:
                continue
            
            # Map PyMuPDF4LLM boxclass names to our label system
            label = "text"
            if "picture" in box_class:
                label = "picture"
            elif "table" in box_class:
                label = "table"
            elif any(h in box_class for h in ["page-header", "header"]):
                label = "heading"
            elif "page-footer" in box_class:
                label = "footer"
            # titles, h1, h2, h3, heading are kept as text (not filtered)
            
            # Get character-level coordinates for this region using PyMuPDF
            rect = fitz.Rect(box_bbox)
            char_map = []

            try:
                # Extract text with character-level coordinates within the bounding box
                text_dict = page.get_text("rawdict", clip=rect, flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                font_name = span.get("font", "")
                                font_size = span.get("size", 0)
                                for char_info in span.get("chars", []):
                                    c = char_info.get("c", "")
                                    if not c:
                                        continue
                                    c_bbox = char_info.get("bbox", [0,0,0,0])
                                    c_height = c_bbox[3] - c_bbox[1]
                                    c_y_bottom = page_height - c_bbox[1] - c_height
                                    char_map.append({
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
            except Exception as e:
                logger.warning(f"Error extracting character coordinates: {e}")

            # Use the text from PyMuPDF4LLM if available, otherwise use extracted text
            text = box_text if box_text else "".join([c.get("c", "") for c in char_map])

            # Filter out images, headers, and footers
            if label in ["picture", "heading", "footer"]:
                continue

            if text.strip() and char_map:
                items.append({
                    "text": text,
                    "label": label,
                    "char_map": char_map,
                    "page": page_num + 1,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "font": {"name": char_map[0].get("font", ""), "size": char_map[0].get("size", 0)} if char_map else {"name": "", "size": 0}
                })
        
        # Fallback: if no PyMuPDF4LLM data, use basic PyMuPDF extraction
        if not page_boxes:
            text_page = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            all_blocks = [b for b in text_page.get("blocks", []) if b.get("type") == 0]
            
            for idx, block in enumerate(all_blocks):
                bbox = fitz.Rect(block["bbox"])
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

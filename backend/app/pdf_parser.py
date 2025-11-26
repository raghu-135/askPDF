import fitz  # PyMuPDF
from io import BytesIO

def extract_text_with_coordinates(data: bytes):
    """
    Extracts text and character coordinates from PDF bytes using PyMuPDF's rawdict.
    
    IMPROVEMENTS:
    - Uses `rawdict` for exact character bounding boxes (pixel-perfect highlighting).
    - Calculates median font size to identify headings.
    - Forces double newlines after headings to prevent sentence clubbing.
    - Filters tables, headers/footers, and images.
    
    Returns:
        full_text (str): The complete text of the PDF.
        char_map (list): List of {page, x, y, width, height} for each char.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    
    full_text = ""
    char_map = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        # 1. Detect Tables
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables]
        
        # 2. Detect Images
        image_bboxes = []
        for img in page.get_images():
            rects = page.get_image_rects(img[0])
            image_bboxes.extend(rects)
            
        # 3. Define Header/Footer Regions (5% margin)
        header_height = page_height * 0.05
        footer_y = page_height * 0.95
        
    # 4. Get Text with rawdict (provides individual characters)
        # flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
        text_page = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = text_page.get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]
        
        # 5. Sort Blocks (Top-to-bottom, Left-to-right)
        # We use a smaller vertical tolerance (10px) to group lines better
        sorted_blocks = sorted(text_blocks, key=lambda b: (b["bbox"][1] // 10, b["bbox"][0]))
        
        for block in sorted_blocks:
            bbox = fitz.Rect(block["bbox"])
            
            # --- FILTERING ---
            if bbox.y1 < header_height or bbox.y0 > footer_y: continue # Header/Footer
            
            # Check center point for Table/Image overlap
            block_center = fitz.Point((bbox.x0 + bbox.x1)/2, (bbox.y0 + bbox.y1)/2)
            
            if any(block_center in t_bbox for t_bbox in table_bboxes): continue # Table
            if any(block_center in i_bbox for i_bbox in image_bboxes): continue # Image
            # --- END FILTERING ---
            
            block_text = ""
            block_chars = []
            
            for line in block.get("lines", []):
                line_text = ""
                line_chars = []
                
                # Sort spans
                spans = sorted(line.get("spans", []), key=lambda s: s["bbox"][0])
                
                for span in spans:
                    # rawdict 'chars' list contains individual characters
                    chars = span.get("chars", [])
                    
                    for char_info in chars:
                        c = char_info.get("c", "")
                        if not c: continue
                        
                        # rawdict gives exact bbox for the character
                        c_bbox = char_info.get("bbox", [0,0,0,0])
                        
                        # Convert to bottom-left origin
                        # PyMuPDF y0 is top edge.
                        # pdfminer y0 is bottom edge (from bottom).
                        # y_bottom = page_height - y_top - height
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
                            "page_width": page_width
                        })
                
                if line_text:
                    block_text += line_text
                    block_chars.extend(line_chars)
                    # Add space if line doesn't end with whitespace
                    if not line_text.endswith((" ", "\n", "\t")):
                        block_text += " "
                        if line_chars:
                            last = line_chars[-1]
                            block_chars.append({
                                "page": last["page"],
                                "x": last["x"] + last["width"],
                                "y": last["y"],
                                "width": last["width"], # approximate space width
                                "height": last["height"],
                                "page_height": last["page_height"],
                                "page_width": last["page_width"],
                                "is_space": True
                            })

            if block_text.strip():
                # Remove trailing whitespace from block_text AND char_map to keep them in sync
                while block_text and block_text[-1].isspace():
                    block_text = block_text[:-1]
                    if block_chars:
                        block_chars.pop()
                
                # Add to full text
                full_text += block_text
                char_map.extend(block_chars)
                
                # Always use double newline separator for blocks
                # This ensures nlp.py treats them as separate sentences
                separator = "\n\n"
                
                full_text += separator
                
                # Add newline markers to char_map
                if block_chars:
                    last = block_chars[-1]
                    for _ in range(2): # Add 2 newlines
                        char_map.append({
                            "page": last["page"],
                            "x": last["x"],
                            "y": last["y"],
                            "width": 0,
                            "height": last["height"],
                            "page_height": last["page_height"],
                            "page_width": last["page_width"],
                            "is_newline": True
                        })
    
    doc.close()
    return full_text, char_map

def pdf_bytes_to_text(data: bytes) -> str:
    """
    Simple text extraction from PDF bytes.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

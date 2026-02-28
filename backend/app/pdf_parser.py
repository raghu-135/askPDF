import io
import os
import logging
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter, DocumentStream, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TextItem

# Configure logging to suppress noisy internal library logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.pdf_parser")

# Suppress noisy dependencies
for logger_name in ["docling", "docling_core", "huggingface_hub", "pypdfium2", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Initialize Docling converter as a module-level singleton
_pipeline_options = PdfPipelineOptions()

def _get_required_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise ValueError(f"{name} environment variable is not set")
    return val

# Tuneable via environment variables for accuracy vs speed
_pipeline_options.do_ocr = _get_required_env("DOCLING_DO_OCR").lower() == "true"
_pipeline_options.do_table_structure = _get_required_env("DOCLING_DO_TABLE_STRUCTURE").lower() == "true"
_pipeline_options.do_formula_enrichment = _get_required_env("DOCLING_DO_FORMULA_ENRICHMENT").lower() == "true"

# Table structure mode (FAST or ACCURATE)
_table_mode = _get_required_env("DOCLING_TABLE_MODE").upper()
if _table_mode == "ACCURATE":
    _pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
else:
    _pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# OCR options
if _pipeline_options.do_ocr:
    _pipeline_options.ocr_options.force_full_page_ocr = _get_required_env("DOCLING_FORCE_FULL_PAGE_OCR").lower() == "true"

_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

# Log the active configuration once during initialization
logger.info("Docling Parser initialized with configuration:")
logger.info(f"  - OCR Enabled: {_pipeline_options.do_ocr}")
if _pipeline_options.do_ocr:
    logger.info(f"  - Force Full Page OCR: {_pipeline_options.ocr_options.force_full_page_ocr}")
logger.info(f"  - Table Structure Extraction: {_pipeline_options.do_table_structure}")
logger.info(f"  - Table Mode: {_table_mode}")
logger.info(f"  - Formula Enrichment: {_pipeline_options.do_formula_enrichment}")

def _get_table_bboxes(page):
    """
    Extract table bounding boxes from a PDF page using PyMuPDF.
    Args:
        page: PyMuPDF page object.
    Returns:
        List of fitz.Rect objects representing table bounding boxes.
    """
    tables = page.find_tables()
    return [fitz.Rect(t.bbox) for t in tables]

def _get_image_bboxes(page):
    """
    Extract image bounding boxes from a PDF page using PyMuPDF.
    Args:
        page: PyMuPDF page object.
    Returns:
        List of fitz.Rect objects representing image bounding boxes.
    """
    image_bboxes = []
    for img in page.get_images():
        rects = page.get_image_rects(img[0])
        image_bboxes.extend(rects)
    return image_bboxes

def _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
    """
    Determine if a text block should be filtered out (header, footer, table, or image).
    Args:
        bbox: fitz.Rect of the block.
        header_height: Height of the header region.
        footer_y: Y-coordinate of the footer region start.
        table_bboxes: List of table bounding boxes.
        image_bboxes: List of image bounding boxes.
    Returns:
        True if the block should be filtered, False otherwise.
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
    Process a line from a text block and extract text, character coordinates, and font metadata.
    Args:
        line: Line dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        Tuple of (line_text, line_chars, font_info)
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
    
    # Return dominant font for the line
    line_font = processed_fonts[0] if processed_fonts else {"name": "", "size": 0}
    return line_text, line_chars, line_font

def _process_block(block, page_num, page_height, page_width):
    """
    Process a text block and extract its text, character coordinates, and font metadata.
    Splits the block into segments if the font changes between lines.
    Args:
        block: Block dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        List of dicts, each with (text, chars, font)
    """
    segments = []
    current_text = ""
    current_chars = []
    current_font = None
    
    # Characters that suggest a URL or Email is being broken across lines
    # If a line ends with these, we don't add a space
    NO_SPACE_ENDINGS = ("/", "@", "-", ".", "_", ":")
    
    for line in block.get("lines", []):
        line_text, line_chars, line_font = _process_line(line, page_num, page_height, page_width)
        if not line_text:
            continue
            
        # Check for font change to start a new segment
        if current_font is not None:
            font_changed = (line_font["name"] != current_font["name"] or 
                          abs(line_font["size"] - current_font["size"]) > 0.5)
            
            if font_changed:
                # Save current segment and start new one
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
            
        # Join line text
        if current_text:
            # Smart joining: avoid space for URLs/Emails
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
        
    # Add final segment
    if current_text.strip():
        segments.append({
            "text": current_text,
            "chars": current_chars,
            "font": current_font
        })
        
    return segments


def extract_text_with_coordinates(data: bytes, filename: str):
    """
    Extracts text and character coordinates from PDF bytes using a hybrid approach:
    1. Docling determines the logical structure and reading order (paragraphs, headings).
    2. PyMuPDF's `rawdict` provides exact character-level bounding boxes.
    
    Returns:
        List of dicts: [
            {"text": str, "label": str, "char_map": list},
            ...
        ]
    """
    from docling.datamodel.document import TextItem, TableItem, PictureItem
    if not filename:
        logger.error("No filename provided to extract_text_with_coordinates.")
        return []

    doc = fitz.open(stream=data, filetype="pdf")
    items = []
    
    # 1. Get Docling conversion for structure and reading order
    docling_doc = None
    try:
        logger.info(f"Docling processing file: {filename}")
        source = DocumentStream(name=filename, stream=io.BytesIO(data))
        docling_result = _docling_converter.convert(source)
        docling_doc = docling_result.document
        logger.info(f"Docling conversion successful: {filename}")
    except Exception as e:
        logger.warning(f"Docling conversion failed, falling back to basic PyMuPDF: {e}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        table_bboxes = _get_table_bboxes(page)
        image_bboxes = _get_image_bboxes(page)
        header_height = page_height * 0.05
        footer_y = page_height * 0.95
        
        # Get all text blocks on this page from PyMuPDF
        text_page = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        all_blocks = [b for b in text_page.get("blocks", []) if b.get("type") == 0]
        processed_indices = set()

        # 2. Process blocks in Docling's reading order
        if docling_doc:
            page_items = []
            for item, _level in docling_doc.iterate_items():
                if (isinstance(item, (TextItem, TableItem, PictureItem))) and item.prov:
                    if item.prov[0].page_no == (page_num + 1):
                        page_items.append(item)
            
            for dl_item in page_items:
                # Get item's bbox and label
                item_label = "text"
                
                # Check for group-level labels
                if hasattr(dl_item, 'parent') and dl_item.parent:
                    # In Docling v2, parent is often a RefItem
                    parent_ref = dl_item.parent
                    if hasattr(parent_ref, 'cref') and parent_ref.cref.startswith('#/groups/'):
                        try:
                            # Try to find the group object in the document
                            group_idx = int(parent_ref.cref.split('/')[-1])
                            if group_idx < len(docling_doc.groups):
                                group = docling_doc.groups[group_idx]
                                if hasattr(group, 'label') and hasattr(group.label, 'value'):
                                    item_label = group.label.value
                        except Exception:
                            pass

                # If no group label, use item label
                if item_label == "text" and hasattr(dl_item, 'label') and hasattr(dl_item.label, 'value'):
                    item_label = dl_item.label.value
                elif isinstance(dl_item, TableItem):
                    item_label = "table"
                elif isinstance(dl_item, PictureItem):
                    item_label = "picture"

                dl_bbox = dl_item.prov[0].bbox
                # Convert Docling bbox (bottom-left origin) to fitz.Rect (top-left origin)
                rect = fitz.Rect(
                    dl_bbox.l,
                    page_height - dl_bbox.t,
                    dl_bbox.r,
                    page_height - dl_bbox.b
                )
                
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
                    
                    # Relaxed matching: 40% intersection OR center point inside OR item contains block
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
                            
                            # Trim trailing whitespace
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

        # 3. Handle any missed blocks or fallback if Docling failed
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
                
                # Trim trailing whitespace
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

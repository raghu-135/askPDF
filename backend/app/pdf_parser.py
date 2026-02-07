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

# Tuneable via environment variables for accuracy vs speed
_pipeline_options.do_ocr = os.environ.get("DOCLING_DO_OCR", "False").lower() == "true"
_pipeline_options.do_table_structure = os.environ.get("DOCLING_DO_TABLE_STRUCTURE", "True").lower() == "true"
_pipeline_options.do_formula_enrichment = os.environ.get("DOCLING_DO_FORMULA_ENRICHMENT", "False").lower() == "true"

# Table structure mode (FAST or ACCURATE)
_table_mode = os.environ.get("DOCLING_TABLE_MODE", "FAST").upper()
if _table_mode == "ACCURATE":
    _pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
else:
    _pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# OCR options
if _pipeline_options.do_ocr:
    _pipeline_options.ocr_options.force_full_page_ocr = os.environ.get("DOCLING_FORCE_FULL_PAGE_OCR", "False").lower() == "true"

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
    Process a line from a text block and extract text and character coordinates.
    Args:
        line: Line dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        Tuple of (line_text, line_chars) where line_text is the string and line_chars is a list of character info dicts.
    """
    line_text = ""
    line_chars = []
    spans = sorted(line.get("spans", []), key=lambda s: s["bbox"][0])
    for span in spans:
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
                "page_width": page_width
            })
    return line_text, line_chars

def _process_block(block, page_num, page_height, page_width):
    """
    Process a text block and extract its text and character coordinates.
    Args:
        block: Block dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        Tuple of (block_text, block_chars) where block_text is the string and block_chars is a list of character info dicts.
    """
    block_text = ""
    block_chars = []
    for line in block.get("lines", []):
        line_text, line_chars = _process_line(line, page_num, page_height, page_width)
        if line_text:
            block_text += line_text
            block_chars.extend(line_chars)
            if not line_text.endswith((" ", "\n", "\t")):
                block_text += " "
                if line_chars:
                    last = line_chars[-1]
                    block_chars.append({
                        "page": last["page"],
                        "x": last["x"] + last["width"],
                        "y": last["y"],
                        "width": last["width"],
                        "height": last["height"],
                        "page_height": last["page_height"],
                        "page_width": last["page_width"],
                        "is_space": True
                    })
    return block_text, block_chars


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
        logger.info(f"Docling conversion successful: {docling_doc}")
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
                    combined_text = ""
                    combined_chars = []
                    for b in item_blocks:
                        b_text, b_chars = _process_block(b, page_num, page_height, page_width)
                        combined_text += b_text
                        combined_chars.extend(b_chars)
                    
                    if combined_text.strip():
                        # Trim trailing whitespace
                        while combined_text and combined_text[-1].isspace():
                            combined_text = combined_text[:-1]
                            if combined_chars:
                                combined_chars.pop()
                        
                        items.append({
                            "text": combined_text,
                            "label": item_label,
                            "char_map": combined_chars,
                            "page": page_num + 1,
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                        })

        # 3. Handle any missed blocks or fallback if Docling failed
        remaining_blocks = [b for i, b in enumerate(all_blocks) if i not in processed_indices]
        sorted_remaining = sorted(remaining_blocks, key=lambda b: (b["bbox"][1] // 10, b["bbox"][0]))
        
        for block in sorted_remaining:
            bbox = fitz.Rect(block["bbox"])
            if _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
                continue
            block_text, block_chars = _process_block(block, page_num, page_height, page_width)
            if block_text.strip():
                # Trim trailing whitespace
                while block_text and block_text[-1].isspace():
                    block_text = block_text[:-1]
                    if block_chars:
                        block_chars.pop()
                
                items.append({
                    "text": block_text,
                    "label": "text",
                    "char_map": block_chars,
                    "page": page_num + 1,
                    "bbox": list(bbox)
                })
            
    doc.close()
    return items

import io
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter, DocumentStream, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TextItem

# Initialize Docling converter as a module-level singleton
_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = False  # Faster, matches PyMuPDF baseline
_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

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

def _finalize_block(block_text, block_chars, full_text, char_map):
    """
    Finalize a processed block by trimming whitespace, updating text and char map, and adding newlines.
    Args:
        block_text: The text of the block.
        block_chars: List of character info dicts for the block.
        full_text: The full text accumulated so far.
        char_map: The char map accumulated so far.
    Returns:
        Updated (full_text, char_map) with the block appended and newlines added.
    """
    if block_text.strip():
        while block_text and block_text[-1].isspace():
            block_text = block_text[:-1]
            if block_chars:
                block_chars.pop()
        full_text += block_text
        char_map.extend(block_chars)
        separator = "\n\n"
        full_text += separator
        if block_chars:
            last = block_chars[-1]
            for _ in range(2):
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
    return full_text, char_map

def extract_text_with_coordinates(data: bytes, filename: str = "input.pdf"):
    """
    Extracts text and character coordinates from PDF bytes using a hybrid approach:
    1. Docling determines the logical structure and reading order (paragraphs, headings).
    2. PyMuPDF's `rawdict` provides exact character-level bounding boxes.
    
    Features:
    - Superior reading order for multi-column and complex layouts.
    - Exact character mapping for frontend highlighting.
    - Filters furniture (headers/footers) and tables/images.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    full_text = ""
    char_map = []
    
    # 1. Get Docling conversion for structure and reading order
    docling_doc = None
    try:
        # Wrap bytes in DocumentStream for Docling validation
        source = DocumentStream(name=filename, stream=io.BytesIO(data))
        docling_result = _docling_converter.convert(source)
        docling_doc = docling_result.document
    except Exception as e:
        import logging
        logging.warning(f"Docling conversion failed, falling back to basic PyMuPDF: {e}")

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
            # Extract items for this page in reading order
            page_items = []
            for item, _level in docling_doc.iterate_items():
                if isinstance(item, TextItem) and item.prov:
                    if item.prov[0].page_no == (page_num + 1):
                        page_items.append(item)
            
            for dl_item in page_items:
                dl_bbox = dl_item.prov[0].bbox
                # Convert Docling bbox (bottom-left origin) to fitz.Rect (top-left origin)
                rect = fitz.Rect(
                    dl_bbox.l,
                    page_height - dl_bbox.t,
                    dl_bbox.r,
                    page_height - dl_bbox.b
                )
                
                # Match PyMuPDF blocks that are substantially inside this Docling item
                item_blocks = []
                for i, b in enumerate(all_blocks):
                    if i in processed_indices:
                        continue
                    b_rect = fitz.Rect(b["bbox"])
                    
                    # Apply existing filtering logic to maintain same accuracy
                    if _is_block_filtered(b_rect, header_height, footer_y, table_bboxes, image_bboxes):
                        processed_indices.add(i)
                        continue
                        
                    intersection = rect.intersect(b_rect)
                    b_center = fitz.Point((b_rect.x0 + b_rect.x1)/2, (b_rect.y0 + b_rect.y1)/2)
                    
                    if intersection.get_area() / b_rect.get_area() > 0.6 or b_center in rect:
                        item_blocks.append(b)
                        processed_indices.add(i)
                
                # Process all matched blocks for this Docling item together
                if item_blocks:
                    # Sort blocks within the item just in case
                    item_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
                    combined_text = ""
                    combined_chars = []
                    for b in item_blocks:
                        b_text, b_chars = _process_block(b, page_num, page_height, page_width)
                        combined_text += b_text
                        combined_chars.extend(b_chars)
                    
                    full_text, char_map = _finalize_block(combined_text, combined_chars, full_text, char_map)

        # 3. Handle any missed blocks or fallback if Docling failed
        remaining_blocks = [b for i, b in enumerate(all_blocks) if i not in processed_indices]
        sorted_remaining = sorted(remaining_blocks, key=lambda b: (b["bbox"][1] // 10, b["bbox"][0]))
        
        for block in sorted_remaining:
            bbox = fitz.Rect(block["bbox"])
            if _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
                continue
            block_text, block_chars = _process_block(block, page_num, page_height, page_width)
            full_text, char_map = _finalize_block(block_text, block_chars, full_text, char_map)
            
    doc.close()
    return full_text, char_map

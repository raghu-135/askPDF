import io
import os
import logging
import json
import spacy
from docling.document_converter import DocumentConverter, DocumentStream, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TextItem

logger = logging.getLogger(__name__)

# Initialize Docling converter as a module-level singleton
_pipeline_options = PdfPipelineOptions()

def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() == "true"

def _get_env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

# Tuneable via environment variables for accuracy vs speed
_pipeline_options.do_ocr = _get_env_bool("DOCLING_DO_OCR", False)
_pipeline_options.do_table_structure = _get_env_bool("DOCLING_DO_TABLE_STRUCTURE", True)
_pipeline_options.do_formula_enrichment = _get_env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True)

# Table structure mode (FAST or ACCURATE)
_table_mode = _get_env_str("DOCLING_TABLE_MODE", "FAST").upper()
if _table_mode == "ACCURATE":
    _pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
else:
    _pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# OCR options
if _pipeline_options.do_ocr:
    _pipeline_options.ocr_options.force_full_page_ocr = _get_env_bool("DOCLING_FORCE_FULL_PAGE_OCR", False)

_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

# Initialize spaCy for sentence splitting
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spacy model: {e}")
    _nlp = None

def _convert_bottomleft_to_topleft(bbox, page_height):
    """Convert Docling BOTTOMLEFT bbox to pdfplumber TOPLEFT."""
    l, t, r, b = bbox
    return (l, page_height - t, r, page_height - b)

def _should_filter_item(item):
    """Filter items based on Docling labels and structure."""
    # Filter by label
    filtered_labels = {"page_header", "page_footer", "footnote", "caption"}
    if hasattr(item, 'label') and item.label.value in filtered_labels:
        return True
    
    # Filter by content_layer
    if hasattr(item, 'content_layer') and item.content_layer.value == "furniture":
        return True
    
    # Filter if parent is picture
    if hasattr(item, 'parent') and item.parent:
        if hasattr(item.parent, 'cref') and 'picture' in item.parent.cref.lower():
            return True
    
    return False

def parse_with_docling(data: bytes, filename: str, write_debug_output: bool = True):
    """
    Parse PDF with Docling to extract structural information and bounding boxes.
    
    Args:
        data: PDF file bytes
        filename: Name of the PDF file
        write_debug_output: Whether to write debug output to /app/tests
        
    Returns:
        Docling document object or None if parsing fails
    """
    if not filename:
        return None
    
    try:
        source = DocumentStream(name=filename, stream=io.BytesIO(data))
        docling_result = _docling_converter.convert(source)
        
        # Write debug output if requested
        if write_debug_output:
            test_dir = "/app/tests"
            os.makedirs(test_dir, exist_ok=True)
            test_file_json = os.path.join(test_dir, f"docling_raw_output_{filename}_dict.json")
            try:
                doc_dict = docling_result.document.export_to_dict()
                with open(test_file_json, "w") as f:
                    json.dump(doc_dict, f, indent=2)
                logger.info(f"Raw Docling output written to {test_file_json}")
            except Exception as e:
                logger.warning(f"Failed to write Docling output: {e}")
        
        return docling_result.document
    except Exception as e:
        logger.warning(f"Docling conversion failed: {e}")
        return None

def _extract_sentences_from_bbox(pdf_page, bbox, label, page_num, page_height, page_width):
    """Extract sentences from a single cropped region using pdfplumber."""
    # Crop to bbox
    cropped = pdf_page.crop(bbox)
    
    # Extract words with font info
    words = cropped.extract_words(
        extra_attrs=["fontname", "size"],
        keep_blank_chars=False,
        use_text_flow=True
    )
    
    # Concatenate text for sentence splitting
    full_text = " ".join(w["text"] for w in words)
    
    # Split into sentences using spaCy
    if not _nlp:
        return []
    
    doc = _nlp(full_text)
    sentences = []
    
    # Pre-calculate character positions for all words
    char_offset = 0
    for word in words:
        word["char_start"] = char_offset
        word["char_end"] = char_offset + len(word["text"])
        char_offset += len(word["text"]) + 1  # +1 for space
    
    # Map sentences back to word bboxes
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        # Find words that belong to this sentence based on char span
        sent_words = []
        for word in words:
            if sent.start_char <= word["char_end"] and sent.end_char >= word["char_start"]:
                sent_words.append(word)
        
        if sent_words:
            # Calculate sentence bbox from word bboxes
            x0 = min(w["x0"] for w in sent_words)
            y0 = min(w["top"] for w in sent_words)
            x1 = max(w["x1"] for w in sent_words)
            y1 = max(w["bottom"] for w in sent_words)
            
            # Extract font info (use first word's font)
            font_name = sent_words[0].get("fontname", "")
            font_size = sent_words[0].get("size", 0)
            
            sentences.append({
                "id": len(sentences),
                "text": sent_text,
                "label": label,
                "page": page_num + 1,
                "bbox": [x0, y0, x1, y1],
                "words": sent_words,
                "font": {"name": font_name, "size": font_size}
            })
    
    return sentences

def _extract_sentences_from_multi_bbox(pdf, item, label):
    """Extract sentences from text spanning multiple bboxes (columns/pages)."""
    all_words = []
    char_offset = 0
    
    # Process each prov entry in order
    for prov in item.prov:
        page_num = prov.page_no - 1  # Convert to 0-indexed
        pdf_page = pdf.pages[page_num]
        page_height = pdf_page.height
        page_width = pdf_page.width
        
        # Convert bbox to TOPLEFT using page-specific height
        bbox = _convert_bottomleft_to_topleft(
            (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b),
            page_height
        )
        
        # Crop and extract words
        cropped = pdf_page.crop(bbox)
        words = cropped.extract_words(
            extra_attrs=["fontname", "size"],
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        # Add page number to each word
        for word in words:
            word["page"] = page_num + 1
            word["char_start"] = char_offset
            word["char_end"] = char_offset + len(word["text"])
            char_offset += len(word["text"]) + 1  # +1 for space
        
        all_words.extend(words)
    
    # Get full text from item
    full_text = item.text
    
    # Split into sentences using spaCy
    if not _nlp:
        return []
    
    doc = _nlp(full_text)
    sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        # Find words that belong to this sentence based on charspan
        sent_words = []
        for word in all_words:
            if sent.start_char <= word["char_end"] and sent.end_char >= word["char_start"]:
                sent_words.append(word)
        
        if sent_words:
            # Calculate sentence bbox (may span multiple pages)
            x0 = min(w["x0"] for w in sent_words)
            y0 = min(w["top"] for w in sent_words)
            x1 = max(w["x1"] for w in sent_words)
            y1 = max(w["bottom"] for w in sent_words)
            
            # Get pages this sentence spans
            pages = sorted(set(w["page"] for w in sent_words))
            
            # Extract font info (use first word's font)
            font_name = sent_words[0].get("fontname", "")
            font_size = sent_words[0].get("size", 0)
            
            sentences.append({
                "id": len(sentences),
                "text": sent_text,
                "label": label,
                "pages": pages,  # List of pages sentence spans
                "bbox": [x0, y0, x1, y1],
                "words": sent_words,
                "font": {"name": font_name, "size": font_size}
            })
    
    return sentences

def parse_with_pdfplumber(data: bytes, docling_doc, filename: str, write_debug_output: bool = True):
    """
    Parse PDF with pdfplumber to extract word-level coordinates within Docling regions.
    
    Args:
        data: PDF file bytes
        docling_doc: Docling document object from parse_with_docling
        filename: Name of the PDF file
        write_debug_output: Whether to write debug output to /app/tests
        
    Returns:
        List of sentence dictionaries with bounding boxes and metadata
    """
    import pdfplumber
    from docling.datamodel.document import TextItem
    
    if not docling_doc:
        return []
    
    all_sentences = []
    
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # Get all text items from Docling
        text_items = []
        for item, _level in docling_doc.iterate_items():
            if isinstance(item, TextItem) and item.prov:
                text_items.append(item)
        
        # Process each Docling text item
        for dl_item in text_items:
            # Skip filtered items
            if _should_filter_item(dl_item):
                continue
            
            # Get label
            item_label = "text"
            if hasattr(dl_item, 'parent') and dl_item.parent:
                parent_ref = dl_item.parent
                if hasattr(parent_ref, 'cref') and parent_ref.cref.startswith('#/groups/'):
                    try:
                        group_idx = int(parent_ref.cref.split('/')[-1])
                        if group_idx < len(docling_doc.groups):
                            group = docling_doc.groups[group_idx]
                            if hasattr(group, 'label') and hasattr(group.label, 'value'):
                                item_label = group.label.value
                    except Exception:
                        pass
            
            if item_label == "text" and hasattr(dl_item, 'label') and hasattr(dl_item.label, 'value'):
                item_label = dl_item.label.value
            
            # Check if item has multiple prov entries (multi-bbox case)
            if len(dl_item.prov) > 1:
                # Handle multi-bbox (multi-column/page) case by processing each prov entry independently
                sentences = []
                for prov in dl_item.prov:
                    page_num = prov.page_no - 1  # Convert to 0-indexed
                    pdf_page = pdf.pages[page_num]
                    page_height = pdf_page.height
                    page_width = pdf_page.width
                    
                    # Convert bbox to TOPLEFT using page-specific height
                    pdf_bbox = _convert_bottomleft_to_topleft(
                        (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b),
                        page_height
                    )
                    
                    prov_sentences = _extract_sentences_from_bbox(
                        pdf_page, pdf_bbox, item_label, page_num, page_height, page_width
                    )
                    sentences.extend(prov_sentences)
            else:
                # Handle single bbox case
                prov = dl_item.prov[0]
                page_num = prov.page_no - 1  # Convert to 0-indexed
                pdf_page = pdf.pages[page_num]
                page_height = pdf_page.height
                page_width = pdf_page.width
                
                # Convert bbox to TOPLEFT using page-specific height
                pdf_bbox = _convert_bottomleft_to_topleft(
                    (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b),
                    page_height
                )
                
                sentences = _extract_sentences_from_bbox(
                    pdf_page, pdf_bbox, item_label, page_num, page_height, page_width
                )
            
            all_sentences.extend(sentences)
    
    # Write pdfplumber parsed output to test folder if requested
    if write_debug_output:
        test_dir = "/app/tests"
        os.makedirs(test_dir, exist_ok=True)
        test_file_pdfplumber = os.path.join(test_dir, f"pdfplumber_parsed_output_{filename}.json")
        try:
            with open(test_file_pdfplumber, "w") as f:
                json.dump(all_sentences, f, indent=2)
            logger.info(f"Pdfplumber parsed output written to {test_file_pdfplumber}")
        except Exception as e:
            logger.warning(f"Failed to write pdfplumber output: {e}")
    
    return all_sentences

def extract_text_with_coordinates(data: bytes, filename: str):
    """
    Extract sentences with bounding boxes using Docling + pdfplumber.
    Docling provides structural labels and paragraph bboxes (potentially multiple).
    pdfplumber provides word/character-level coordinates within those bboxes.
    
    This is the main orchestration function that combines Docling and pdfplumber parsing.
    """
    if not filename:
        return []
    
    # Parse with Docling
    docling_doc = parse_with_docling(data, filename, write_debug_output=True)
    
    if not docling_doc:
        return []
    
    # Parse with pdfplumber using Docling regions
    all_sentences = parse_with_pdfplumber(data, docling_doc, filename, write_debug_output=True)
    
    return all_sentences

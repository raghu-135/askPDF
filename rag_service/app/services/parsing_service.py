import io
import os
import logging
import json
import pdfplumber

logger = logging.getLogger(__name__)

# OpenDataLoader configuration via environment variables
OPEN_DATALOADER_FORMAT = os.environ.get("OPEN_DATALOADER_FORMAT", "json")

def _is_block_filtered(bbox, header_height, footer_y):
    """
    Determine if a text block should be filtered out (e.g., headers, footers).
    """
    if bbox[3] < header_height or bbox[1] > footer_y:
        return True
    return False

def _extract_word_coordinates(page, page_num, original_page_height, original_page_width):
    """
    Extract word-level coordinates from a pdfplumber page.
    Returns a list of word dictionaries with text and bounding box coordinates.
    """
    words = page.extract_words(extra_attrs=["fontname", "size"])
    cropped_page_width = page.width
    cropped_page_height = page.height
    
    word_boxes = []
    for word in words:
        word_text = word.get("text", "")
        if not word_text.strip():
            continue
            
        # pdfplumber coordinates: (x0, top, x1, bottom) in top-down format (y from top)
        x0, top, x1, bottom = word["x0"], word["top"], word["x1"], word["bottom"]
        
        # Convert to bottom-up coordinates (y from bottom) for frontend
        # Use ORIGINAL page dimensions for conversion, not cropped page dimensions
        # Frontend does page_height - bbox.y1, so it expects y1 to be distance from bottom
        y0 = original_page_height - bottom
        y1 = original_page_height - top
        
        word_box = {
            "word": word_text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "page": page_num + 1,
            "page_width": original_page_width,
            "page_height": original_page_height,
            "font": word.get("fontname", ""),
            "size": word.get("size", 0)
        }
        
        word_boxes.append(word_box)
    
    return word_boxes

def _crop_to_bbox(page, bbox):
    """
    Crop a pdfplumber page to the specified bounding box.
    bbox: [x0, y0, x1, y1] in bottom-up coordinates (y from bottom)
    """
    page_height = page.height
    # Convert from bottom-up to pdfplumber's top-down coordinates
    x0, y0, x1, y1 = bbox
    top = page_height - y1  # Convert top edge (from bottom) to distance from top
    bottom = page_height - y0  # Convert bottom edge (from bottom) to distance from top
    
    return page.crop((x0, top, x1, bottom))


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
    Use OpenDataLoader v2.0 to extract structural elements with bounding boxes.
    Returns a flat list of elements with type, content, page number, and bounding box.
    
    Args:
        data: PDF file data as bytes
    """
    try:
        import opendataloader_pdf
        
        # Create a temporary file for OpenDataLoader
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name
        
        # Use persistent output directory for the JSON result (mounted volume)
        output_dir = "/data/opendataloader_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique subdirectory for this run
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmp_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(tmp_dir, exist_ok=True)
        
        try:
            # Call OpenDataLoader convert function
            logger.info(f"Calling OpenDataLoader v2.0 convert on {tmp_path}")
            opendataloader_pdf.convert(
                tmp_path,
                output_dir=tmp_dir,
                format=OPEN_DATALOADER_FORMAT,
                use_struct_tree=True
            )
            
            # Find the JSON file in the output directory
            json_files = [f for f in os.listdir(tmp_dir) if f.endswith('.json')]
            if not json_files:
                logger.warning(f"No JSON file found in output directory {tmp_dir}")
                return []
            
            json_path = os.path.join(tmp_dir, json_files[0])
            logger.info(f"Reading OpenDataLoader result from {json_path}")
            logger.info(f"Raw output will be preserved at: {json_path}")
            
            # Read the JSON result
            with open(json_path, 'r') as f:
                result = json.load(f)
            
            logger.info(f"OpenDataLoader raw output: {json.dumps(result, indent=2)}")
            logger.info(f"OpenDataLoader result type: {type(result)}")
            logger.info(f"OpenDataLoader result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            if not result:
                logger.warning("OpenDataLoader returned no result or empty result")
                return []
            
            # Flatten the hierarchical structure
            flat_elements = []
            _flatten_odl_elements(result, flat_elements)
            
            logger.info(f"Flattened {len(flat_elements)} elements from OpenDataLoader")
            if flat_elements:
                logger.info(f"Sample element: {flat_elements[0]}")
            
            return flat_elements
            
        finally:
            # Clean up only the input PDF temp file, preserve the output directory
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            # Note: We do NOT delete tmp_dir anymore - it's preserved for inspection
    except Exception as e:
        logger.error(f"Error loading OpenDataLoader elements: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _map_odl_type_to_label(element: dict) -> str:
    """
    Map OpenDataLoader element type to our label system.
    Returns None for types that should be ignored (tables, images, charts).
    """
    element_type = element.get('type', 'text').lower()
    
    # Types to ignore completely
    ignored_types = {'table', 'figure', 'picture', 'image', 'chart'}
    if element_type in ignored_types:
        return None
    
    type_mapping = {
        'heading': 'heading',
        'title': 'heading',
        'sectionheader': 'heading',
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
    combined with pdfplumber for word-level coordinates.
    """
    if not filename:
        return []

    # Create a temporary file for pdfplumber
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name
    
    try:
        with pdfplumber.open(tmp_path) as pdf:
            items = []
            
            # Load structural elements from OpenDataLoader v2.0
            logger.info("Loading OpenDataLoader v2.0 elements")
            odl_elements = _load_opendataloader_elements(data)
            logger.info(f"Loaded {len(odl_elements)} odl_elements from OpenDataLoader v2.0")

            for page_num, page in enumerate(pdf.pages):
                page_height = page.height
                page_width = page.width
                header_height = page_height * 0.05
                footer_y = page_height * 0.95
                
                processed_odl_indices = set()
                processed_odl_bboxes = []

                if odl_elements:
                    # Filter elements for current page (OpenDataLoader uses 1-based page numbers)
                    page_elements = [
                        elem for elem in odl_elements 
                        if elem.get('page number', 0) == (page_num + 1)
                    ]
                    
                    # Track which regions have been processed
                    processed_odl_bboxes = []
                    
                    for elem in page_elements:
                        item_label = _map_odl_type_to_label(elem)
                        
                        # Skip ignored types (tables, images, charts)
                        if item_label is None:
                            continue
                        
                        # Get bounding box from OpenDataLoader [x1, y1, x2, y2]
                        bbox = elem.get('bounding box', [0, 0, 0, 0])
                        if len(bbox) >= 4:
                            # OpenDataLoader bbox is [left, bottom, right, top] in PDF coordinates
                            # where (0,0) is bottom-left (same as bottom-up coordinates)
                            # Direct mapping to bottom-up coordinates for frontend
                            rect_bbox = [
                                bbox[0],  # left -> x0
                                bbox[1],  # bottom -> y0
                                bbox[2],  # right -> x1
                                bbox[3]   # top -> y1
                            ]
                        else:
                            continue
                        
                        # Filter out headers/footers
                        if _is_block_filtered(rect_bbox, header_height, footer_y):
                            continue
                        
                        # Crop page to the ODL bounding box and extract word coordinates
                        try:
                            cropped_page = _crop_to_bbox(page, rect_bbox)
                            word_boxes = _extract_word_coordinates(cropped_page, page_num, page_height, page_width)
                            
                            if word_boxes:
                                # Combine words into text
                                text = " ".join([w["word"] for w in word_boxes])
                                if text.strip():
                                    items.append({
                                        "text": text,
                                        "label": item_label,
                                        "word_boxes": word_boxes,
                                        "page": page_num + 1,
                                        "bbox": rect_bbox
                                    })
                            processed_odl_bboxes.append(rect_bbox)
                        except Exception as e:
                            logger.warning(f"Failed to extract words from ODL region: {e}")
                            continue

                # Extract remaining text from page not covered by ODL
                # Extract all words and filter out those in ODL regions
                try:
                    all_word_boxes = _extract_word_coordinates(page, page_num, page_height, page_width)
                    
                    # Filter out words that are in processed ODL regions
                    remaining_words = []
                    for word_box in all_word_boxes:
                        word_center_x = (word_box["x0"] + word_box["x1"]) / 2
                        word_center_y = (word_box["y0"] + word_box["y1"]) / 2
                        
                        in_odl_region = False
                        for odl_rect in processed_odl_bboxes:
                            if (odl_rect[0] <= word_center_x <= odl_rect[2] and
                                odl_rect[1] <= word_center_y <= odl_rect[3]):
                                in_odl_region = True
                                break
                        
                        if not in_odl_region:
                            remaining_words.append(word_box)
                    
                    # Sort remaining words by reading order (top to bottom, left to right)
                    remaining_words.sort(key=lambda w: (w["y0"], w["x0"]))
                    
                    # Group remaining words into text blocks by proximity
                    if remaining_words:
                        # Simple grouping: all remaining words as one block
                        # This can be improved later for better segmentation
                        text = " ".join([w["word"] for w in remaining_words])
                        if text.strip():
                            # Calculate overall bbox
                            min_x = min(w["x0"] for w in remaining_words)
                            min_y = min(w["y0"] for w in remaining_words)
                            max_x = max(w["x1"] for w in remaining_words)
                            max_y = max(w["y1"] for w in remaining_words)
                            
                            items.append({
                                "text": text,
                                "label": "text",
                                "word_boxes": remaining_words,
                                "page": page_num + 1,
                                "bbox": [min_x, min_y, max_x, max_y]
                            })
                except Exception as e:
                    logger.warning(f"Failed to extract remaining words from page {page_num}: {e}")
            
            return items
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

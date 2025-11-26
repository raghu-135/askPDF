from io import BytesIO
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextBox, LTTextLine

def extract_text_with_coordinates(data: bytes):
    """
    Extracts text and character coordinates from PDF bytes.
    Returns:
        full_text (str): The complete text of the PDF.
        char_map (list): A list of dicts, where each dict corresponds to a character
                         in full_text and contains {page, x, y, width, height}.
    """
    buf = BytesIO(data)
    full_text = ""
    char_map = []

    for page_layout in extract_pages(buf):
        page_num = page_layout.pageid
        
        for element in page_layout:
            # Handle TextBoxes (which contain TextLines)
            if isinstance(element, LTTextBox):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        for char in text_line:
                            if isinstance(char, LTChar):
                                full_text += char.get_text()
                                char_map.append({
                                    "page": page_num,
                                    "x": char.x0,
                                    "y": char.y0,
                                    "width": char.width,
                                    "height": char.height,
                                    "page_height": page_layout.height,
                                    "page_width": page_layout.width
                                })
            # Handle TextLines directly if they appear at top level
            elif isinstance(element, LTTextLine):
                for char in element:
                    if isinstance(char, LTChar):
                        full_text += char.get_text()
                        char_map.append({
                            "page": page_num,
                            "x": char.x0,
                            "y": char.y0,
                            "width": char.width,
                            "height": char.height,
                            "page_height": page_layout.height,
                            "page_width": page_layout.width
                        })

    return full_text, char_map

def pdf_bytes_to_text(data: bytes) -> str:
    # Simple, reliable extraction
    buf = BytesIO(data)
    text = extract_text(buf)
    return text.strip()

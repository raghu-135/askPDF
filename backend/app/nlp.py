
import spacy
import re

_nlp = spacy.load("en_core_web_sm")

# URL and Email regex for protection - more tolerant of common PDF artifacts
# Matches typical URLs/Emails and also fragments that might be separated by single spaces
URL_PATTERN = re.compile(r'(?:https?://|www\.)\S+|(?:[\w\.-]+)\s*@\s*(?:[\w\.-]+)\.\w+')

def split_into_sentences(items: list[dict]):
    """
    Splits structured items into sentences, joining consecutive paragraphs if they
    seem to be part of the same sentence.
    
    Args:
        items: List of dicts with 'text', 'label', 'char_map', and 'font'.
    
    Returns:
        list[dict]: Sentences with 'id', 'text', and 'bboxes' (char_map).
    """
    if not items:
        return []

    # 1. Join consecutive items if they flow together
    processed_items = []
    # Labels that are eligible for joining if they "flow"
    text_labels = {"paragraph", "text", "list_item", "note", "caption"}
    
    # Labels that strictly block any joining with neighbors
    blocked_labels = {"key_value_area", "key_value", "table", "page_header", "page_footer", "heading", "title", "section_header"}
    
    # Characters that suggest a URL or Email - don't add space between items if these are present
    NO_SPACE_CHARS = ("-", "/", "@", ".", "_", ":", "(", "[")

    for item in items:
        label = item.get("label", "text")
        text = item.get("text", "").strip()
        bbox = item.get("bbox") # [x0, y0, x1, y1]
        page = item.get("page", 0)
        font = item.get("font", {"name": "", "size": 0})
        
        should_join = False
        if (processed_items and 
            processed_items[-1]["label"] in text_labels and 
            label in text_labels and
            processed_items[-1]["label"] not in blocked_labels and
            label not in blocked_labels):
            
            prev_item = processed_items[-1]
            prev_text = prev_item["text"].strip()
            prev_bbox = prev_item.get("bbox")
            prev_page = prev_item.get("page", 0)
            prev_font = prev_item.get("font", {"name": "", "size": 0})
            
            # SPATIAL HEURISTICS
            spatial_safe = True
            if bbox and prev_bbox:
                if page == prev_page:
                    # Same page: Check gaps
                    v_gap = bbox[1] - prev_bbox[3]
                    # Line height approximation
                    line_height = prev_bbox[3] - prev_bbox[1]
                    
                    if v_gap < -2: # Small overlap or same line
                        spatial_safe = True
                    elif v_gap > line_height * 1.3: # Slightly more relaxed
                        spatial_safe = False
                    else:
                        h_overlap = min(bbox[2], prev_bbox[2]) - max(bbox[0], prev_bbox[0])
                        h_width = prev_bbox[2] - prev_bbox[0]
                        if h_overlap < h_width * 0.35:
                            spatial_safe = False
                else:
                    # Different page
                    spatial_safe = True

            # FONT HEURISTICS
            font_safe = True
            if prev_font and font:
                size_diff = abs(prev_font.get("size", 0) - font.get("size", 0))
                # Relaxed font size check (some PDFs vary slightly across pages/lines)
                if size_diff > 2.0:
                    font_safe = False

            if spatial_safe and font_safe:
                # TEXTUAL HEURISTICS:
                no_punc = not prev_text.endswith((".", "!", "?"))
                starts_lower = text and text[0].islower()
                ends_hyphen = prev_text.endswith("-")
                
                starts_connective = False
                if text:
                    first_word = text.split()[0].lower()
                    starts_connective = first_word in {"and", "or", "but", "nor", "yet", "so", "with", "from", "for", "to", "is"}

                if (no_punc and (starts_lower or starts_connective)) or ends_hyphen:
                    should_join = True

        if should_join:
            prev = processed_items[-1]
            
            # Check if we need to add a space
            prev_trimmed = prev["text"].strip()
            item_start = item["text"].lstrip()
            needs_space = not (prev_trimmed.endswith(NO_SPACE_CHARS) or 
                              item_start.startswith(NO_SPACE_CHARS) or 
                              item_start.startswith((" ",)))
            
            if needs_space:
                prev["text"] += " "
                if prev["char_map"]:
                    last_char = prev["char_map"][-1]
                    space_char = last_char.copy()
                    space_char.update({"is_space": True, "width": 0, "c": " "})
                    prev["char_map"].append(space_char)
            
            prev["text"] += item["text"]
            prev["char_map"].extend(item["char_map"])
            
            # Update combined bbox
            if bbox and prev.get("bbox"):
                prev["bbox"] = [
                    min(prev["bbox"][0], bbox[0]),
                    min(prev["bbox"][1], bbox[1]),
                    max(prev["bbox"][2], bbox[2]),
                    max(prev["bbox"][3], bbox[3])
                ]
        else:
            processed_items.append(item.copy())

    # 2. Split into sentences and map character coordinates
    sentences = []
    global_id = 0
    
    for item in processed_items:
        text = item["text"]
        char_map = item["char_map"]
        
        # URL/Email Protection
        protected_spans = []
        for match in URL_PATTERN.finditer(text):
            protected_spans.append((match.start(), match.end()))
            
        doc = _nlp(text)
        
        # Merge protected spans
        with doc.retokenize() as retokenizer:
            for start, end in protected_spans:
                span = doc.char_span(start, end, alignment_mode="expand")
                if span:
                    retokenizer.merge(span)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            start_idx = sent.start_char
            end_idx = sent.end_char
            
            # Map indices to char_map
            # Note: The char_map should have the same length as the text 
            # if we account for spaces correctly during extraction.
            # In our extract_text_with_coordinates, we trim whitespace but 
            # preserve internal spaces.
            
            sent_bboxes = char_map[start_idx:end_idx]
            
            sentences.append({
                "id": global_id,
                "text": sent_text,
                "bboxes": sent_bboxes
            })
            global_id += 1
            
    return sentences

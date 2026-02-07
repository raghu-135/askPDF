
import spacy

_nlp = spacy.load("en_core_web_sm")

def split_into_sentences(items: list[dict]):
    """
    Splits structured items into sentences, joining consecutive paragraphs if they
    seem to be part of the same sentence.
    
    Args:
        items: List of dicts with 'text', 'label', and 'char_map'.
    
    Returns:
        list[dict]: Sentences with 'id', 'text', and 'bboxes' (char_map).
    """
    if not items:
        return []

    # 1. Join consecutive paragraphs if they flow together
    processed_items = []
    # Labels that are eligible for joining if they "flow"
    text_labels = {"paragraph", "text", "list_item", "note", "caption"}
    
    # Labels that strictly block any joining with neighbors (receipt specific)
    blocked_labels = {"key_value_area", "key_value", "table", "page_header", "page_footer"}
    
    for item in items:
        label = item.get("label", "text")
        text = item.get("text", "").strip()
        bbox = item.get("bbox") # [x0, y0, x1, y1]
        page = item.get("page", 0)
        
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
            
            # SPATIAL HEURISTICS
            spatial_safe = True
            if bbox and prev_bbox:
                if page == prev_page:
                    # Same page: Check gaps
                    v_gap = bbox[1] - prev_bbox[3]
                    # Line height approximation
                    line_height = prev_bbox[3] - prev_bbox[1]
                    
                    if v_gap < 0:
                        # Potential column wrap (bottom-left to top-right)
                        # In reading order, this is usually a continuation
                        spatial_safe = True
                    elif v_gap > line_height * 1.5:
                        # Significant vertical gap. Likely new section.
                        spatial_safe = False
                    else:
                        # Normal vertical gap: check horizontal alignment
                        # If overlap is very low, they are side-by-side fields
                        h_overlap = min(bbox[2], prev_bbox[2]) - max(bbox[0], prev_bbox[0])
                        if h_overlap < (prev_bbox[2] - prev_bbox[0]) * 0.3:
                            spatial_safe = False
                else:
                    # Different page: Trust Docling's reading order/iterate_items()
                    # Cross-page joining is usually safe if labels match
                    spatial_safe = True

            if spatial_safe:
                # TEXTUAL HEURISTICS:
                # 1. If previous doesn't end in sentence-ending punctuation
                no_punc = not prev_text.endswith((".", "!", "?", ":", ";"))
                
                # 2. If current starts with a lowercase letter (strong signal of continuation)
                starts_lower = text and text[0].islower()
                
                # 3. If previous ends with a hyphen (word break)
                ends_hyphen = prev_text.endswith("-")
                
                # 4. If current starts with a common connective
                starts_connective = False
                if text:
                    first_word = text.split()[0].lower()
                    starts_connective = first_word in {"and", "or", "but", "nor", "yet", "so"}

                # Join if (no punctuation AND (starts lower OR starts connective)) OR ends hyphen
                if (no_punc and (starts_lower or starts_connective)) or ends_hyphen:
                    should_join = True

        if should_join:
            prev = processed_items[-1]
            
            # Check if we need to add a space between items
            # If it ends in hyphen, we might NOT want a space (word break)
            needs_space = not (prev["text"].endswith(("-", " ")) or item["text"].startswith(" "))
            
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
            
            # Preserve the most descriptive label
            if prev["label"] == "text" and label != "text":
                prev["label"] = label
        else:
            processed_items.append(item.copy())

    # 2. Split into sentences and map character coordinates
    sentences = []
    global_id = 0
    
    for item in processed_items:
        text = item["text"]
        char_map = item["char_map"]
        
        doc = _nlp(text)
        
        # We need to track the character position in the original text to match with char_map
        # spaCy tokens have idx (start char offset)
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

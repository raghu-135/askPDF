
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
    # Broaden labels that can be joined
    #text_labels = {"paragraph", "text", "list_item", "note", "caption"}
    text_labels = {"paragraph", "text"}
    
    for item in items:
        label = item.get("label", "text")
        if (processed_items and 
            processed_items[-1]["label"] in text_labels and 
            label in text_labels and
            not processed_items[-1]["text"].strip().endswith((".", "!", "?", ":", ";"))):
            
            prev = processed_items[-1]
            
            # Check if we need to add a space between items
            # items from extract_text_with_coordinates are already trimmed of trailing whitespace
            needs_space = not (prev["text"].endswith(" ") or item["text"].startswith(" "))
            
            if needs_space:
                prev["text"] += " "
                # Add a space char to char_map
                if prev["char_map"]:
                    last_char = prev["char_map"][-1]
                    space_char = last_char.copy()
                    space_char.update({
                        "is_space": True,
                        "width": 0,
                        "c": " "
                    })
                    prev["char_map"].append(space_char)
            
            prev["text"] += item["text"]
            prev["char_map"].extend(item["char_map"])
            
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

import spacy
import re
import logging

logger = logging.getLogger(__name__)

# Initialize Spacy model as a singleton
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spacy model: {e}")
    # Fallback or initialization handled by Dockerfile download
    _nlp = None

# URL and Email regex for protection
URL_PATTERN = re.compile(r'(?:https?://|www\.)\S+|(?:[\w\.-]+)\s*@\s*(?:[\w\.-]+)\.\w+')

def split_into_sentences(items: list[dict]):
    """
    Pass-through function for backward compatibility.
    Sentence splitting is now handled in parsing_service.py.
    This function returns the input items unchanged.
    """
    return items

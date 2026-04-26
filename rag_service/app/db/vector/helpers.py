"""
vector/helpers.py - Helper functions for validation and metadata handling.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _validate_not_empty(value: Any, field_name: str) -> None:
    """Validate that a value is not empty.
    
    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        
    Raises:
        ValueError: If the value is empty.
    """
    if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
        raise ValueError(f"{field_name} cannot be empty")


def _validate_embeddings_match_texts(texts: List[str], embeddings: List[List[float]]) -> None:
    """Validate that embeddings list matches texts list length.
    
    Args:
        texts: List of text strings.
        embeddings: List of embedding vectors.
        
    Raises:
        ValueError: If lengths don't match or embeddings are invalid.
    """
    if len(texts) != len(embeddings):
        raise ValueError(f"Texts count ({len(texts)}) must match embeddings count ({len(embeddings)})")
    for i, emb in enumerate(embeddings):
        if not emb or len(emb) == 0:
            raise ValueError(f"Embedding at index {i} is empty")


def _metadata_json(metadata: Optional[Dict[str, Any]]) -> str:
    """Serialize metadata dict to JSON; return `{}` on empty/invalid input.
    
    Args:
        metadata: Optional metadata dictionary.
        
    Returns:
        str: JSON string representation.
    """
    if not metadata:
        return "{}"
    try:
        return json.dumps(metadata)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize metadata: {e}")
        return "{}"


def _parse_metadata(raw: Optional[str]) -> Dict[str, Any]:
    """Parse metadata JSON string into a dict; return empty dict on failure.
    
    Args:
        raw: JSON string to parse.
        
    Returns:
        Dict[str, Any]: Parsed metadata dictionary.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse metadata JSON: {e}")
        return {}


def _score(obj: Any) -> float:
    """Extract retrieval score (or distance fallback) from Weaviate result metadata.
    
    Args:
        obj: Weaviate result object.
        
    Returns:
        float: Retrieval score or 0.0 if unavailable.
    """
    meta = getattr(obj, "metadata", None)
    if not meta:
        return 0.0
    score = getattr(meta, "score", None)
    if score is None:
        score = getattr(meta, "distance", None)
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to convert score to float: {e}")
        return 0.0

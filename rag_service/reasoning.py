"""
Utilities for extracting provider-specific reasoning/thinking traces from AI responses.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _text_from_content_item(item: Any) -> str:
    """Best-effort extraction of visible text from OpenAI-compatible content blocks."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""

    if item.get("type") == "text":
        text = item.get("text")
        if isinstance(text, str):
            return text

    # Responses-style nested output text blocks
    nested = item.get("content")
    if isinstance(nested, list):
        parts = [_text_from_content_item(x) for x in nested]
        return "\n".join([p for p in parts if p]).strip()

    return ""


def _reasoning_from_content_item(item: Any) -> str:
    """Extract reasoning text from structured content blocks."""
    if not isinstance(item, dict):
        return ""

    block_type = str(item.get("type", "")).lower()
    if "reason" in block_type or "think" in block_type:
        # Common direct fields
        direct_fields = [
            item.get("text"),
            item.get("content"),
            item.get("reasoning"),
            item.get("reasoning_content"),
            item.get("thinking"),
            item.get("summary"),
            item.get("output_text"),
        ]
        for value in direct_fields:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                flattened = []
                for v in value:
                    if isinstance(v, str):
                        flattened.append(v)
                    elif isinstance(v, dict):
                        t = v.get("text")
                        if isinstance(t, str):
                            flattened.append(t)
                joined = "\n".join([x for x in flattened if x]).strip()
                if joined:
                    return joined

    return ""


def _collect_structured_reasoning(message: Any) -> str:
    """Find reasoning content in common metadata and structured content locations."""
    additional_kwargs = getattr(message, "additional_kwargs", None)
    response_metadata = getattr(message, "response_metadata", None)
    metadata_sources = [
        additional_kwargs if isinstance(additional_kwargs, dict) else {},
        response_metadata if isinstance(response_metadata, dict) else {},
    ]

    keys = [
        "thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "reasoning_summary",
        "thoughts",
    ]

    for source in metadata_sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                joined = "\n".join(str(v) for v in value if v).strip()
                if joined:
                    return joined
            if isinstance(value, dict):
                try:
                    dumped = json.dumps(value, ensure_ascii=True)
                    if dumped and dumped != "{}":
                        return dumped
                except Exception:
                    continue

    if isinstance(message.content, list):
        for item in message.content:
            extracted = _reasoning_from_content_item(item)
            if extracted:
                return extracted

    return ""


def _extract_from_think_tags(text: str) -> Tuple[str, str]:
    """Extract and strip <think>...</think> sections from plain text output."""
    if not text:
        return "", ""

    matches = _THINK_TAG_PATTERN.findall(text)
    if not matches:
        return "", text.strip()

    reasoning = "\n\n".join(match.strip() for match in matches if match.strip()).strip()
    answer = _THINK_TAG_PATTERN.sub("", text).strip()
    return reasoning, answer


def normalize_ai_response(message: Optional[Any]) -> Dict[str, Any]:
    """
    Normalize an AIMessage into answer + optional reasoning trace fields.
    """
    if message is None:
        return {
            "answer": "",
            "reasoning": "",
            "reasoning_available": False,
            "reasoning_format": "none",
        }

    content = getattr(message, "content", "")
    if isinstance(content, str):
        raw_answer = content
    elif isinstance(content, list):
        parts = [_text_from_content_item(item) for item in content]
        raw_answer = "\n".join([p for p in parts if p]).strip()
    else:
        raw_answer = str(content or "")

    structured_reasoning = _collect_structured_reasoning(message)
    if structured_reasoning:
        return {
            "answer": raw_answer.strip(),
            "reasoning": structured_reasoning,
            "reasoning_available": True,
            "reasoning_format": "structured",
        }

    tag_reasoning, cleaned_answer = _extract_from_think_tags(raw_answer)
    
    # Strip LLM-mimicked "Q: ... A: ..." prefixes if present
    if cleaned_answer.startswith("Q:") and "\nA:" in cleaned_answer:
        parts = cleaned_answer.split("\nA:", 1)
        if len(parts) == 2:
            cleaned_answer = parts[1].strip()
    elif cleaned_answer.startswith("A:"):
        cleaned_answer = cleaned_answer[2:].strip()

    if tag_reasoning:
        return {
            "answer": cleaned_answer,
            "reasoning": tag_reasoning,
            "reasoning_available": True,
            "reasoning_format": "tagged_text",
        }

    return {
        "answer": cleaned_answer,
        "reasoning": "",
        "reasoning_available": False,
        "reasoning_format": "none",
    }

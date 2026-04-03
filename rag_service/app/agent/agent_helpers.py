import json
import logging
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), MessagesPlaceholder("messages")]
    )


import re

def parse_intent_response(raw: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    content = (raw or "").strip()
    
    # helper to extract first occurrence of an XML tag
    def extract_tag(tag: str, text: str) -> Optional[str]:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    # extract fields
    route = extract_tag("route", content)
    rewritten_query = extract_tag("rewritten_query", content)
    reference_type = extract_tag("reference_type", content)
    context_coverage = extract_tag("context_coverage", content)
    
    # extract clarification options
    clarification_options = None
    clarify_block = extract_tag("clarification_options", content)
    if clarify_block:
        options = re.findall(r"<option>(.*?)</option>", clarify_block, re.IGNORECASE | re.DOTALL)
        if options:
            cleaned = [opt.strip() for opt in options if opt.strip()]
            meta_prefixes = (
                "are you asking about",
                "are you asking for",
                "are you referring to",
                "did you mean",
                "do you mean",
                "is your question about",
                "is this about",
                "are you looking for",
                "would you like",
                "do you want",
            )
            if any(opt.lower().startswith(meta_prefixes) for opt in cleaned):
                logger.warning("Intent Agent clarification options looked like meta-questions; rejecting to force retry.")
                clarification_options = None
            else:
                clarification_options = cleaned

    # Defaults or validation parsing
    if not route:
        route = "ANSWER"
    else:
        route = route.upper()
        
    if not rewritten_query:
        # If it failed to emit an XML rewritten query, we can't reliably proceed as an intent rewrite
        return None

    if not reference_type:
        reference_type = "NONE"
    else:
        reference_type = reference_type.upper()

    if not context_coverage:
        context_coverage = "PARTIAL"
    else:
        context_coverage = context_coverage.upper()

    # validations
    if route not in {"ANSWER", "CLARIFY"}:
        route = "ANSWER"
    if reference_type not in {"NONE", "SEMANTIC", "TEMPORAL", "ENTITY"}:
        reference_type = "NONE"
    if context_coverage not in {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}:
        context_coverage = "PARTIAL"

    if route == "CLARIFY":
        if not clarification_options or len(clarification_options) < 2:
            return None
    else:
        clarification_options = None

    return {
        "route": route,
        "rewritten_query": rewritten_query,
        "reference_type": reference_type,
        "context_coverage": context_coverage,
        "clarification_options": clarification_options,
    }


def evidence_insufficient(state: Dict[str, Any]) -> bool:
    document_sources = state.get("document_sources") or []
    web_sources = state.get("web_sources") or []
    used_chat_ids = state.get("used_chat_ids") or []
    return not document_sources and not web_sources and not used_chat_ids


def collect_tool_sources(
    content: str,
    document_sources: list,
    web_sources: list,
    used_chat_ids: list,
) -> None:
    if not isinstance(content, str) or not content.startswith("{"):
        return
    try:
        data = json.loads(content)
    except Exception:
        return
    if "__document_sources__" in data:
        document_sources.extend(data["__document_sources__"])
    if "__web_sources__" in data:
        web_sources.extend(data["__web_sources__"])
    if "__used_chat_ids__" in data:
        used_chat_ids.extend(data["__used_chat_ids__"])

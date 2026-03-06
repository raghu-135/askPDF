import json
import logging
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), MessagesPlaceholder("messages")]
    )


def parse_intent_response(raw: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    content = (raw or "").strip()
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse intent JSON: {e}")
        return None

    if not isinstance(data, dict):
        return None

    required = {"status", "rewritten_query", "reference_type", "context_coverage", "clarification_options"}
    if not required.issubset(set(data.keys())):
        return None

    status_ok = data["status"] in {"CLEAR_STANDALONE", "CLEAR_FOLLOWUP", "AMBIGUOUS"}
    ref_ok = data["reference_type"] in {"NONE", "SEMANTIC", "TEMPORAL", "ENTITY"}
    cov_ok = data["context_coverage"] in {"SUFFICIENT", "PROBABLY_SUFFICIENT", "INSUFFICIENT"}
    if not (status_ok and ref_ok and cov_ok):
        return None

    clar = data.get("clarification_options")
    if clar is not None and not isinstance(clar, list):
        data["clarification_options"] = None

    if not isinstance(data.get("rewritten_query"), str) or not data["rewritten_query"].strip():
        return None

    return data


def looks_like_followup(question: str) -> bool:
    q = (question or "").strip().lower()
    return len(q.split()) <= 6 and any(q.startswith(p) for p in ("what about", "and", "also", "that", "it", "this"))


def evidence_insufficient(state: Dict[str, Any]) -> bool:
    pdf_sources = state.get("pdf_sources") or []
    web_sources = state.get("web_sources") or []
    used_chat_ids = state.get("used_chat_ids") or []
    return not pdf_sources and not web_sources and not used_chat_ids


def collect_tool_sources(
    content: str,
    pdf_sources: list,
    web_sources: list,
    used_chat_ids: list,
) -> None:
    if not isinstance(content, str) or not content.startswith("{"):
        return
    try:
        data = json.loads(content)
    except Exception:
        return
    if "__pdf_sources__" in data:
        pdf_sources.extend(data["__pdf_sources__"])
    if "__web_sources__" in data:
        web_sources.extend(data["__web_sources__"])
    if "__used_chat_ids__" in data:
        used_chat_ids.extend(data["__used_chat_ids__"])

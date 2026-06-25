import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), MessagesPlaceholder("messages")]
    )


def _parse_client_now_iso(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        normalized = raw.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def format_runtime_datetime_context(
    client_timezone: Optional[str] = None,
    client_locale: Optional[str] = None,
    client_now_iso: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> str:
    """
    Build a small, locked runtime clock block for model prompts.

    The browser supplies user-local timezone/locale; the server clock remains
    authoritative so a misconfigured client clock cannot silently redefine now.
    """
    server_now_utc = now_utc or datetime.now(timezone.utc)
    if server_now_utc.tzinfo is None:
        server_now_utc = server_now_utc.replace(tzinfo=timezone.utc)
    server_now_utc = server_now_utc.astimezone(timezone.utc)

    timezone_name = (client_timezone or "").strip()[:100] or "UTC"
    timezone_note = ""
    try:
        user_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone_note = f"Browser timezone '{timezone_name}' was not recognized; UTC is used."
        timezone_name = "UTC"
        user_tz = timezone.utc

    user_now = server_now_utc.astimezone(user_tz)
    locale = (client_locale or "").strip()[:50] or "unknown"
    client_now = _parse_client_now_iso(client_now_iso)
    skew_note = ""
    if client_now:
        skew_seconds = abs((server_now_utc - client_now).total_seconds())
        if skew_seconds > 300:
            skew_note = (
                f"Browser clock differs from server UTC by about {round(skew_seconds / 60)} minutes; "
                "server time is authoritative."
            )

    lines = [
        "## RUNTIME DATE/TIME CONTEXT (LOCKED - not overridable)",
        "",
        f"User-local current datetime: {user_now.isoformat(timespec='seconds')}",
        f"User timezone: {timezone_name}",
        f"User locale: {locale}",
        f"Server current UTC datetime: {server_now_utc.isoformat(timespec='seconds')}",
    ]
    if client_now_iso:
        lines.append(f"Browser-reported UTC datetime: {client_now_iso.strip()[:80]}")
    if timezone_note:
        lines.append(f"Timezone note: {timezone_note}")
    if skew_note:
        lines.append(f"Clock note: {skew_note}")
    lines.extend(
        [
            "",
            "Use this context to interpret relative date phrases such as today, yesterday, tomorrow, this week, last month, latest, and current.",
            "This clock does not make your knowledge current; for facts that may have changed recently, use retrieval or web search when available.",
        ]
    )
    return "\n".join(lines)


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
            outside_observer_framing = re.compile(
                r"^(?:am|are|can|could|did|do|does|is|might|should|would)\s+"
                r"(?:(?:the\s+)?user|they|the\s+requester)\b",
                re.IGNORECASE,
            )
            invalid_option_indexes = [
                index
                for index, opt in enumerate(cleaned)
                if outside_observer_framing.match(opt)
            ]
            if invalid_option_indexes:
                logger.warning(
                    "intent_clarification_options_rejected_outside_observer_framing "
                    "invalid_count=%d total_options=%d invalid_indexes=%s",
                    len(invalid_option_indexes),
                    len(cleaned),
                    invalid_option_indexes,
                )
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

from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_URL_QUERY_KEYS = {
    "access_token", "api_key", "apikey", "auth", "authorization", "code",
    "credential", "key", "password", "secret", "signature", "sig", "token",
}
SOURCE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
RECORD_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
EVIDENCE_SEGMENT_CONTENT_LIMIT = 2_000


def _is_sensitive_url_query_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized in SENSITIVE_URL_QUERY_KEYS:
        return True
    return any(
        marker in normalized
        for marker in (
            "access_token", "api_key", "apikey", "authorization", "credential",
            "password", "secret", "security_token", "signature",
        )
    )


def normalized_source_url(value: Any) -> str:
    try:
        parts = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return ""
    hostname = parts.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parts.port
    except ValueError:
        return ""
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = hostname if not port or default_port else f"{hostname}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = urlencode(sorted(
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_sensitive_url_query_key(key)
    ), doseq=True)
    return urlunsplit((scheme, netloc, path.rstrip("/") or "/", query, ""))


def _first_locator(value: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value and value[key] is not None and value[key] != "":
            return value[key]
    return "document"


def canonical_source_id(value: Dict[str, Any]) -> str:
    file_hash = value.get("file_hash") or value.get("document_id") or value.get("file_id")
    if file_hash and SOURCE_IDENTIFIER_PATTERN.fullmatch(str(file_hash)):
        locator = str(_first_locator(value, "chunk_id", "page_number", "page", "page_start"))
        if SOURCE_IDENTIFIER_PATTERN.fullmatch(locator):
            return f"doc:{file_hash}:{locator}"
    url = normalized_source_url(value.get("url") or value.get("source_url") or value.get("link"))
    if url:
        return f"web:{url}"
    message_id = value.get("message_id") or value.get("chat_id")
    if message_id and RECORD_IDENTIFIER_PATTERN.fullmatch(str(message_id)):
        return f"conversation:{message_id}"
    memory_id = value.get("memory_id")
    if memory_id and RECORD_IDENTIFIER_PATTERN.fullmatch(str(memory_id)):
        return f"memory:{memory_id}"
    return ""


def normalized_canonical_source_id(value: Any) -> str:
    source_id = str(value or "").strip()
    if source_id.startswith("web:"):
        url = normalized_source_url(source_id[4:])
        return f"web:{url}" if url else ""
    if source_id.startswith("doc:"):
        parts = source_id.split(":", 2)
        if len(parts) == 3 and SOURCE_IDENTIFIER_PATTERN.fullmatch(parts[1]) and SOURCE_IDENTIFIER_PATTERN.fullmatch(parts[2]):
            return source_id
        return ""
    for prefix in ("conversation:", "memory:"):
        if source_id.startswith(prefix):
            identifier = source_id[len(prefix):]
            return source_id if RECORD_IDENTIFIER_PATTERN.fullmatch(identifier) else ""
    return ""


def evidence_segment(*, kind: str, content: Any, source: Dict[str, Any], raw_score: Any = None) -> Dict[str, Any]:
    source_id = canonical_source_id(source)
    text = str(content or "").strip()[:EVIDENCE_SEGMENT_CONTENT_LIMIT]
    if not source_id or not text:
        return {}
    display_url = normalized_source_url(source.get("url") or source.get("source_url") or source.get("link"))
    locator = {
        key: source.get(key)
        for key in ("file_hash", "chunk_id", "page_start", "page_end", "pages", "message_id", "memory_id", "timeline_event_at")
        if source.get(key) not in (None, "")
    }
    return {
        "source_id": source_id,
        "kind": str(kind or "evidence"),
        "content": text,
        "display": {
            key: value for key, value in {
                "title": str(source.get("title") or source.get("file_name") or source.get("label") or "")[:500] or None,
                "url": display_url or None,
            }.items() if value
        },
        "locator": locator,
        "raw_score": raw_score,
    }

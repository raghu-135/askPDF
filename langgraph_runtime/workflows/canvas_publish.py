"""Answer-node MCP emit for publish_canvas. Evidence workers stay query-only."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Literal, Mapping

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from langgraph_runtime.agent.canvas_layout_skills import canvas_emit_enabled
from langgraph_runtime.canvas_spec_contract import CANVAS_SPEC_V1_FIELD_HINT
from langgraph_runtime.agent.reasoning import normalize_ai_response
from langgraph_runtime.agent.tool_contract import normalize_tool_result
from langgraph_runtime.mcp_client import create_mcp_langchain_tool
from langgraph_runtime.workflows.cancellation import ChatRunCancellationRequested
from langgraph_runtime.workflows.enums import ToolName
from langgraph_runtime.workflows.runtime_invocation import (
    append_tool_event_for_node,
    invoke_llm_for_node,
    invoke_tool_for_node,
    llm_result_metadata,
    llm_retry_observer,
    tool_config_for_node,
)

_PUBLISH_TOOL = ToolName.PUBLISH_CANVAS.value
_FAKE_FENCE = re.compile(r"```(?:\w+)?\s*publish_canvas[\s\S]*?```", re.IGNORECASE)
_FAKE_CALL = re.compile(r"publish_canvas\s*\([^)]*\)", re.IGNORECASE)
_MAX_ROUNDS = 3
_BLOCK_TYPES = frozenset({
    "stat",
    "table",
    "callout",
    "markdown",
    "sources",
    "dag",
    "quote",
    "timeline",
    "claim_pair",
})
_BLOCK_FIELDS = {
    "stat": frozenset({"type", "value", "label", "tone"}),
    "table": frozenset({"type", "caption", "headers", "rows"}),
    "callout": frozenset({"type", "tone", "title", "body"}),
    "markdown": frozenset({"type", "text"}),
    "sources": frozenset({"type", "title", "citations"}),
    "dag": frozenset({"type", "title", "nodes", "edges"}),
    "quote": frozenset({"type", "file_hash", "sentence_id", "label"}),
    "timeline": frozenset({"type", "title", "events"}),
    "claim_pair": frozenset({"type", "relation", "left", "right"}),
}
_TOOL_ARG_KEYS = frozenset({"spec", "supersedes_id", "idempotency_key"})
_CANVAS_SPEC_ENVELOPE_KEYS = frozenset({"schema_version", "title", "summary", "sections"})
_DEFAULT_CANVAS_TITLE = "Research canvas"
_DEFAULT_SECTION_TITLE = "Findings"
_SPEC_COACHING = (
    f"publish_canvas failed. Required shape: {CANVAS_SPEC_V1_FIELD_HINT} "
    "Do not use from/to on dag edges, content instead of sections[].blocks, or title on dag nodes (use label). "
    "Every canvas needs a sources block: {\"type\": \"sources\", \"citations\": [{\"kind\": \"document\", \"label\": \"...\", \"file_hash\": \"...\"}]} "
    "or at least one quote block. Blocks must use type (not markdown/stat/sources wrapper keys). "
    "Include citations in sources, timeline events, claim_pair sides, or quote blocks. "
    "Omit supersedes_id unless superseding a canvas id returned by a prior successful publish_canvas on this thread; "
    "never use file_hash or document titles as supersedes_id."
)

_CANVAS_ARTIFACT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_FILE_HASH_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def sanitize_supersedes_id_for_publish(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if _FILE_HASH_RE.fullmatch(raw):
        return None
    if not _CANVAS_ARTIFACT_ID_RE.fullmatch(raw):
        return None
    return raw


class CanvasSpecNormalizeError(ValueError):
    """Raised when model spec cannot be repaired into a non-empty canvas_spec_v1."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


_DAG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_DAG_LABEL_CHARS = 48


def _slug_dag_node_id(raw_id: str, *, used: dict[str, str]) -> str:
    original = str(raw_id or "").strip()
    if original in used:
        return used[original]
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._-")
    if not candidate or not candidate[0].isalnum():
        candidate = f"n_{candidate}" if candidate else "node"
    candidate = candidate[:64]
    if not _DAG_ID_PATTERN.fullmatch(candidate):
        candidate = f"n_{hashlib.sha256(original.encode()).hexdigest()[:8]}"
    base = candidate
    suffix = 1
    while candidate in used.values():
        suffix += 1
        candidate = f"{base[: max(1, 60 - len(str(suffix)))]}_{suffix}"
    used[original] = candidate
    return candidate


def _unwrap_canvas_spec_body(body: Mapping[str, Any]) -> dict[str, Any]:
    envelope = dict(body)
    nested = _parse_json_object(envelope.get("canvas_spec_v1"))
    if isinstance(nested, Mapping):
        return dict(nested)
    return envelope


def _repair_dag_block(block: dict[str, Any]) -> None:
    nodes = block.get("nodes")
    if not isinstance(nodes, list):
        return
    id_map: dict[str, str] = {}
    repaired_nodes: list[Any] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            repaired_nodes.append(node)
            continue
        node_dict = dict(node)
        raw_id = str(node_dict.get("id") or node_dict.get("label") or f"node_{len(repaired_nodes) + 1}")
        node_dict["id"] = _slug_dag_node_id(raw_id, used=id_map)
        label = node_dict.get("label")
        if not isinstance(label, str) or not label.strip():
            for key in ("title", "name"):
                candidate = node_dict.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    label = candidate.strip()
                    break
            if not isinstance(label, str) or not label.strip():
                label = raw_id
        node_dict["label"] = str(label).strip()[:_MAX_DAG_LABEL_CHARS]
        repaired_nodes.append(node_dict)
    block["nodes"] = repaired_nodes
    edges = block.get("edges")
    if not isinstance(edges, list):
        return
    repaired_edges: list[Any] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            repaired_edges.append(edge)
            continue
        edge_dict = dict(edge)
        if not edge_dict.get("source") and edge_dict.get("from") is not None:
            edge_dict["source"] = edge_dict.pop("from")
        else:
            edge_dict.pop("from", None)
        if not edge_dict.get("target") and edge_dict.get("to") is not None:
            edge_dict["target"] = edge_dict.pop("to")
        else:
            edge_dict.pop("to", None)
        source = str(edge_dict.get("source") or "")
        target = str(edge_dict.get("target") or "")
        if source in id_map:
            edge_dict["source"] = id_map[source]
        if target in id_map:
            edge_dict["target"] = id_map[target]
        repaired_edges.append(edge_dict)
    block["edges"] = repaired_edges


def strip_prose_publish_canvas(text: str) -> str:
    cleaned = _FAKE_FENCE.sub("", str(text or ""))
    cleaned = _FAKE_CALL.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def canvas_idempotency_key(state: Mapping[str, Any]) -> str:
    return str(state.get("agent_run_id") or state.get("run_id") or "").strip()


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    calls = getattr(response, "tool_calls", None) or []
    return [dict(call) for call in calls if str(call.get("name") or "") == _PUBLISH_TOOL]


def _tool_call_args(call: Mapping[str, Any]) -> dict[str, Any]:
    raw = call.get("args")
    if isinstance(raw, Mapping):
        return dict(raw)
    parsed = _parse_json_object(raw)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _parse_json_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed


def _citation_list(value: Mapping[str, Any]) -> list[Any] | None:
    for key in ("citations", "sources", "items"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return candidate
    return None


_CITATION_FIELDS = frozenset({"kind", "label", "file_hash", "sentence_id", "url", "memory_id", "message_id"})


def _prose_from_mapping(block: Mapping[str, Any]) -> str | None:
    for key in ("text", "content", "body", "message", "summary", "description"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _lift_wrapped_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """Turn {markdown: \"...\"} / {stat: {...}} / {sources: {...}} into typed blocks."""

    if block.get("type"):
        return dict(block)
    wrapper_keys = [key for key in block if key in _BLOCK_TYPES]
    if len(wrapper_keys) != 1:
        return dict(block)
    block_type = wrapper_keys[0]
    inner = block[block_type]
    if block_type == "markdown":
        if isinstance(inner, str):
            return {"type": "markdown", "text": inner.strip()}
        if isinstance(inner, Mapping):
            lifted = dict(inner)
            lifted["type"] = "markdown"
            return lifted
        if inner is not None:
            return {"type": "markdown", "text": str(inner).strip()}
        return dict(block)
    if isinstance(inner, Mapping):
        lifted = dict(inner)
        lifted["type"] = block_type
        return lifted
    if block_type == "stat" and isinstance(inner, (int, float)):
        return {"type": "stat", "value": str(inner)}
    return dict(block)


def _resolve_document_file_hash(
    candidate: str,
    source_id_to_file_hash: Mapping[str, str] | None,
) -> str | None:
    raw = str(candidate or "").strip()
    if not raw:
        return None
    if _FILE_HASH_RE.fullmatch(raw):
        return raw.lower()
    if raw.startswith("src_") and source_id_to_file_hash:
        resolved = source_id_to_file_hash.get(raw)
        if resolved and _FILE_HASH_RE.fullmatch(resolved):
            return resolved.lower()
    if raw.startswith("doc:") and source_id_to_file_hash:
        resolved = source_id_to_file_hash.get(raw)
        if resolved and _FILE_HASH_RE.fullmatch(resolved):
            return resolved.lower()
        parts = raw.split(":", 2)
        if len(parts) == 3 and _FILE_HASH_RE.fullmatch(parts[1]):
            return parts[1].lower()
    return None


def source_id_to_file_hash_from_state(state: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def absorb(source: Mapping[str, Any]) -> None:
        file_hash = str(source.get("file_hash") or source.get("document_id") or source.get("file_id") or "").strip().lower()
        if not _FILE_HASH_RE.fullmatch(file_hash):
            return
        for key in ("source_id", "src_id"):
            sid = source.get(key)
            if isinstance(sid, str) and sid.strip():
                mapping[sid.strip()] = file_hash
        canonical = str(source.get("source_id") or "").strip()
        if canonical.startswith("doc:"):
            mapping.setdefault(canonical, file_hash)

    for collection_key in ("evidence_packets", "document_sources"):
        collection = state.get(collection_key)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("sources") or item.get("matches") or item.get("document_sources")
            if isinstance(nested, list):
                for source in nested:
                    if isinstance(source, Mapping):
                        absorb(source)
            absorb(item)
    return mapping


def _normalize_citation(
    value: Any,
    *,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    citation = dict(value)
    if citation.get("kind") == "document" and not citation.get("file_hash"):
        for key in ("source_id", "file_id", "document_id", "src_id"):
            candidate = citation.get(key)
            if isinstance(candidate, str) and candidate.strip():
                resolved = _resolve_document_file_hash(candidate.strip(), source_id_to_file_hash)
                if resolved:
                    citation["file_hash"] = resolved
                    break
    elif citation.get("kind") == "document" and citation.get("file_hash"):
        resolved = _resolve_document_file_hash(str(citation.get("file_hash")), source_id_to_file_hash)
        if not resolved:
            return None
        citation["file_hash"] = resolved
    label = citation.get("label")
    if not isinstance(label, str) or not label.strip():
        for key in ("title", "name", "file_name"):
            candidate = citation.get(key)
            if isinstance(candidate, str) and candidate.strip():
                citation["label"] = candidate.strip()[:160]
                break
    kind = citation.get("kind")
    if kind not in {"document", "web", "memory", "conversation"}:
        if citation.get("file_hash"):
            citation["kind"] = "document"
        elif citation.get("url"):
            citation["kind"] = "web"
        else:
            return None
    if not isinstance(citation.get("label"), str) or not str(citation.get("label")).strip():
        return None
    if citation.get("kind") == "document":
        file_hash = str(citation.get("file_hash") or "")
        if not _FILE_HASH_RE.fullmatch(file_hash):
            return None
    return {key: citation[key] for key in _CITATION_FIELDS if key in citation}


def _coerce_block(
    value: Any,
    *,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> Any | None:
    if isinstance(value, str):
        text = value.strip()
        return {"type": "markdown", "text": text} if text else None
    if not isinstance(value, Mapping):
        return None
    block = _lift_wrapped_block(value)
    block_type = str(block.get("type") or "")
    prose = _prose_from_mapping(block)
    if not block_type and prose:
        block_type = "markdown"
        block["type"] = "markdown"
    if block_type == "markdown":
        if not block.get("text") and prose:
            block["text"] = prose
        if not isinstance(block.get("text"), str) or not str(block.get("text")).strip():
            return None
    if block_type == "callout":
        if not block.get("title"):
            block["title"] = str(block.get("heading") or "Note")[:120]
        if not block.get("body") and prose:
            block["body"] = prose[:1200]
        if not isinstance(block.get("body"), str) or not str(block.get("body")).strip():
            if prose:
                return {"type": "markdown", "text": prose}
            return None
        if block.get("tone") not in {"info", "warning"}:
            block["tone"] = "info"
    if block_type == "stat":
        if not block.get("label") and block.get("name"):
            block["label"] = str(block.get("name"))
        if not block.get("label") and block.get("title"):
            block["label"] = str(block.get("title"))
        if block.get("value") is not None and not isinstance(block.get("value"), str):
            block["value"] = str(block.get("value"))
        if not block.get("value") or not block.get("label"):
            return None
    if block_type == "sources":
        if "citations" not in block:
            citations = _citation_list(block)
            if citations is not None:
                block["citations"] = citations
        raw_citations = block.get("citations")
        if isinstance(raw_citations, list):
            citations = [
                item
                for item in (
                    _normalize_citation(c, source_id_to_file_hash=source_id_to_file_hash)
                    for c in raw_citations
                )
                if item
            ]
            if not citations:
                return None
            block["citations"] = citations
        if not block.get("title"):
            block["title"] = "Sources"
    if block_type == "markdown" and "text" not in block and block.get("content"):
        block["text"] = str(block.pop("content"))
    if block_type == "dag":
        _repair_dag_block(block)
    if block_type == "quote":
        resolved = _resolve_document_file_hash(str(block.get("file_hash") or ""), source_id_to_file_hash)
        if not resolved:
            return None
        block["file_hash"] = resolved
    allowed = _BLOCK_FIELDS.get(block_type)
    if allowed is None:
        return block
    coerced = {key: block[key] for key in allowed if key in block}
    return coerced if coerced.get("type") else None


def _coerce_blocks(
    blocks: list[Any],
    *,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> list[Any]:
    repaired = [
        _coerce_block(block, source_id_to_file_hash=source_id_to_file_hash)
        for block in blocks
    ]
    return [block for block in repaired if isinstance(block, Mapping) and block.get("type")]


def _blocks_from_payload(
    body: Mapping[str, Any],
    *,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> list[Any] | None:
    block_type = str(body.get("type") or "")
    citations = _citation_list(body) if block_type != "sources" else None
    if block_type in _BLOCK_TYPES:
        blocks = _coerce_blocks([body], source_id_to_file_hash=source_id_to_file_hash)
        if citations:
            blocks.extend(_coerce_blocks(
                [{"type": "sources", "citations": citations}],
                source_id_to_file_hash=source_id_to_file_hash,
            ))
        return blocks or None
    if body.get("text") or body.get("content"):
        blocks = _coerce_blocks(
            [{**dict(body), "type": "markdown"}],
            source_id_to_file_hash=source_id_to_file_hash,
        )
        if citations:
            blocks.extend(_coerce_blocks(
                [{"type": "sources", "citations": citations}],
                source_id_to_file_hash=source_id_to_file_hash,
            ))
        return blocks or None
    return None


def _resolve_canvas_title(body: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> str:
    for candidate in (
        body.get("title"),
        body.get("subtitle"),
        body.get("name"),
        (payload or {}).get("title"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:160]
    sections = body.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            section_title = section.get("title") or section.get("name")
            if isinstance(section_title, str) and section_title.strip():
                return section_title.strip()[:160]
    return _DEFAULT_CANVAS_TITLE


def _resolve_schema_version(body: Mapping[str, Any]) -> Literal[1]:
    if body.get("schema_version") == 1:
        return 1
    version = body.get("version")
    if version in {1, "1"}:
        return 1
    return 1


def _normalize_section(
    section: Any,
    *,
    index: int,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> Any:
    if isinstance(section, str):
        text = section.strip()
        if not text:
            return None
        return {
            "title": _DEFAULT_SECTION_TITLE if index == 0 else f"Section {index + 1}",
            "blocks": [{"type": "markdown", "text": text}],
        }
    if not isinstance(section, Mapping):
        return None
    normalized: dict[str, Any] = {}
    section_title = section.get("title") or section.get("name")
    if isinstance(section_title, str) and section_title.strip():
        normalized["title"] = section_title.strip()[:120]
    else:
        normalized["title"] = _DEFAULT_SECTION_TITLE if index == 0 else f"Section {index + 1}"
    blocks = section.get("blocks")
    coerced: list[Any] = []
    if isinstance(blocks, list):
        coerced = _coerce_blocks(blocks, source_id_to_file_hash=source_id_to_file_hash)
    content = section.get("content")
    if not coerced:
        if isinstance(content, str) and content.strip():
            coerced = [{"type": "markdown", "text": content.strip()}]
        elif isinstance(content, list):
            coerced = _coerce_blocks(content, source_id_to_file_hash=source_id_to_file_hash)
    if coerced:
        normalized["blocks"] = coerced
    if not normalized.get("blocks"):
        return None
    return normalized


def _sanitize_canvas_spec_envelope(
    body: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    if "summary" in body and body.get("summary") is not None:
        summary = body.get("summary")
        if isinstance(summary, str) and summary.strip():
            spec["summary"] = summary.strip()[:400]
    sections = body.get("sections")
    normalized_sections: list[Any] = []
    if isinstance(sections, list):
        normalized_sections = [
            section
            for index, section in enumerate(sections)
            for section in [_normalize_section(
                section,
                index=index,
                source_id_to_file_hash=source_id_to_file_hash,
            )]
            if section is not None and section.get("blocks")
        ]
        spec["sections"] = normalized_sections
    title_source = {**body, "sections": normalized_sections} if normalized_sections else body
    spec["schema_version"] = _resolve_schema_version(body)
    spec["title"] = _resolve_canvas_title(title_source, payload)
    if not normalized_sections:
        dropped = []
        if isinstance(sections, list):
            for index, section in enumerate(sections):
                if isinstance(section, Mapping):
                    label = section.get("title") or section.get("name") or f"section_{index + 1}"
                    dropped.append(str(label))
                elif isinstance(section, str):
                    dropped.append(section[:40] or f"section_{index + 1}")
        detail = ", ".join(dropped) if dropped else "all sections"
        raise CanvasSpecNormalizeError(
            "Canvas spec has no publishable sections after repair. "
            f"Dropped or empty: {detail}. "
            "Use sections[].blocks (not content alone at the root). "
            "DAG edges need source/target; DAG nodes and stats need label."
        )
    return spec


def _canvas_envelope(
    *,
    title: str,
    summary: Any,
    blocks: list[Any],
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": 1,
        "title": (title or "Research canvas").strip()[:160] or "Research canvas",
        "sections": [{
            "title": "Findings",
            "blocks": _coerce_blocks(blocks, source_id_to_file_hash=source_id_to_file_hash),
        }],
    }
    if isinstance(summary, str) and summary.strip():
        spec["summary"] = summary.strip()[:400]
    return spec


def normalize_publish_canvas_spec(
    args: Any,
    *,
    source_id_to_file_hash: Mapping[str, str] | None = None,
) -> Any:
    """Lift common model mistakes into canvas_spec_v1 before MCP validation."""

    payload = dict(args or {}) if isinstance(args, Mapping) else {}
    spec = _parse_json_object(payload.get("spec"))
    if spec is None:
        leftover = {key: value for key, value in payload.items() if key not in _TOOL_ARG_KEYS}
        spec = leftover or None
    spec = _parse_json_object(spec)
    if isinstance(spec, list):
        envelope = _canvas_envelope(
            title=str(payload.get("title") or ""),
            summary=payload.get("summary"),
            blocks=spec,
            source_id_to_file_hash=source_id_to_file_hash,
        )
        if not envelope.get("sections") or not envelope["sections"][0].get("blocks"):
            raise CanvasSpecNormalizeError(
                "Canvas spec has no publishable blocks. Use sections[].blocks with valid block types."
            )
        return envelope
    if not isinstance(spec, Mapping):
        return spec
    body = _unwrap_canvas_spec_body(spec)
    if isinstance(body.get("sections"), list):
        return _sanitize_canvas_spec_envelope(
            body,
            payload=payload,
            source_id_to_file_hash=source_id_to_file_hash,
        )
    lifted = _blocks_from_payload(body, source_id_to_file_hash=source_id_to_file_hash)
    if lifted is not None:
        envelope = _canvas_envelope(
            title=str(payload.get("title") or body.get("title") or ""),
            summary=body.get("summary"),
            blocks=lifted,
            source_id_to_file_hash=source_id_to_file_hash,
        )
        if not envelope.get("sections") or not envelope["sections"][0].get("blocks"):
            raise CanvasSpecNormalizeError(
                "Canvas spec has no publishable blocks. Use sections[].blocks with valid block types."
            )
        return envelope
    blocks = body.get("blocks")
    if isinstance(blocks, list):
        envelope = _canvas_envelope(
            title=str(body.get("title") or payload.get("title") or ""),
            summary=body.get("summary"),
            blocks=blocks,
            source_id_to_file_hash=source_id_to_file_hash,
        )
        if not envelope.get("sections") or not envelope["sections"][0].get("blocks"):
            raise CanvasSpecNormalizeError(
                "Canvas spec has no publishable blocks. Use sections[].blocks with valid block types."
            )
        return envelope
    return body


def _tool_ok(result: Mapping[str, Any]) -> bool:
    if result.get("error"):
        return False
    return bool(result.get("ok", True))


def _tool_message_body(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if content:
        return str(content)
    error = result.get("error")
    if isinstance(error, Mapping):
        return json.dumps(error, ensure_ascii=True)
    return json.dumps({"ok": _tool_ok(result)}, ensure_ascii=True)


async def synthesize_with_canvas_publish(
    llm: Any,
    messages: List[Any],
    *,
    state: Mapping[str, Any],
    config: RunnableConfig,
    node: str,
    started: float,
    failure_data: Dict[str, Any] | None = None,
    consume_usage: Callable[..., Awaitable[Any] | Any] | None = None,
) -> Dict[str, Any]:
    """Invoke the answer LLM, optionally binding publish_canvas for one persist."""

    conversation = list(messages)
    retry_attempts, retry_observer = llm_retry_observer()
    emit_enabled = canvas_emit_enabled(state.get("allowed_tool_ids")) and hasattr(llm, "bind_tools")
    citation_source_map = source_id_to_file_hash_from_state(state)
    published = False
    failed_attempts = 0
    publish_calls = 0
    model_calls = 0
    model_tokens = 0
    last_publish_error: str | None = None
    last_response: Any = None
    bound = llm.bind_tools([create_mcp_langchain_tool(_PUBLISH_TOOL)]) if emit_enabled else llm

    async def _meter(**usage: int) -> None:
        if consume_usage is None:
            return
        result = consume_usage(**usage)
        if hasattr(result, "__await__"):
            await result

    async def _invoke_model(model: Any) -> Any:
        nonlocal last_response, model_calls, model_tokens
        last_response = await invoke_llm_for_node(
            model.ainvoke,
            conversation,
            state=state,
            config=config,
            node=node,
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data=failure_data,
        )
        model_calls += 1
        metadata = llm_result_metadata(last_response, model_name=state.get("llm_model"), retry_attempts=retry_attempts)
        token_counts = metadata.get("token_counts") if isinstance(metadata.get("token_counts"), dict) else {}
        tokens = int(token_counts.get("total") or 0)
        model_tokens += tokens
        await _meter(model_calls=1, model_tokens=tokens)
        return last_response

    for _round in range(_MAX_ROUNDS):
        model = llm if (published or not emit_enabled) else bound
        await _invoke_model(model)
        calls = _tool_calls(last_response) if emit_enabled else []
        if not calls or published:
            break
        conversation.append(last_response)
        for call in calls:
            call_id = str(call.get("id") or f"{node}:{_PUBLISH_TOOL}")
            if published:
                conversation.append(ToolMessage(content="Canvas already published for this run.", tool_call_id=call_id))
                continue
            args = _tool_call_args(call)
            publish_calls += 1
            await _meter(tool_calls=1)
            tool_started = time.perf_counter()
            tool_runtime = tool_config_for_node(
                state, config, caller_node=node, tool_name=_PUBLISH_TOOL, started=tool_started
            )
            try:
                normalized_spec = normalize_publish_canvas_spec(
                    args,
                    source_id_to_file_hash=citation_source_map,
                )
                if isinstance(normalized_spec, str):
                    normalized_spec = _parse_json_object(normalized_spec)
                if not isinstance(normalized_spec, Mapping):
                    raise CanvasSpecNormalizeError(
                        "Canvas spec must be a JSON object with schema_version, title, and sections."
                    )
            except CanvasSpecNormalizeError as exc:
                tool_call_id = str((tool_runtime.get("configurable") or {}).get("tool_call_id") or call_id)
                last_publish_error = exc.message
                normalized = {
                    "ok": False,
                    "content": exc.message,
                    "error": {"code": "publish_canvas_invalid_spec", "message": exc.message},
                    "trace": {
                        "tool_name": _PUBLISH_TOOL,
                        "tool_call_id": tool_call_id,
                        "caller_node": node,
                    },
                }
                append_tool_event_for_node(
                    state,
                    {**normalized, "tool_name": _PUBLISH_TOOL, "caller_node": node},
                    tool_input={"spec": args.get("spec") if isinstance(args, Mapping) else args},
                    config=tool_runtime,
                )
                tool_body = f"{exc.message}\n{_SPEC_COACHING}"
                conversation.append(ToolMessage(content=tool_body, tool_call_id=call_id))
                failed_attempts += 1
                continue
            tool_input: dict[str, Any] = {"spec": normalized_spec}
            key = canvas_idempotency_key(state)
            if key:
                tool_input["idempotency_key"] = key
            sanitized_supersedes = sanitize_supersedes_id_for_publish(args.get("supersedes_id"))
            if sanitized_supersedes:
                tool_input["supersedes_id"] = sanitized_supersedes
            try:
                raw = await invoke_tool_for_node(
                    _PUBLISH_TOOL,
                    tool_input,
                    state=state,
                    config=tool_runtime,
                    node=node,
                    started=tool_started,
                )
                normalized = normalize_tool_result(raw, tool_name=_PUBLISH_TOOL, config=tool_runtime)
            except (ChatRunCancellationRequested, GraphBubbleUp):
                raise
            except Exception as exc:
                tool_call_id = str((tool_runtime.get("configurable") or {}).get("tool_call_id") or call_id)
                normalized = {
                    "ok": False,
                    "content": str(exc),
                    "error": {"code": "publish_canvas_failed", "message": str(exc)},
                    "trace": {
                        "tool_name": _PUBLISH_TOOL,
                        "tool_call_id": tool_call_id,
                        "caller_node": node,
                    },
                }
            append_tool_event_for_node(state, {**normalized, "tool_name": _PUBLISH_TOOL, "caller_node": node}, tool_input=tool_input, config=tool_runtime)
            tool_body = _tool_message_body(normalized)
            if not _tool_ok(normalized):
                last_publish_error = (
                    str((normalized.get("error") or {}).get("message") or normalized.get("content") or tool_body)
                )[:700]
                tool_body = f"{tool_body}\n{_SPEC_COACHING}"
            conversation.append(ToolMessage(content=tool_body, tool_call_id=call_id))
            if _tool_ok(normalized):
                published = True
            else:
                failed_attempts += 1
        if published or failed_attempts:
            continue
        break

    normalized = normalize_ai_response(last_response)
    answer = strip_prose_publish_canvas(normalized["answer"])
    if emit_enabled and publish_calls > 0 and not published and not str(answer or "").strip():
        await _invoke_model(llm)
        normalized = normalize_ai_response(last_response)
        answer = strip_prose_publish_canvas(normalized["answer"])
    if emit_enabled and publish_calls > 0 and not published:
        notice = "The research canvas could not be published for this answer."
        if last_publish_error:
            notice = f"{notice} {last_publish_error}"
        answer = f"{answer}\n\n{notice}".strip() if answer else notice
    return {
        "answer": answer,
        "response": last_response,
        "published": published,
        "reasoning": normalized["reasoning"],
        "reasoning_available": normalized["reasoning_available"],
        "reasoning_format": normalized["reasoning_format"],
        "retry_attempts": retry_attempts,
        "model_calls": model_calls,
        "model_tokens": model_tokens,
        "publish_tool_calls": publish_calls,
    }

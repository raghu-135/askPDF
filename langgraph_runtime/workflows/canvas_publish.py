"""Answer-node MCP emit for publish_canvas. Evidence workers stay query-only."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Literal, Mapping

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from langgraph_runtime.agent.canvas_layout_skills import canvas_emit_enabled
from langgraph_runtime.agent.reasoning import normalize_ai_response
from langgraph_runtime.agent.tool_contract import normalize_tool_result
from langgraph_runtime.mcp_client import create_mcp_langchain_tool
from langgraph_runtime.workflows.cancellation import ChatRunCancellationRequested
from langgraph_runtime.workflows.enums import ToolName
from langgraph_runtime.workflows.runtime_invocation import (
    append_tool_event_for_node,
    invoke_llm_for_node,
    invoke_tool_for_node,
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
    "publish_canvas requires spec as canvas_spec_v1: "
    '{"schema_version": 1, "title": "...", "sections": [{"title": "...", "blocks": [...]}]}. '
    "Allowed blocks: stat, table, callout, markdown, sources, dag, quote, timeline, claim_pair. "
    "Table cells may be strings or {text, tone} with tone supported|contradicted|partial|absent. "
    "DAG edges may include relation supports|contradicts|precedes. "
    "Include citations in a sources block, timeline events, claim_pair sides, or a quote block. "
    "Do not pass a single block as spec."
)


def strip_prose_publish_canvas(text: str) -> str:
    cleaned = _FAKE_FENCE.sub("", str(text or ""))
    cleaned = _FAKE_CALL.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def canvas_idempotency_key(state: Mapping[str, Any]) -> str:
    return str(state.get("agent_run_id") or state.get("run_id") or "").strip()


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    calls = getattr(response, "tool_calls", None) or []
    return [dict(call) for call in calls if str(call.get("name") or "") == _PUBLISH_TOOL]


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


def _normalize_citation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    citation = dict(value)
    if citation.get("kind") == "document" and not citation.get("file_hash"):
        for key in ("source_id", "file_id", "document_id", "src_id"):
            candidate = citation.get(key)
            if isinstance(candidate, str) and candidate.strip():
                citation["file_hash"] = candidate.strip()
                break
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
    return {key: citation[key] for key in _CITATION_FIELDS if key in citation}


def _coerce_block(value: Any) -> Any | None:
    if isinstance(value, str):
        text = value.strip()
        return {"type": "markdown", "text": text} if text else None
    if not isinstance(value, Mapping):
        return None
    block = dict(value)
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
            citations = [item for item in (_normalize_citation(c) for c in raw_citations) if item]
            if not citations:
                return None
            block["citations"] = citations
        if not block.get("title"):
            block["title"] = "Sources"
    if block_type == "markdown" and "text" not in block and block.get("content"):
        block["text"] = str(block.pop("content"))
    allowed = _BLOCK_FIELDS.get(block_type)
    if allowed is None:
        return block
    coerced = {key: block[key] for key in allowed if key in block}
    return coerced if coerced.get("type") else None


def _coerce_blocks(blocks: list[Any]) -> list[Any]:
    repaired = [_coerce_block(block) for block in blocks]
    return [block for block in repaired if isinstance(block, Mapping) and block.get("type")]


def _blocks_from_payload(body: Mapping[str, Any]) -> list[Any] | None:
    block_type = str(body.get("type") or "")
    citations = _citation_list(body) if block_type != "sources" else None
    if block_type in _BLOCK_TYPES:
        blocks = _coerce_blocks([body])
        if citations:
            blocks.extend(_coerce_blocks([{"type": "sources", "citations": citations}]))
        return blocks or None
    if body.get("text") or body.get("content"):
        blocks = _coerce_blocks([{**dict(body), "type": "markdown"}])
        if citations:
            blocks.extend(_coerce_blocks([{"type": "sources", "citations": citations}]))
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


def _normalize_section(section: Any, *, index: int) -> Any:
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
    if isinstance(blocks, list):
        coerced = _coerce_blocks(blocks)
        if not coerced and isinstance(section.get("content"), str) and section["content"].strip():
            coerced = [{"type": "markdown", "text": section["content"].strip()}]
        normalized["blocks"] = coerced
    elif "blocks" in section:
        normalized["blocks"] = section["blocks"]
    if not normalized.get("blocks"):
        return None
    return normalized


def _sanitize_canvas_spec_envelope(body: Mapping[str, Any], *, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
            for section in [_normalize_section(section, index=index)]
            if section is not None and section.get("blocks")
        ]
        spec["sections"] = normalized_sections
    title_source = {**body, "sections": normalized_sections} if normalized_sections else body
    spec["schema_version"] = _resolve_schema_version(body)
    spec["title"] = _resolve_canvas_title(title_source, payload)
    return spec


def _canvas_envelope(*, title: str, summary: Any, blocks: list[Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": 1,
        "title": (title or "Research canvas").strip()[:160] or "Research canvas",
        "sections": [{"title": "Findings", "blocks": _coerce_blocks(blocks)}],
    }
    if isinstance(summary, str) and summary.strip():
        spec["summary"] = summary.strip()[:400]
    return spec


def normalize_publish_canvas_spec(args: Any) -> Any:
    """Lift common model mistakes into canvas_spec_v1 before MCP validation."""

    payload = dict(args or {}) if isinstance(args, Mapping) else {}
    spec = _parse_json_object(payload.get("spec"))
    if spec is None:
        leftover = {key: value for key, value in payload.items() if key not in _TOOL_ARG_KEYS}
        spec = leftover or None
    spec = _parse_json_object(spec)
    if isinstance(spec, list):
        return _canvas_envelope(title=str(payload.get("title") or ""), summary=payload.get("summary"), blocks=spec)
    if not isinstance(spec, Mapping):
        return spec
    body = dict(spec)
    if isinstance(body.get("sections"), list):
        return _sanitize_canvas_spec_envelope(body, payload=payload)
    lifted = _blocks_from_payload(body)
    if lifted is not None:
        return _canvas_envelope(
            title=str(payload.get("title") or body.get("title") or ""),
            summary=body.get("summary"),
            blocks=lifted,
        )
    blocks = body.get("blocks")
    if isinstance(blocks, list):
        return _canvas_envelope(title=str(body.get("title") or payload.get("title") or ""), summary=body.get("summary"), blocks=blocks)
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
) -> Dict[str, Any]:
    """Invoke the answer LLM, optionally binding publish_canvas for one persist."""

    conversation = list(messages)
    retry_attempts, retry_observer = llm_retry_observer()
    emit_enabled = canvas_emit_enabled(state.get("allowed_tool_ids")) and hasattr(llm, "bind_tools")
    published = False
    failed_attempts = 0
    last_response: Any = None
    bound = llm.bind_tools([create_mcp_langchain_tool(_PUBLISH_TOOL)]) if emit_enabled else llm

    for _round in range(_MAX_ROUNDS):
        model = llm if (published or not emit_enabled) else bound
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
        calls = _tool_calls(last_response) if emit_enabled else []
        if not calls or published:
            break
        conversation.append(last_response)
        for call in calls:
            call_id = str(call.get("id") or f"{node}:{_PUBLISH_TOOL}")
            if published:
                conversation.append(ToolMessage(content="Canvas already published for this run.", tool_call_id=call_id))
                continue
            args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
            tool_input: dict[str, Any] = {"spec": normalize_publish_canvas_spec(args)}
            key = canvas_idempotency_key(state)
            if key:
                tool_input["idempotency_key"] = key
            if args.get("supersedes_id"):
                tool_input["supersedes_id"] = str(args.get("supersedes_id"))
            tool_started = time.perf_counter()
            tool_runtime = tool_config_for_node(
                state, config, caller_node=node, tool_name=_PUBLISH_TOOL, started=tool_started
            )
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
    return {
        "answer": strip_prose_publish_canvas(normalized["answer"]),
        "response": last_response,
        "published": published,
        "reasoning": normalized["reasoning"],
        "reasoning_available": normalized["reasoning_available"],
        "reasoning_format": normalized["reasoning_format"],
        "retry_attempts": retry_attempts,
    }

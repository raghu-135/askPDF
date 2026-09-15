"""Answer-node MCP emit for publish_canvas. Evidence workers stay query-only."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Mapping

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
_BLOCK_TYPES = frozenset({"stat", "table", "callout", "markdown", "sources", "dag"})
_BLOCK_FIELDS = {
    "stat": frozenset({"type", "value", "label", "tone"}),
    "table": frozenset({"type", "caption", "headers", "rows"}),
    "callout": frozenset({"type", "tone", "title", "body"}),
    "markdown": frozenset({"type", "text"}),
    "sources": frozenset({"type", "title", "citations"}),
    "dag": frozenset({"type", "title", "nodes", "edges"}),
}
_TOOL_ARG_KEYS = frozenset({"spec", "supersedes_id", "idempotency_key"})
_SPEC_COACHING = (
    "publish_canvas requires spec as canvas_spec_v1: "
    '{"schema_version": 1, "title": "...", "sections": [{"title": "...", "blocks": [...]}]}. '
    "Allowed blocks: stat, table, callout, markdown, sources, dag. "
    "Include a sources block with citations. Do not pass a single block as spec."
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


def _coerce_block(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    block = dict(value)
    block_type = str(block.get("type") or "")
    if not block_type and (block.get("text") or block.get("content")):
        block_type = "markdown"
        block["type"] = "markdown"
    if block_type == "markdown" and "text" not in block and block.get("content"):
        block["text"] = str(block.pop("content"))
    if block_type == "sources" and "citations" not in block:
        citations = _citation_list(block)
        if citations is not None:
            block["citations"] = citations
    allowed = _BLOCK_FIELDS.get(block_type)
    if allowed is None:
        return block
    return {key: block[key] for key in allowed if key in block}


def _blocks_from_payload(body: Mapping[str, Any]) -> list[Any] | None:
    block_type = str(body.get("type") or "")
    citations = _citation_list(body) if block_type != "sources" else None
    if block_type in _BLOCK_TYPES:
        blocks = [_coerce_block(body)]
        if citations:
            blocks.append(_coerce_block({"type": "sources", "citations": citations}))
        return blocks
    if body.get("text") or body.get("content"):
        blocks = [_coerce_block({**dict(body), "type": "markdown"})]
        if citations:
            blocks.append(_coerce_block({"type": "sources", "citations": citations}))
        return blocks
    return None


def _canvas_envelope(*, title: str, summary: Any, blocks: list[Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": 1,
        "title": (title or "Research canvas").strip()[:160] or "Research canvas",
        "sections": [{"title": "Findings", "blocks": [_coerce_block(block) for block in blocks]}],
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
        return body
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
                normalized = {
                    "ok": False,
                    "content": str(exc),
                    "error": {"code": "publish_canvas_failed", "message": str(exc)},
                    "trace": {"tool_name": _PUBLISH_TOOL},
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

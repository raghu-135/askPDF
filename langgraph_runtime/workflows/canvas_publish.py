"""Answer-node MCP emit for publish_canvas. Evidence workers stay query-only."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Mapping

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from langgraph_runtime.agent.canvas_layout_skills import canvas_emit_enabled
from langgraph_runtime.canvas_draft_contract import CANVAS_PUBLISH_DRAFT_HINT
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
_DRAFT_COACHING = (
    f"publish_canvas failed. {CANVAS_PUBLISH_DRAFT_HINT} "
    "Each claim needs text and source_ids copied from the Citation menu. "
    "Timeline layout requires time on every claim. "
    "Omit supersedes_id unless superseding a canvas id from Current canvases on this thread."
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


class CanvasDraftNormalizeError(ValueError):
    """Raised when publish_canvas draft arguments are incomplete."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


CanvasSpecNormalizeError = CanvasDraftNormalizeError  # tests import legacy name


def prepare_publish_canvas_tool_input(args: Mapping[str, Any]) -> dict[str, Any]:
    layout = str(args.get("layout") or "").strip().lower()
    title = str(args.get("title") or "").strip()
    claims = args.get("claims")
    if layout not in {"compare", "matrix", "timeline"}:
        raise CanvasDraftNormalizeError(f"layout must be compare, matrix, or timeline (got {layout!r})")
    if not title:
        raise CanvasDraftNormalizeError("title is required")
    if not isinstance(claims, list) or not claims:
        raise CanvasDraftNormalizeError("claims must be a non-empty list")
    payload: dict[str, Any] = {"layout": layout, "title": title, "claims": claims}
    summary = args.get("summary")
    if isinstance(summary, str) and summary.strip():
        payload["summary"] = summary.strip()[:400]
    return payload


def strip_prose_publish_canvas(text: str) -> str:
    cleaned = _FAKE_FENCE.sub("", str(text or ""))
    cleaned = _FAKE_CALL.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def canvas_idempotency_key(state: Mapping[str, Any], *, supersedes_id: str | None = None) -> str:
    run_id = str(state.get("agent_run_id") or state.get("run_id") or "").strip()
    if not run_id:
        return ""
    suffix = str(supersedes_id or "").strip() or "create"
    return f"publish_canvas:{run_id}:{suffix}"


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
                tool_input = prepare_publish_canvas_tool_input(args)
            except CanvasDraftNormalizeError as exc:
                tool_call_id = str((tool_runtime.get("configurable") or {}).get("tool_call_id") or call_id)
                last_publish_error = exc.message
                normalized = {
                    "ok": False,
                    "content": exc.message,
                    "error": {"code": "publish_canvas_invalid_draft", "message": exc.message},
                    "trace": {
                        "tool_name": _PUBLISH_TOOL,
                        "tool_call_id": tool_call_id,
                        "caller_node": node,
                    },
                }
                append_tool_event_for_node(
                    state,
                    {**normalized, "tool_name": _PUBLISH_TOOL, "caller_node": node},
                    tool_input=dict(args),
                    config=tool_runtime,
                )
                tool_body = f"{exc.message}\n{_DRAFT_COACHING}"
                conversation.append(ToolMessage(content=tool_body, tool_call_id=call_id))
                failed_attempts += 1
                continue
            sanitized_supersedes = sanitize_supersedes_id_for_publish(args.get("supersedes_id"))
            if sanitized_supersedes:
                tool_input["supersedes_id"] = sanitized_supersedes
            key = canvas_idempotency_key(state, supersedes_id=sanitized_supersedes)
            if key:
                tool_input["idempotency_key"] = key
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
                tool_body = f"{tool_body}\n{_DRAFT_COACHING}"
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

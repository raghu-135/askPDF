"""Shared Selkies+Brave workspace tools (Pelorus + CDP via capture service)."""

from __future__ import annotations

from starlette.background import BackgroundTasks

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.db import get_thread
from app.db.enums import FileSourceType
from app.services.browser_capture_client import (
    BrowserCaptureClientError,
    capture_current_page,
    capture_navigate,
    capture_read_page,
    pelorus_control,
    pelorus_screenshot,
    pelorus_state,
)
from app.services.file_processing_service import queue_file_processing
from app.tools.contracts import BrowserActRequest, BrowserNavigateRequest, EmptyBrowserRequest
from app.tools.context import ToolInvocationContext


def _browser_disabled(tool_name: str, context: ToolInvocationContext, started):
    return make_tool_result(
        tool_name=tool_name,
        content="Interactive browser access is not enabled for this session.",
        context=context,
        started=started,
        warnings=[ToolWarningCode.BROWSER_ACCESS_DISABLED],
    )


async def browser_navigate(request: BrowserNavigateRequest, context: ToolInvocationContext):
    started = tool_started()
    tool_name = "browser_navigate"
    if not context.use_browser_access:
        return _browser_disabled(tool_name, context, started)
    try:
        payload = await capture_navigate(request.url)
        page_text = str(payload.get("text") or "").strip()
        if not page_text:
            read = await capture_read_page()
            page_text = str(read.get("text") or "").strip()
            payload = {**payload, "text": page_text, "url": read.get("url") or payload.get("url"), "title": read.get("title") or payload.get("title")}
        body_preview = page_text[:12000]
        content = "\n".join(
            part
            for part in [
                f"Navigated to {payload.get('title') or payload.get('url')} ({payload.get('url')})",
                body_preview,
            ]
            if part
        )
        return make_tool_result(
            tool_name=tool_name,
            content=content,
            context=context,
            started=started,
            artifacts={"browser_page": payload, "browser_page_text": {"url": payload.get("url"), "title": payload.get("title"), "text": body_preview}},
            sources=[{"url": payload.get("url"), "title": payload.get("title"), "text": body_preview[:2000]}],
        )
    except BrowserCaptureClientError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)


async def browser_snapshot(_request: EmptyBrowserRequest, context: ToolInvocationContext):
    started = tool_started()
    tool_name = "browser_snapshot"
    if not context.use_browser_access:
        return _browser_disabled(tool_name, context, started)
    try:
        state = await pelorus_state()
        screenshot = await pelorus_screenshot()
        page = await capture_read_page()
        desktop_ok = bool(state.get("desktop_control", True)) and bool(
            screenshot.get("desktop_control", True)
        )
        summary = str(state.get("summary") or state.get("state") or state)[:4000]
        shot = (
            str(screenshot.get("image") or screenshot.get("screenshot") or "")[:8000]
            if desktop_ok
            else ""
        )
        warnings = (
            []
            if desktop_ok
            else [ToolWarningCode.BROWSER_DESKTOP_CONTROL_UNAVAILABLE]
        )
        content = "\n".join(
            part
            for part in [
                f"URL: {page.get('url')}",
                f"Title: {page.get('title')}",
                summary,
                (page.get("text") or "")[:6000],
            ]
            if part
        )
        return make_tool_result(
            tool_name=tool_name,
            content=content,
            context=context,
            started=started,
            artifacts={
                "browser_state": state,
                "browser_screenshot_b64": shot,
                "browser_page_text": page,
            },
            sources=[{"url": page.get("url"), "title": page.get("title")}],
            warnings=warnings,
        )
    except BrowserCaptureClientError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)


async def browser_act(request: BrowserActRequest, context: ToolInvocationContext):
    started = tool_started()
    tool_name = "browser_act"
    if not context.use_browser_access:
        return _browser_disabled(tool_name, context, started)
    action = "left_click" if request.action == "click" else request.action
    body = {"action": action}
    if request.coordinate is not None:
        body["coordinate"] = request.coordinate
    if request.text is not None:
        body["text"] = request.text
    if request.scroll_direction is not None:
        body["scroll_direction"] = request.scroll_direction
    if request.scroll_amount is not None:
        body["scroll_amount"] = request.scroll_amount
    if request.duration is not None:
        body["duration"] = request.duration
    try:
        result = await pelorus_control(body)
        if result.get("desktop_control") is False:
            return make_tool_result(
                tool_name=tool_name,
                content=str(
                    result.get("message")
                    or "Desktop control is unavailable; use browser_navigate and browser_read instead."
                ),
                context=context,
                started=started,
                artifacts={"browser_control_result": result},
                warnings=[ToolWarningCode.BROWSER_DESKTOP_CONTROL_UNAVAILABLE],
            )
        return make_tool_result(
            tool_name=tool_name,
            content=f"Browser action {request.action} completed.",
            context=context,
            started=started,
            artifacts={"browser_control_result": result},
        )
    except BrowserCaptureClientError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)


async def browser_read(_request: EmptyBrowserRequest, context: ToolInvocationContext):
    started = tool_started()
    tool_name = "browser_read"
    if not context.use_browser_access:
        return _browser_disabled(tool_name, context, started)
    try:
        page = await capture_read_page()
        text = str(page.get("text") or "")[:12000]
        content = f"Title: {page.get('title')}\nURL: {page.get('url')}\n\n{text}"
        return make_tool_result(
            tool_name=tool_name,
            content=content,
            context=context,
            started=started,
            artifacts={"browser_page_text": page},
            sources=[{"url": page.get("url"), "title": page.get("title"), "text": text[:2000]}],
        )
    except BrowserCaptureClientError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)


async def browser_save_to_thread(_request: EmptyBrowserRequest, context: ToolInvocationContext):
    started = tool_started()
    tool_name = "browser_save_to_thread"
    if not context.use_browser_access:
        return _browser_disabled(tool_name, context, started)
    if not context.thread_id:
        return make_tool_error_result(
            tool_name=tool_name,
            error=ValueError("thread_id is required to save a browser page"),
            context=context,
            started=started,
        )
    try:
        capture = await capture_current_page()
        url = str(capture.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return make_tool_error_result(
                tool_name=tool_name,
                error=ValueError("Active page is not a capturable http(s) URL."),
                context=context,
                started=started,
            )
        thread = await get_thread(context.thread_id)
        if thread is None:
            return make_tool_error_result(
                tool_name=tool_name,
                error=ValueError("Thread not found"),
                context=context,
                started=started,
            )
        background = BackgroundTasks()
        await queue_file_processing(
            background_tasks=background,
            thread=thread,
            file_hash=capture["file_hash"],
            file_name=f"{capture.get('title')} - {url}",
            file_path=url,
            source_type=FileSourceType.BROWSER.value,
            indexing_metadata={
                "source_kind": "browser_capture",
                "url": url,
                "title": capture.get("title"),
            },
        )
        for task in background.tasks:
            await task()
        return make_tool_result(
            tool_name=tool_name,
            content=f"Saved browser page to thread as {capture['file_hash']}.pdf ({url}).",
            context=context,
            started=started,
            artifacts={"browser_capture": capture},
            sources=[{"url": url, "title": capture.get("title"), "file_hash": capture["file_hash"]}],
            warnings=[ToolWarningCode.INDEXING_IN_PROGRESS],
        )
    except BrowserCaptureClientError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started)
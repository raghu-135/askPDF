"""Reconnectable event delivery for the pinned Hermes Runs API.

v0.23 replaces the single-consumer queue with ``_RunStream.attach()``. The native
run stream already replays via backlog and per-subscriber queues; this module only
extends transport TTL for live approvals and triggers profile-local MCP retirement
after the terminal SSE frame.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class _Replay:
    events: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False
    subscriber: asyncio.Lock = field(default_factory=asyncio.Lock)


def sweep_run_events(adapter, original_sweep, now=None):
    now = time.time() if now is None else now
    for run_id, task in adapter._active_run_tasks.items():
        if not task.done() and run_id in adapter._run_streams_created:
            adapter._run_streams_created[run_id] = now
    original_sweep(adapter, now)
    replays = getattr(adapter, "_askpdf_event_replays", None)
    if replays is None:
        return
    for run_id in list(replays):
        if run_id not in adapter._run_streams:
            replays.pop(run_id)


async def serve_run_events(
    adapter,
    request,
    *,
    web,
    encode_frame,
    on_terminal: Callable[[], Awaitable[None]],
    _api_server=None,
):
    auth_err = adapter._check_run_auth(request, permission="status") if hasattr(adapter, "_check_run_auth") else adapter._check_auth(request)
    if auth_err is not None:
        return auth_err

    run_id = request.match_info["run_id"]
    if hasattr(adapter, "_request_owns_run") and not adapter._request_owns_run(request, run_id):
        from gateway.platforms.api_server_runs import _run_not_found

        return _run_not_found(_api_server._openai_error, run_id)

    for _ in range(20):
        if run_id in adapter._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        if _api_server is not None:
            from gateway.platforms.api_server_runs import _run_not_found

            return _run_not_found(_api_server._openai_error, run_id)
        return web.json_response({"error": {"code": "run_not_found", "message": f"Run not found: {run_id}"}}, status=404)

    stream = adapter._run_streams[run_id]
    if hasattr(stream, "attach") and _api_server is not None:
        return await _serve_v023(
            adapter,
            request,
            web=web,
            _api_server=_api_server,
            on_terminal=on_terminal,
            run_id=run_id,
            stream=stream,
        )
    return await _serve_legacy(
        adapter, request, web=web, encode_frame=encode_frame, on_terminal=on_terminal, run_id=run_id
    )


async def _serve_v023(adapter, request, *, web, _api_server, on_terminal, run_id: str, stream: Any):
    from gateway.platforms.api_server_runs import _RUN_STREAM_SUBSCRIBER_OVERFLOW, _run_event

    raw_last_seq = request.headers.get("Last-Event-ID") or request.query.get("last_seq")
    try:
        last_seq = max(-1, int(str(raw_last_seq).strip())) if raw_last_seq is not None else -1
    except (TypeError, ValueError):
        last_seq = -1
    queue, replay = stream.attach(last_seq)
    headers = adapter._sse_headers(request) if hasattr(adapter, "_sse_headers") else {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    response = web.StreamResponse(status=200, headers=headers)
    retired = False

    async def _write(data: bytes) -> None:
        await response.write(data)

    async def _write_event(seq: int, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload["seq"] = seq
        await _write(_api_server._sse_frame(payload, id=seq))

    async def _retire_once() -> None:
        nonlocal retired
        if retired:
            return
        retired = True
        await on_terminal()

    prepared = False
    try:
        await response.prepare(request)
        prepared = True
        await _write(b": open\n\n")
        if replay and replay[0][0] > last_seq + 1:
            truncation = _run_event(
                run_id,
                "replay.truncated",
                oldest_retained_seq=replay[0][0],
            )
            if raw_last_seq is not None:
                truncation["requested_seq"] = last_seq
            await _write(_api_server._sse_frame(truncation))
        for seq, event in replay:
            if event is None:
                await _write(b": stream closed\n\n")
                await _retire_once()
                return response
            await _write_event(seq, event)
        if stream.terminal and queue.empty():
            await _write(b": stream closed\n\n")
            await _retire_once()
            return response
        while True:
            try:
                seq, event = await asyncio.wait_for(
                    queue.get(),
                    timeout=_api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS,
                )
            except asyncio.TimeoutError:
                await _write(b": keepalive\n\n")
                continue
            if event is _RUN_STREAM_SUBSCRIBER_OVERFLOW:
                break
            if event is None:
                await _write(b": stream closed\n\n")
                await _retire_once()
                break
            await _write_event(seq, event)
    except ConnectionError:
        pass
    finally:
        with suppress(Exception):
            stream.detach(queue)
        if hasattr(adapter, "_release_run_owner_if_forgotten"):
            with suppress(Exception):
                adapter._release_run_owner_if_forgotten(run_id)
    return response


async def _serve_legacy(adapter, request, *, web, encode_frame, on_terminal, run_id: str):
    if not hasattr(adapter, "_askpdf_event_replays"):
        adapter._askpdf_event_replays = {}
    if not hasattr(adapter, "_run_stream_subscribers"):
        adapter._run_stream_subscribers = set()
    replay = adapter._askpdf_event_replays.setdefault(run_id, _Replay())
    async with replay.subscriber:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await response.prepare(request)
        adapter._run_stream_subscribers.add(run_id)
        try:
            index = 0
            while True:
                if index == len(replay.events):
                    if replay.ended:
                        await response.write(b": stream closed\n\n")
                        await on_terminal()
                        break
                    queue = adapter._run_streams[run_id]
                    if not hasattr(queue, "get"):
                        raise AttributeError("incompatible Hermes run stream transport")
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        await response.write(b": keepalive\n\n")
                        continue
                    if event is None:
                        replay.ended = True
                        continue
                    event = dict(event)
                    event.setdefault("event_id", f"{run_id}:{len(replay.events)}")
                    replay.events.append(event)
                await response.write(encode_frame(replay.events[index]))
                index += 1
        except ConnectionError:
            pass
        finally:
            adapter._run_stream_subscribers.discard(run_id)
        return response

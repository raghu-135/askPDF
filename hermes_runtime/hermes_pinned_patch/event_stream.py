"""Reconnectable event delivery for the pinned Hermes Runs API.

The native queue is single-consumer and the native run remains authoritative.
Retain delivered frames for the lifetime of its generated profile so a human
pause can close and reopen HTTP without losing a tool completion or run result.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Replay:
    events: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False
    subscriber: asyncio.Lock = field(default_factory=asyncio.Lock)


def sweep_run_events(adapter, original_sweep, now=None):
    now = time.time() if now is None else now
    # A native approval can outlast the transport's five-minute idle TTL.
    # Keep live executors reconnectable; retain the native terminal TTL.
    for run_id, task in adapter._active_run_tasks.items():
        if not task.done() and run_id in adapter._run_streams_created:
            adapter._run_streams_created[run_id] = now
    original_sweep(adapter, now)
    replays = getattr(adapter, "_askpdf_event_replays", {})
    for run_id in list(replays):
        if run_id not in adapter._run_streams:
            replays.pop(run_id)


async def serve_run_events(adapter, request, *, web, encode_frame, on_terminal):
    auth_error = adapter._check_auth(request)
    if auth_error is not None:
        return auth_error
    run_id = request.match_info["run_id"]
    # Preserve the pinned API's registration race window.
    for _ in range(20):
        if run_id in adapter._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        return web.json_response({"error": {"code": "run_not_found", "message": f"Run not found: {run_id}"}}, status=404)
    if not hasattr(adapter, "_askpdf_event_replays"):
        adapter._askpdf_event_replays = {}
    replay = adapter._askpdf_event_replays.setdefault(run_id, _Replay())
    # A reconnect can arrive before the previous socket notices disconnect.
    # Serial readers preserve native queue order during that handover.
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
                    try:
                        event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        await response.write(b": keepalive\n\n")
                        continue
                    if event is None:
                        replay.ended = True
                        continue
                    event = dict(event)
                    event.setdefault("event_id", f"{run_id}:{len(replay.events)}")
                    replay.events.append(event)
                # Record before writing: a broken socket has an unknown
                # delivery outcome. The gateway deduplicates these stable IDs.
                await response.write(encode_frame(replay.events[index]))
                index += 1
        except ConnectionError:
            pass  # Keep the queue and replay log for the next subscriber.
        finally:
            adapter._run_stream_subscribers.discard(run_id)
        return response

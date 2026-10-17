import asyncio

import pytest

from langgraph_runtime.workflows.parallel_contracts import await_worker_with_progress_extension


@pytest.mark.asyncio
async def test_progress_extension_allows_slow_worker_to_finish():
    progress_events: list[dict] = []

    async def slow_worker() -> str:
        await asyncio.sleep(0.05)
        return "done"

    async def on_progress(payload: dict) -> None:
        progress_events.append(payload)

    result = await await_worker_with_progress_extension(
        slow_worker(),
        timeout_seconds=0.02,
        on_progress=on_progress,
        poll_interval_seconds=0.01,
        extension_seconds=0.1,
        max_extensions=1,
    )
    assert result == "done"
    assert progress_events

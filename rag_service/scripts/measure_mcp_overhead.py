#!/usr/bin/env python3
"""Measure MCP descriptor discovery and per-call session overhead.

This intentionally does not cache descriptors. It measures the current
adapter/transport behavior so caching can be justified by data.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time


def _summary(values: list[float]) -> str:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return (
        f"median={statistics.median(values):.2f}ms "
        f"p95={ordered[p95_index]:.2f}ms"
    )


async def _run() -> None:
    from app.mcp import discovery
    logging.basicConfig(level=logging.CRITICAL)
    from app.mcp import langchain_adapter
    from app.mcp.transport import InProcessMCPClient

    client = InProcessMCPClient()
    warmup = 5
    samples = 40
    list_timings: list[float] = []
    call_timings: list[float] = []
    uncached_adapter_timings: list[float] = []
    cached_adapter_timings: list[float] = []

    for index in range(warmup + samples):
        started = time.perf_counter()
        await client.request("tools/list", {})
        list_elapsed = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        await client.request("tools/call", {"name": "get_thread_shape", "arguments": {}})
        call_elapsed = (time.perf_counter() - started) * 1000

        discovery.clear_discovery_cache()
        started = time.perf_counter()
        await langchain_adapter.call_mcp_tool("get_thread_shape", {})
        uncached_adapter_elapsed = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        await langchain_adapter.call_mcp_tool("get_thread_shape", {})
        cached_adapter_elapsed = (time.perf_counter() - started) * 1000

        if index >= warmup:
            list_timings.append(list_elapsed)
            call_timings.append(call_elapsed)
            uncached_adapter_timings.append(uncached_adapter_elapsed)
            cached_adapter_timings.append(cached_adapter_elapsed)

    print(f"transport=in_process warmup={warmup} samples={samples} tool=get_thread_shape")
    print(f"direct tools/list: {_summary(list_timings)}")
    print(f"direct tools/call: {_summary(call_timings)}")
    print(f"adapter call without cache: {_summary(uncached_adapter_timings)}")
    print(f"adapter call with descriptor cache: {_summary(cached_adapter_timings)}")
    print(
        "discovery share estimate before cache: "
        f"{statistics.median(list_timings) / statistics.median(uncached_adapter_timings) * 100:.1f}% median"
    )


if __name__ == "__main__":
    asyncio.run(_run())

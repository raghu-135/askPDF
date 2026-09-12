import asyncio

import pytest

from langgraph_runtime.workflows import deep_research_execution as execution
from langgraph_runtime.workflows.deep_research_execution import RuntimeBudgetMeter


def _budget(limit=90_000):
    dimensions = {
        "model_calls": 10,
        "model_tokens": 10_000,
        "tool_calls": 10,
        "elapsed_active_ms": limit,
    }
    zero = {key: 0 for key in dimensions}
    return {
        "tranche_index": 1,
        "tranche_limits": dimensions,
        "tranche_usage": zero,
        "lifetime_usage": dict(zero),
    }


@pytest.mark.asyncio
async def test_parallel_spans_charge_one_wall_clock_interval(monkeypatch):
    meter = RuntimeBudgetMeter(_budget(), {})
    ticks = iter((0.0, 60.0))
    monkeypatch.setattr(execution.time, "perf_counter", lambda: next(ticks))

    async def worker():
        async with meter.execution_span():
            await asyncio.sleep(0)

    await asyncio.gather(worker(), worker())
    snapshot = await meter.snapshot()
    assert snapshot["lifetime_usage"]["elapsed_active_ms"] == 60_000


@pytest.mark.asyncio
async def test_sequential_spans_and_checkpoint_continuation_accumulate_once(monkeypatch):
    first = RuntimeBudgetMeter(_budget(), {})
    ticks = iter((0.0, 10.0, 20.0, 35.0))
    monkeypatch.setattr(execution.time, "perf_counter", lambda: next(ticks))
    async with first.execution_span():
        pass
    continued = RuntimeBudgetMeter(await first.snapshot(), {})
    async with continued.execution_span():
        pass
    snapshot = await continued.snapshot()
    assert snapshot["lifetime_usage"]["elapsed_active_ms"] == 25_000


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension", ["elapsed_active_ms", "model_tokens", "model_calls", "tool_calls"])
async def test_all_dimensions_use_the_same_repeatable_boundary(dimension):
    budget = _budget()
    for tranche in range(1, 8):
        meter = RuntimeBudgetMeter(budget, {})
        limit = budget["tranche_limits"][dimension]
        assert await meter.boundary() is None
        await meter.consume(**{dimension: limit})
        boundary = await meter.boundary()
        assert boundary == {"status": "requested", "dimensions": [dimension], "tranche_index": tranche}
        budget = dict(await meter.snapshot())
        assert budget["lifetime_usage"][dimension] == limit * tranche
        # Continuation resets tranche counters, never lifetime accounting.
        budget.update(tranche_index=tranche + 1, boundary=None)
        budget["tranche_usage"] = {key: 0 for key in budget["tranche_limits"]}


@pytest.mark.asyncio
async def test_time_boundary_retains_atomic_work_and_excludes_provisional_synthesis(monkeypatch):
    meter = RuntimeBudgetMeter(_budget(limit=1000), {})
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(execution.time, "perf_counter", lambda: next(ticks))
    evidence = []
    async with meter.execution_span():
        evidence.append("completed atomic evidence")
        assert await meter.boundary() is None
    assert evidence == ["completed atomic evidence"]
    assert (await meter.boundary())["dimensions"] == ["elapsed_active_ms"]
    before = await meter.snapshot()
    services = execution.RuntimeExecutionServices(
        todos=None, artifacts=None, budgets=None, cancellation=None,
        events=None, memory=None, runtime_budget_meter=meter,
    )
    async with services.execution_span(enabled=False):
        evidence.append("provisional answer")
    assert await meter.snapshot() == before

from langgraph_runtime.workflows.parallel_contracts import normalized_parallel_policy


def test_router_parallel_policy_uses_extended_worker_timeout():
    policy = normalized_parallel_policy({
        "default_worker_timeout_ms": 120_000,
        "dispatch_timeout_ms": 180_000,
    })
    assert policy["default_worker_timeout_ms"] >= 120_000
    assert policy["dispatch_timeout_ms"] >= 180_000


def test_normalized_parallel_policy_never_below_defaults_without_config():
    policy = normalized_parallel_policy({})
    assert policy["default_worker_timeout_ms"] >= 30_000

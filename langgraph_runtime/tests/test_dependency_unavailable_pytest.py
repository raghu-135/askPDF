from langgraph_runtime.dependencies import DependencyMonitor


def test_provider_missing_model_is_reported_by_monitor():
    monitor = DependencyMonitor()
    monitor._snapshots["provider"] = {
        "state": "available",
        "capability_ids": ["gpt-4"],
        "_last_success_epoch": 1.0,
    }
    failure = monitor.unavailable({"provider": {"missing-model"}})
    assert failure is not None
    assert failure["dependency"] == "provider"
    assert failure["missing_capability_ids"] == ["missing-model"]

from langgraph_runtime.workflows.workflow_runtime import replan_loop_policy


def test_coverage_bonus_extends_web_worker_visit_limit():
    config = {
        "replans": 1,
        "coverage_iterations": {"web_search_iteration_bonus": 3},
        "graph": {
            "nodes": [
                {"id": "web-1", "type": "web_worker"},
                {"id": "serial", "type": "serial_dispatch"},
            ],
            "edges": [],
        },
    }
    spec = {"config": config}
    base = replan_loop_policy(spec, config)
    web_limit = base["node_visit_limits"]["web-1"]
    config_without = {**config, "coverage_iterations": {}}
    without = replan_loop_policy({"config": config_without}, config_without)
    assert web_limit == without["node_visit_limits"]["web-1"] + 3
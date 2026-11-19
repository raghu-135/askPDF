"""Read effective runtime settings from an agent run payload (sweep verification)."""

from __future__ import annotations

from typing import Any


def _dig(data: Any, *keys: str) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def supplement_from_thread_settings(
    thread_settings: dict[str, Any] | None,
    effective: dict[str, Any],
) -> dict[str, Any]:
    """Merge persisted thread settings when the run trace omits budget ratios."""
    if not isinstance(thread_settings, dict):
        return effective
    out = dict(effective)
    if out.get("context_budget") is None and isinstance(thread_settings.get("context_budget"), dict):
        out["context_budget"] = thread_settings["context_budget"]
    if out.get("context_window") is None and thread_settings.get("context_window") is not None:
        out["context_window"] = thread_settings["context_window"]
    cov = dict(out.get("coverage_iterations") or {})
    for key in (
        "max_iterations_sufficient_coverage",
        "max_iterations_probably_sufficient_coverage",
        "web_search_iteration_bonus",
    ):
        if cov.get(key) is None and thread_settings.get(key) is not None:
            cov[key] = thread_settings[key]
    thread_cov = thread_settings.get("coverage_iterations")
    if isinstance(thread_cov, dict):
        for key, value in thread_cov.items():
            if cov.get(key) is None and value is not None:
                cov[key] = value
    if cov:
        out["coverage_iterations"] = cov
    return out


def effective_settings_from_run(run: dict[str, Any]) -> dict[str, Any]:
    """Best-effort effective settings from GET /agent-runs trace projection."""
    resolved = run.get("resolved_spec_json") if isinstance(run.get("resolved_spec_json"), dict) else {}
    cfg = resolved.get("config") if isinstance(resolved.get("config"), dict) else {}
    debug = run.get("debug") if isinstance(run.get("debug"), dict) else {}
    details = debug.get("details") if isinstance(debug.get("details"), list) else []
    prefetch_budget = None
    context_window_trace = _dig(debug, "trace", "attributes", "askpdf.context_window")

    for row in details:
        if not isinstance(row, dict):
            continue
        if row.get("node_id") == "context_loader" or row.get("node") == "context_loader":
            before = row.get("checkpoint_before") if isinstance(row.get("checkpoint_before"), dict) else {}
            if before.get("context_budget") and prefetch_budget is None:
                prefetch_budget = before.get("context_budget")
            if before.get("context_window") and not cfg.get("context_window"):
                cfg = {**cfg, "context_window": before.get("context_window")}
        ps = row.get("prefetch_summary")
        if isinstance(ps, dict) and isinstance(ps.get("context_budget"), dict):
            prefetch_budget = ps.get("context_budget")

    coverage = cfg.get("coverage_iterations") if isinstance(cfg.get("coverage_iterations"), dict) else {}
    if not coverage:
        for key in (
            "max_iterations_sufficient_coverage",
            "max_iterations_probably_sufficient_coverage",
            "web_search_iteration_bonus",
        ):
            if cfg.get(key) is not None:
                coverage[key] = cfg.get(key)

    return {
        "context_window": cfg.get("context_window") or context_window_trace,
        "context_budget": prefetch_budget or cfg.get("context_budget"),
        "coverage_iterations": coverage or None,
        "loop_policy": cfg.get("loop_policy"),
    }


def assert_override_applied(
    expected: dict[str, Any],
    effective: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """Return (ok, reason) for sweep gate."""
    if not expected:
        return True, "no expected overrides"
    for key, want in expected.get("context_budget", {}).items() if isinstance(expected.get("context_budget"), dict) else []:
        got = (effective.get("context_budget") or {}).get(key)
        if got is None:
            return False, f"context_budget.{key} missing in effective settings"
        if abs(float(got) - float(want)) > tolerance:
            return False, f"context_budget.{key}: expected {want}, got {got}"
    if expected.get("context_window") is not None:
        got = effective.get("context_window")
        if got is None:
            return False, "context_window missing in effective settings"
        if int(got) != int(expected["context_window"]):
            return False, f"context_window: expected {expected['context_window']}, got {got}"
    exp_cov = expected.get("coverage_iterations") if isinstance(expected.get("coverage_iterations"), dict) else {}
    eff_cov = effective.get("coverage_iterations") if isinstance(effective.get("coverage_iterations"), dict) else {}
    for key, want in exp_cov.items():
        got = eff_cov.get(key)
        if got is None:
            return False, f"coverage_iterations.{key} missing"
        if int(got) != int(want):
            return False, f"coverage_iterations.{key}: expected {want}, got {got}"
    return True, "ok"
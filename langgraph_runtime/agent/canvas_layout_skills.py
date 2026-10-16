"""Prompt-only canvas layout skills. Not separate runtimes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from langgraph_runtime.prompts.loaders import load_runtime_prompt


CANVAS_LAYOUT_SKILL_IDS = (
    "canvas_compare_papers",
    "canvas_evidence_matrix",
    "canvas_timeline",
)
CANVAS_PUBLISH_TOOL_IDS = frozenset({"research_canvas_publish", "publish_canvas"})


def canvas_emit_enabled(allowed_tool_ids: Iterable[Any] | None) -> bool:
    if not isinstance(allowed_tool_ids, (list, tuple, set, frozenset)):
        return False
    allowed = {str(item) for item in allowed_tool_ids if item}
    return bool(allowed & CANVAS_PUBLISH_TOOL_IDS)


def canvas_layout_skills_section(allowed_tool_ids: Iterable[Any] | None) -> str:
    if not canvas_emit_enabled(allowed_tool_ids):
        return ""
    return load_runtime_prompt("canvas_layout_skills.md").strip()


def canvas_publish_synthesis_enabled(state: Mapping[str, Any]) -> bool:
    """Bind publish_canvas at synthesis only when the objective asks for a canvas artifact."""

    if not canvas_emit_enabled(state.get("allowed_tool_ids")):
        return False
    orchestration = state.get("task_orchestration") if isinstance(state.get("task_orchestration"), Mapping) else {}
    if orchestration.get("publish_canvas") is True:
        return True
    text = str(state.get("question") or state.get("objective") or "").lower()
    markers = (
        "canvas",
        "comparison",
        "compare papers",
        "timeline",
        "evidence map",
        "evidence matrix",
    )
    return any(marker in text for marker in markers)

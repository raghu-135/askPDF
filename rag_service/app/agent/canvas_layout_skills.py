"""Prompt-only canvas layout skills. Not separate runtimes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.prompts.loaders import load_prompt


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


def canvas_layout_skills_markdown() -> str:
    return load_prompt("product_orchestration/canvas_layout_skills.md").strip()


def canvas_layout_skills_section(allowed_tool_ids: Iterable[Any] | None) -> str:
    if not canvas_emit_enabled(allowed_tool_ids):
        return ""
    return canvas_layout_skills_markdown()


def apply_canvas_layout_skills(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Attach layout skill ids and prompt text only when publish_canvas is admitted."""

    updated = dict(config or {})
    allowed = updated.get("allowed_tool_ids")
    skills = [str(item) for item in (updated.get("skills") or []) if str(item).strip()]
    if not canvas_emit_enabled(allowed):
        updated["skills"] = [item for item in skills if item not in CANVAS_LAYOUT_SKILL_IDS]
        return updated
    for skill_id in CANVAS_LAYOUT_SKILL_IDS:
        if skill_id not in skills:
            skills.append(skill_id)
    updated["skills"] = skills
    prompt = str(updated.get("system_prompt") or "").strip()
    body = canvas_layout_skills_markdown()
    if body and body not in prompt:
        updated["system_prompt"] = f"{prompt}\n\n{body}".strip() if prompt else body
    return updated

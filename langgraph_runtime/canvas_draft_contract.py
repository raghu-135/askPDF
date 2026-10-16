"""Agent-facing publish_canvas draft shape (no rag_service import)."""

CANVAS_PUBLISH_DRAFT_HINT = (
    "publish_canvas draft: layout compare|matrix|timeline, title, optional summary, "
    "claims[{text, source_ids[], optional time (timeline), tone supported|contradicted|partial|absent, "
    "optional group (source column name)}]. "
    "Copy source_id values exactly from the Citation menu in the prompt. "
    "Optional supersedes_id only when superseding a canvas id from Current canvases on this thread."
)

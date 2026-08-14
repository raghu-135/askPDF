"""Framework-neutral thread-shape implementation."""

import logging
from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.agent.tool_contract import ToolResult, make_tool_error_result, make_tool_result, tool_started

logger = logging.getLogger(__name__)


class ThreadShapeRequest(BaseModel):
    pass


async def invoke_thread_shape(request: ThreadShapeRequest, context: ToolInvocationContext) -> ToolResult:
    del request
    started = tool_started()
    if not context.thread_id:
        return make_tool_result(
            tool_name="get_thread_shape", content="No thread context found.", context=context,
            started=started, warnings=["missing_thread_id"],
        )

    try:
        from app.db import get_thread_shape

        shape = await get_thread_shape(context.thread_id)
        qa_pairs = shape["total_qa_pairs"]
        avg_qa = shape["avg_qa_chars"]
        total_qa = shape["total_qa_chars"]
        docs = shape["documents"]
        lines = ["[THREAD SHAPE]"]
        lines.append(
            f"QA History  : {qa_pairs} pair(s) | {avg_qa:,.0f} avg chars/pair | {total_qa:,} total chars"
        )
        if docs:
            lines.append(f"Documents   : {len(docs)} source(s)")
            for i, (file_hash, meta) in enumerate(docs.items(), 1):
                status = meta.get("indexing_status", "unknown")
                chunks = meta.get("chunk_count", 0)
                chars = meta.get("total_chars", 0)
                words = meta.get("word_count")
                pages = meta.get("page_count")
                sentences = meta.get("sentence_count")
                name = meta.get("file_name", file_hash)
                source_type = meta.get("source_type", "pdf")
                available_at = meta.get("document_available_in_thread_at")
                counts = []
                if pages not in (None, ""):
                    counts.append(f"{pages} pages")
                if words not in (None, ""):
                    counts.append(f"{words:,} words")
                if sentences not in (None, ""):
                    counts.append(f"{sentences:,} sentences")
                counts_text = f" | {', '.join(counts)}" if counts else ""
                availability = f" | added_to_thread_at={available_at}" if available_at else ""
                lines.append(
                    f"  {i}. file_name={name} | file_hash={file_hash} | source_type={source_type} | "
                    f"{chunks} chunks | {chars:,} chars{counts_text} | {status}{availability}"
                )
        else:
            lines.append("Documents   : none uploaded yet")
        return make_tool_result(
            tool_name="get_thread_shape", content="\n".join(lines), context=context,
            started=started, artifacts={"thread_shape": shape},
        )
    except Exception as exc:
        logger.error("Error reading thread shape: %s", exc, exc_info=True)
        return make_tool_error_result(
            tool_name="get_thread_shape", error=exc, context=context, started=started,
            user_message="Error reading thread shape. No thread-shape evidence was returned.",
        )

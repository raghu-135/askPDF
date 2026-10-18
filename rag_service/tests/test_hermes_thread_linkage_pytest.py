import os
import uuid

import pytest

from app.mcp.execution_context_token import decode_execution_context_token, issue_execution_context_token
from app.tools.context import ToolInvocationContext


@pytest.mark.asyncio
async def test_hermes_execution_context_carries_thread_for_document_tools():
    thread_id = f"hermes-thread-linkage-{uuid.uuid4().hex}"
    run_id = f"hermes-run-{uuid.uuid4().hex}"
    context_window = int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"])
    token = issue_execution_context_token(
        ToolInvocationContext(
            thread_id=thread_id,
            run_id=run_id,
            embedding_model="test-embed",
            context_window=context_window,
        ),
        task_id=run_id,
        allowed_tools=["search_knowledge"],
    )
    context = decode_execution_context_token(token)
    assert context.thread_id == thread_id

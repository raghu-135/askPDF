from types import SimpleNamespace

import pytest

from app.tools import retrieval_knowledge


@pytest.mark.asyncio
async def test_schedule_document_repair_skips_failed_thread_job_without_raising():
    scheduled = await retrieval_knowledge._schedule_document_repair(
        file_hash="bad-file",
        embedding_model="model-a",
        thread_id="thread-a",
        readiness={
            "reason": "thread_job_failed",
            "thread_job": SimpleNamespace(status="failed", error="blank capture"),
        },
    )
    assert scheduled is False

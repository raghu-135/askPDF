"""
test_stats_repository_pytest.py - Tests for thread-column-backed stats.
"""

import os
import sys

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlmodel import select

    from app.db.models_sqlmodel import ChatTurn, ThreadFile
    from app.db.repositories.stats_repo_sqlmodel import StatsRepository

    SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestStatsRepository:
    @pytest_asyncio.fixture
    async def repo(self, engine):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            yield StatsRepository(repo_session)

    @pytest.mark.asyncio
    async def test_increment_qa_stats_updates_thread_columns(self, repo, session, sample_thread):
        await repo.increment_qa_stats(sample_thread.id, 120)
        await repo.increment_qa_stats(sample_thread.id, 80)

        await session.refresh(sample_thread)
        assert sample_thread.total_qa_pairs == 2
        assert sample_thread.total_qa_chars == 200
        assert sample_thread.avg_qa_chars == 100
        assert sample_thread.last_qa_at is not None

    @pytest.mark.asyncio
    async def test_document_metadata_round_trip(self, repo, session, sample_thread, sample_file):
        await repo.upsert_document_in_stats(
            sample_thread.id,
            sample_file.file_hash,
            {"chunk_count": 12, "indexing_status": "completed"},
        )
        await repo.upsert_document_in_stats(
            sample_thread.id,
            sample_file.file_hash,
            {"total_chars": 5000},
        )

        stats = await repo.get_stats(sample_thread.id)
        assert stats["documents"][sample_file.file_hash]["chunk_count"] == 12
        assert stats["documents"][sample_file.file_hash]["total_chars"] == 5000

        await repo.remove_document_from_stats(sample_thread.id, sample_file.file_hash)
        await session.refresh(sample_thread)
        assert sample_thread.documents_meta == {}

    @pytest.mark.asyncio
    async def test_recompute_qa_stats_counts_answered_turns(self, repo, session, sample_thread):
        turns = [
            ChatTurn(
                id="turn-completed",
                thread_id=sample_thread.id,
                status="completed",
                payload={"question": "Q1", "answer": "abcd"},
            ),
            ChatTurn(
                id="turn-clarification",
                thread_id=sample_thread.id,
                status="clarification",
                payload={"question": "Q2", "answer": "pick one"},
            ),
            ChatTurn(
                id="turn-failed",
                thread_id=sample_thread.id,
                status="failed",
                payload={"question": "Q3", "answer": "fallback"},
            ),
            ChatTurn(
                id="turn-empty",
                thread_id=sample_thread.id,
                status="completed",
                payload={"question": "Q4", "answer": ""},
            ),
        ]
        session.add_all(turns)
        await session.commit()

        await repo.recompute_qa_stats(sample_thread.id)
        await session.refresh(sample_thread)

        assert sample_thread.total_qa_pairs == 2
        assert sample_thread.total_qa_chars == len("abcd") + len("pick one")
        assert sample_thread.avg_qa_chars == 6

    @pytest.mark.asyncio
    async def test_thread_shape_uses_thread_files_as_source_of_truth(
        self,
        repo,
        session,
        sample_thread,
        sample_file,
    ):
        sample_thread.documents_meta = {
            sample_file.file_hash: {
                "chunk_count": 4,
                "file_name": "stale-name.pdf",
            },
            "detached-file": {"file_name": "not-attached.pdf"},
        }
        sample_thread.total_qa_pairs = 3
        sample_thread.total_qa_chars = 90
        sample_thread.avg_qa_chars = 30
        session.add(
            ThreadFile(thread_id=sample_thread.id, file_hash=sample_file.file_hash)
        )
        await session.commit()

        shape = await repo.get_thread_shape(sample_thread.id)

        assert shape["total_qa_pairs"] == 3
        assert set(shape["documents"].keys()) == {sample_file.file_hash}
        assert shape["documents"][sample_file.file_hash]["chunk_count"] == 4
        assert shape["documents"][sample_file.file_hash]["file_name"] == sample_file.file_name

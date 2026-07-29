"""Project knowledge association and effective inventory tests."""

import pytest

from app.db.repositories.project_file_repo_sqlmodel import ProjectFileRepository
from app.db.repositories.thread_file_repo_sqlmodel import ThreadFileRepository


@pytest.mark.asyncio
async def test_project_association_is_idempotent(session, sample_thread, sample_file):
    repo = ProjectFileRepository(session)
    project_id = sample_thread.project_id
    file_hash = sample_file.file_hash
    await session.rollback()

    assert await repo.add(project_id, file_hash)
    assert await repo.add(project_id, file_hash)

    rows = await repo.get_files(project_id)
    assert [row.file_hash for row in rows] == [file_hash]


@pytest.mark.asyncio
async def test_effective_files_include_project_knowledge(session, sample_thread, sample_file):
    repo = ProjectFileRepository(session)
    project_id = sample_thread.project_id
    thread_id = sample_thread.id
    file_hash = sample_file.file_hash
    await session.rollback()
    await repo.add(project_id, file_hash)

    rows = await repo.get_effective_thread_files(thread_id)

    assert len(rows) == 1
    assert rows[0].association_scope == "project"
    assert rows[0].is_project_knowledge is True


@pytest.mark.asyncio
async def test_direct_association_wins_without_duplication(session, sample_thread, sample_file):
    project_repo = ProjectFileRepository(session)
    thread_repo = ThreadFileRepository(session)
    project_id = sample_thread.project_id
    thread_id = sample_thread.id
    file_hash = sample_file.file_hash
    await session.rollback()
    await project_repo.add(project_id, file_hash)
    await thread_repo.add(thread_id, file_hash)

    rows = await project_repo.get_effective_thread_files(thread_id)

    assert await project_repo.is_file_in_project_thread(project_id, file_hash)
    assert len(rows) == 1
    assert rows[0].association_scope == "thread"
    assert rows[0].is_project_knowledge is True


@pytest.mark.asyncio
async def test_project_removal_preserves_direct_association(session, sample_thread, sample_file):
    project_repo = ProjectFileRepository(session)
    thread_repo = ThreadFileRepository(session)
    project_id = sample_thread.project_id
    thread_id = sample_thread.id
    file_hash = sample_file.file_hash
    await session.rollback()
    await project_repo.add(project_id, file_hash)
    await thread_repo.add(thread_id, file_hash)

    assert await project_repo.remove(project_id, file_hash)
    rows = await project_repo.get_effective_thread_files(thread_id)

    assert len(rows) == 1
    assert rows[0].association_scope == "thread"
    assert rows[0].is_project_knowledge is False

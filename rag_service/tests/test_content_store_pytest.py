from __future__ import annotations

import hashlib
import os

import pytest

from app.services.content_store import (
    SharedVolumeContentStore,
    pdf_content_key,
    task_artifact_content_key,
)


@pytest.mark.asyncio
async def test_shared_volume_content_store_crud_and_hash_verification(tmp_path):
    store = SharedVolumeContentStore(tmp_path)
    body = b"durable research artifact"
    digest = hashlib.sha256(body).hexdigest()
    key = task_artifact_content_key("task", "run", "artifact")

    result = await store.put(key, body, expected_sha256=digest)

    assert result.size == len(body)
    assert result.sha256 == digest
    assert await store.exists(key) is True
    assert await store.read(key) == body
    assert (await store.stat(key)).sha256 == digest
    assert store.internal_path(key).is_file()
    assert await store.list_keys("agent-tasks") == [key]

    await store.delete(key)
    await store.delete(key)
    assert await store.exists(key) is False


@pytest.mark.asyncio
async def test_shared_volume_content_store_rejects_traversal_symlinks_and_bad_hashes(tmp_path):
    store = SharedVolumeContentStore(tmp_path)
    with pytest.raises(ValueError, match="normalized relative"):
        await store.put("../escape", b"no")
    with pytest.raises(ValueError, match="SHA-256"):
        await store.put("safe/object", b"body", expected_sha256="0" * 64)
    assert not (tmp_path / "safe" / "object").exists()

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "linked")
    with pytest.raises(ValueError, match="symbolic link"):
        await store.put("linked/object", b"no")


@pytest.mark.asyncio
async def test_shared_volume_content_store_put_if_absent_never_overwrites(tmp_path):
    store = SharedVolumeContentStore(tmp_path)
    key = task_artifact_content_key("task", "run", "artifact")

    created, first = await store.put_if_absent(key, b"first")
    reused, second = await store.put_if_absent(key, b"second")

    assert created is True
    assert reused is False
    assert first.sha256 == second.sha256
    assert await store.read(key) == b"first"


def test_content_key_contracts():
    assert pdf_content_key("0" * 32) == f"{'0' * 32}.pdf"
    with pytest.raises(ValueError, match="PDF content hash"):
        pdf_content_key("../../pdf")
    with pytest.raises(ValueError, match="artifact identity"):
        task_artifact_content_key("../task", "run", "artifact")

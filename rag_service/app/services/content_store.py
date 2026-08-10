from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


CONTENT_ROOT = Path(os.getenv("ASKPDF_CONTENT_ROOT", "/static"))
_PDF_HASH = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class ContentStat:
    size: int
    sha256: str


class ContentStore(ABC):
    @abstractmethod
    async def put(self, key: str, content: bytes | BinaryIO, *, expected_sha256: str | None = None) -> ContentStat: ...

    @abstractmethod
    async def read(self, key: str) -> bytes: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def stat(self, key: str) -> ContentStat: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def internal_path(self, key: str) -> Path: ...


class SharedVolumeContentStore(ContentStore):
    """Root-contained content storage for the single-host shared-volume deployment."""

    def __init__(self, root: Path | str = CONTENT_ROOT) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o750)

    def _resolve(self, key: str, *, create_parents: bool = False) -> Path:
        if not isinstance(key, str) or not key.strip() or "\\" in key:
            raise ValueError("content key is invalid")
        relative = PurePosixPath(key)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("content key must be a normalized relative path")
        candidate = self.root.joinpath(*relative.parts)
        current = self.root
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError("content key traverses a symbolic link")
            if create_parents:
                current.mkdir(mode=0o750, exist_ok=True)
        if candidate.is_symlink():
            raise ValueError("content key resolves to a symbolic link")
        resolved_parent = candidate.parent.resolve(strict=False)
        if resolved_parent != self.root and self.root not in resolved_parent.parents:
            raise ValueError("content key escapes the configured root")
        return candidate

    async def put(self, key: str, content: bytes | BinaryIO, *, expected_sha256: str | None = None) -> ContentStat:
        def write() -> ContentStat:
            target = self._resolve(key, create_parents=True)
            source = content if hasattr(content, "read") else None
            body = None if source is not None else bytes(content)
            digest = hashlib.sha256()
            size = 0
            fd, temporary_name = tempfile.mkstemp(prefix=".content-", dir=target.parent)
            try:
                os.fchmod(fd, 0o640)
                with os.fdopen(fd, "wb") as output:
                    if body is not None:
                        output.write(body)
                        digest.update(body)
                        size = len(body)
                    else:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                actual = digest.hexdigest()
                if expected_sha256 is not None and actual != expected_sha256:
                    raise ValueError("content SHA-256 does not match metadata")
                os.replace(temporary_name, target)
                return ContentStat(size=size, sha256=actual)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

        return await asyncio.to_thread(write)

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self._resolve(key).read_bytes)

    async def exists(self, key: str) -> bool:
        def check() -> bool:
            path = self._resolve(key)
            return path.is_file() and not path.is_symlink()

        return await asyncio.to_thread(check)

    async def stat(self, key: str) -> ContentStat:
        def inspect() -> ContentStat:
            path = self._resolve(key)
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as content:
                for chunk in iter(lambda: content.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            return ContentStat(size=size, sha256=digest.hexdigest())

        return await asyncio.to_thread(inspect)

    async def delete(self, key: str) -> None:
        def remove() -> None:
            path = self._resolve(key)
            try:
                path.unlink()
            except FileNotFoundError:
                return
            parent = path.parent
            while parent != self.root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

        await asyncio.to_thread(remove)

    async def list_keys(self, prefix: str = "") -> list[str]:
        def scan() -> list[str]:
            base = self.root if not prefix else self._resolve(prefix)
            if not base.exists():
                return []
            if base.is_symlink():
                raise ValueError("content prefix resolves to a symbolic link")
            return sorted(
                path.relative_to(self.root).as_posix()
                for path in base.rglob("*")
                if path.is_file() and not path.is_symlink()
            )

        return await asyncio.to_thread(scan)

    def internal_path(self, key: str) -> Path:
        return self._resolve(key)


_content_store: ContentStore | None = None


def get_content_store() -> ContentStore:
    global _content_store
    if _content_store is None:
        _content_store = SharedVolumeContentStore()
    return _content_store


def set_content_store(store: ContentStore | None) -> None:
    global _content_store
    _content_store = store


def pdf_content_key(file_hash: str) -> str:
    normalized = str(file_hash).strip().lower()
    if not _PDF_HASH.fullmatch(normalized):
        raise ValueError("invalid PDF content hash")
    return f"{normalized}.pdf"


def task_artifact_content_key(task_id: str, run_id: str, artifact_id: str, version: int = 1) -> str:
    values = (task_id, run_id, artifact_id)
    if any(not value or "/" in value or "\\" in value or value in {".", ".."} for value in values):
        raise ValueError("invalid task artifact identity")
    if version < 1:
        raise ValueError("artifact version must be positive")
    return f"agent-tasks/{task_id}/{run_id}/{artifact_id}/{version}"

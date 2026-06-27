"""Best-effort mirror writes from primary SQLModel Postgres to Supabase."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.db import supabase_client

logger = logging.getLogger(__name__)


def _dt(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    return value


async def _swallow(operation: str, coro):
    if not supabase_client.supabase_enabled():
        if hasattr(coro, "close"):
            coro.close()
        return None
    try:
        return await coro
    except Exception as exc:
        logger.warning("Supabase dual-write skipped for %s: %s", operation, exc)
        return None


async def mirror_thread(thread: Any) -> None:
    if not thread:
        return
    await _swallow(
        "thread",
        supabase_client.execute(
            """
            insert into public.threads (id, name, embed_model, settings, created_at, updated_at)
            values ($1, $2, $3, $4::jsonb, $5, $6)
            on conflict (id) do update
            set name = excluded.name,
                embed_model = excluded.embed_model,
                settings = excluded.settings,
                updated_at = excluded.updated_at
            """,
            thread.id,
            thread.name,
            thread.embed_model,
            supabase_client.json_arg(getattr(thread, "settings", {}) or {}),
            _dt(getattr(thread, "created_at", None)),
            _dt(getattr(thread, "updated_at", None)),
        ),
    )


async def mirror_thread_settings(thread_id: str, settings: dict[str, Any]) -> None:
    await _swallow(
        "thread_settings",
        supabase_client.execute(
            """
            update public.threads
            set settings = $2::jsonb, updated_at = now()
            where id = $1
            """,
            thread_id,
            supabase_client.json_arg(settings or {}),
        ),
    )


async def delete_thread(thread_id: str) -> None:
    await _swallow("delete_thread", supabase_client.execute("delete from public.threads where id = $1", thread_id))


async def mirror_file(file: Any) -> None:
    if not file:
        return
    await _swallow(
        "file",
        supabase_client.execute(
            """
            insert into public.files (
              file_hash, file_name, file_path, source_type, file_status,
              parsed_sentences_json, created_at
            )
            values ($1, $2, $3, $4, $5::jsonb, $6, $7)
            on conflict (file_hash) do update
            set file_name = excluded.file_name,
                file_path = coalesce(excluded.file_path, public.files.file_path),
                source_type = excluded.source_type,
                file_status = excluded.file_status,
                parsed_sentences_json = excluded.parsed_sentences_json
            """,
            file.file_hash,
            file.file_name,
            getattr(file, "file_path", None),
            getattr(file, "source_type", "pdf"),
            supabase_client.json_arg(getattr(file, "file_status", {}) or {}),
            getattr(file, "parsed_sentences_json", None),
            _dt(getattr(file, "created_at", None)),
        ),
    )


async def update_file_status(file_hash: str, status: dict[str, Any]) -> None:
    await _swallow(
        "file_status",
        supabase_client.execute(
            "update public.files set file_status = $2::jsonb where file_hash = $1",
            file_hash,
            supabase_client.json_arg(status or {}),
        ),
    )


async def update_file_parsed_sentences(file_hash: str, parsed_data_json: str) -> None:
    await _swallow(
        "file_parsed_sentences",
        supabase_client.execute(
            "update public.files set parsed_sentences_json = $2 where file_hash = $1",
            file_hash,
            parsed_data_json,
        ),
    )


async def delete_file(file_hash: str) -> None:
    await _swallow("delete_file", supabase_client.execute("delete from public.files where file_hash = $1", file_hash))


async def add_thread_file(thread_id: str, file_hash: str, added_at: Any = None) -> None:
    await _swallow(
        "thread_file",
        supabase_client.execute(
            """
            insert into public.thread_files (thread_id, file_hash, added_at)
            values ($1, $2, coalesce($3, now()))
            on conflict (thread_id, file_hash) do update
            set added_at = public.thread_files.added_at
            """,
            thread_id,
            file_hash,
            _dt(added_at),
        ),
    )


async def remove_thread_file(thread_id: str, file_hash: str) -> None:
    await _swallow(
        "remove_thread_file",
        supabase_client.execute(
            "select public.remove_thread_file_pair($1, $2)",
            thread_id,
            file_hash,
        ),
    )


async def mirror_annotations(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    await _swallow(
        "annotations",
        supabase_client.execute(
            """
            insert into public.thread_file_annotations (
              thread_id, file_hash, annotations_json, created_at, updated_at
            )
            values ($1, $2, $3, coalesce($4, now()), coalesce($5, now()))
            on conflict (thread_id, file_hash) do update
            set annotations_json = excluded.annotations_json,
                updated_at = excluded.updated_at
            """,
            payload["thread_id"],
            payload["file_hash"],
            supabase_client.json_arg(payload.get("annotations", [])),
            _dt(payload.get("created_at")),
            _dt(payload.get("updated_at")),
        ),
    )


async def delete_annotations(thread_id: str, file_hash: str | None = None) -> None:
    if file_hash:
        sql = "delete from public.thread_file_annotations where thread_id = $1 and file_hash = $2"
        await _swallow("delete_annotations", supabase_client.execute(sql, thread_id, file_hash))
    else:
        await _swallow(
            "delete_annotations",
            supabase_client.execute("delete from public.thread_file_annotations where thread_id = $1", thread_id),
        )


async def mirror_message(message: Any) -> None:
    if not message:
        return
    role = getattr(message, "role", None)
    role = role.value if hasattr(role, "value") else role
    web_sources = getattr(message, "web_sources", None)
    await _swallow(
        "message",
        supabase_client.execute(
            """
            insert into public.messages (
              id, thread_id, role, content, context_compact, reasoning,
              reasoning_available, reasoning_format, web_sources, created_at
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
            on conflict (id) do update
            set content = excluded.content,
                context_compact = excluded.context_compact,
                reasoning = excluded.reasoning,
                reasoning_available = excluded.reasoning_available,
                reasoning_format = excluded.reasoning_format,
                web_sources = excluded.web_sources
            """,
            message.id,
            message.thread_id,
            role,
            message.content,
            getattr(message, "context_compact", None),
            getattr(message, "reasoning", None),
            bool(getattr(message, "reasoning_available", False)),
            getattr(message, "reasoning_format", "none"),
            supabase_client.json_arg(web_sources) if web_sources is not None else None,
            _dt(getattr(message, "created_at", None)),
        ),
    )


async def update_message_context(message_id: str, context_compact: str) -> None:
    await _swallow(
        "message_context",
        supabase_client.execute(
            "update public.messages set context_compact = $2 where id = $1",
            message_id,
            context_compact,
        ),
    )


async def delete_message(message_id: str) -> None:
    await _swallow("delete_message", supabase_client.execute("delete from public.messages where id = $1", message_id))


async def delete_message_pair(message_id: str) -> None:
    await _swallow(
        "delete_message_pair",
        supabase_client.execute("select public.delete_message_pair($1)", message_id),
    )


async def upsert_stats(thread_id: str, stats: dict[str, Any]) -> None:
    await _swallow(
        "thread_stats",
        supabase_client.execute(
            """
            insert into public.thread_stats (
              thread_id, total_qa_pairs, total_qa_chars, avg_qa_chars,
              last_qa_at, documents_meta, last_updated_at
            )
            values ($1, $2, $3, $4, $5, $6::jsonb, coalesce($7::timestamptz, now()))
            on conflict (thread_id) do update
            set total_qa_pairs = excluded.total_qa_pairs,
                total_qa_chars = excluded.total_qa_chars,
                avg_qa_chars = excluded.avg_qa_chars,
                last_qa_at = excluded.last_qa_at,
                documents_meta = excluded.documents_meta,
                last_updated_at = excluded.last_updated_at
            """,
            thread_id,
            int(stats.get("total_qa_pairs", 0) or 0),
            int(stats.get("total_qa_chars", 0) or 0),
            float(stats.get("avg_qa_chars", 0.0) or 0.0),
            _dt(stats.get("last_qa_at")),
            supabase_client.json_arg(stats.get("documents") or stats.get("documents_meta") or {}),
            _dt(stats.get("last_updated_at")),
        ),
    )


async def recompute_qa_stats(thread_id: str) -> None:
    await _swallow(
        "recompute_qa_stats",
        supabase_client.execute("select public.recompute_qa_stats($1)", thread_id),
    )

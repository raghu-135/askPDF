"""Harden durable memory constraints

Revision ID: f6a1b2c3d4e5
Revises: e5a9c2d7b8f1
Create Date: 2026-07-25 00:00:00.000000

"""
from __future__ import annotations

from alembic import op


revision = "f6a1b2c3d4e5"
down_revision = "e5a9c2d7b8f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_projects_embedding_model_nonempty",
        "projects",
        "length(btrim(embedding_model)) > 0",
    )
    op.execute(
        """
        create function prevent_project_embedding_model_change()
        returns trigger as $$
        begin
            if new.embedding_model is distinct from old.embedding_model then
                raise exception 'project embedding_model is immutable';
            end if;
            return new;
        end;
        $$ language plpgsql
        """
    )
    op.execute(
        """
        create trigger trg_projects_embedding_model_immutable
        before update of embedding_model on projects
        for each row execute function prevent_project_embedding_model_change()
        """
    )
    op.create_check_constraint(
        "ck_memories_scope_type",
        "memories",
        "scope_type in ('user', 'project', 'thread')",
    )
    op.create_check_constraint(
        "ck_memories_memory_type",
        "memories",
        "memory_type in ('semantic', 'episodic', 'procedural')",
    )
    op.create_check_constraint(
        "ck_memories_status",
        "memories",
        "status = 'active'",
    )
    op.create_check_constraint(
        "ck_memories_visibility",
        "memories",
        "visibility in ('private', 'project', 'internal')",
    )
    op.create_check_constraint(
        "ck_memories_confidence",
        "memories",
        "confidence >= 0 and confidence <= 1",
    )
    op.create_check_constraint(
        "ck_memories_scope_id_nonempty",
        "memories",
        "length(btrim(scope_id)) > 0",
    )
    op.create_check_constraint(
        "ck_memories_content_nonempty",
        "memories",
        "length(btrim(content)) > 0",
    )
    op.create_check_constraint(
        "ck_memories_embedding_model_nonempty",
        "memories",
        "length(btrim(embedding_model)) > 0",
    )
    op.create_check_constraint(
        "ck_memories_content_hash_nonempty",
        "memories",
        "length(btrim(content_hash)) > 0",
    )
    op.create_check_constraint(
        "ck_memories_index_status",
        "memories",
        "index_status in ('pending', 'indexing', 'indexed', 'failed')",
    )
    op.create_check_constraint(
        "ck_memories_index_attempts",
        "memories",
        "index_attempts >= 0",
    )
    op.create_check_constraint(
        "ck_memory_candidates_scope_type",
        "memory_candidates",
        "proposed_scope_type in ('user', 'project', 'thread')",
    )
    op.create_check_constraint(
        "ck_memory_candidates_memory_type",
        "memory_candidates",
        "memory_type in ('semantic', 'episodic', 'procedural')",
    )
    op.create_check_constraint(
        "ck_memory_candidates_status",
        "memory_candidates",
        "status in ('pending', 'approved', 'rejected', 'auto_approved')",
    )
    op.create_check_constraint(
        "ck_memory_candidates_confidence",
        "memory_candidates",
        "confidence >= 0 and confidence <= 1",
    )
    op.create_check_constraint(
        "ck_memory_candidates_scope_id_nonempty",
        "memory_candidates",
        "length(btrim(proposed_scope_id)) > 0",
    )
    op.create_check_constraint(
        "ck_memory_candidates_content_nonempty",
        "memory_candidates",
        "length(btrim(content)) > 0",
    )


def downgrade() -> None:
    for name, table in (
        ("ck_memories_index_attempts", "memories"),
        ("ck_memories_index_status", "memories"),
        ("ck_memories_content_hash_nonempty", "memories"),
        ("ck_memories_embedding_model_nonempty", "memories"),
        ("ck_memory_candidates_content_nonempty", "memory_candidates"),
        ("ck_memory_candidates_scope_id_nonempty", "memory_candidates"),
        ("ck_memory_candidates_confidence", "memory_candidates"),
        ("ck_memory_candidates_status", "memory_candidates"),
        ("ck_memory_candidates_memory_type", "memory_candidates"),
        ("ck_memory_candidates_scope_type", "memory_candidates"),
        ("ck_memories_content_nonempty", "memories"),
        ("ck_memories_scope_id_nonempty", "memories"),
        ("ck_memories_confidence", "memories"),
        ("ck_memories_visibility", "memories"),
        ("ck_memories_status", "memories"),
        ("ck_memories_memory_type", "memories"),
        ("ck_memories_scope_type", "memories"),
    ):
        op.drop_constraint(name, table, type_="check")
    op.execute("drop trigger if exists trg_projects_embedding_model_immutable on projects")
    op.execute("drop function if exists prevent_project_embedding_model_change()")
    op.drop_constraint("ck_projects_embedding_model_nonempty", "projects", type_="check")

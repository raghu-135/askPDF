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
        "status in ('active', 'archived', 'deleted', 'rejected')",
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

"""Reconcile databases stamped before the project-memory migrations were reshaped.

Revision ID: a7c4e9f2b1d6
Revises: f6a1b2c3d4e5
Create Date: 2026-07-25 00:00:00.000000

"""
from __future__ import annotations

import hashlib
import uuid

from alembic import op
import sqlalchemy as sa


revision = "a7c4e9f2b1d6"
down_revision = "f6a1b2c3d4e5"
branch_labels = None
depends_on = None


PROJECT_MODEL = "BAAI/bge-m3"


def _column_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def _constraint_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    names = {
        constraint.get("name")
        for constraint in inspector.get_check_constraints(table_name)
    }
    names.update(
        constraint.get("name")
        for constraint in inspector.get_foreign_keys(table_name)
    )
    names.update(
        constraint.get("name")
        for constraint in inspector.get_unique_constraints(table_name)
    )
    return {name for name in names if name}


def _drop_constraint_if_present(
    bind: sa.Connection,
    table_name: str,
    constraint_name: str,
    constraint_type: str,
) -> None:
    if constraint_name in _constraint_names(bind, table_name):
        op.drop_constraint(constraint_name, table_name, type_=constraint_type)


def upgrade() -> None:
    bind = op.get_bind()

    project_columns = _column_names(bind, "projects")
    if "embedding_model" not in project_columns:
        op.add_column(
            "projects",
            sa.Column(
                "embedding_model",
                sa.String(),
                nullable=True,
                server_default=PROJECT_MODEL,
            ),
        )

    op.execute(
        sa.text(
            """
            update projects
            set embedding_model = :model
            where embedding_model is null or btrim(embedding_model) = ''
            """
        ).bindparams(model=PROJECT_MODEL)
    )
    op.alter_column(
        "projects",
        "embedding_model",
        existing_type=sa.String(),
        nullable=False,
        server_default=None,
    )

    project_count = bind.scalar(sa.text("select count(*) from projects"))
    if not project_count:
        bind.execute(
            sa.text(
                """
                insert into projects
                    (id, name, description, embedding_model, settings_json, created_at)
                values
                    (:id, 'Personal', 'Default project.', :model, '{}'::jsonb, now())
                """
            ),
            {"id": str(uuid.uuid4()), "model": PROJECT_MODEL},
        )

    default_project_id = bind.scalar(
        sa.text(
            """
            select id
            from projects
            order by case when name = 'Personal' then 0 else 1 end, created_at, id
            limit 1
            """
        )
    )
    op.execute(
        sa.text("update threads set project_id = :project_id where project_id is null")
        .bindparams(project_id=default_project_id)
    )
    op.execute(
        """
        update threads as t
        set embedding_model = p.embedding_model
        from projects as p
        where p.id = t.project_id
          and t.embedding_model is distinct from p.embedding_model
        """
    )
    op.alter_column(
        "threads",
        "project_id",
        existing_type=sa.String(),
        nullable=False,
    )

    _drop_constraint_if_present(
        bind,
        "threads",
        "fk_threads_project_id_projects",
        "foreignkey",
    )
    _drop_constraint_if_present(
        bind,
        "threads",
        "fk_threads_project_embedding_model",
        "foreignkey",
    )
    if "uq_projects_id_embedding_model" not in _constraint_names(bind, "projects"):
        op.create_unique_constraint(
            "uq_projects_id_embedding_model",
            "projects",
            ["id", "embedding_model"],
        )
    op.create_foreign_key(
        "fk_threads_project_embedding_model",
        "threads",
        "projects",
        ["project_id", "embedding_model"],
        ["id", "embedding_model"],
        ondelete="RESTRICT",
    )

    memory_columns = _column_names(bind, "memories")
    memory_additions = (
        ("embedding_model", sa.String(), None),
        ("content_hash", sa.String(), None),
        ("index_status", sa.String(), "pending"),
        ("index_attempts", sa.Integer(), "0"),
        ("indexed_at", sa.DateTime(timezone=True), None),
        ("index_error", sa.String(), None),
    )
    for name, column_type, default in memory_additions:
        if name not in memory_columns:
            op.add_column(
                "memories",
                sa.Column(
                    name,
                    column_type,
                    nullable=True,
                    server_default=default,
                ),
            )

    op.execute(
        sa.text(
            """
            update memories as m
            set embedding_model = case
                when m.scope_type = 'thread' then coalesce(
                    (select t.embedding_model from threads as t where t.id = m.scope_id),
                    :model
                )
                when m.scope_type = 'project' then coalesce(
                    (select p.embedding_model from projects as p where p.id = m.scope_id),
                    :model
                )
                else :model
            end
            where m.embedding_model is null or btrim(m.embedding_model) = ''
            """
        ).bindparams(model=PROJECT_MODEL)
    )
    missing_hashes = bind.execute(
        sa.text(
            """
            select id, content
            from memories
            where content_hash is null or btrim(content_hash) = ''
            """
        )
    ).all()
    for memory_id, content in missing_hashes:
        bind.execute(
            sa.text("update memories set content_hash = :content_hash where id = :id"),
            {
                "id": memory_id,
                "content_hash": hashlib.sha256((content or "").encode("utf-8")).hexdigest(),
            },
        )
    op.execute("update memories set index_status = 'pending' where index_status is null")
    op.execute("update memories set index_attempts = 0 where index_attempts is null")

    # The redesigned lifecycle has no archived/rejected durable rows.
    op.execute("delete from memories where status <> 'active'")

    for name, column_type in (
        ("embedding_model", sa.String()),
        ("content_hash", sa.String()),
        ("index_status", sa.String()),
        ("index_attempts", sa.Integer()),
    ):
        op.alter_column(
            "memories",
            name,
            existing_type=column_type,
            nullable=False,
            server_default=None,
        )

    candidate_columns = _column_names(bind, "memory_candidates")
    for name, column_type in (
        ("promoted_memory_id", sa.String()),
        ("resolved_by", sa.String()),
        ("resolved_at", sa.DateTime(timezone=True)),
    ):
        if name not in candidate_columns:
            op.add_column(
                "memory_candidates",
                sa.Column(name, column_type, nullable=True),
            )
    if (
        "fk_memory_candidates_promoted_memory_id_memories"
        not in _constraint_names(bind, "memory_candidates")
    ):
        op.create_foreign_key(
            "fk_memory_candidates_promoted_memory_id_memories",
            "memory_candidates",
            "memories",
            ["promoted_memory_id"],
            ["id"],
            ondelete="SET NULL",
        )

    for constraint_name, table_name, expression in (
        (
            "ck_projects_embedding_model_nonempty",
            "projects",
            "length(btrim(embedding_model)) > 0",
        ),
        ("ck_memories_status", "memories", "status = 'active'"),
        (
            "ck_memories_embedding_model_nonempty",
            "memories",
            "length(btrim(embedding_model)) > 0",
        ),
        (
            "ck_memories_content_hash_nonempty",
            "memories",
            "length(btrim(content_hash)) > 0",
        ),
        (
            "ck_memories_index_status",
            "memories",
            "index_status in ('pending', 'indexing', 'indexed', 'failed')",
        ),
        ("ck_memories_index_attempts", "memories", "index_attempts >= 0"),
    ):
        _drop_constraint_if_present(
            bind,
            table_name,
            constraint_name,
            "check",
        )
        op.create_check_constraint(
            constraint_name,
            table_name,
            expression,
        )

    op.execute(
        """
        create or replace function prevent_project_embedding_model_change()
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
    op.execute("drop trigger if exists trg_projects_embedding_model_immutable on projects")
    op.execute(
        """
        create trigger trg_projects_embedding_model_immutable
        before update of embedding_model on projects
        for each row execute function prevent_project_embedding_model_change()
        """
    )

    for statement in (
        "create index if not exists ix_projects_embedding_model on projects (embedding_model)",
        "create index if not exists ix_memories_embedding_model on memories (embedding_model)",
        "create index if not exists ix_memories_content_hash on memories (content_hash)",
        "create index if not exists ix_memories_index_status on memories (index_status)",
        "create index if not exists idx_memory_index_retry on memories (index_status, updated_at)",
        "create index if not exists ix_memory_candidates_promoted_memory_id on memory_candidates (promoted_memory_id)",
        "create index if not exists ix_memory_candidates_resolved_by on memory_candidates (resolved_by)",
    ):
        op.execute(statement)


def downgrade() -> None:
    # This revision reconciles two historical physical schemas at the same
    # Alembic revision. Reversing it cannot be made deterministic.
    raise RuntimeError("Schema reconciliation migration cannot be downgraded")
